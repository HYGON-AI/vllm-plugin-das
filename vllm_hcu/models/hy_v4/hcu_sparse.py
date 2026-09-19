# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sink-capable FlashMLA sparse backend for HY V4 on HCU.

HY V4 adds a per-head learnable attention sink on top of sparse MLA. The
vendored FlashMLA kernels already accept an ``attn_sink`` argument, but vLLM's
shared ``FLASHMLA_SPARSE`` backend neither advertises sink support nor forwards
the tensor, so the bias would be silently dropped.

This module supplies the missing wiring inside the model package, mirroring how
`vllm.models.deepseek_v4.nvidia.flashinfer_sparse` hands ``sinks`` to the
FlashInfer sparse MLA kernels: subclass the platform backend, declare
`supports_sink`, and thread the sink into every kernel call.

The subclass intentionally keeps the inherited ``get_name()``
(``"FLASHMLA_SPARSE"``). Several shared code paths key off that exact string —
`_canonicalize_sparse_mla_kv_cache_dtype` promotes a quantized KV cache to
``fp8_ds_mla`` for it, and `FlashMLASparseImpl` asserts that layout — so a new
name would silently change KV cache behaviour. Only``supports_sink`` and the
two kernel wrappers differ from the parent.
"""

import math
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import get_dcp_group
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_filter_and_convert_dcp_index,
)
import vllm_hcu.platforms.envs as henvs
from vllm_hcu.models.hy_v4.fp8_kv_dequant import gather_dequantize_fp8_ds_mla_cache
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseImpl,
    FlashMLASparseMetadata,
)
from vllm_hcu.v1.attention.backends.mla.flashmla_sparse import (
    HcuFlashMLASparseBackend,
)
from vllm_hcu.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)


def _tokens_per_request(speculative_config: object | None) -> int:
    """Return decode rows per request, including the current token."""
    num_speculative_tokens = int(
        getattr(speculative_config, "num_speculative_tokens", 0) or 0
    )
    return num_speculative_tokens + 1 if num_speculative_tokens > 0 else 1


class HYV4FlashMLASparseImpl(FlashMLASparseImpl):
    """FlashMLA sparse impl that applies HY V4's per-head learnable sink.

    The sink enters as the ``sinks`` impl kwarg of
    `vllm.model_executor.layers.attention.MLAAttention` and is consumed by the
    FlashMLA kernels, which fold it into the softmax denominator:
    ``out *= exp(lse) / (exp(lse) + exp(sink))``.
    """

    supports_pcp: bool = True
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        # ``SparseMLACommonImpl`` takes explicit keyword arguments only, so the
        # sink has to be removed before the base classes see``mla_args``.
        sinks: torch.Tensor | None = mla_args.pop("sinks", None)
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            topk_indices_buffer=topk_indices_buffer,
            indexer=indexer,
            **mla_args,
        )
        self._validate_sinks(sinks, num_heads)
        self.sinks = sinks
        self._dcp_sinks: torch.Tensor | None = None
        # MTP width: query rows per request in a decode batch. LightOp's gather
        # groups ``num_tokens`` rows into requests with this stride. MTP3 has
        # three speculative rows plus the current decode row, so it passes 4;
        # plain decoding passes 1.
        speculative_config = getattr(
            get_current_vllm_config(), "speculative_config", None
        )
        self.tokens_per_request = _tokens_per_request(speculative_config)

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Prepare MLA weights and gather static sink shards for DCP."""
        super().process_weights_after_loading(act_dtype)
        self._dcp_sinks = None
        if self.sinks is None or self.dcp_world_size <= 1:
            return

        gathered_sinks = get_dcp_group().all_gather(self.sinks, dim=0)
        expected_heads = self.num_heads * self.dcp_world_size
        if gathered_sinks.ndim != 1 or gathered_sinks.shape[0] != expected_heads:
            raise ValueError(
                "HYV4 FlashMLA DCP attention sinks must gather to shape "
                f"({expected_heads},), but got {tuple(gathered_sinks.shape)}."
            )
        self._dcp_sinks = gathered_sinks

    @staticmethod
    def _validate_sinks(sinks: torch.Tensor | None, num_heads: int) -> None:
        """Reject sink tensors the FlashMLA kernels cannot consume.

        Args:
            sinks: Candidate sink tensor, or None when the layer has no sink.
            num_heads: Local (TP-sharded) query head count of this layer.

        Raises:
            ValueError: If the dtype is not float32 or the shape is not
                ``(num_heads,)``.
        """
        if sinks is None:
            return
        if sinks.dtype != torch.float32:
            raise ValueError(
                "HYV4 FlashMLA sparse attention sinks must have dtype "
                f"torch.float32, but got {sinks.dtype}."
            )
        if sinks.ndim != 1 or sinks.shape[0] != num_heads:
            raise ValueError(
                "HYV4 FlashMLA sparse attention sinks must have shape "
                f"({num_heads},), but got {tuple(sinks.shape)}."
            )

    def _sinks_for_query(
        self,
        q: torch.Tensor,
        head_dim: int,
        kernel_heads: int,
    ) -> torch.Tensor | None:
        """Return the sink laid out for the kernel's query head count.

        Args:
            q: Query tensor, before any head padding.
            head_dim: Axis of ``q`` holding the query heads.
            kernel_heads: Head count the kernel is invoked with, which may
                exceed the query head count because of padding.

        Returns:
            The sink tensor padded to ``kernel_heads`` with ``-inf`` (a no-op
            sink) for the padded lanes, or None when the layer has no sink.

        Raises:
            ValueError: If the sink and query head layouts disagree, or if they
                live on different devices.
        """
        sinks = self.sinks
        if sinks is None:
            return None

        query_heads = q.shape[head_dim]
        dcp_sinks = getattr(self, "_dcp_sinks", None)
        if dcp_sinks is not None and dcp_sinks.shape[0] == query_heads:
            # Each DCP rank includes the virtual sink in its local softmax.
            # Divide its exponential contribution across ranks so the later
            # LSE reduction counts that shared sink exactly once.
            sinks = dcp_sinks - math.log(self.dcp_world_size)
        if sinks.shape[0] != query_heads:
            raise ValueError(
                "HYV4 FlashMLA sparse attention sink head count must match the "
                f"runtime query layout: sinks={sinks.shape[0]}, "
                f"query_heads={query_heads}. The sink must use the same "
                "unpadded head layout as the query."
            )
        if sinks.device != q.device:
            raise ValueError(
                "HYV4 FlashMLA sparse attention sinks and query must be on the "
                f"same device, but got sinks={sinks.device}, query={q.device}."
            )
        if kernel_heads < query_heads:
            raise ValueError(
                "HYV4 FlashMLA sparse kernel head count cannot be smaller than "
                f"the runtime query layout: query_heads={query_heads}, "
                f"kernel_heads={kernel_heads}."
            )
        if kernel_heads == query_heads:
            return sinks

        # Mirror the query padding the kernels require. Reading ``sinks`` here
        # (rather than caching a padded copy at construction time) keeps the
        # values correct for weights loaded after the module is built, and the
        # allocation plus copy are captured in the CUDA graph.
        padded_sinks = sinks.new_full((kernel_heads,), float("-inf"))
        padded_sinks[:query_heads] = sinks
        return padded_sinks

    def _fp8_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        kernel_metadata: FlashMLASparseMetadata.FP8KernelMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # The fp8 FlashMLA kernel hardcodes DeepSeek's fp8_ds_mla geometry
        # (pe_dim == 64). When VLLM_HCU_HYV4_FP8_KV_DEQUANT is set, dequantize
        # the fp8 KV cache to BF16 and run the BF16 sparse kernel instead. In
        # HY V4's mixed-batch mode this interception point handles both prefill
        # and decode tokens, so the env var covers both paths.
        if henvs.VLLM_HCU_HYV4_FP8_KV_DEQUANT:
            return self._dequant_bf16_attn(q, kv_c_and_k_pe_cache, topk_indices)

        # q shape: (batch, seq_len, num_heads, head_dim)
        actual_num_heads = q.size(2)
        padded_num_heads = self.fp8_decode_padded_heads
        attn_sink = self._sinks_for_query(q, head_dim=2, kernel_heads=padded_num_heads)

        # Pad query if needed (kernel only supports h_q = 64 or 128)
        if actual_num_heads < padded_num_heads:
            logger.warning_once(
                f"Padding num_heads from {actual_num_heads} to "
                f"{padded_num_heads} for FP8 sparse decode kernel"
            )
            q_padded = q.new_zeros((q.size(0), q.size(1), padded_num_heads, q.size(3)))
            q_padded[:, :, :actual_num_heads, :] = q
            q = q_padded

        out, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
            block_table=kernel_metadata.dummy_block_table,
            head_dim_v=512,
            cache_seqlens=kernel_metadata.cache_lens,
            tile_scheduler_metadata=kernel_metadata.scheduler_metadata,
            is_fp8_kvcache=True,
            indices=topk_indices,
            softmax_scale=self.softmax_scale,
            attn_sink=attn_sink,
        )

        # Slice output back to actual head count if we padded
        if actual_num_heads < padded_num_heads:
            out = out[:, :, :actual_num_heads, :]

        return out, lse

    def _dequant_bf16_attn(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Dequantize the fp8 KV cache to BF16 and run the BF16 sparse kernel.

        Replaces the fp8 FlashMLA kernel (which hardcodes the fp8_ds_mla
        geometry) with the geometry-agnostic BF16 sparse kernel. In HY V4's
        mixed-batch mode this runs for prefill and decode tokens alike, so both
        paths go through ``flash_mla_sparse_fwd`` on the upconverted cache.

        Only the ``num_tokens * topk`` slots the sparse kernel actually reads are
        dequantized, into a compact ``(num_tokens * topk, head_size)`` BF16
        buffer, and ``topk_indices`` (already converted to global cache slots,
        with -1 marking invalid entries) is remapped to that buffer's rows.
        Both ``topk`` and ``num_tokens`` are fixed per step, so the buffer shape
        is fixed and the decode path stays CUDA-graph-capturable.
        ``flash_mla_sparse_fwd`` handles the -1 indices natively.
        """
        num_tokens_b, seq_len, num_heads, head_dim = q.shape
        rope_dim = self.head_size - self.kv_lora_rank

        # Flatten tokens; indices are already global cache slots.
        q_flat = q.reshape(num_tokens_b * seq_len, num_heads, head_dim)
        idx_flat = topk_indices.reshape(num_tokens_b * seq_len, -1)

        # Gather + dequantize only the selected slots; remap indices to the
        # compact buffer's rows.
        kv_bf16, new_idx = gather_dequantize_fp8_ds_mla_cache(
            kv_c_and_k_pe_cache,
            idx_flat,
            self.kv_lora_rank,
            rope_dim,
            self.tokens_per_request,
        )

        out = self._bf16_flash_mla_kernel(q_flat, kv_bf16, new_idx)
        out = out.reshape(num_tokens_b, seq_len, num_heads, out.shape[-1])
        return out, None

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output, _ = self._bf16_flash_mla_kernel_with_lse(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            topk_length=topk_length,
        )
        return output

    def _bf16_flash_mla_kernel_with_lse(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        query_heads = q.shape[1]
        needs_padding = query_heads % self.prefill_padding != 0
        kernel_heads = (
            math.ceil(query_heads / self.prefill_padding) * self.prefill_padding
            if needs_padding
            else query_heads
        )
        attn_sink = self._sinks_for_query(q, head_dim=1, kernel_heads=kernel_heads)

        # NOTE(Chen): kernel requires num_local_head to be a multiple of
        # 64 on hopper and 128 on blackwell
        if needs_padding:
            logger.warning_once(
                f"Padding num_heads from {query_heads} to "
                f"{kernel_heads} for BF16 sparse prefill kernel"
            )
            # Zero (not new_empty) the padded lanes: topk_indices is shared by
            # all heads, so the kernel reduces across the head group and NaNs
            # from uninitialized memory would leak into the real heads.
            q_padded = q.new_zeros((q.shape[0], kernel_heads, q.shape[2]))
            q_padded[:, :query_heads, :] = q
            q = q_padded

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output, _, lse = flash_mla_sparse_fwd(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            self.softmax_scale,
            attn_sink=attn_sink,
            topk_length=topk_length,
        )
        if lse is None:
            raise RuntimeError("HYV4 sparse MLA DCP kernel did not return LSE")

        output = output[:, :query_heads, :]
        lse = lse[:, :query_heads]
        if attn_sink is not None:
            # FlashMLA applies the sink denominator to `output` but documents
            # that its returned LSE excludes the sink. DCP correction needs
            # the denominator that produced `output`, so fold the same sink
            # logit into LSE before the cross-rank log-sum-exp reduction.
            lse = torch.logaddexp(
                lse,
                attn_sink[:query_heads].view(1, query_heads),
            )
        return output, lse

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.dcp_world_size <= 1:
            return super().forward_mqa(
                q,
                kv_c_and_k_pe_cache,
                attn_metadata,
                layer,
            )

        if isinstance(q, tuple):
            from vllm import _custom_ops as ops

            ql_nope, q_pe = q
            q = self.q_concat_buffer[: ql_nope.shape[0]]
            ops.concat_mla_q(ql_nope, q_pe, q)

        num_actual_toks = q.shape[0]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        topk_indices, topk_length = triton_filter_and_convert_dcp_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            dcp_size=self.dcp_world_size,
            dcp_rank=self.dcp_rank,
            cp_kv_cache_interleave_size=(
                attn_metadata.cp_kv_cache_interleave_size
            ),
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )

        if self.kv_cache_dtype == "fp8_ds_mla":
            rope_dim = self.head_size - self.kv_lora_rank
            cache, kernel_indices = gather_dequantize_fp8_ds_mla_cache(
                kv_c_and_k_pe_cache,
                topk_indices,
                self.kv_lora_rank,
                rope_dim,
                self.tokens_per_request,
            )
        else:
            cache = kv_c_and_k_pe_cache
            kernel_indices = topk_indices

        attn_out, lse = self._bf16_flash_mla_kernel_with_lse(
            q,
            cache,
            kernel_indices,
            topk_length=topk_length,
        )
        empty_rows = topk_length == 0
        attn_out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
        if self.sinks is None:
            lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
        return attn_out, lse


class HYV4FlashMLASparseBackend(HcuFlashMLASparseBackend):
    """``FLASHMLA_SPARSE`` with attention-sink support for HY V4.

    Keeps the parent's name, metadata and builder; only the impl class and the
    sink capability differ. See the module docstring for why the name is reused.
    """

    @staticmethod
    def get_impl_cls() -> type[HYV4FlashMLASparseImpl]:
        return HYV4FlashMLASparseImpl

    @classmethod
    def supports_sink(cls) -> bool:
        return True
