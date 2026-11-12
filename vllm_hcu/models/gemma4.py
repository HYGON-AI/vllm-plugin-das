# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Opt-in Gemma4 FA backend and model-local registration."""

from __future__ import annotations

import functools
import os
from types import ModuleType

import torch
from vllm import envs
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionType, CommonAttentionMetadata
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVQuantMode

logger = init_logger(__name__)

# Native paged-KV layout selectors accepted by pagedkv_to_contiguouskv.
_NATIVE_LAYOUT_HND = 1  # [num_blocks, num_kv_heads, block_size, head_dim]
_NATIVE_LAYOUT_NHD = 2  # [num_blocks, block_size, num_kv_heads, head_dim]

# Attribute used to carry the CPU-side seq-len upper bound to the impl.
_GEMMA4_SEQ_LENS_BOUND = "_vllm_hcu_seq_lens_cpu_upper_bound"


_NATIVE_WARP_SIZE = 64  # paged_attention.h; gfx936 wave size


def _gemma4_bshd_stride_is_zero(row_width: int, itemsize: int) -> bool:
    """Would ``write_one_block_bshd`` advance by zero for this row width?

    The NHD selector routes to ``write_one_block_bshd``, whose loop reads

        vec_size      = 16 / sizeof(cache_t)
        thread_per_dim = head_size / vec_size
        i += WARP_SIZE / thread_per_dim * num_head * head_size

    with integer division. ``head_size`` here is the gather's row width, and
    this model's pool interleaves K and V, so it is ``2 * head_size`` = 1024.
    At one byte per element that gives thread_per_dim 64 and a stride of one
    row group; at two bytes it gives 128, and ``WARP_SIZE / 128`` truncates to
    zero -- the loop then never advances and the worker hangs on a non-empty
    page. ``write_one_block_bhsd``, which HND uses, strides by
    ``WARP_SIZE * vec_size`` and cannot reach zero, so this applies to NHD
    only.
    """

    vec_size = 16 // itemsize
    if vec_size <= 0:
        return True
    thread_per_dim = row_width // vec_size
    return thread_per_dim > _NATIVE_WARP_SIZE


def _gemma4_paged_view(cache: torch.Tensor) -> tuple[torch.Tensor, int] | None:
    """Describe the pool to the gather in the order it is actually stored in.

    The per-layer view is always [blocks, kv_heads, block_size, head_dim]
    whatever the resolved KV cache layout is; only the strides differ. LBNHC
    keeps tokens before heads in memory, so the view transposed back to
    [blocks, block_size, kv_heads, head_dim] is the contiguous one and NHD
    describes it; LBHNC stores heads first, so the view itself is contiguous
    and HND describes it.

    Read this off the strides rather than the config: the strides are what the
    kernel dereferences, and it takes a bare pointer, so a layout it was not
    given yields wrong values with no error. Padded pages leave neither form
    contiguous; return None so the caller can fall back.

    Both branches are metadata only -- one tensor object, no copy, no kernel.
    """
    transposed = cache.transpose(1, 2)
    if transposed.is_contiguous():
        return transposed, _NATIVE_LAYOUT_NHD
    if cache.is_contiguous():
        return cache, _NATIVE_LAYOUT_HND
    return None


def _gemma4_kv_to_query_dtype(
    cache: torch.Tensor, query_dtype: torch.dtype
) -> torch.Tensor:
    """Decode FP8 KV storage and cast it to the BF16 FA input dtype."""
    if cache.dtype == torch.uint8:
        cache = cache.view(torch.float8_e4m3fn)
    if cache.dtype != query_dtype:
        cache = cache.to(query_dtype)
    return cache


def _gemma4_gathered_kv_rows(
    *,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    seq_lens_bound: torch.Tensor | None,
) -> int:
    """Upper-bound the rows the paged gather will write, without a device sync.

    The host needs a row count only to size the gather's output buffer: the
    gather itself writes seqused_k rows and publishes cu_seqlens_k, and
    varlen_fwd reads cu_seqlens_k alone, so over-allocating is harmless while
    reading the exact sum off the device would drain the launch pipeline.
    """
    batch_size = cu_seqlens_q.numel() - 1
    if seq_lens_bound is not None:
        # Per-row CPU upper bound from the runner; exact for prefill rows and
        # never below the real length for async spec-decode rows.
        return int(seq_lens_bound[:batch_size].sum())
    return batch_size * max_seqlen_k


def _gemma4_flash_attention(
    *,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    head_size: int,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
    block_table: torch.Tensor,
    seq_lens_bound: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the admitted BF16 D512 causal path over the layer's paged KV pool.

    The pool holds every block of the layer, this batch's and everyone else's,
    so work on the batch's own rows only: gather them out of FP8 storage, then
    cast just those rows to BF16. pagedkv_to_contiguouskv is agnostic to the
    row dtype and width, so the 2*head_size row carrying K and V interleaved
    goes through as a single head dim and never needs splitting.

    The caller checks attention semantics, layout and that FA is not running
    under graph capture.
    """
    from flash_attn.flash_attn_interface import flash_attn_cuda

    cache = kv_cache
    if cache.dtype == torch.uint8:
        cache = cache.view(torch.float8_e4m3fn)

    view = _gemma4_paged_view(cache)
    if view is None:
        raise RuntimeError(
            "Gemma4 FA needs a paged KV pool contiguous as either "
            "[blocks, block_size, kv_heads, 2*head_size] or "
            f"[blocks, kv_heads, block_size, 2*head_size]; got shape "
            f"{tuple(kv_cache.shape)} stride {kv_cache.stride()}. "
            "_can_use_fa should have rejected this layout."
        )
    paged, native_layout = view
    # Heads and the K|V row width come from the logical view, whose dimension
    # order is fixed; paged's own axes move with the layout.
    num_kv_heads, row_width = cache.shape[1], cache.shape[3]

    total_k = _gemma4_gathered_kv_rows(
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        seq_lens_bound=seq_lens_bound,
    )
    gathered = torch.empty(
        (total_k, num_kv_heads, row_width),
        device=paged.device,
        dtype=paged.dtype,
    )
    cu_seqlens_k = torch.empty_like(cu_seqlens_q)
    # Both destinations receive the same interleaved K|V rows; aliasing them
    # keeps a single gather instead of splitting K and V into two buffers.
    flash_attn_cuda.pagedkv_to_contiguouskv(
        gathered,
        gathered,
        paged,
        paged,
        block_table,
        seqused_k,
        cu_seqlens_k,
        max_seqlen_k,
        native_layout,
    )
    kv = _gemma4_kv_to_query_dtype(gathered, q.dtype)
    # varlen_fwd reads K/V strides, so the halves go in as views. Rows past
    # cu_seqlens_k[-1] are never read and are left uninitialised on purpose.
    # Use the HCU native entry point to preserve the caller's output buffer.
    # The public non-paged autograd wrapper does not forward its out argument.
    flash_attn_cuda.varlen_fwd(
        q,
        kv[..., :head_size],
        kv[..., head_size:],
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        None,
        None,
        None,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        softmax_scale,
        False,
        True,
        -1,
        -1,
        0.0,
        False,
        None,
        None,
        None,
        None,
        None,
    )
    return out


class Gemma4FAMetadataBuilder(TritonAttentionMetadataBuilder):
    """Attach the runner's CPU seq-len bound to Triton's metadata.

    TritonAttentionMetadata carries seq_lens on the device only. The FA path
    needs a row count on the host to size its gather buffer, and reading the
    device tensor there costs a full stream drain per full-attention layer.
    The runner already computes a CPU per-row upper bound, so pass it along;
    everything else about the metadata is unchanged.
    """

    _logged_bound_source = False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonAttentionMetadata:
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build
        )
        bound = common_attn_metadata.seq_lens_cpu_upper_bound
        setattr(metadata, _GEMMA4_SEQ_LENS_BOUND, bound)
        if not Gemma4FAMetadataBuilder._logged_bound_source:
            Gemma4FAMetadataBuilder._logged_bound_source = True
            logger.info(
                "[GEMMA4_FA_BUILDER] active; seq_lens_cpu_upper_bound=%s "
                "(None falls back to batch_size*max_seq_len)",
                "present" if bound is not None else "None",
            )
        return metadata


class Gemma4FAAttentionBackend(TritonAttentionBackend):
    """Preserve Triton's cache, metadata, graph and backend-name contracts."""

    @staticmethod
    def get_impl_cls() -> type[Gemma4FAAttentionImpl]:
        return Gemma4FAAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Gemma4FAMetadataBuilder]:
        return Gemma4FAMetadataBuilder


class Gemma4FAAttentionImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._flash_attention = _gemma4_flash_attention
        # limited varified scenios, other cases currently fallback to triton backend
        static_checks = {
            "attn_type": self.attn_type == AttentionType.DECODER,
            "head_size": self.head_size == 512,
            "full_attention": self.sliding_window == (-1, -1),
            "kv_quant_mode": self._kv_quant_mode
            in (KVQuantMode.NONE, KVQuantMode.FP8_PER_TENSOR),
            "scale": self.scale == 1.0,
            "soft_cap": self.logits_soft_cap == 0.0,
            "alibi": self.alibi_slopes is None,
            "sinks": self.sinks is None,
            "alibi_sqrt": not self.use_alibi_sqrt,
            "chunk_lookback": self.chunk_lookback == -1,
            "batch_invariant": not envs.VLLM_BATCH_INVARIANT,
        }
        self._fa_static_failures = [
            name for name, passed in static_checks.items() if not passed
        ]
        self._fa_supported = not self._fa_static_failures
        # debug log config, set "HCU_DEBUG_ATTN_Q_LEN" to 1 to enable logs
        self._debug_routes = os.environ.get("HCU_DEBUG_ATTN_Q_LEN", "0").lower() in (
            "1",
            "all",
        )
        self._logged_routes: set[tuple] = set()
        self._logged_failures: set[tuple] = set()

    def _log_fa_failures(
        self,
        failures: list[str],
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: TritonAttentionMetadata | None,
        layer_name: str | None = None,
    ) -> None:
        if not failures:
            return
        if not self._debug_routes:
            return
        q_len = getattr(metadata, "max_query_len", None)
        # One line per distinct rejection, as the route log does. Every layer
        # of every decode step reaches this, so logging each occurrence floods
        # the hot path and depresses throughput for as long as the switch is on.
        log_key = (
            layer_name or getattr(self, "layer_name", "unknown"),
            tuple(failures),
            q_len,
            tuple(query.shape),
        )
        if log_key in self._logged_failures:
            return
        self._logged_failures.add(log_key)
        logger.info(
            ("[GEMMA4_FA_FALSE] layer=%s q_len=%s failures=%s "
            "q_shape=%s q_dtype=%s kv_shape=%s kv_dtype=%s "
            "q_stride=%s kv_stride=%s actual_tokens=%s "
            "head_size=%s heads=%s/%s window=%s kv_quant_mode=%s "
            "scale=%s softcap=%s chunk_lookback=%s "
            "attn_type=%s alibi=%s sinks=%s alibi_sqrt=%s batch_invariant=%s") % (
                layer_name or getattr(self, "layer_name", "unknown"),
                q_len,
                ",".join(failures),
                tuple(query.shape),
                query.dtype,
                tuple(kv_cache.shape),
                kv_cache.dtype,
                query.stride(),
                kv_cache.stride(),
                getattr(metadata, "num_actual_tokens", None),
                self.head_size,
                self.num_heads,
                self.num_kv_heads,
                self.sliding_window,
                self._kv_quant_mode,
                self.scale,
                self.logits_soft_cap,
                self.chunk_lookback,
                self.attn_type,
                self.alibi_slopes is not None,
                self.sinks is not None,
                self.use_alibi_sqrt,
                envs.VLLM_BATCH_INVARIANT,
            ),
        )

    def _can_use_fa(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: TritonAttentionMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
        layer_name: str | None = None,
    ) -> bool:
        failures: list[str] = []
        failures.extend(getattr(self, "_fa_static_failures", ()))
        if torch.cuda.is_current_stream_capturing():
            # A captured graph bakes in both the gather buffer's address and
            # its size, while the rows needed at replay are sum(seq_lens),
            # bounded only by max_num_seqs * max_model_len. No reservation
            # that size fits in memory, so captured batches belong to Triton.
            # FA serves eager prefill and mixed batches; the vendor kernel
            # rejects paged decode above head_dim 256 regardless.
            failures.append("cudagraph_capturing")
        if metadata is None:
            failures.append("metadata_none")
            self._log_fa_failures(failures, query, kv_cache, metadata, layer_name)
            return False

        mm_ranges = metadata.mm_prefix_range
        # The runner supplies a CPU dictionary even for text-only prefix-LM
        # batches. It is authoritative for its generated mask tensor; inspect
        # empty lists here rather than reading device data during capture.
        q_len = metadata.max_query_len
        checks = {
            "q_len<=16": q_len <= 16,
            "no_actual_tokens": metadata.num_actual_tokens <= 0,
            "cascade": metadata.use_cascade,
            "non_causal": getattr(metadata, "causal", True) is not True,
            "mm_prefix": (
                any(mm_ranges.values())
                if isinstance(mm_ranges, dict)
                else not (mm_ranges is None and metadata.mm_prefix_range_tensor is None)
            ),
            "rswa_prefix_lens": getattr(metadata, "rswa_prefix_lens", None) is not None,
            "rswa_window": getattr(metadata, "rswa_window", None) is not None,
            "output_scale": output_scale is not None,
            "output_block_scale": output_block_scale is not None,
            "query_output_not_bf16": not (
                query.dtype == output.dtype == torch.bfloat16
            ),
            "kv_dtype": not (
                (
                    self._kv_quant_mode == KVQuantMode.NONE
                    and kv_cache.dtype == torch.bfloat16
                )
                or (
                    self._kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
                    and kv_cache.dtype
                    in (torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2)
                )
            ),
            "query_ndim": query.ndim != 3,
            "query_shape": query.shape[1:] != (self.num_heads, self.head_size),
            "query_stride": query.ndim < 1 or query.stride(-1) != 1,
            "kv_ndim": kv_cache.ndim != 4,
            "kv_heads": kv_cache.ndim == 4
            and kv_cache.shape[1] != self.num_kv_heads,
            "kv_block_size": kv_cache.ndim == 4
            and kv_cache.shape[2] not in (32, 64),
            "kv_shape": kv_cache.ndim == 4
            and kv_cache.shape[3] != 2 * self.head_size,
            "kv_stride_last": kv_cache.ndim < 1 or kv_cache.stride(-1) != 1,
            # The gather kernel reads a bare pointer and silently produces
            # wrong values for a layout other than the one it is told, so
            # require a pool whose strides name one of the two it accepts.
            # Given the shape, contiguity of either form pins every stride,
            # which is why no individual stride needs its own check; padded
            # pages satisfy neither and route to Triton.
            "kv_layout_unsupported": kv_cache.ndim == 4
            and _gemma4_paged_view(kv_cache) is None,
            # The NHD gather's loop stride truncates to zero once an element
            # is wide enough, which hangs the worker rather than returning a
            # wrong answer. BF16 KV at this model's 1024-wide interleaved row
            # hits exactly that; see _gemma4_bshd_stride_is_zero.
            "kv_bshd_zero_stride": kv_cache.ndim == 4
            and (view := _gemma4_paged_view(kv_cache)) is not None
            and view[1] == _NATIVE_LAYOUT_NHD
            and _gemma4_bshd_stride_is_zero(
                kv_cache.shape[3], kv_cache.element_size()
            ),
        }
        failures.extend(name for name, failed in checks.items() if failed)
        self._log_fa_failures(failures, query, kv_cache, metadata, layer_name)
        return not failures

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        use_fa = self._can_use_fa(
            query,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
            getattr(layer, "layer_name", None),
        )
        if use_fa and self._kv_quant_mode == KVQuantMode.FP8_PER_TENSOR:
            # This path is valid only for the calibrated scale=1 experiment.
            fp8_scale_failures = []
            if getattr(layer, "_k_scale_float", None) != 1.0:
                fp8_scale_failures.append("k_scale!=1")
            if getattr(layer, "_v_scale_float", None) != 1.0:
                fp8_scale_failures.append("v_scale!=1")
            if fp8_scale_failures:
                self._log_fa_failures(
                    fp8_scale_failures,
                    query,
                    kv_cache,
                    attn_metadata,
                    getattr(layer, "layer_name", None),
                )
                use_fa = False
        if self._debug_routes and attn_metadata is not None:
            capturing = torch.cuda.is_current_stream_capturing()
            layer_name = (
                getattr(layer, "layer_name", None)
                or getattr(layer, "layer_id", None)
                or "unknown"
            )
            window = getattr(self, "sliding_window", None)
            attn_kind = "full" if window == (-1, -1) else "swa"
            route = "flash" if use_fa else "triton"
            log_key = (
                layer_name,
                attn_kind,
                route,
                attn_metadata.max_query_len,
                tuple(query.shape),
                capturing,
            )
            if log_key not in self._logged_routes:
                self._logged_routes.add(log_key)
                logger.info(
                    ("[GEMMA4_ATTN] layer=%s kind=%s route=%s "
                    "head_size=%s heads=%s/%s window=%s max_q=%s "
                    "total_q=%s capturing=%s") % (
                        layer_name,
                        attn_kind,
                        route,
                        self.head_size,
                        self.num_heads,
                        self.num_kv_heads,
                        window,
                        attn_metadata.max_query_len,
                        attn_metadata.num_actual_tokens,
                        capturing,
                    ),
                )
                if not use_fa and attn_metadata.max_query_len in (5, 17, 2048):
                    logger.info(
                        ("[GEMMA4_FA_ELIGIBILITY] supported=%s heads=%d/%d "
                        "window=%s scale=%s softcap=%s chunk=%s "
                        "q=%s/%s kv=%s/%s stride=%s out=%s/%s "
                        "cascade=%s causal=%s mm=%s/%s rswa=%s/%s "
                        "output_scale=%s/%s") % (
                            self._fa_supported,
                            self.num_heads,
                            self.num_kv_heads,
                            self.sliding_window,
                            self.scale,
                            self.logits_soft_cap,
                            self.chunk_lookback,
                            tuple(query.shape),
                            query.dtype,
                            tuple(kv_cache.shape),
                            kv_cache.dtype,
                            kv_cache.stride(),
                            tuple(output.shape),
                            output.dtype,
                            attn_metadata.use_cascade,
                            getattr(attn_metadata, "causal", True),
                            attn_metadata.mm_prefix_range is not None,
                            attn_metadata.mm_prefix_range_tensor is not None,
                            getattr(attn_metadata, "rswa_prefix_lens", None) is not None,
                            getattr(attn_metadata, "rswa_window", None),
                            output_scale is not None,
                            output_block_scale is not None,
                        ),
                    )
        if not use_fa:
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        num_tokens = attn_metadata.num_actual_tokens
        self._flash_attention(
            q=query[:num_tokens],
            kv_cache=kv_cache,
            head_size=self.head_size,
            out=output[:num_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            block_table=attn_metadata.block_table,
            seq_lens_bound=getattr(attn_metadata, _GEMMA4_SEQ_LENS_BOUND, None),
        )
        return output


_GEMMA4_MODULE = "vllm.model_executor.models.gemma4"
_GEMMA4_PATCH_ID = "model.gemma4.fa_attention"
_GEMMA4_MARKER = "_vllm_hcu_gemma4_attention_injection"


def _use_gemma4_fa(module: ModuleType) -> bool:
    """Bind only Gemma4's local Attention; its caller passes three positional args."""
    if module.__name__ != _GEMMA4_MODULE:
        raise ValueError(f"Expected {_GEMMA4_MODULE}, got {module.__name__}")
    original = module.Attention
    if getattr(original, _GEMMA4_MARKER, False):
        return False

    class Gemma4Attention(original):
        @functools.wraps(original.__init__)
        def __init__(self, num_heads, head_size, scale, **kwargs):
            if (
                head_size == 512
                and kwargs.get("per_layer_sliding_window") is None
                and kwargs.get("attn_backend") is None
            ):
                from vllm.config import get_current_vllm_config

                backend = get_current_vllm_config().attention_config.backend
                if getattr(backend, "name", backend) not in (None, "TRITON_ATTN"):
                    from vllm_hcu.utils.gemma4_switches import SWITCH

                    raise ValueError(
                        f"{SWITCH}=1 conflicts with attention "
                        f"backend {backend!r}; requires Triton-compatible metadata."
                    )
                kwargs["attn_backend"] = Gemma4FAAttentionBackend
            super().__init__(num_heads, head_size, scale, **kwargs)

    setattr(Gemma4Attention, _GEMMA4_MARKER, True)
    module.Attention = Gemma4Attention
    return True


def register_attention():
    """Called by the HCU model-registration entry point only when opted in."""
    from vllm_hcu.patch import register_post_import

    return register_post_import(
        _GEMMA4_PATCH_ID,
        _GEMMA4_MODULE,
        _use_gemma4_fa,
        targets=f"{_GEMMA4_MODULE}.Attention",
    )
