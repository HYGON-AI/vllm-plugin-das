# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

from __future__ import annotations

import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
    FlashMLASparseMetadata,
    FlashMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm_hcu.models.hy_v4.fp8_kv_dequant import (
    LightOpKVReuseState,
    gather_dequantize_fp8_ds_mla_cache,
)
import vllm_hcu.platforms.envs as henvs


def _has_uniform_query_width(
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
) -> bool:
    if num_tokens > 0 and max_query_len <= 0:
        raise ValueError(
            "HCU sparse attention requires a positive max_query_len when "
            f"tokens are present; got num_tokens={num_tokens}, "
            f"max_query_len={max_query_len}."
        )
    return (
        num_reqs > 0
        and max_query_len > 0
        and num_tokens == num_reqs * max_query_len
    )


def _lightop_tokens_per_request(
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
) -> int:
    """Return a uniform LightOp grouping width, or one for ragged input."""
    if num_reqs <= 0:
        raise ValueError("HCU sparse attention requires at least one request")
    if _has_uniform_query_width(num_tokens, num_reqs, max_query_len):
        return max_query_len
    return 1


def _lightop_mapping_reuse_group_size(metadata: FlashMLASparseMetadata) -> int:
    """Return the request width eligible for compact-index mapping reuse."""
    width = _lightop_tokens_per_request(
        metadata.num_actual_tokens,
        metadata.num_reqs,
        metadata.max_query_len,
    )
    return width if width > 1 else 1


def bind_lightop_kv_reuse_state(
    mla_wrapper,
    state: LightOpKVReuseState | None,
    *,
    is_indexer_producer: bool,
) -> None:
    """Bind model-owned compact-index state to the sparse attention impl."""
    impl = mla_wrapper.mla_attn.impl
    impl._lightop_kv_reuse_state = state
    impl._is_indexer_producer = is_indexer_producer


class HcuFlashMLASparseMetadataBuilder(FlashMLASparseMetadataBuilder):
    """Attach LightOp grouping metadata for both prefill and decode."""

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build: bool = False,
    ) -> FlashMLASparseMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
        )
        metadata.lightop_kv_group_size = _lightop_mapping_reuse_group_size(metadata)
        return metadata


class HcuFlashMLASparseImpl(FlashMLASparseImpl):
    supports_pcp: bool = True
    can_return_lse_for_decode: bool = True

    def _forward_lightop_fp8_kv(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> tuple[torch.Tensor, None]:
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            q = self.q_concat_buffer[: ql_nope.shape[0]]
            ops.concat_mla_q(ql_nope, q_pe, q)

        num_actual_toks = q.shape[0]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        topk_indices, topk_length = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )

        tokens_per_request = _lightop_tokens_per_request(
            num_actual_toks,
            attn_metadata.num_reqs,
            attn_metadata.max_query_len,
        )
        reuse_state = getattr(self, "_lightop_kv_reuse_state", None)
        reuse_kwargs = (
            {
                "reuse_state": reuse_state,
                "allow_mapping_reuse": not getattr(
                    self, "_is_indexer_producer", True
                ),
                "mapping_reuse_group_size": getattr(
                    attn_metadata,
                    "lightop_kv_group_size",
                    1,
                ),
            }
            if reuse_state is not None
            else {}
        )
        cache, compact_indices = gather_dequantize_fp8_ds_mla_cache(
            kv_c_and_k_pe_cache,
            topk_indices,
            self.kv_lora_rank,
            self.head_size - self.kv_lora_rank,
            tokens_per_request,
            **reuse_kwargs,
        )
        output = self._bf16_flash_mla_kernel(
            q,
            cache,
            compact_indices,
            topk_length,
        )
        return output, None

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (
            self.dcp_world_size <= 1
            and self.kv_cache_dtype == "fp8_ds_mla"
            and henvs.VLLM_HCU_HYV4_FP8_KV_DEQUANT
        ):
            return self._forward_lightop_fp8_kv(
                q,
                kv_c_and_k_pe_cache,
                attn_metadata,
            )
        if self.dcp_world_size <= 1:
            return super().forward_mqa(
                q,
                kv_c_and_k_pe_cache,
                attn_metadata,
                layer,
            )
        if self.kv_cache_dtype == "fp8_ds_mla":
            raise RuntimeError("HCU sparse MLA DCP does not support FP8 KV cache")

        from vllm import _custom_ops as ops
        from vllm.v1.attention.backends.mla.sparse_utils import (
            triton_filter_and_convert_dcp_index,
        )
        from vllm_hcu.v1.attention.ops.flashmla import flash_mla_sparse_fwd

        # DCP gathers the query-head dimension before this call. Keep every
        # gathered head and return the kernel LSE so the common MLA runtime can
        # perform its numerically stable cross-rank reduction.
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            q = self.q_concat_buffer[: ql_nope.shape[0]]
            ops.concat_mla_q(ql_nope, q_pe, q)

        num_actual_toks = q.shape[0]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        topk_indices, topk_length = (
            triton_filter_and_convert_dcp_index(
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
        )
        cache = kv_c_and_k_pe_cache.view(
            -1,
            1,
            kv_c_and_k_pe_cache.shape[-1],
        )
        indices = topk_indices.view(num_actual_toks, 1, -1)
        attn_out, _, lse = flash_mla_sparse_fwd(
            q,
            cache,
            indices,
            self.softmax_scale,
            topk_length=topk_length,
        )
        if lse is None:
            raise RuntimeError("HCU sparse MLA DCP kernel did not return LSE")
        empty_rows = topk_length == 0
        attn_out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
        lse.masked_fill_(empty_rows.view(-1, 1), float("-inf"))
        return attn_out, lse


class HcuFlashMLASparseBackend(FlashMLASparseBackend):
    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type[HcuFlashMLASparseImpl]:
        return HcuFlashMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type[HcuFlashMLASparseMetadataBuilder]:
        return HcuFlashMLASparseMetadataBuilder


__all__ = [
    "HcuFlashMLASparseBackend",
    "HcuFlashMLASparseImpl",
    "HcuFlashMLASparseMetadataBuilder",
    "bind_lightop_kv_reuse_state",
]
