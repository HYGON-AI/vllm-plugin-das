# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

from __future__ import annotations

import torch

from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
)


class HcuFlashMLASparseImpl(FlashMLASparseImpl):
    supports_pcp: bool = True
    can_return_lse_for_decode: bool = True

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
            triton_convert_req_index_to_global_index,
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
            triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token,
                attn_metadata.block_table,
                topk_indices,
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
        return attn_out, lse


class HcuFlashMLASparseBackend(FlashMLASparseBackend):
    @staticmethod
    def get_name() -> str:
        return "FLASHMLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type[HcuFlashMLASparseImpl]:
        return HcuFlashMLASparseImpl


__all__ = ["HcuFlashMLASparseBackend", "HcuFlashMLASparseImpl"]
