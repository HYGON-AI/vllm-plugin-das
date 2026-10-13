# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Gemma4 full-attention FA path using Triton metadata and cache management."""

from __future__ import annotations

import os

import torch
from vllm import envs
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import KVQuantMode

logger = init_logger(__name__)


class Gemma4FAAttentionBackend(TritonAttentionBackend):
    """Preserve Triton's cache, metadata, graph and backend-name contracts."""

    @staticmethod
    def get_impl_cls() -> type[Gemma4FAAttentionImpl]:
        return Gemma4FAAttentionImpl


class Gemma4FAAttentionImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from .fa_utils import gemma4_flash_attention

        self._flash_attention = gemma4_flash_attention
        # limited varified scenios, other cases currently fallback to triton backend
        self._fa_supported = (
            self.attn_type == AttentionType.DECODER
            and self.head_size == 512
            and self.sliding_window == (-1, -1)
            and self._kv_quant_mode == KVQuantMode.NONE
            and self.scale == 1.0
            and self.logits_soft_cap == 0.0
            and self.alibi_slopes is None
            and self.sinks is None
            and not self.use_alibi_sqrt
            and self.chunk_lookback == -1
            and not envs.VLLM_BATCH_INVARIANT
        )
        # debug log config, set "HCU_DEBUG_ATTN_Q_LEN" to 1 to enable logs
        self._debug_routes = os.environ.get("HCU_DEBUG_ATTN_Q_LEN", "0").lower() in (
            "1",
            "all",
        )
        self._logged_routes: set[tuple] = set()

    def _can_use_fa(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: TritonAttentionMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
    ) -> bool:
        if not self._fa_supported or metadata is None:
            return False
        q_len = metadata.max_query_len
        return (
            (q_len > 16 or q_len == 5)
            and metadata.num_actual_tokens > 0
            and not metadata.use_cascade
            and getattr(metadata, "causal", True) is True
            and metadata.mm_prefix_range is None
            and metadata.mm_prefix_range_tensor is None
            and getattr(metadata, "rswa_prefix_lens", None) is None
            and getattr(metadata, "rswa_window", None) is None
            and output_scale is None
            and output_block_scale is None
            and query.dtype == kv_cache.dtype == output.dtype == torch.bfloat16
            and query.ndim == 3
            and query.shape[1:] == (self.num_heads, self.head_size)
            and query.stride(-1) == 1
            and kv_cache.ndim == 5
            and kv_cache.shape[1] == 2
            # Hybrid page-size unification can grow a requested block32 to 64.
            and kv_cache.shape[2] in (32, 64)
            and kv_cache.shape[3:] == (self.num_kv_heads, self.head_size)
            and kv_cache.stride(-1) == 1
            and kv_cache.stride(-2) == self.head_size
            and kv_cache.stride(-3) == self.num_kv_heads * self.head_size
        )

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
            query, kv_cache, attn_metadata, output, output_scale, output_block_scale
        )
        if self._debug_routes and attn_metadata is not None:
            capturing = torch.cuda.is_current_stream_capturing()
            route = "flash" if use_fa else "triton"
            log_key = (route, attn_metadata.max_query_len, query.shape, capturing)
            if log_key not in self._logged_routes:
                self._logged_routes.add(log_key)
                logger.info(
                    "[GEMMA4_FA_ROUTE] layer=%s route=%s max_seqlen_q=%d "
                    "num_seqs=%d total_q=%d capturing=%s",
                    getattr(layer, "layer_name", "unknown"),
                    route,
                    attn_metadata.max_query_len,
                    attn_metadata.query_start_loc.numel() - 1,
                    attn_metadata.num_actual_tokens,
                    capturing,
                )
                if not use_fa and attn_metadata.max_query_len in (5, 17, 2048):
                    logger.info(
                        "[GEMMA4_FA_ELIGIBILITY] supported=%s heads=%d/%d "
                        "window=%s scale=%s softcap=%s chunk=%s "
                        "q=%s/%s kv=%s/%s stride=%s out=%s/%s "
                        "cascade=%s causal=%s mm=%s/%s rswa=%s/%s "
                        "output_scale=%s/%s",
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
        key_cache, value_cache = kv_cache.unbind(1)
        self._flash_attention(
            q=query[:num_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            block_table=attn_metadata.block_table,
        )
        return output
