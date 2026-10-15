# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Default-on V4.1 decode over the existing packed FP8 caches.

Keep KV writers and source-layer selection in the model. Map local top-k to
dense physical indices with the model's audited utility, without KV gathering.
Unsupported metadata falls back before launching native attention; native
errors deliberately propagate rather than concealing a failed HIP launch.
"""

from __future__ import annotations

import functools
import logging

import torch

from vllm_hcu.platforms import envs as henvs
from vllm_hcu.v1.attention.ops.deepseek_v41_flashmla_metadata import DecodeScheduler

logger = logging.getLogger(__name__)
_SUPPORTED_ARCHES = frozenset({"gfx938"})


def _supported_device(q: torch.Tensor) -> bool:
    return (
        q.device.type == "cuda"
        and torch.version.hip is not None
        and torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0]
        in _SUPPORTED_ARCHES
    )


@functools.cache
def _decode_bindings():
    try:
        from vllm_hcu.v1.attention.ops.flashmla import (
            flash_mla_with_kvcache,
            get_mla_metadata,
            is_flashmla_sparse_supported,
        )
        from vllm.models.deepseek_v41.common.ops import (
            compute_global_topk_indices_and_lens,
        )
    except ImportError:
        return None
    if not is_flashmla_sparse_supported()[0]:
        return None
    return flash_mla_with_kvcache, compute_global_topk_indices_and_lens, get_mla_metadata


@functools.cache
def _log_fallback(reason: str) -> None:
    logger.info("V4.1 FlashMLA decode: using original operator (%s)", reason)


@functools.cache
def _log_selected() -> None:
    logger.info("V4.1 decode: using native FlashMLA sparse FP8 attention")


def _packed_cache_supported(cache: torch.Tensor, q: torch.Tensor) -> bool:
    # A page contains all token data first, followed by the UE8M0 scales.
    # The row shape is a descriptor, NOT a per-token interleaved byte layout.
    return (
        isinstance(cache, torch.Tensor)
        and cache.ndim == 3
        and cache.shape[0] > 0
        and cache.shape[1] in (32, 64, 128)
        and cache.shape[2] == 584
        and cache.dtype == torch.uint8
        and cache.device == q.device
        and cache.stride(2) == 1
        and cache.stride(1) == 584
        and cache.stride(0) >= cache.shape[1] * 584
        and cache.stride(0) % 576 == 0
    )


def _indices_supported(indices: torch.Tensor, q: torch.Tensor) -> bool:
    return (
        isinstance(indices, torch.Tensor)
        and indices.ndim in (2, 3)
        and indices.shape[0] == q.shape[0]
        and (indices.ndim == 2 or indices.shape[1] == 1)
        and indices.shape[-1] > 0
        and indices.shape[-1] % 64 == 0
        and indices.dtype == torch.int32
        and indices.device == q.device
        and indices.stride(-1) == 1
    )


def try_flashmla_decode(
    *,
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    scheduler,
    scale: float,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    max_decode_query_len: int = 1,
    kv_cache_dtype: str = "fp8_ds_mla",
    topk_indices: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    token_to_req_indices: torch.Tensor | None = None,
    is_valid_token: torch.Tensor | None = None,
) -> bool:
    """Write caller-owned output on selection; leave it untouched on fallback."""
    if not henvs.dsv41_flashmla_decode_enabled():
        return False
    managed_spec = isinstance(scheduler, DecodeScheduler) and scheduler.speculative
    supported_query = max_decode_query_len == 1 or (
        managed_spec and henvs.dsv41_flashmla_dspark_enabled()
        and 1 <= max_decode_query_len <= scheduler.query_limit
    )
    if (
        not supported_query
        or (managed_spec and not henvs.dsv41_flashmla_dspark_enabled())
        or kv_cache_dtype != "fp8_ds_mla"
        or (head_dim, nope_head_dim, rope_head_dim) != (512, 448, 64)
        or scheduler is None
    ):
        _log_fallback("decode needs supported query rows, D512 packed FP8 and scheduler")
        return False
    if (
        q.ndim != 3 or q.shape[0] == 0 or q.shape[1] not in (64, 128)
        or q.shape[2] != 512 or q.dtype != torch.bfloat16 or q.stride(-1) != 1
        or output.shape != q.shape or output.dtype != q.dtype
        or output.device != q.device
    ):
        _log_fallback("unsupported query/output shape, dtype, or stride")
        return False
    # V4.1 owns a 32-token SWA page independently of --block-size (history).
    # Pass the original descriptor/stride through; do not reinterpret as 64.
    if not _packed_cache_supported(swa_cache, q) or swa_cache.shape[1] not in (32, 64):
        _log_fallback("unsupported packed SWA page layout")
        return False
    if not _indices_supported(swa_indices, q) or (
        swa_lens is None or swa_lens.shape != (q.shape[0],) or swa_lens.dtype != torch.int32
        or swa_lens.device != q.device
    ):
        _log_fallback("unsupported SWA indices/lengths")
        return False
    if attn_sink is not None and (
        attn_sink.ndim != 1 or attn_sink.numel() < q.shape[1]
        or attn_sink.dtype != torch.float32 or attn_sink.device != q.device
    ):
        _log_fallback("unsupported attention sink")
        return False
    if kv_cache is not None:
        if (
            not _packed_cache_supported(kv_cache, q)
            or topk_indices is None or not _indices_supported(topk_indices, q)
            or topk_indices.ndim != 2
            or block_table is None or block_table.ndim != 2
            or block_table.shape[0] == 0 or block_table.shape[1] == 0
            or block_table.dtype != torch.int32 or block_table.device != q.device
            or block_table.stride(-1) != 1
            or token_to_req_indices is None
            or token_to_req_indices.shape != (q.shape[0],)
            or token_to_req_indices.dtype != torch.int32
            or token_to_req_indices.device != q.device
            or not token_to_req_indices.is_contiguous()
            or is_valid_token is None or is_valid_token.shape != (q.shape[0],)
            or is_valid_token.dtype != torch.bool or is_valid_token.device != q.device
            or not is_valid_token.is_contiguous()
        ):
            _log_fallback("unsupported compressed cache or top-k mapping metadata")
            return False
    if not _supported_device(q):
        _log_fallback("only gfx938 decode is validated")
        return False
    if torch.cuda.is_current_stream_capturing() and not (
        henvs.dsv41_flashmla_decode_graph_enabled()
        and isinstance(scheduler, DecodeScheduler) and scheduler.allow_graph
    ):
        _log_fallback("decode capture needs opt-in and a framework-owned fresh plan")
        return False
    bindings = _decode_bindings()
    if bindings is None:
        _log_fallback("native FlashMLA decode unavailable")
        return False
    kernel, map_indices, get_metadata = bindings
    # Framework metadata owns per-layer leases. Raw holders remain supported
    # for standalone callers and pre-patch metadata; never initialize a shared
    # per-type holder in place. Consume only after every fallback check passes.
    plan = scheduler.take() if isinstance(scheduler, DecodeScheduler) else get_metadata()[0]
    extra_indices, extra_lens = None, None
    if kv_cache is not None:
        extra_indices, extra_lens = map_indices(
            topk_indices, token_to_req_indices, block_table, kv_cache.shape[1],
            is_valid_token,
        )
        # The model utility counts but does not compact scattered -1 holes.
        # Sort valid physical slots before the sentinels, keeping duplicates.
        sentinel = torch.iinfo(torch.int32).max
        extra_indices = torch.where(extra_indices >= 0, extra_indices, sentinel)
        extra_indices = extra_indices.sort(dim=-1).values.view(q.shape[0], 1, -1)
        columns = torch.arange(extra_indices.shape[-1], device=q.device)
        extra_indices = torch.where(
            columns[None, None, :] < extra_lens[:, None, None], extra_indices, -1,
        )
    main_indices = swa_indices.reshape(q.shape[0], 1, -1)
    columns = torch.arange(main_indices.shape[-1], device=q.device)
    main_indices = torch.where(
        columns[None, None, :] < swa_lens[:, None, None], main_indices, -1,
    )
    result, _ = kernel(
        # One native batch row per query token, including DSpark verify/draft.
        # Non-causal future visibility is already encoded in the SWA indices;
        # never impose a triangular mask or regroup rows by request here.
        q=q.unsqueeze(1), k_cache=swa_cache.unsqueeze(-2),
        block_table=None, cache_seqlens=None, head_dim_v=512,
        # A fresh per-invocation plan is required even for equal-shaped layers.
        tile_scheduler_metadata=plan, softmax_scale=scale,
        is_fp8_kvcache=True, indices=main_indices,
        topk_length=swa_lens.contiguous(),
        attn_sink=None if attn_sink is None else attn_sink[:q.shape[1]].contiguous(),
        extra_k_cache=None if kv_cache is None else kv_cache.unsqueeze(-2),
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_lens,
    )
    output.copy_(result.squeeze(1))
    _log_selected()
    return True
