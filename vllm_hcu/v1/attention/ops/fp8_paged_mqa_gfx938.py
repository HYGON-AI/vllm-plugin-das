# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct sparse-indexer logits from HIPC 16-by-16 shuffled FP8 pages."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _paged_logits(
    Q,
    Cache,
    Weights,
    Lengths,
    Tables,
    Out,
    Q_BATCH: tl.constexpr,
    Q_NEXT: tl.constexpr,
    Q_HEAD: tl.constexpr,
    W_ROW: tl.constexpr,
    W_HEAD: tl.constexpr,
    L_BATCH: tl.constexpr,
    L_NEXT: tl.constexpr,
    T_BATCH: tl.constexpr,
    T_PAGE: tl.constexpr,
    NEXT_N: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    MAX_LEN: tl.constexpr,
    PER_QUERY: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // NEXT_N
    query = row % NEXT_N
    tokens = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    heads = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    if PER_QUERY:
        end = tl.load(Lengths + batch * L_BATCH + query * L_NEXT)
    else:
        end = tl.load(Lengths + batch * L_BATCH) - NEXT_N + 1 + query
    logical_page = tokens // PAGE_SIZE
    valid = (tokens < MAX_LEN) & (tokens < end) & (logical_page < TABLE_WIDTH)
    page = tl.load(
        Tables + batch * T_BATCH + logical_page * T_PAGE, valid, other=-1
    ).to(tl.int64)
    valid = valid & (page >= 0) & (page < NUM_PAGES)
    slot = tokens % PAGE_SIZE
    page_base = page * (PAGE_SIZE * (DIM + 4))
    offset = (
        page_base[None, :]
        + (slot[None, :] // 16) * (16 * DIM)
        + (dims[:, None] // 16) * 256
        + (slot[None, :] % 16) * 16
        + dims[:, None] % 16
    )
    keys = tl.load(Cache + offset, (dims[:, None] < DIM) & valid[None, :], 0.0)
    query_values = tl.load(
        Q + batch * Q_BATCH + query * Q_NEXT + heads[:, None] * Q_HEAD + dims[None, :],
        (heads[:, None] < HEADS) & (dims[None, :] < DIM),
        0.0,
    )
    # FP8 values are exactly representable in FP16; accumulate QK in FP32.
    dots = tl.dot(query_values.to(tl.float16), keys.to(tl.float16))
    weights = tl.load(Weights + row * W_ROW + heads * W_HEAD, heads < HEADS, 0)
    scale_ptr = (Cache + page_base + PAGE_SIZE * DIM + slot * 4).to(
        tl.pointer_type(tl.float32)
    )
    scales = tl.load(scale_ptr, valid, 0)
    scores = tl.sum(tl.maximum(dots, 0) * weights[:, None], axis=0) * scales
    tl.store(
        Out + row * MAX_LEN + tokens,
        tl.where(valid, scores, -float("inf")),
        tokens < MAX_LEN,
    )


def gfx938_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute weighted logits without materializing token-major K storage.

    Args:
        q_fp8: FP8 queries with shape [batch, next_n, heads, head_dim].
        kv_cache_fp8: Contiguous physical HIPC pages [pages, page_size, 1, D+4].
        weights: Float32 per-query head weights [batch * next_n, heads].
        context_lens: Int32/int64 final lengths [batch] or exact ends [batch,next_n].
        block_tables: Int32/int64 logical-to-physical page IDs [batch, pages].
        max_model_len: Number of output token positions.

    Returns:
        Float32 logits [batch * next_n, max_model_len], with invalid positions
        set to negative infinity, including missing or out-of-range pages.
    """
    if q_fp8.ndim != 4 or q_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError("q_fp8 must be a rank-4 float8_e4m3fn tensor")
    batch, next_n, heads, dim = q_fp8.shape
    if min(batch, next_n, heads, dim) <= 0 or dim % 16 or q_fp8.stride(-1) != 1:
        raise ValueError("queries require nonempty axes and contiguous 16-wide D")
    if (
        kv_cache_fp8.ndim != 4
        or kv_cache_fp8.shape[0] < 1
        or kv_cache_fp8.shape[1] not in (16, 32, 64)
        or kv_cache_fp8.shape[2:] != (1, dim + 4)
        or not kv_cache_fp8.is_contiguous()
        or kv_cache_fp8.dtype not in (torch.uint8, q_fp8.dtype)
    ):
        raise ValueError("cache must contain contiguous physical HIPC FP8 pages")
    if weights.shape != (batch * next_n, heads) or weights.dtype != torch.float32:
        raise ValueError("weights must be float32 [batch * next_n, heads]")
    if context_lens.shape not in ((batch,), (batch, next_n)):
        raise ValueError("context_lens must have shape [batch] or [batch, next_n]")
    if (
        block_tables.ndim != 2
        or block_tables.shape[0] != batch
        or block_tables.shape[1] < 1
    ):
        raise ValueError("block_tables must have shape [batch, positive page count]")
    if any(
        t.dtype not in (torch.int32, torch.int64) for t in (context_lens, block_tables)
    ):
        raise ValueError("lengths and page IDs must be int32 or int64")
    tensors = (kv_cache_fp8, weights, context_lens, block_tables)
    if not q_fp8.is_cuda or any(t.device != q_fp8.device for t in tensors):
        raise ValueError("all inputs must be on the same GPU")
    if not isinstance(max_model_len, int) or max_model_len < 0:
        raise ValueError("max_model_len must be a nonnegative integer")
    output = torch.empty(
        (batch * next_n, max_model_len), dtype=torch.float32, device=q_fp8.device
    )
    if max_model_len == 0:
        return output
    cache = kv_cache_fp8.view(q_fp8.dtype)
    _paged_logits[(batch * next_n, triton.cdiv(max_model_len, 64))](
        q_fp8,
        cache,
        weights,
        context_lens,
        block_tables,
        output,
        *q_fp8.stride()[:3],
        *weights.stride(),
        context_lens.stride(0),
        context_lens.stride(1) if context_lens.ndim == 2 else 0,
        *block_tables.stride(),
        next_n,
        heads,
        dim,
        cache.shape[1],
        cache.shape[0],
        block_tables.shape[1],
        max_model_len,
        context_lens.ndim == 2,
        max(16, triton.next_power_of_2(heads)),
        triton.next_power_of_2(dim),
        64,
        num_warps=4,
    )
    return output
