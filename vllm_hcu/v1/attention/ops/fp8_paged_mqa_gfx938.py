# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct sparse-indexer logits from HIPC 16-by-16 shuffled FP8 pages."""

import torch
from vllm.triton_utils import tl, triton

# Per-row program count of the decode grid, which skips the -inf tail fill.
_LIVE_TILE_PROGRAMS = 32


class _FlatAddressCache:
    """Keep HCU's pointer specialization from narrowing large page offsets."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.dtype = tensor.dtype
        self.device = tensor.device

    def data_ptr(self):
        return self.tensor.data_ptr()

    def ptr_range(self):
        # The HCU buffer path treats 2--4 GiB byte offsets as signed int32.
        # No 32-bit range promise: retain the kernel's explicit int64 math.
        return 1 << 63


@triton.jit
def _load_indexer_tile(
    Cache, Tables, tokens, dims, batch, end,
    T_BATCH: tl.constexpr, T_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr, DIM: tl.constexpr,
    CACHE_PAGE_STRIDE: tl.constexpr, NUM_PAGES: tl.constexpr,
    TABLE_WIDTH: tl.constexpr, MAX_LEN: tl.constexpr, TOKEN_MAJOR: tl.constexpr,
):
    # Keep physical addresses in int64 and clamp masked OOB pages on HCU.
    logical_page = tokens // PAGE_SIZE
    valid = (tokens < MAX_LEN) & (tokens < end) & (logical_page < TABLE_WIDTH)
    page = tl.load(
        Tables + batch * T_BATCH + logical_page * T_PAGE, valid, other=-1
    ).to(tl.int64)
    valid = valid & (page >= 0) & (page < NUM_PAGES)
    page = tl.where(valid, page, 0)
    slot = tokens % PAGE_SIZE
    page_base = page.to(tl.int64) * CACHE_PAGE_STRIDE
    if TOKEN_MAJOR:
        in_page_offset = slot[None, :] * DIM + dims[:, None]
    else:
        in_page_offset = (
            (slot[None, :] // 16) * (16 * DIM)
            + (dims[:, None] // 16) * 256
            + (slot[None, :] % 16) * 16
            + dims[:, None] % 16
        )
    offset = page_base[None, :] + in_page_offset.to(tl.int64)
    keys = tl.load(Cache + offset, (dims[:, None] < DIM) & valid[None, :], 0.0)
    return keys, valid, page_base, slot


@triton.jit
def _store_indexer_tile(
    Cache, Out, dots, weights, valid, page_base, slot, row, tokens,
    PAGE_SIZE: tl.constexpr, DIM: tl.constexpr, MAX_LEN: tl.constexpr,
):
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
    CACHE_PAGE_STRIDE: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    MAX_LEN: tl.constexpr,
    PER_QUERY: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    FILL_TAIL: tl.constexpr,
    TOKEN_MAJOR: tl.constexpr,
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
    if tl.program_id(1) * BLOCK_T >= end:
        # Tiles past this query's context only need the -inf fill; skipping
        # the page loads and dot keeps cost proportional to the context
        # instead of max_model_len.  Callers whose top-k honors per-row ends
        # skip the fill too: with a 1M max_model_len it alone writes
        # rows * 4 MB per layer.
        if FILL_TAIL:
            tl.store(
                Out + row * MAX_LEN + tokens,
                tl.full((BLOCK_T,), -float("inf"), tl.float32),
                tokens < MAX_LEN,
            )
        return
    keys, valid, page_base, slot = _load_indexer_tile(
        Cache, Tables, tokens, dims, batch, end,
        T_BATCH, T_PAGE, PAGE_SIZE, DIM, CACHE_PAGE_STRIDE,
        NUM_PAGES, TABLE_WIDTH, MAX_LEN, TOKEN_MAJOR,
    )
    query_values = tl.load(
        Q + batch * Q_BATCH + query * Q_NEXT + heads[:, None] * Q_HEAD + dims[None, :],
        (heads[:, None] < HEADS) & (dims[None, :] < DIM),
        0.0,
    )
    dots = tl.dot(query_values, keys)
    weights = tl.load(Weights + row * W_ROW + heads * W_HEAD, heads < HEADS, 0)
    _store_indexer_tile(
        Cache, Out, dots, weights, valid, page_base, slot, row, tokens,
        PAGE_SIZE, DIM, MAX_LEN,
    )


@triton.jit
def _paged_logits_live_tiles(
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
    CACHE_PAGE_STRIDE: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    MAX_LEN: tl.constexpr,
    PER_QUERY: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    TILE_PROGRAMS: tl.constexpr,
    TOKEN_MAJOR: tl.constexpr,
):
    """_paged_logits without the -inf tail, on a fixed [rows, TILE_PROGRAMS] grid.

    Each program loads its query and head weights once and strides over the
    row's live tiles.  A grid sized by max_model_len spends ~200 us per layer
    on programs past the context end at 1M context, even though they exit
    immediately.
    """
    row = tl.program_id(0)
    batch = row // NEXT_N
    query = row % NEXT_N
    heads = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    if PER_QUERY:
        end = tl.load(Lengths + batch * L_BATCH + query * L_NEXT)
    else:
        end = tl.load(Lengths + batch * L_BATCH) - NEXT_N + 1 + query
    query_values = tl.load(
        Q + batch * Q_BATCH + query * Q_NEXT + heads[:, None] * Q_HEAD + dims[None, :],
        (heads[:, None] < HEADS) & (dims[None, :] < DIM),
        0.0,
    )
    weights = tl.load(Weights + row * W_ROW + heads * W_HEAD, heads < HEADS, 0)
    for tile in range(tl.program_id(1), tl.cdiv(end, BLOCK_T), TILE_PROGRAMS):
        tokens = tile * BLOCK_T + tl.arange(0, BLOCK_T)
        keys, valid, page_base, slot = _load_indexer_tile(
            Cache, Tables, tokens, dims, batch, end,
            T_BATCH, T_PAGE, PAGE_SIZE, DIM, CACHE_PAGE_STRIDE,
            NUM_PAGES, TABLE_WIDTH, MAX_LEN, TOKEN_MAJOR,
        )
        dots = tl.dot(query_values, keys)
        _store_indexer_tile(
            Cache, Out, dots, weights, valid, page_base, slot, row, tokens,
            PAGE_SIZE, DIM, MAX_LEN,
        )


def gfx938_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    fill_tail: bool = True,
    token_major: bool = False,
) -> torch.Tensor:
    """Compute weighted logits without materializing token-major K storage.

    Args:
        q_fp8: FP8 queries with shape [batch, next_n, heads, head_dim].
        kv_cache_fp8: Physical HIPC pages [pages, page_size, 1, D+4]. Pages
            may be separated by a larger stride when the KV layout interleaves
            layers inside each block; every individual page must remain dense.
        weights: Float32 per-query head weights [batch * next_n, heads].
        context_lens: Int32/int64 final lengths [batch] or exact ends [batch,next_n].
        block_tables: Int32/int64 logical-to-physical page IDs [batch, pages].
        max_model_len: Number of output token positions.
        token_major: Pages hold token-major rows (slot * D + d) instead of
            16x16 tiles.  DeepSeek-V4's compressor writes its indexer cache
            token-major and its prefill gather reads it that way.
        fill_tail: Write -inf past each row's context end.  When False those
            positions are left unwritten and consumers must bound reads by the
            per-row end, as the decode top-k selectors do.

    Returns:
        Float32 logits [batch * next_n, max_model_len], with invalid positions
        set to negative infinity, including missing or out-of-range pages
        (positions past the row end only when ``fill_tail``).
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
        or kv_cache_fp8.stride(0) < kv_cache_fp8.shape[1] * (dim + 4)
        or kv_cache_fp8.stride(1) != dim + 4
        or kv_cache_fp8.stride(-1) != 1
        or kv_cache_fp8.dtype not in (torch.uint8, q_fp8.dtype)
    ):
        raise ValueError("cache must contain dense physical HIPC FP8 pages")
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
    num_tiles = triton.cdiv(max_model_len, 64)
    if fill_tail:
        kernel, grid, tile_args = _paged_logits, (batch * next_n, num_tiles), (fill_tail,)
    else:
        # The tail is left unwritten, so a fixed per-row program count
        # striding over live tiles covers any context length.
        tile_programs = min(num_tiles, _LIVE_TILE_PROGRAMS)
        kernel = _paged_logits_live_tiles
        grid, tile_args = (batch * next_n, tile_programs), (tile_programs,)
    physical_bytes = (
        (cache.shape[0] - 1) * cache.stride(0) + cache.shape[1] * (dim + 4)
    )
    cache_arg = _FlatAddressCache(cache) if physical_bytes >= (1 << 31) else cache
    kernel[grid](
        q_fp8,
        cache_arg,
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
        cache.stride(0),
        cache.shape[0],
        block_tables.shape[1],
        max_model_len,
        context_lens.ndim == 2,
        max(16, triton.next_power_of_2(heads)),
        triton.next_power_of_2(dim),
        64,
        *tile_args,
        token_major,
        num_warps=4,
    )
    return output
