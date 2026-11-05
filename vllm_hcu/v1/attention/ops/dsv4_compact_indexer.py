# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage live token-major pages for LightOp without large cross-layer strides.

Preserve the FP8 payload and FP32 scales byte-for-byte. The packed page stride
keeps LightOp offsets in its supported range even for layer-interleaved caches.
"""

import torch
from vllm.triton_utils import tl, triton

_WORKSPACES = {}


@triton.jit
def _stage_pages(Cache, Compact, Lengths, Tables, NewLengths, NewTables,
                 LB: tl.constexpr, LN: tl.constexpr, TB: tl.constexpr,
                 TP: tl.constexpr, NEXT: tl.constexpr, WIDTH: tl.constexpr,
                 PS: tl.constexpr, CS_WORDS: tl.constexpr,
                 PAGES: tl.constexpr, COMPACT_PAGES: tl.constexpr,
                 USE_PHYSICAL: tl.constexpr, MAXLEN: tl.constexpr,
                 PER_QUERY: tl.constexpr, PROGRAMS: tl.constexpr,
                 WORDS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    batch, query = row // NEXT, row % NEXT
    if PER_QUERY:
        end = tl.load(Lengths + batch * LB + query * LN)
    else:
        end = tl.load(Lengths + batch * LB) - NEXT + 1 + query
    end = tl.minimum(tl.maximum(end, 0), tl.minimum(MAXLEN, WIDTH * PS))
    if tl.program_id(1) == 0:
        tl.store(NewLengths + row, end)
    words = tl.arange(0, BLOCK)
    source = Cache.to(tl.pointer_type(tl.uint32))
    dest = Compact.to(tl.pointer_type(tl.uint32))
    for logical in range(tl.program_id(1), tl.cdiv(end, PS), PROGRAMS):
        page = tl.load(Tables + batch * TB + logical * TP).to(tl.int64)
        valid = (page >= 0) & (page < PAGES)
        # Give every referenced logical page in this batch its own compact
        # slot.  The slot count is bounded by rows * WIDTH instead of the
        # entire physical KV-cache capacity, and remains graph-safe because
        # the mapping is produced on device from the live block table.
        compact_page = tl.where(
            USE_PHYSICAL,
            page,
            row * WIDTH + logical,
        )
        tl.store(
            NewTables + row * WIDTH + logical,
            tl.where(valid, compact_page, COMPACT_PAGES),
        )
        if valid:
            values = tl.load(
                source + page.to(tl.int64) * CS_WORDS + words,
                words < WORDS,
                0,
            )
            tl.store(
                dest + compact_page * WORDS + words,
                values,
                words < WORDS,
            )


@triton.jit
def _mask_missing(Out, Lengths, Tables, LB: tl.constexpr, LN: tl.constexpr,
                  TB: tl.constexpr, TP: tl.constexpr, NEXT: tl.constexpr,
                  WIDTH: tl.constexpr, PS: tl.constexpr, PAGES: tl.constexpr,
                  MAXLEN: tl.constexpr, OUT_ROW: tl.constexpr, PER_QUERY: tl.constexpr,
                  PROGRAMS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    batch, query = row // NEXT, row % NEXT
    if PER_QUERY:
        end = tl.load(Lengths + batch * LB + query * LN)
    else:
        end = tl.load(Lengths + batch * LB) - NEXT + 1 + query
    end = tl.minimum(tl.maximum(end, 0), MAXLEN)
    for tile in range(tl.program_id(1), tl.cdiv(end, BLOCK), PROGRAMS):
        positions = tile * BLOCK + tl.arange(0, BLOCK)
        logical = positions // PS
        covered = (positions < end) & (logical < WIDTH)
        safe = tl.where(covered, logical, 0)
        page = tl.load(Tables + batch * TB + safe * TP, covered, -1)
        invalid = (positions < end) & ((page < 0) | (page >= PAGES) | (logical >= WIDTH))
        if tl.sum(invalid.to(tl.int32), 0) > 0:
            tl.store(Out + row * OUT_ROW + positions, -float("inf"), invalid)


def compact_indexer_logits(q, cache, weights, lengths, table, max_len):
    from .fp8_paged_mqa_gfx938 import _FlatAddressCache
    from .rocm_aiter_mla_sparse import _get_lightop_attention

    batch, next_n, heads, dim = q.shape
    rows = batch * next_n
    pages, page_size = cache.shape[:2]
    width = table.shape[1]
    # Separate streams may run independently; each owns its staging buffers.
    # Include the page shape in the key because a stream can serve more than
    # one indexer configuration during worker lifetime.
    key = (
        q.device, torch.cuda.current_stream(q.device).cuda_stream,
        pages, page_size, width, dim,
    )
    state = _WORKSPACES.get(key)
    required_slots = max(1, rows * width)
    use_physical = required_slots >= pages
    required_pages = pages if use_physical else required_slots
    if state is None:
        state = {
            "compact": None,
            "mapped": None,
            "new_lengths": None,
            "page_capacity": 0,
            "row_capacity": 0,
            "old": [],
        }
        _WORKSPACES[key] = state

    if (
        state["page_capacity"] < required_pages
        or state["row_capacity"] < rows
    ):
        # Earlier CUDA graphs may still reference the old buffers. Keep them
        # alive after replacement so graph replay never observes freed storage.
        if state["compact"] is not None:
            state["old"].append(
                (state["compact"], state["mapped"], state["new_lengths"])
            )
        if state["page_capacity"] < required_pages:
            page_capacity = (
                required_pages
                if use_physical
                else min(pages, triton.next_power_of_2(required_pages))
            )
            state["compact"] = torch.empty(
                (page_capacity + 1, page_size, 1, dim + 4),
                device=q.device,
                dtype=torch.uint8,
            )
            state["compact"][-1].zero_()  # invalid-page sentinel
            state["page_capacity"] = page_capacity
        if state["row_capacity"] < rows:
            row_capacity = triton.next_power_of_2(rows)
            state["mapped"] = torch.empty(
                (row_capacity, width), device=q.device, dtype=torch.int32
            )
            state["new_lengths"] = torch.empty(
                row_capacity, device=q.device, dtype=torch.int32
            )
            state["row_capacity"] = row_capacity

    compact = state["compact"]
    mapped = state["mapped"][:rows]
    new_lengths = state["new_lengths"][:rows]
    compact_pages = state["page_capacity"]
    span = (pages - 1) * cache.stride(0) + page_size * (dim + 4)
    cache_arg = _FlatAddressCache(cache) if span >= (1 << 31) else cache
    words = page_size * (dim + 4) // 4
    _stage_pages[(rows, 32)](
        cache_arg, compact, lengths, table, new_lengths, mapped,
        lengths.stride(0), lengths.stride(1) if lengths.ndim == 2 else 0,
        *table.stride(), next_n, width, page_size, cache.stride(0) // 4,
        pages, compact_pages, use_physical, max_len,
        lengths.ndim == 2, 32, words, triton.next_power_of_2(words), num_warps=4,
    )
    result = _get_lightop_attention().paged_mqa_logits(
        q.reshape(rows, 1, heads, dim), compact, weights.float().contiguous(),
        new_lengths, mapped, schedule_meta=None,
        max_context_len=max_len, clean_logits=False,
    ).reshape(rows, max_len)
    _mask_missing[(rows, 32)](
        result, lengths, table, lengths.stride(0),
        lengths.stride(1) if lengths.ndim == 2 else 0,
        *table.stride(), next_n, width, page_size, pages, max_len,
        result.stride(0), lengths.ndim == 2, 32, 256,
        num_warps=4,
    )
    return result
