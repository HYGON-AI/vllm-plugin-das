# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""Request-major QSA scoring for packed prefill batches."""

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton


_TILE_R = 64
_BLOCK_N = 64
_K_TILES = 16


@triton.jit(do_not_specialize=["num_rows", "query_offset"])
def _qsa_mqa_paged_prefill_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    query_start_loc_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    stride_logits_column,
    num_rows,
    query_offset,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
) -> None:
    num_columns: tl.constexpr = PAGE_TABLE_WIDTH * PAGE_SIZE
    num_heads_padded: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    block_d: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))

    request = tl.program_id(0)
    query_base = tl.load(query_start_loc_ptr)
    query_end = query_offset + num_rows
    request_start = tl.maximum(
        tl.load(query_start_loc_ptr + request) - query_base,
        query_offset,
    )
    request_end = tl.minimum(
        tl.load(query_start_loc_ptr + request + 1) - query_base,
        query_end,
    )
    absolute_row_start = request_start + tl.program_id(1) * TILE_R
    if absolute_row_start >= request_end:
        return

    lanes = tl.arange(0, TILE_R)
    absolute_rows = absolute_row_start + lanes
    output_rows = absolute_rows - query_offset
    valid_rows = absolute_rows < request_end
    visible = tl.load(
        visible_blocks_ptr + absolute_rows,
        mask=valid_rows,
        other=0,
    )
    max_visible = tl.max(visible, axis=0)
    k_tile_start = tl.program_id(2) * K_TILES
    if k_tile_start * BLOCK_N >= max_visible:
        return
    k_tile_end = tl.minimum(
        k_tile_start + K_TILES,
        tl.cdiv(max_visible, BLOCK_N),
    )
    k_tile_end = tl.minimum(k_tile_end, tl.cdiv(num_columns, BLOCK_N))

    dims = tl.arange(0, block_d)
    query_lanes = tl.arange(0, TILE_R * num_heads_padded)
    query_row_offsets = query_lanes // num_heads_padded
    query_rows = absolute_row_start + query_row_offsets
    heads = query_lanes % num_heads_padded
    query = tl.load(
        q_ptr
        + query_rows[None, :] * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None] * stride_q_dim,
        mask=(heads[None, :] < NUM_HEADS)
        & (query_rows[None, :] < request_end)
        & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )

    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(k_tile_start, k_tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr
            + request * stride_table_req
            + logical_page * stride_table_page,
            mask=live,
            other=0,
        )
        keys = tl.load(
            k_cache_ptr
            + physical_page[:, None].to(tl.int64) * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.reshape(scores, (BLOCK_N, TILE_R, num_heads_padded))
        score = tl.sum(tl.maximum(scores, 0.0), axis=2) / HEAD_DIM**0.5
        store_mask = (
            valid_rows[None, :]
            & (columns[:, None] < visible[None, :])
            & (columns[:, None] < num_columns)
        )
        tl.store(
            logits_ptr
            + output_rows[None, :] * stride_logits_row
            + columns[:, None] * stride_logits_column,
            score,
            mask=store_mask,
        )


def _prefill_chunks(
    rows: int,
    columns: int,
    max_logits_bytes: int,
) -> tuple[tuple[int, int], ...]:
    """Partition score rows so each FP32 logits chunk fits the byte budget."""

    if rows < 0:
        raise ValueError("rows must be non-negative")
    if columns <= 0:
        raise ValueError("columns must be positive")
    if max_logits_bytes <= 0:
        raise ValueError("max_logits_bytes must be positive")
    rows_per_chunk = max(1, max_logits_bytes // (columns * 4))
    return tuple(
        (start, min(start + rows_per_chunk, rows))
        for start in range(0, rows, rows_per_chunk)
    )


def _validate_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    max_query_len: int,
    query_offset: int,
    num_queries: int,
) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA query must be [rows, heads, head_dim]")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA query and cache dimensions must match")
    if page_table.ndim != 2 or not all(page_table.shape):
        raise ValueError("QSA page table must be a nonempty matrix")
    if query_start_loc.shape != (page_table.shape[0] + 1,):
        raise ValueError("query_start_loc must have one terminal request offset")
    if visible_blocks.shape != (q.shape[0],):
        raise ValueError("visible_blocks must have one value per query row")
    if max_query_len <= 0:
        raise ValueError("max_query_len must be positive")
    if query_offset < 0 or num_queries < 0 or query_offset + num_queries > q.shape[0]:
        raise ValueError("query slice is outside the packed Q rows")
    tensors = (q, k_cache, page_table, query_start_loc, visible_blocks)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("QSA prefill tensors must share one device")
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16:
        raise ValueError("QSA prefill scoring requires BF16 Q and K")
    if page_table.dtype != torch.int32:
        raise ValueError("QSA page table must use int32")
    if query_start_loc.dtype != torch.int32 or visible_blocks.dtype != torch.int32:
        raise ValueError("QSA prefill offsets and visible counts must use int32")
    if not query_start_loc.is_contiguous() or not visible_blocks.is_contiguous():
        raise ValueError("QSA prefill row metadata must be contiguous")
    if not q.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA prefill scoring requires an HCU GPU and Triton")


def qsa_mqa_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    max_query_len: int,
    *,
    query_offset: int = 0,
    num_queries: int | None = None,
) -> torch.Tensor:
    """Return paged MQA scores for one packed prefill row slice."""

    if num_queries is None:
        num_queries = q.shape[0] - query_offset
    _validate_inputs(
        q,
        k_cache,
        page_table,
        query_start_loc,
        visible_blocks,
        max_query_len,
        query_offset,
        num_queries,
    )
    columns = page_table.shape[1] * k_cache.shape[1]
    logits = torch.empty(
        (num_queries, columns),
        dtype=torch.float32,
        device=q.device,
    )
    if num_queries == 0:
        return logits

    grid = (
        page_table.shape[0],
        triton.cdiv(min(num_queries, max_query_len), _TILE_R),
        triton.cdiv(columns, _BLOCK_N * _K_TILES),
    )
    _qsa_mqa_paged_prefill_kernel[grid](
        q,
        k_cache,
        page_table,
        query_start_loc,
        visible_blocks,
        logits,
        *q.stride(),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        *page_table.stride(),
        *logits.stride(),
        num_queries,
        query_offset,
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        TILE_R=_TILE_R,
        BLOCK_N=_BLOCK_N,
        K_TILES=_K_TILES,
        STAGES=2,
        num_warps=4,
    )
    return logits


__all__ = ["qsa_mqa_paged_prefill"]

