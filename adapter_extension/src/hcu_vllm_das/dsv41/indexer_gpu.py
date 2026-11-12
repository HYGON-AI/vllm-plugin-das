# SPDX-License-Identifier: Apache-2.0
"""HCU SHUFFLE-cache decode scoring; scheduling and top-k stay in Core."""
import torch
from vllm.triton_utils import tl, triton

@triton.jit
def _paged_scores(q, cache_values, cache_scales, weights, ends,
                  tables, logits, error,
                  q_row_stride, q_head_stride, q_dim_stride,
                  weight_row_stride, weight_head_stride,
                  cache_page_bytes, table_row_stride, table_col_stride,
                  WIDTH: tl.constexpr, TABLE_ROWS: tl.constexpr,
                  TABLE_COLS: tl.constexpr, PAGES: tl.constexpr,
                  PAGE_SIZE: tl.constexpr, HEADS: tl.constexpr,
                  HEAD_TILE: tl.constexpr, KV_TILE: tl.constexpr,
                  SPLITS: tl.constexpr):
    row = tl.program_id(0)
    split = tl.program_id(1)
    req = row
    end = tl.load(ends + row)
    valid_row = (req >= 0) & (req < TABLE_ROWS)
    if (end < 0) | (end > WIDTH) | (end > TABLE_COLS * PAGE_SIZE) | ~valid_row:
        tl.atomic_or(error, 2)
    length = tl.minimum(tl.maximum(end, 0), tl.minimum(WIDTH, TABLE_COLS * PAGE_SIZE))
    h = tl.arange(0, HEAD_TILE)
    d = tl.arange(0, 128)
    query = tl.load(q + row * q_row_stride + h[:, None] * q_head_stride
                    + d[None, :] * q_dim_stride, h[:, None] < HEADS, other=0.0).to(tl.float32)
    weight = tl.load(weights + row * weight_row_stride + h * weight_head_stride,
                     h < HEADS, other=0).to(tl.float32)
    k = tl.arange(0, KV_TILE)
    for first in tl.range(split * KV_TILE, length, SPLITS * KV_TILE):
        pos = first + k
        logical_page = pos // PAGE_SIZE
        valid_pos = (pos < length) & valid_row
        page = tl.load(tables + req * table_row_stride + logical_page * table_col_stride,
                       valid_pos, other=-1)
        safe = valid_pos & (page >= 0) & (page < PAGES)
        if tl.sum((valid_pos & ~safe).to(tl.int32), 0) != 0:
            tl.atomic_or(error, 4)
        page = tl.where(safe, page, 0).to(tl.int64)
        slot = pos % PAGE_SIZE
        if PAGE_SIZE == 1:
            in_page = d[:, None] + tl.zeros((1, KV_TILE), tl.int32)
        else:
            # Values are [block/16, D/16, 16 tokens, 16 bytes], then FP32 scales.
            in_page = ((slot[None, :] // 16) * (16 * 128)
                       + (d[:, None] // 16) * 256
                       + (slot[None, :] % 16) * 16 + d[:, None] % 16)
        key = tl.load(cache_values + page[None, :] * cache_page_bytes + in_page,
                      safe[None, :], other=0.0).to(tl.float32)
        scale = tl.load(cache_scales + page * (cache_page_bytes // 4)
                        + PAGE_SIZE * 128 // 4 + slot, safe, other=0)
        dot = tl.dot(query, key, input_precision='ieee')
        score = tl.sum(tl.maximum(dot, 0.0) * weight[:, None], 0) * scale
        tl.store(logits + row * WIDTH + pos,
                 tl.where(safe, score, -float('inf')), pos < length)


def paged_logits(q, cache, weights, ends, tables, width):
    """Score compact FP8 queries without host reads or a full-cache temporary."""
    rows, heads, dim = q.shape
    if dim != 128 or heads <= 0 or q.stride(-1) != 1:
        raise ValueError('V41 GPU scoring requires FP8 [tokens,heads,128]')
    if (tables.ndim != 2 or tables.shape[0] != rows or ends.shape != (rows,)
            or weights.shape != (rows, heads) or width < 0 or width > 2147483647):
        raise ValueError('V41 GPU scoring row/table/weight shapes disagree')
    for tensor in (tables, ends):
        if tensor.device != q.device or tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError('V41 GPU scoring metadata requires same-device integer tensors')
    logits = torch.full((rows, width), -float('inf'), dtype=torch.float32, device=q.device)
    if not rows or not width:
        return logits
    error = torch.zeros((), dtype=torch.int32, device=q.device)
    splits = max(1, min(triton.cdiv(width, 32), max(1, 256 // rows)))
    _paged_scores[(rows, splits)](
        q, cache.view(q.dtype), cache.view(torch.float32), weights, ends,
        tables, logits, error, *q.stride(), *weights.stride(), cache.stride(0),
        *tables.stride(), WIDTH=width, TABLE_ROWS=tables.shape[0],
        TABLE_COLS=tables.shape[1], PAGES=cache.shape[0], PAGE_SIZE=cache.shape[1],
        HEADS=heads, HEAD_TILE=max(16, triton.next_power_of_2(heads)), KV_TILE=32,
        SPLITS=splits, num_warps=4, enable_fp_fusion=False)
    # Device-side assertion preserves fail-closed page checks without D2H.
    torch._assert_async(error == 0, 'V41 indexer invalid context length or cache page')
    return logits
