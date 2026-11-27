# SPDX-License-Identifier: Apache-2.0
"""Scatter fixed HT INT8 receives without reading unused payload rows."""

import triton
import triton.language as tl


@triton.jit
def _aligned_starts(counts, starts, EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.arange(0, BLOCK)
    count = tl.load(counts + expert, expert < EXPERTS, other=0)
    padded = tl.cdiv(count, 256) * 256
    tl.store(starts + expert, tl.cumsum(padded) - padded, expert < EXPERTS)


@triton.jit
def _scatter_fixed_int8(
    data, scales, routes, starts, dest_data, dest_scales, indices, inverse,
    rows, data_stride, scale_stride, route_stride0, route_stride1,
    dest_stride, dest_scale_stride, inverse_stride,
    HIDDEN: tl.constexpr, TOPK: tl.constexpr, EXPERTS: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_D)
    slots = tl.arange(0, BLOCK_K)
    for row in range(tl.program_id(0), rows, tl.num_programs(0)):
        route_row = routes + row * route_stride0
        ids = tl.load(route_row + slots * route_stride1, slots < TOPK, other=-1)
        valid = (ids >= 0) & (ids < EXPERTS)
        # DeepEP initializes routing IDs in the unused tail, but not data/scales.
        if tl.sum(valid.to(tl.int32), 0) > 0:
            row_offset = tl.cast(row, tl.int64) * data_stride
            value = tl.load(data + row_offset + cols, cols < HIDDEN, other=0)
            scale = tl.load(scales + row * scale_stride)
            for slot in tl.range(TOPK):
                expert = tl.load(route_row + slot * route_stride1)
                if 0 <= expert and expert < EXPERTS:
                    dest = tl.atomic_add(starts + expert, 1, sem="relaxed")
                    tl.store(dest_data + dest.to(tl.int64) * dest_stride + cols, value, cols < HIDDEN)
                    tl.store(dest_scales + dest * dest_scale_stride, scale)
                    tl.store(indices + dest, expert)
                    tl.store(inverse + row * inverse_stride + slot, dest)


def scatter_fixed_int8(data, scales, routes, counts, starts, dest_data,
                       dest_scales, indices, inverse):
    rows, hidden = data.shape
    experts = counts.numel()
    topk = routes.shape[1]
    _aligned_starts[(1,)](counts, starts, experts, triton.next_power_of_2(experts))
    _scatter_fixed_int8[(min(rows, 8192),)](
        data, scales, routes, starts, dest_data, dest_scales, indices, inverse,
        rows, data.stride(0), scales.stride(0), routes.stride(0), routes.stride(1),
        dest_data.stride(0), dest_scales.stride(0), inverse.stride(0),
        hidden, topk, experts, triton.next_power_of_2(hidden),
        triton.next_power_of_2(topk), num_warps=8,
    )
