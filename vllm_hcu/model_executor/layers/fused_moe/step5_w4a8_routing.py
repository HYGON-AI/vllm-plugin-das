"""Private64-row W4 permutation: block histograms, stable offsets, token gather."""

import torch
import triton
import triton.language as tl
from .step5_w4a8_policy import padded_route_rows


@triton.jit
def _histogram(
    ids,
    mapping,
    partial,
    indices,
    inverse,
    R: tl.constexpr,
    E: tl.constexpr,
    G: tl.constexpr,
    BOUND: tl.constexpr,
    NB: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
):
    group = tl.program_id(0)
    r = group * B + tl.arange(0, B)
    gid = tl.load(ids + r, mask=r < R, other=-1)
    local = tl.load(mapping + gid, mask=(r < R) & (gid >= 0) & (gid < G), other=-1)
    local = tl.where((local >= 0) & (local < E), local, E)
    counts = tl.histogram(local, H)
    e = tl.arange(0, H)
    tl.store(partial + group * E + e, counts, mask=e < E)
    tl.store(inverse + r, -1, mask=r < R)
    tl.store(indices + r, -1, mask=r < BOUND)
    tail = r + NB * B
    tl.store(indices + tail, -1, mask=tail < BOUND)


@triton.jit
def _block_prefixes(
    partial,
    prefix,
    E: tl.constexpr,
    NB: tl.constexpr,
    RB: tl.constexpr,
    EH: tl.constexpr,
):
    r = tl.arange(0, RB)
    e = tl.arange(0, EH)
    values = tl.load(
        partial + r[:, None] * E + e[None, :],
        mask=(r[:, None] < NB) & (e[None, :] < E),
        other=0,
    )
    counts = tl.sum(values, axis=0)
    padded = tl.cdiv(counts, 64) * 64
    starts = tl.cumsum(padded, axis=0) - padded
    block_starts = starts[None, :] + tl.cumsum(values, axis=0) - values
    tl.store(
        prefix + r[:, None] * E + e[None, :],
        block_starts,
        mask=(r[:, None] < NB) & (e[None, :] < E),
    )


@triton.jit
def _assign_positions(
    ids,
    mapping,
    prefix,
    inverse,
    indices,
    R: tl.constexpr,
    E: tl.constexpr,
    G: tl.constexpr,
    B: tl.constexpr,
):
    group = tl.program_id(0)
    expert = tl.program_id(1)
    r = group * B + tl.arange(0, B)
    inside = r < R
    gid = tl.load(ids + r, mask=inside, other=-1)
    local = tl.load(mapping + gid, mask=inside & (gid >= 0) & (gid < G), other=-1)
    mine = inside & (local == expert)
    if tl.sum(mine.to(tl.int32), 0) > 0:
        start = tl.load(prefix + group * E + expert)
        pos = start + tl.cumsum(mine.to(tl.int32), axis=0) - 1
        tl.store(inverse + r, pos, mask=mine)
        tl.store(indices + pos, expert, mask=mine)


@triton.jit
def _gather_tokens(
    a,
    scale,
    inverse,
    out,
    out_scale,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    slots = tl.arange(0, triton.next_power_of_2(TOPK))
    positions = tl.load(inverse + token * TOPK + slots, mask=slots < TOPK, other=-1)
    if tl.sum((positions >= 0).to(tl.int32), 0) > 0:
        value = tl.load(a + token.to(tl.int64) * K + col, mask=col < K, other=0)
        s = tl.load(scale + token)
        for slot in tl.static_range(0, TOPK):
            pos = tl.load(inverse + token * TOPK + slot)
            if pos >= 0:
                tl.store(out + pos.to(tl.int64) * K + col, value, mask=col < K)
                tl.store(out_scale + pos, s)


def permute64(a, scale, ids, mapping, experts, out):
    assert a.is_contiguous() and ids.is_contiguous() and mapping.is_contiguous()
    assert scale.is_contiguous() and scale.shape[1] == 1 and a.dtype == torch.int8
    routes = ids.numel()
    groups = triton.cdiv(routes, 1024)
    bound = padded_route_rows(ids.shape[0], ids.shape[1], experts)
    assert out.shape == (bound, a.shape[1])
    partial = torch.empty((groups, experts), device=a.device, dtype=torch.int32)
    prefix = torch.empty_like(partial)
    indices = torch.empty((bound,), device=a.device, dtype=torch.int32)
    inverse = torch.empty_like(ids)
    scales = torch.empty((bound, 1), device=a.device, dtype=torch.float32)
    _histogram[(groups,)](
        ids,
        mapping,
        partial,
        indices,
        inverse,
        routes,
        experts,
        mapping.numel(),
        bound,
        groups,
        1024,
        triton.next_power_of_2(experts + 1),
        num_warps=4,
    )
    _block_prefixes[(1,)](
        partial,
        prefix,
        experts,
        groups,
        triton.next_power_of_2(groups),
        triton.next_power_of_2(experts),
        num_warps=4,
    )
    _assign_positions[(groups, experts)](
        ids,
        mapping,
        prefix,
        inverse,
        indices,
        routes,
        experts,
        mapping.numel(),
        1024,
        num_warps=4,
    )
    _gather_tokens[(ids.shape[0],)](
        a,
        scale,
        inverse,
        out,
        scales,
        a.shape[1],
        ids.shape[1],
        triton.next_power_of_2(a.shape[1]),
        num_warps=4,
    )
    return out, scales, indices, inverse, 64
