import triton
import torch
import triton.language as tl
from triton.language.extra import libdevice
from .step5_w4a8_policy import padded_route_rows


@triton.jit
def _row_value(ptr, row, col, mask, I: tl.constexpr):
    offset = row[:, None].to(tl.int64) * (2 * I) + col[None, :]
    gate = tl.load(ptr + offset, mask=mask, other=0).to(tl.float32)
    up = tl.load(ptr + offset + I, mask=mask, other=0).to(tl.float32)
    return (
        (
            tl.minimum(gate * tl.sigmoid(gate), 7.0)
            * tl.minimum(tl.maximum(up, -7.0), 7.0)
        )
        .to(tl.bfloat16)
        .to(tl.float32)
    )


@triton.jit
def activation_persistent(
    gateup,
    quant,
    scales,
    experts,
    ROWS: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    CH: tl.constexpr,
):
    tl.static_assert(I == 3 * CH)
    col = tl.arange(0, CH)
    step = tl.num_programs(0) * BM
    for start in range(tl.program_id(0) * BM, ROWS, step):
        row = start + tl.arange(0, BM)
        inside = row < ROWS
        valid = inside & (tl.load(experts + row, mask=inside, other=-1) >= 0)
        if tl.sum(valid.to(tl.int32), 0) > 0:
            mask = valid[:, None]
            y0 = _row_value(gateup, row, col, mask, I)
            y1 = _row_value(gateup, row, col + CH, mask, I)
            y2 = _row_value(gateup, row, col + 2 * CH, mask, I)
            maximum = tl.maximum(
                tl.maximum(tl.max(tl.abs(y0), 1), tl.max(tl.abs(y1), 1)),
                tl.max(tl.abs(y2), 1),
            )
            scale = tl.where(maximum > 0, maximum / 127.0, 1.0)
            offset = row[:, None].to(tl.int64) * I + col[None, :]
            q0 = tl.minimum(
                tl.maximum(libdevice.nearbyint(y0 / scale[:, None]), -127.0), 127.0
            ).to(tl.int8)
            q1 = tl.minimum(
                tl.maximum(libdevice.nearbyint(y1 / scale[:, None]), -127.0), 127.0
            ).to(tl.int8)
            q2 = tl.minimum(
                tl.maximum(libdevice.nearbyint(y2 / scale[:, None]), -127.0), 127.0
            ).to(tl.int8)
            tl.store(quant + offset, q0, mask=mask)
            tl.store(quant + offset + CH, q1, mask=mask)
            tl.store(quant + offset + 2 * CH, q2, mask=mask)
            tl.store(scales + row, tl.where(valid, scale, 1.0), mask=inside)
        else:
            tl.store(scales + row, 1.0, mask=inside)


@triton.jit
def _small_route_meta(
    ids,
    mapping,
    inverse,
    expert_ids,
    ROUTES: tl.constexpr,
    GLOBAL_E: tl.constexpr,
    E: tl.constexpr,
    ALIGN: tl.constexpr,
    BOUND: tl.constexpr,
    BLOCK_R: tl.constexpr,
    EBINS: tl.constexpr,
    BLOCK_BOUND: tl.constexpr,
):
    expert = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    inside = r < ROUTES
    gid = tl.load(ids + r, mask=inside, other=-1)
    local = tl.load(
        mapping + gid, mask=inside & (gid >= 0) & (gid < GLOBAL_E), other=-1
    )
    local = tl.where((local >= 0) & (local < E), local, E)
    counts = tl.histogram(local, EBINS)
    e = tl.arange(0, EBINS)
    padded = tl.where(e < E, tl.cdiv(counts, ALIGN) * ALIGN, 0)
    if expert < E:
        start = tl.sum(tl.where(e < expert, padded, 0), 0)
        n = tl.sum(tl.where(e == expert, counts, 0), 0)
        npad = tl.cdiv(n, ALIGN) * ALIGN
        mine = inside & (local == expert)
        position = start + tl.cumsum(mine.to(tl.int32), 0) - 1
        tl.store(inverse + r, position, mask=mine)
        tl.store(expert_ids + position, expert, mask=mine)
        j = tl.arange(0, ALIGN)
        pad = start + n + j
        tl.store(expert_ids + pad, -1, mask=n + j < npad)
    else:
        tl.store(inverse + r, -1, mask=inside & (local == E))
        total = tl.sum(padded, 0)
        tail = tl.arange(0, BLOCK_BOUND)
        tl.store(expert_ids + tail, -1, mask=(tail >= total) & (tail < BOUND))


@triton.jit
def _small_route_gather(
    a,
    scale,
    inverse,
    out,
    out_scale,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    route = tl.program_id(0)
    position = tl.load(inverse + route)
    if position >= 0:
        token = route // TOPK
        col = tl.arange(0, BLOCK)
        value = tl.load(a + token.to(tl.int64) * K + col, mask=col < K, other=0)
        tl.store(out + position.to(tl.int64) * K + col, value, mask=col < K)
        tl.store(out_scale + position, tl.load(scale + token * SCALE_STRIDE))


def small_permute(a, scale, ids, mapping, experts, out, global_experts=352):
    routes = ids.numel()
    bound = padded_route_rows(ids.shape[0], ids.shape[1], experts)
    assert out.shape == (bound, a.shape[1]) and 0 < routes <= 2048
    indices = torch.empty((bound,), device=a.device, dtype=torch.int32)
    inverse = torch.empty_like(ids)
    scales = torch.empty((bound, 1), device=a.device, dtype=torch.float32)
    _small_route_meta[(experts + 1,)](
        ids,
        mapping,
        inverse,
        indices,
        routes,
        global_experts,
        experts,
        64,
        bound,
        triton.next_power_of_2(routes),
        triton.next_power_of_2(experts + 1),
        triton.next_power_of_2(bound),
        num_warps=1 if routes <= 256 else 8,
    )
    _small_route_gather[(routes,)](
        a,
        scale,
        inverse,
        out,
        scales,
        a.shape[1],
        ids.shape[1],
        scale.stride(0),
        triton.next_power_of_2(a.shape[1]),
        num_warps=4,
    )
    return out, scales, indices, inverse, 64


@triton.jit
def _activation(gateup, quant, scales, experts, I: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    if tl.load(experts + row) >= 0:
        g = tl.load(gateup + row * (2 * I) + col, col < I, other=0).to(tl.float32)
        u = tl.load(gateup + row * (2 * I) + I + col, col < I, other=0).to(tl.float32)
        y = (
            (tl.minimum(g * tl.sigmoid(g), 7.0) * tl.minimum(tl.maximum(u, -7.0), 7.0))
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        maximum = tl.max(tl.where(col < I, tl.abs(y), 0), 0)
        scale = tl.where(maximum > 0, maximum / 127.0, 1.0)
        q = tl.minimum(tl.maximum(libdevice.nearbyint(y / scale), -127.0), 127.0).to(
            tl.int8
        )
        tl.store(quant + row * I + col, q, col < I)
        tl.store(scales + row, scale)
    else:
        tl.store(scales + row, 1.0)


@triton.jit
def _weighted_output(
    a,
    inverse,
    weights,
    mapping,
    ids,
    output,
    shared,
    M: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    GLOBAL_E: tl.constexpr,
    HAS_MAP: tl.constexpr,
    ADD_SHARED: tl.constexpr,
    SHARED_START: tl.constexpr,
    SHARED_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for slot in tl.static_range(0, TOPK):
        e = tl.load(ids + row * TOPK + slot)
        valid = (e >= 0) & (e < GLOBAL_E)
        if HAS_MAP:
            local = tl.load(mapping + e, mask=valid, other=-1)
            valid = valid & (local >= 0)
        pos = tl.load(inverse + row * TOPK + slot, mask=valid, other=-1)
        # Padded all-local65536x8 routes can exceed2^31 elements at H4096.
        # Promote before multiplying, not after an overflowing int32 product.
        value = tl.load(
            a + pos.to(tl.int64) * K + cols,
            mask=valid & (pos >= 0) & (cols < K),
            other=0,
        ).to(tl.float32)
        w = tl.load(weights + row * TOPK + slot)
        acc += value * w
    # Match the separate BF16 routed-output/shared-add rounding boundary.
    rounded = acc.to(output.dtype.element_ty)
    if ADD_SHARED:
        own = (row >= SHARED_START) & (row < SHARED_START + SHARED_ROWS)
        value = tl.load(
            shared + (row - SHARED_START) * K + cols, mask=own & (cols < K), other=0
        ).to(tl.float32)
        rounded = (rounded.to(tl.float32) + value).to(output.dtype.element_ty)
    tl.store(output + row * K + cols, rounded, cols < K)
