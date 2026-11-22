"""DeepEP alignment=1 exact-count pack with runtime count bounds.

CPU dispatch counts supply allocation size. No nonzero/item/bool(tensor) or
atomic index_add. Local IDs enter Bolt directly with expert_map=None, topk=1.
The expert callback must apply route weights BEFORE its BF16 output store.
Shared experts and outer routed scale remain owned by the model runner.
"""
import torch
import triton
import triton.language as tl


def route_capacity(counts, rows, topk, *, alignment=1):
    if alignment != 1:
        raise ValueError('counts must be unpadded DeepEP normal alignment=1')
    if not isinstance(counts, (tuple, list)) or len(counts) != 16:
        raise ValueError('expected 16 host expert counts')
    if any(type(n) is not int or n < 0 for n in counts):
        raise ValueError('expert counts must be nonnegative host ints')
    total = sum(counts)
    if total > rows * topk:
        raise ValueError('expert route count exceeds received route slots')
    return total


@triton.jit(do_not_specialize=['SLOTS'])
def _count_routes(Ids, Counts, SLOTS, E: tl.constexpr,
                  B: tl.constexpr):
    slots = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    ids = tl.load(Ids + slots, slots < SLOTS, other=-1)
    valid = (slots < SLOTS) & (ids >= 0) & (ids < E)
    tl.store(Counts + tl.program_id(0), tl.sum(valid.to(tl.int32), 0))


@triton.jit(do_not_specialize=['NB'])
def _scan_counts(Counts, Offsets, Total, NB, B: tl.constexpr):
    p = tl.arange(0, B)
    c = tl.load(Counts + p, p < NB, other=0)
    inclusive = tl.cumsum(c, 0)
    tl.store(Offsets + p, inclusive - c, p < NB)
    tl.store(Total, tl.sum(c, 0))


@triton.jit(do_not_specialize=['SLOTS', 'CAP'])
def _pack_routes(X, Ids, Weights, Offsets, PackedX, PackedIds, PackedWeights,
                 Inverse, SLOTS, TOPK: tl.constexpr,
                 H: tl.constexpr, CAP, E: tl.constexpr,
                 B: tl.constexpr, C: tl.constexpr):
    p = tl.program_id(0)
    slot = p.to(tl.int64) * B + tl.arange(0, B)
    ids = tl.load(Ids + slot, slot < SLOTS, other=-1)
    valid = (slot < SLOTS) & (ids >= 0) & (ids < E)
    dst = (tl.load(Offsets + p) + tl.cumsum(valid.to(tl.int32), 0) - 1).to(tl.int64)
    dst_ok = valid & (dst >= 0) & (dst < CAP)
    col = tl.program_id(1) * C + tl.arange(0, C)
    x = tl.load(X + (slot // TOPK)[:, None] * H + col[None, :],
                valid[:, None] & (col[None, :] < H), other=0)
    tl.store(PackedX + dst[:, None] * H + col[None, :], x,
             dst_ok[:, None] & (col[None, :] < H))
    if tl.program_id(1) == 0:
        tl.store(Inverse + slot, tl.where(dst_ok, dst, -1), slot < SLOTS)
        tl.store(PackedIds + dst, ids, dst_ok)
        w = tl.load(Weights + slot, valid, other=0)
        tl.store(PackedWeights + dst, w, dst_ok)


@triton.jit(do_not_specialize=['CAP'])
def _restore_routes(PackedY, Inverse, Y, H: tl.constexpr, TOPK: tl.constexpr,
                    CAP, C: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * C + tl.arange(0, C)
    acc = tl.zeros((C,), tl.float32)
    # Match Bolt moe_sum_kernel: FP32 sequential topk accumulation of BF16
    # weighted expert rows, then one BF16 store; no additional route weight.
    for k in range(TOPK):
        route = tl.load(Inverse + row * TOPK + k).to(tl.int64)
        if (route >= 0) & (route < CAP):
            value = tl.load(PackedY + route * H + col, col < H, other=0)
            acc += value.to(tl.float32)
    tl.store(Y + row * H + col, acc.to(Y.dtype.element_ty, fp_downcast_rounding='rtne'), col < H)


def run_received_local_routes(hidden, ids, weights, counts, expert_fn, *, alignment=1):
    """Return received-token output for DeepEP combine, not source-rank output.

    `expert_fn(packed_x, packed_weights[R,1], local_ids[R,1])` must call the
    existing W4A8 expert kernel with expert_map=None and return BF16 [R,H].
    No global expert ID remapping is performed. Device checks remain async.
    """
    if (hidden.ndim != 2 or hidden.dtype != torch.bfloat16 or not hidden.is_cuda
            or ids.ndim != 2 or ids.shape != weights.shape or ids.shape[0] != hidden.shape[0]
            or ids.shape[1] != 4 or ids.dtype not in (torch.int32, torch.int64)
            or weights.dtype != torch.float32):
        raise ValueError('expected BF16 [recv,H], local IDs/FP32 weights [recv,4]')
    if any(t.device != hidden.device or not t.is_contiguous() for t in (hidden, ids, weights)):
        raise ValueError('received tensors must be contiguous on one device')
    rows, h = hidden.shape
    cap = route_capacity(counts, rows, 4, alignment=alignment)
    torch._assert_async(((ids >= -1) & (ids < 16)).all(), 'DeepEP local IDs outside [-1,16)')
    if rows == 0:
        return torch.empty_like(hidden)
    slots = rows * 4
    block = 128
    nb = triton.cdiv(slots, block)
    block_counts = torch.empty((nb,), dtype=torch.int32, device=hidden.device)
    offsets = torch.empty_like(block_counts)
    total = torch.empty((), dtype=torch.int32, device=hidden.device)
    _count_routes[(nb,)](ids, block_counts, slots, 16, block)
    _scan_counts[(1,)](block_counts, offsets, total, nb, triton.next_power_of_2(nb))
    torch._assert_async(total == cap, 'DeepEP host counts do not match received valid routes')
    if cap == 0:
        return torch.zeros_like(hidden)
    px = torch.empty((cap,h), dtype=hidden.dtype, device=hidden.device)
    pi = torch.empty((cap,1), dtype=torch.int32, device=hidden.device)
    pw = torch.empty((cap,1), dtype=torch.float32, device=hidden.device)
    inv = torch.empty_like(ids, dtype=torch.int32)
    _pack_routes[(nb,triton.cdiv(h,64))](hidden,ids,weights,offsets,px,pi,pw,inv,
                                     slots,4,h,cap,16,block,64,num_warps=8)
    py = expert_fn(px,pw,pi)
    if (py.shape != px.shape or py.dtype != hidden.dtype or py.device != hidden.device
            or not py.is_contiguous()):
        raise ValueError('expert callback must return weighted contiguous BF16 route output')
    result = torch.empty_like(hidden)
    _restore_routes[(rows,triton.cdiv(h,256))](py,inv,result,h,4,cap,256)
    return result
