"""EP-aware replacement for boltops fused_experts_impl (fp8 W8A8 per-channel, expert_map set).

With an expert_map, boltops fused_experts_impl does work proportional to *all* M*topk routes
although only ~1/EP of them are local:
  - torch.zeros of cache13 (M*topk*4096) and intermediate_cache2 (M*topk*1536), plus fill_(0)
  - activation and per-token quant over all M*topk rows (fused act+quant is disabled for EP)
  - moe_sum reading all M*topk*4096 values
At M=16384 that is ~4.5 ms of a 7.6 ms MoE call.

This path always uses the expert-sorted bridge layout (GEMM1 c_sorted, GEMM2 a_sorted), so
local routes are contiguous in [0, num_tokens_post_padded). Then:
  - fused swiglustep + fp8 per-token quant runs only on rows < num_tokens_post_padded (read on
    device: CUDA-graph safe), folding the router weight into the per-row scale
  - GEMM2 writes route rows (t*topk+k) of an uninitialised buffer
  - a masked moe_sum reads only routes whose expert is local
  - nothing is zero-filled
Padding rows in the sorted layout hold garbage; every kernel treats rows independently and the
GEMMs mask padding rows on store, so garbage never reaches the output.
"""
import functools
import json
import os

import torch
import triton
import triton.language as tl

from boltops.fused_moe.triton import moe_op as _moe_op
from boltops.fused_moe.triton.moe_op import fused_moe as invoke_fused_moe_kernel
from boltops.fused_moe.triton.moe_config_utils import try_get_optimal_moe_config
from boltops.fused_moe.triton.moe_compat import per_token_quant_triton
from boltops.fused_moe.triton.moe_sorting_hipc import moe_sorting_fwd
from vllm_hcu.platforms import envs as henvs

FP8 = torch.float8_e4m3fn
_announced = False
STEP5_EXPERT_GRID = henvs.optional_custom_op_enabled(
    os.environ.get("HCU_STEP5_EXPERT_GRID", "0") == "1"
)
_step5_grid_announced = False
STEP5_PREFILL_G2 = henvs.optional_custom_op_enabled(
    os.environ.get("HCU_STEP5_PREFILL_G2", "0") == "1"
)
STEP5_PREFILL_ASM_G1 = henvs.optional_custom_op_enabled(
    os.environ.get("HCU_STEP5_PREFILL_ASM_G1", "0") == "1"
)


@triton.jit
def _ep_swiglu_chunk(in_ptr, offs, m2, LIMIT: tl.constexpr, D: tl.constexpr):
    x = tl.load(in_ptr + offs, mask=m2, other=0.0)
    up = tl.load(in_ptr + offs + D, mask=m2, other=0.0)
    xf = x.to(tl.float32)
    # fdiv_fast: the IEEE divide alone costs ~20% of this memory-bound kernel.
    silu = tl.math.fdiv_fast(xf, 1.0 + tl.exp(-xf))
    y = tl.minimum(silu, LIMIT) * tl.minimum(tl.maximum(up.to(tl.float32), -LIMIT), LIMIT)
    return y.to(x.dtype)


@triton.jit
def _ep_swiglustep_quant_kernel(
    q_ptr, scale_ptr, in_ptr, w_ptr, nvalid_ptr,
    D: tl.constexpr, LIMIT: tl.constexpr, DTYPE_MAX: tl.constexpr,
    BLOCK_M: tl.constexpr, CH: tl.constexpr,
):
    # Same math as boltops activation_quant_kernel(GATED_SWIGLU_STEP, row multiplier),
    # but grid-strided over the device-side valid row count. D = 3*CH: a power-of-2
    # BLOCK_D=2048 tile for D=1536 masks a quarter of the lanes and runs at ~0.6 TB/s.
    tl.static_assert(D == 3 * CH)
    n = tl.load(nvalid_ptr)
    cols = tl.arange(0, CH)
    step = tl.num_programs(0) * BLOCK_M
    for start in range(tl.program_id(0) * BLOCK_M, n, step):
        rows = start + tl.arange(0, BLOCK_M)
        row_mask = rows < n
        m2 = row_mask[:, None]
        offs = rows[:, None].to(tl.int64) * (2 * D) + cols[None, :]
        y0 = _ep_swiglu_chunk(in_ptr, offs, m2, LIMIT, D)
        y1 = _ep_swiglu_chunk(in_ptr, offs + CH, m2, LIMIT, D)
        y2 = _ep_swiglu_chunk(in_ptr, offs + 2 * CH, m2, LIMIT, D)
        absmax = tl.maximum(tl.maximum(tl.max(tl.abs(y0), axis=-1), tl.max(tl.abs(y1), axis=-1)),
                            tl.max(tl.abs(y2), axis=-1))
        scale = tl.maximum(absmax.to(tl.float32), 1.0e-10) / DTYPE_MAX
        inv = (1.0 / scale)[:, None]
        w = tl.load(w_ptr + rows, mask=row_mask, other=0.0)
        q_offs = rows[:, None].to(tl.int64) * D + cols[None, :]
        tl.store(q_ptr + q_offs, (y0 * inv).to(q_ptr.dtype.element_ty), mask=m2, cache_modifier=".cs")
        tl.store(q_ptr + q_offs + CH, (y1 * inv).to(q_ptr.dtype.element_ty), mask=m2, cache_modifier=".cs")
        tl.store(q_ptr + q_offs + 2 * CH, (y2 * inv).to(q_ptr.dtype.element_ty), mask=m2, cache_modifier=".cs")
        tl.store(scale_ptr + rows, scale * w, mask=row_mask)


@triton.jit
def _ep_swiglustep_quant_deepgemm_kernel(
    q_ptr, scale_ptr, in_ptr, expert_ids_ptr, M,
    D: tl.constexpr, LIMIT: tl.constexpr, DTYPE_MAX: tl.constexpr,
    BLOCK_M: tl.constexpr, CH: tl.constexpr,
):
    # DeepGEMM marks padding in a per-row array, including holes between
    # experts. Reuse native Step5's post-SiLU clamp and BF16 rounding before
    # calculating amax; LightOp's fused activation clamps before SiLU.
    tl.static_assert(D == 3 * CH)
    cols = tl.arange(0, CH)
    step = tl.num_programs(0) * BLOCK_M
    for start in range(tl.program_id(0) * BLOCK_M, M, step):
        rows = start + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        valid = tl.load(expert_ids_ptr + rows, mask=row_mask, other=-1) >= 0
        m2 = (row_mask & valid)[:, None]
        offs = rows[:, None].to(tl.int64) * (2 * D) + cols[None, :]
        y0 = _ep_swiglu_chunk(in_ptr, offs, m2, LIMIT, D)
        y1 = _ep_swiglu_chunk(in_ptr, offs + CH, m2, LIMIT, D)
        y2 = _ep_swiglu_chunk(in_ptr, offs + 2 * CH, m2, LIMIT, D)
        absmax = tl.maximum(tl.maximum(tl.max(tl.abs(y0), axis=-1), tl.max(tl.abs(y1), axis=-1)),
                            tl.max(tl.abs(y2), axis=-1))
        scale = tl.maximum(absmax.to(tl.float32), 1.0e-10) / DTYPE_MAX
        inv = (1.0 / scale)[:, None]
        q_offs = rows[:, None].to(tl.int64) * D + cols[None, :]
        store_mask = row_mask[:, None]
        tl.store(q_ptr + q_offs, (y0 * inv).to(q_ptr.dtype.element_ty), mask=store_mask, cache_modifier=".cs")
        tl.store(q_ptr + q_offs + CH, (y1 * inv).to(q_ptr.dtype.element_ty), mask=store_mask, cache_modifier=".cs")
        tl.store(q_ptr + q_offs + 2 * CH, (y2 * inv).to(q_ptr.dtype.element_ty), mask=store_mask, cache_modifier=".cs")
        # Router weights belong to DeepGEMM's unpermute/reduce after GEMM2.
        # Invalid rows store zero q and the harmless native zero-row scale.
        tl.store(scale_ptr + rows, scale, mask=row_mask)


@triton.jit
def _ep_masked_moe_sum_kernel(
    out_ptr, in_ptr, ids_ptr, emap_ptr, stride_ids, add_ptr, stride_add, add_start, add_end,
    N: tl.constexpr, TOPK: tl.constexpr, SCALE: tl.constexpr, BLOCK_N: tl.constexpr,
    HAS_ADD: tl.constexpr,
):
    t = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        e = tl.load(ids_ptr + t * stride_ids + k)
        if tl.load(emap_ptr + e) >= 0:
            acc += tl.load(in_ptr + (t.to(tl.int64) * TOPK + k) * N + offs).to(tl.float32)
    if SCALE != 1.0:
        acc *= SCALE
    if HAS_ADD:
        if (t >= add_start) & (t < add_end):
            acc += tl.load(add_ptr + (t - add_start).to(tl.int64) * stride_add + offs).to(tl.float32)
    tl.store(out_ptr + t.to(tl.int64) * N + offs, acc.to(out_ptr.dtype.element_ty))


# Direct combine for large M (HCU_EP_DIRECT_SUM=min M, 0 disables), used when the
# caller provides scratch rows right after out (direct_scratch, set by the DP MoE pipe).
# GEMM2 stores each token's first local route straight into out[token] and only the
# further routes of tokens with 2+ local experts into compact scratch rows;
# _ep_direct_fix_kernel adds those (and zero-fills tokens with no local route). Same
# fp32 sum order as ep_masked_moe_sum, so bit-exact, without the (M, topk, K) round trip.
# If the scratch would overflow (very skewed routing), device-side gates run the
# cache3 + masked-sum path instead, so no host sync is needed.
DIRECT_MIN = (
    int(os.environ.get("HCU_EP_DIRECT_SUM", "0"))
    if henvs.custom_ops_enabled() else 0
)
direct_scratch = None
_DIRECT_BIG = 2**31 - 1
DIRECT_TPP = 1
_direct_announced = False


@triton.jit
def _ep_direct_info_kernel(ids_ptr, emap_ptr, info_ptr, inv_ptr, ctr_ptr, M,
                           TOPK: tl.constexpr, BLOCK: tl.constexpr):
    # info[t] = first local k (TOPK if none) | local route count << 4; inv[t, :] = -1
    # (routes the sort drops, e.g. duplicate ids in padding rows, stay -1).
    if tl.program_id(0) == 0:
        tl.store(ctr_ptr, 0)
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tm = t < M
    c = tl.zeros((BLOCK,), tl.int32)
    f = tl.full((BLOCK,), TOPK, tl.int32)
    for k in tl.static_range(TOPK):
        e = tl.load(ids_ptr + t * TOPK + k, mask=tm, other=0)
        loc = tl.load(emap_ptr + e, mask=tm, other=-1) >= 0
        f = tl.where(loc & (f == TOPK), k, f)
        c += loc.to(tl.int32)
        tl.store(inv_ptr + t * TOPK + k, -1, mask=tm)
    tl.store(info_ptr + t, f | (c << 4), mask=tm)


@triton.jit
def _ep_direct_rows_kernel(sids_ptr, nvalid_ptr, info_ptr, rows_ptr, inv_ptr, ctr_ptr, M, XOFF,
                           BOUND, TOPK: tl.constexpr, BIG: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pm = p < BOUND
    s = tl.load(sids_ptr + p, mask=pm, other=0)
    t = s & 0x00FFFFFF
    k = s >> 24
    ok = pm & (p < tl.load(nvalid_ptr)) & (t < M)
    f = tl.load(info_ptr + t, mask=ok, other=0) & 15
    extra = ok & (k != f)
    e32 = extra.to(tl.int32)
    slot = tl.atomic_add(ctr_ptr, tl.sum(e32, 0)) + tl.cumsum(e32, 0) - e32
    row = tl.where(extra, XOFF + slot, t)
    tl.store(rows_ptr + p, tl.where(ok, row, BIG), mask=pm)
    tl.store(inv_ptr + t * TOPK + k, slot, mask=extra)


@triton.jit
def _ep_direct_gate_kernel(nvalid_ptr, ctr_ptr, gate_ptr, CAP):
    # gate = [nvalid for the direct GEMM2, nvalid for the fallback GEMM2, fallback on]
    nv = tl.load(nvalid_ptr)
    over = tl.load(ctr_ptr) > CAP
    tl.store(gate_ptr, tl.where(over, 0, nv))
    tl.store(gate_ptr + 1, tl.where(over, nv, 0))
    tl.store(gate_ptr + 2, over.to(tl.int32))


@triton.jit
def _ep_direct_fix_kernel(out_ptr, x_ptr, c3_ptr, ids_ptr, emap_ptr, info_ptr, inv_ptr, gate_ptr,
                          M, N: tl.constexpr, TOPK: tl.constexpr, BLOCK_N: tl.constexpr,
                          TPP: tl.constexpr):
    # TPP tokens per program: most tokens have exactly one local route and need nothing.
    over = tl.load(gate_ptr + 2) != 0
    offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    for i in range(TPP):
        t = tl.program_id(0) * TPP + i
        if t < M:
            info = tl.load(info_ptr + t)
            c = info >> 4
            if over | (c != 1):
                acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
                if over:  # cache3 fallback: ep_masked_moe_sum
                    for k in tl.static_range(TOPK):
                        e = tl.load(ids_ptr + t * TOPK + k)
                        if tl.load(emap_ptr + e) >= 0:
                            acc += tl.load(c3_ptr + (t.to(tl.int64) * TOPK + k) * N
                                           + offs).to(tl.float32)
                elif c > 1:  # first local route is already in out[t]
                    f = info & 15
                    acc += tl.load(out_ptr + t.to(tl.int64) * N + offs).to(tl.float32)
                    for k in tl.static_range(TOPK):
                        if k > f:
                            j = tl.load(inv_ptr + t * TOPK + k).to(tl.int64)
                            if j >= 0:
                                acc += tl.load(x_ptr + j * N + offs).to(tl.float32)
                tl.store(out_ptr + t.to(tl.int64) * N + offs, acc.to(out_ptr.dtype.element_ty))


@functools.lru_cache(maxsize=None)
def _num_cus(dev):
    return torch.cuda.get_device_properties(dev).multi_processor_count


ACT_CFG = {"BLOCK_M": 2, "num_warps": 1, "CTAS_PER_CU": 64}
SUM_CFG = {"BLOCK_N": 4096, "num_warps": 4}


def ep_swiglustep_quant(cache1, sorted_weights, nvalid, rows, limit=7.0, q=None):
    D = cache1.shape[1] // 2
    if q is None:
        q = torch.empty((rows, D), dtype=FP8, device=cache1.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=cache1.device)
    bm = ACT_CFG["BLOCK_M"]
    grid = (max(1, min(triton.cdiv(rows, bm), _num_cus(cache1.device.index) * ACT_CFG["CTAS_PER_CU"])),)
    assert D % 3 == 0 and triton.next_power_of_2(D // 3) == D // 3
    _ep_swiglustep_quant_kernel[grid](
        q, scale, cache1, sorted_weights, nvalid,
        D=D, LIMIT=float(limit), DTYPE_MAX=torch.finfo(FP8).max,
        BLOCK_M=bm, CH=D // 3, num_warps=ACT_CFG["num_warps"], num_stages=1)
    return q, scale


def ep_swiglustep_quant_deepgemm(cache1, expert_ids, q, limit=7.0):
    """Native Step5 activation/quantization for DeepGEMM's padded row layout.

    ``q`` is caller-owned scratch; only per-row scales are allocated here.
    Input and output must be disjoint contiguous buffers. Negative expert ids
    mask input loads and produce deterministic zero padding, without a device
    row-count scalar or a host synchronization.
    """
    if (cache1.dtype != torch.bfloat16 or cache1.ndim != 2
            or cache1.shape[1] != 3072 or not cache1.is_contiguous()):
        raise ValueError("Step5 DeepGEMM activation requires contiguous BF16 [M, 3072]")
    rows, n = cache1.shape
    d = n // 2
    if (expert_ids.dtype not in (torch.int32, torch.int64)
            or expert_ids.shape != (rows,) or not expert_ids.is_contiguous()
            or expert_ids.device != cache1.device):
        raise ValueError("Step5 DeepGEMM requires contiguous per-row expert ids on the input device")
    if (q.dtype != FP8 or q.shape != (rows, d) or not q.is_contiguous()
            or q.device != cache1.device):
        raise ValueError("Step5 DeepGEMM requires contiguous FP8 [M, 1536] output on the input device")
    if rows:
        start, other = cache1.data_ptr(), q.data_ptr()
        if (start < other + q.numel() * q.element_size()
                and other < start + cache1.numel() * cache1.element_size()):
            raise ValueError("Step5 DeepGEMM activation input and output must not overlap")
    scale = torch.empty((rows, 1), dtype=torch.float32, device=cache1.device)
    if rows:
        bm = ACT_CFG["BLOCK_M"]
        grid = (max(1, min(triton.cdiv(rows, bm),
                           _num_cus(cache1.device.index) * ACT_CFG["CTAS_PER_CU"])),)
        _ep_swiglustep_quant_deepgemm_kernel[grid](
            q, scale, cache1, expert_ids, rows,
            D=d, LIMIT=float(limit), DTYPE_MAX=torch.finfo(FP8).max,
            BLOCK_M=bm, CH=d // 3, num_warps=ACT_CFG["num_warps"], num_stages=1)
    return q, scale


def ep_masked_moe_sum(cache3, out, topk_ids, expert_map, scale=1.0, add=None, add_start=0):
    """out[t] = scale * sum of the local-expert routes of token t, plus
    add[t - add_start] for the rows add covers."""
    M, topk, N = cache3.shape
    bn = SUM_CFG["BLOCK_N"]
    assert N % bn == 0 and topk_ids.stride(1) == 1
    if add is not None:
        assert add.shape[1] == N and add.stride(1) == 1
        assert 0 <= add_start and add_start + add.shape[0] <= M
    _ep_masked_moe_sum_kernel[(M, N // bn)](
        out, cache3, topk_ids, expert_map, topk_ids.stride(0),
        out if add is None else add, 0 if add is None else add.stride(0),
        add_start, add_start + (0 if add is None else add.shape[0]),
        N=N, TOPK=topk, SCALE=float(scale), BLOCK_N=bn, HAS_ADD=add is not None,
        num_warps=SUM_CFG["num_warps"])


# Set by the MoE runner under VLLM_HCU_EP_FUSED_COMBINE: given the (M, K) output
# buffer, returns (shared_output, row_start) when the running layer wants its
# shared-expert output added into those rows, else None.
shared_add_provider = None


@triton.jit
def _ep_sort_kernel(ids_ptr, w_ptr, emap_ptr, sids_ptr, sw_ptr, eids_ptr, nvalid_ptr, M,
                    TOPK: tl.constexpr, LE_P: tl.constexpr, BM: tl.constexpr,
                    BLOCK_R: tl.constexpr, WRITE_LAYOUT: tl.constexpr = False,
                    layout_ptr=None):
    # Same layout as moe_sorting_fwd with a local expert mask, for small M: one
    # program per local expert x. Every program histograms all routes to find
    # where x's block starts, then writes x's routes (in route order), its
    # padding rows and its expert_ids entries. Non-local routes go to bin LE_P.
    x = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    rm = r < M * TOPK
    le = tl.load(emap_ptr + tl.load(ids_ptr + r, mask=rm, other=0), mask=rm, other=-1)
    le = tl.where(le < 0, LE_P, le)
    cnt = tl.histogram(le, 2 * LE_P)
    xs = tl.arange(0, 2 * LE_P)
    padded = tl.where(xs < LE_P, (cnt + BM - 1) // BM * BM, 0)
    start = tl.sum(tl.where(xs < x, padded, 0), axis=0)
    n = tl.sum(tl.where(xs == x, cnt, 0), axis=0)
    npad = (n + BM - 1) // BM * BM
    mine = le == x
    pos = start + tl.cumsum(mine.to(tl.int32), axis=0) - 1
    tok = r // TOPK
    tl.store(sids_ptr + pos, tok | ((r - tok * TOPK) << 24), mask=mine)
    tl.store(sw_ptr + pos, tl.load(w_ptr + r, mask=mine, other=0.0), mask=mine)
    j = tl.arange(0, BM)
    pm = n + j < npad
    tl.store(sids_ptr + start + n + j, tl.zeros((BM,), tl.int32) + (M | (TOPK << 24)), mask=pm)
    tl.store(sw_ptr + start + n + j, tl.zeros((BM,), tl.float32), mask=pm)
    for b in range(start // BM, (start + npad) // BM):
        tl.store(eids_ptr + b, x)
    if WRITE_LAYOUT:
        tl.store(layout_ptr + x * 2, start // BM)
        tl.store(layout_ptr + x * 2 + 1, npad // BM)
    if x == 0:
        tl.store(nvalid_ptr, tl.sum(padded, axis=0))


# Largest M*topk sorted by _ep_sort_kernel (all routes live in one tile); above
# it moe_sorting_fwd is used. At decode sizes the Triton sort takes 3-6 us vs
# 13-27 us for the CK multi-phase sort plus its workspace memset.
EP_SORT_MAX_ROUTES = (
    int(os.environ.get("EP_FUSED_MOE_SORT_MAX_ROUTES", "2048"))
    if henvs.custom_ops_enabled() else 2048
)


def ep_sorting(topk_ids, topk_weights, expert_mask, global_E, local_E, bm, expert_map=None,
               expert_layout=None):
    """moe_sorting_ck with buffers bounded by the *local* expert count.

    Only local experts occupy the sorted layout, so its length is at most
    M*topk + min(M*topk, local_E)*(bm-1) rounded up to bm; the GEMM grid is sized from
    sorted_ids.size(0), which in boltops is padded for all global experts.
    No moe_buf: the sorting kernel would only zero it, and nothing here reads it.
    With expert_map and few routes, sorts with _ep_sort_kernel (same output).
    """
    dev = topk_ids.device
    M, topk = topk_ids.shape
    routes = M * topk
    full = routes + global_E * bm - topk
    bound = min(full, triton.cdiv(routes + min(routes, local_E) * (bm - 1), bm) * bm)
    nvalid = torch.empty((1,), dtype=torch.int32, device=dev)
    if (expert_map is not None and 0 < routes <= EP_SORT_MAX_ROUTES
            and topk_ids.is_contiguous() and topk_weights.is_contiguous()):
        sorted_ids = torch.empty((bound,), dtype=torch.int32, device=dev)
        sorted_w = torch.empty((bound,), dtype=torch.float32, device=dev)
        expert_ids = torch.empty((bound // bm,), dtype=torch.int32, device=dev)
        _ep_sort_kernel[(local_E,)](
            topk_ids, topk_weights, expert_map, sorted_ids, sorted_w, expert_ids, nvalid, M,
            TOPK=topk, LE_P=triton.next_power_of_2(local_E), BM=bm,
            BLOCK_R=triton.next_power_of_2(routes),
            WRITE_LAYOUT=expert_layout is not None, layout_ptr=expert_layout,
            num_warps=1 if routes <= 256 else 8)
        return sorted_ids, sorted_w, expert_ids, nvalid, bound
    assert expert_layout is None, "Expert-grid metadata requires the small-route sorter"
    sorted_ids = torch.empty((full,), dtype=torch.int32, device=dev)
    sorted_w = torch.empty((full,), dtype=torch.float32, device=dev)
    expert_ids = torch.empty((triton.cdiv(full, bm),), dtype=torch.int32, device=dev)
    tpe = torch.empty((global_E * 2,), dtype=torch.int32, device=dev)
    moe_sorting_fwd(topk_ids, topk_weights, sorted_ids, sorted_w, expert_ids, tpe, nvalid,
                    None, global_E, bm, expert_mask)
    return sorted_ids[:bound], sorted_w, expert_ids, nvalid, bound


def lib_configs(w1, w2, topk, M):
    c1 = try_get_optimal_moe_config(tuple(w1.shape), topk, "fp8_w8a8", M)
    c2, _ = try_get_optimal_moe_config(tuple(w2.shape), topk, "fp8_w8a8", M, is_bottom=True)
    return c1, c2


# Tuned for this path by bench_moe/tune_ep.py + select_ep.py; the Step5 decode entry (M=128)
# by bench_moe/tune_real.py + tune_persist.py on captured routing (clumped, unlike tune_ep's
# uniform random routing). A cfg with PERSISTENT_PROGRAMS runs the persistent kernel.
# {"E=..,N=..,K=..,topk=..": {M: [gemm1_cfg, gemm2_cfg]}}, nearest M wins (same rule as boltops).
# Shapes without an entry fall back to the library tables.
EP_TABLE_PATH = (
    os.environ.get("EP_FUSED_MOE_TABLE", "")
    if henvs.custom_ops_enabled() else ""
)


@functools.lru_cache(maxsize=None)
def _ep_table(key):
    if not henvs.custom_ops_enabled() or not EP_TABLE_PATH:
        return None
    if not os.path.exists(EP_TABLE_PATH):
        return None
    t = json.load(open(EP_TABLE_PATH)).get(key)
    return {int(m): (c1, c2) for m, (c1, c2) in t.items()} if t else None


@functools.lru_cache(maxsize=None)
def _device_arch(device_index):
    return torch.cuda.get_device_properties(device_index).gcnArchName.split(":", 1)[0]


def pick_configs(w1, w2, topk, M):
    E, N, K = w1.shape
    arch = _device_arch(w1.device.index)
    t = (
        _ep_table(f"E={E},N={N},K={K},topk={topk}")
        if arch == "gfx938" and henvs.custom_ops_enabled() else None
    )
    if t:
        return t[min(t, key=lambda x: abs(x - M))]
    return lib_configs(w1, w2, topk, M)


def _invoke_persistent(A, B, C, A_scale, B_scale, topk_weights, topk_ids, sorted_ids,
                       sorted_w, expert_ids, nvalid, top_k, compute_type, c_sorted=False,
                       a_sorted=False, bottom_a_use_mls_load=False, ck_topk=8, config=None):
    """boltops' persistent fused_moe launch (fp8 per-channel, ck sorting) with its program
    count taken from config["PERSISTENT_PROGRAMS"] instead of 2 x CUs.

    The regular kernel's grid covers the static sorted-layout bound (~105 M-blocks at
    M=128, EP8) while decode routing fills ~22, and the programs past nvalid only exit:
    12 us of a 180 us GEMM1 and 30 us of a 110 us GEMM2. The persistent kernel strides
    over the tiles below nvalid, so a program count near the usual valid tile count
    keeps one tile per program without the exit-only tail.
    """
    cfg = dict(config)
    nprog = cfg.pop("PERSISTENT_PROGRAMS")
    for k in ("N_TILES_PER_CTA", "SPLIT_K"):
        cfg.pop(k, None)
    bk = cfg.pop("BLOCK_SIZE_K")
    combine = cfg.pop("COMBINE_SCALE_LOAD", False)
    cfg.setdefault("USE_MLS_LOAD", False)
    lim = 2**32 - 1
    EM = sorted_ids.size(0)
    N = B.size(1)
    grid = (min(nprog, triton.cdiv(EM, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"])),)
    _moe_op.fused_moe_persistent_kernel[grid](
        A, B, C, topk_weights, nvalid, expert_ids, sorted_ids, sorted_w, A_scale, B_scale,
        N, B.size(2), EM, topk_ids.numel(),
        A.stride(0), A.stride(1), B.stride(0), B.stride(2), B.stride(1),
        C.stride(-2), C.stride(-1),
        A_scale.stride(0) if A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale.ndim >= 2 else 0,
        A.size(0), 0, 0,
        MUL_ROUTED_WEIGHT=False,
        USE_ADDR_OFFSET_INT64_A=A.numel() * A.element_size() >= lim,
        USE_ADDR_OFFSET_INT64_B=B.numel() * B.element_size() >= lim,
        USE_ADDR_OFFSET_INT64_C=C.numel() * C.element_size() >= lim,
        top_k=top_k, compute_type=compute_type, use_fp8_w8a8=True, use_int8_w8a8=False,
        use_int8_w8a16=False, per_channel_quant=True, c_sorted=c_sorted, a_sorted=a_sorted,
        bottom_a_use_mls_load=bottom_a_use_mls_load, ck_sorting=True, ck_topk=ck_topk,
        NUM_SMS=nprog, NUM_XCDS=1, BLOCK_SIZE_K=bk, COMBINE_SCALE_LOAD=combine, **cfg)


def _gemm(A, B, C, A_scale, B_scale, topk_weights, topk_ids, sorted_ids, sorted_w,
          expert_ids, nvalid, top_k, compute_type, config, expert_layout=None, **kw):
    if expert_layout is not None:
        from .step5_expert_grid import invoke_expert_grid

        return invoke_expert_grid(
            A, B, C, A_scale, B_scale, sorted_ids, expert_layout, topk_ids, config,
            a_sorted=kw.get("a_sorted", False), c_sorted=kw.get("c_sorted", False))
    if "PERSISTENT_PROGRAMS" in config:
        kw.pop("scale_bias_with_routed_weight", None)  # only acts on a bias; there is none
        _invoke_persistent(A, B, C, A_scale, B_scale, topk_weights, topk_ids, sorted_ids,
                           sorted_w, expert_ids, nvalid, top_k, compute_type, config=config,
                           **{k: v for k, v in kw.items() if k != "ck_sorting"})
        return
    invoke_fused_moe_kernel(A, B, C, A_scale, B_scale, None, topk_weights, topk_ids,
                            sorted_ids, sorted_w, expert_ids, nvalid, False, top_k,
                            compute_type=compute_type, use_fp8_w8a8=True,
                            per_channel_quant=True, config=config, **kw)


# Largest M per call; vLLM's MoE chunk size, so normally a single chunk.

@triton.jit
def _ep_gather_sorted_kernel(a_ptr, s_ptr, sids_ptr, nvalid_ptr, out_ptr, outs_ptr, M,
                             K: tl.constexpr, ROWS: tl.constexpr, BK: tl.constexpr):
    r0 = tl.program_id(0) * ROWS
    if r0 >= tl.load(nvalid_ptr):
        return
    rows = r0 + tl.arange(0, ROWS)
    tok = tl.load(sids_ptr + rows) & 0xFFFFFF
    ok = tok < M
    tok = tl.where(ok, tok, 0)
    tl.store(outs_ptr + rows, tl.load(s_ptr + tok, mask=ok, other=0.0))
    for k in tl.static_range(0, K, BK):
        cols = k + tl.arange(0, BK)
        cm = cols[None, :] < K
        v = tl.load(a_ptr + tok[:, None].to(tl.int64) * K + cols[None, :], mask=ok[:, None] & cm,
                    other=0)
        tl.store(out_ptr + rows[:, None].to(tl.int64) * K + cols[None, :], v, mask=cm)


def ep_gather_sorted(a_q, a_s, sorted_ids, nvalid, rows, ws=None):
    """A rows (fp8) + per-token scales in sorted (BM-padded) order, for GEMM1 a_sorted.
    Padding rows inside nvalid are zero; rows past nvalid are not written. With real
    (scattered) routing the gathering GEMM1 is A-load bound; on contiguous A a different
    config wins: M=32768 GEMM1 2.87 -> 2.22 ms + 0.20 ms gather, full call -7.5%, bitwise
    identical. (A token-sorted microbench hides this: there the gather is already contiguous.)"""
    M, K = a_q.shape
    buf = _carve(ws, (rows * K + 4 * rows,), torch.uint8, a_q.device)
    a_g = buf[:rows * K].view(a_q.dtype).view(rows, K)
    s_g = buf[rows * K:].view(torch.float32).view(rows, 1)
    ROWS = 32
    _ep_gather_sorted_kernel[(triton.cdiv(rows, ROWS),)](
        a_q.view(torch.uint8), a_s.reshape(-1), sorted_ids, nvalid, a_g.view(torch.uint8), s_g,
        M, K, ROWS, min(2048, triton.next_power_of_2(K)), num_warps=4)
    return a_g, s_g


CHUNK = (
    int(os.environ.get("EP_FUSED_MOE_CHUNK", "32768"))
    if henvs.custom_ops_enabled() else 32768
)


def _carve(ws, shape, dtype, device, *, avoid=None):
    """`shape` of `dtype` at the start of workspace `ws` (any dtype), or fresh if it won't fit."""
    nbytes = dtype.itemsize
    for d in shape:
        nbytes *= d
    # WorkspaceManager returns disjoint views of one backing allocation.
    # Compare the actual byte intervals, including nonzero output offsets.
    aliases_output = False
    if ws is not None and avoid is not None and ws.device == avoid.device:
        if ws.is_contiguous() and avoid.is_contiguous():
            start, other = ws.data_ptr(), avoid.data_ptr()
            aliases_output = (
                start < other + avoid.numel() * avoid.element_size()
                and other < start + nbytes
            )
        else:
            aliases_output = (
                ws.untyped_storage().data_ptr() == avoid.untyped_storage().data_ptr()
            )
    if (ws is None or not ws.is_contiguous()
            or ws.numel() * ws.element_size() < nbytes or aliases_output):
        return torch.empty(shape, dtype=dtype, device=device)
    return ws.view(-1).view(torch.uint8)[:nbytes].view(dtype).view(shape)


def _ep_fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, expert_map,
                     w1_scale, w2_scale, a1_scale=None, global_num_experts=-1,
                     output_dtype=None, gemm1_limit=None, routed_scaling_factor=1.0,
                     configs=None, expert_mask=None, out=None, workspace13=None,
                     workspace2=None, preserve_chunk_output=None):
    # out/workspace13/workspace2: caller-owned buffers (vLLM modular-kernel workspaces).
    # cache1 then cache3 live in workspace13, act_q in workspace2.
    out_dtype = output_dtype or (out.dtype if out is not None else hidden_states.dtype)
    compute_type = tl.bfloat16 if out_dtype == torch.bfloat16 else (
        tl.float16 if out_dtype == torch.float16 else tl.float32)
    M = hidden_states.shape[0]
    E, N, _ = w1.shape
    K = w2.shape[1]
    topk = topk_ids.shape[1]
    G = global_num_experts if global_num_experts > 0 else expert_map.numel()
    if expert_mask is None:
        expert_mask = (expert_map >= 0).to(torch.int32)
    cfg1, cfg2 = configs if configs is not None else pick_configs(w1, w2, topk, M)
    bm = cfg1["BLOCK_SIZE_M"]
    assert cfg2["BLOCK_SIZE_M"] == bm
    use_mls = bool(cfg2.get("bottom_a_use_mls_load", False))
    cfg2 = {k: v for k, v in cfg2.items() if k not in ("bottom_a_use_mls_load", "a_sorted")}
    presort = bool(cfg1.get("a_presort", False))
    cfg1 = {k: v for k, v in cfg1.items() if k != "a_presort"}

    if hidden_states.dtype in (torch.bfloat16, torch.float16):
        a_q, a_s = per_token_quant_triton(hidden_states, quant_dtype=FP8)
    else:
        a_q, a_s = hidden_states, a1_scale.view(-1, 1) if a1_scale.numel() > 1 else a1_scale

    expert_layout = None
    if (M == 256 and configs is None and STEP5_EXPERT_GRID
            and EP_SORT_MAX_ROUTES >= M * topk):
        from .step5_expert_grid import can_use_expert_grid, expert_grid_configs

        if (topk_weights.is_contiguous()
                and can_use_expert_grid(a_q, a_s, w1, w2, w1_scale, w2_scale,
                                       topk_ids, out_dtype)):
            cfg1, cfg2 = expert_grid_configs()
            bm, presort, use_mls = 16, False, False
            expert_layout = torch.empty((E, 2), dtype=torch.int32, device=a_q.device)
            global _step5_grid_announced
            if not _step5_grid_announced:
                _step5_grid_announced = True
                print("[step5_expert_grid] active: E22 M256 BM16 workers4", flush=True)
    sorting_args = (topk_ids, topk_weights, expert_mask, G, E, bm, expert_map)
    if expert_layout is None:
        sorted_ids, sorted_w, expert_ids, nvalid, rows = ep_sorting(*sorting_args)
    else:
        sorted_ids, sorted_w, expert_ids, nvalid, rows = ep_sorting(
            *sorting_args, expert_layout=expert_layout)

    dev = hidden_states.device
    # Later chunks must preserve every previously completed output slice.
    # Modular workspaces may share the output's storage, including at an offset.
    cache1 = _carve(workspace13, (rows, N), out_dtype, dev,
                    avoid=preserve_chunk_output)
    asm_activation = None
    if (STEP5_PREFILL_ASM_G1 and presort and a_s.numel() == M
            and M == 65536 and topk == 8 and bm == 256):
        from vllm_hcu.ops.step5.prefill_asm_g1 import try_step5_prefill_asm_g1

        asm_activation = try_step5_prefill_asm_g1(
            a_q, w1, w2, cache1, a_s, w1_scale, sorted_ids, sorted_w,
            expert_ids, nvalid, num_tokens=M, top_k=topk, sort_block_m=bm,
            limit=7.0 if gemm1_limit is None else gemm1_limit, workspace2=workspace2,
        )
    if asm_activation is not None:
        act_q, act_s = asm_activation
    elif presort and a_s.numel() == M:
        # A gathered into sorted order first (lives in workspace2, dead before act_q).
        a_g, s_g = ep_gather_sorted(a_q, a_s, sorted_ids, nvalid, rows, workspace2)
        _gemm(a_g, w1, cache1, s_g, w1_scale, topk_weights, topk_ids, sorted_ids, sorted_w,
              expert_ids, nvalid, topk, compute_type, cfg1, c_sorted=True, ck_sorting=True,
              ck_topk=topk, a_sorted=True)
        del a_g, s_g
    else:
        _gemm(a_q, w1, cache1, a_s, w1_scale, topk_weights, topk_ids, sorted_ids, sorted_w,
              expert_ids, nvalid, topk, compute_type, cfg1, c_sorted=True, ck_sorting=True,
              ck_topk=topk, expert_layout=expert_layout)
    if asm_activation is None:
        act_q, act_s = ep_swiglustep_quant(cache1, sorted_w, nvalid, rows,
                                           7.0 if gemm1_limit is None else gemm1_limit,
                                           q=_carve(workspace2, (rows, N // 2), FP8, dev))
    del cache1
    if out is None:
        out = torch.empty((M, K), dtype=out_dtype, device=dev)
    add = shared_add_provider(out) if shared_add_provider is not None else None
    if (expert_layout is None and DIRECT_MIN and M >= DIRECT_MIN and routed_scaling_factor == 1.0 and add is None
            and "PERSISTENT_PROGRAMS" not in cfg2):
        xoff = _ep_direct_xoff(out, topk_ids)
        if xoff is not None:
            global _direct_announced
            if not _direct_announced:
                _direct_announced = True
                print(f"[ep_fused_moe] direct combine: M={M} scratch={direct_scratch.shape[0]}",
                      flush=True)
            _ep_direct_combine(act_q, act_s, w2, w2_scale, topk_weights, topk_ids, expert_map,
                               sorted_ids, sorted_w, expert_ids, nvalid, compute_type, cfg2,
                               use_mls, out, xoff, workspace13, preserve_chunk_output)
            return out
    cache3 = _carve(workspace13, (M, topk, K), out_dtype, dev,
                    avoid=preserve_chunk_output if preserve_chunk_output is not None else out)
    _gemm(act_q, w2, cache3, act_s, w2_scale, topk_weights, topk_ids, sorted_ids, sorted_w,
          expert_ids, nvalid, 1, compute_type, cfg2, a_sorted=True,
          bottom_a_use_mls_load=use_mls, ck_sorting=True, ck_topk=topk,
          scale_bias_with_routed_weight=True, expert_layout=expert_layout)
    ep_masked_moe_sum(cache3, out, topk_ids, expert_map, routed_scaling_factor,
                      *(add or ()))
    return out


def _ep_direct_xoff(out, topk_ids):
    """Row offset of direct_scratch from out when the direct combine applies, else None."""
    x = direct_scratch
    M, K = out.shape
    if (x is None or x.dtype != out.dtype or x.device != out.device or x.dim() != 2
            or x.shape[1] != K or not x.is_contiguous() or not out.is_contiguous()
            or not topk_ids.is_contiguous()):
        return None
    rb = K * out.element_size()
    d = x.data_ptr() - out.data_ptr()
    # GEMM2 addresses scratch rows from out's pointer with 32-bit offsets.
    if d < M * rb or d % rb or d + x.numel() * x.element_size() >= 2**31:
        return None
    return d // rb


def _ep_direct_combine(act_q, act_s, w2, w2_scale, topk_weights, topk_ids, expert_map,
                       sorted_ids, sorted_w, expert_ids, nvalid, compute_type, cfg2, use_mls,
                       out, xoff, workspace13, preserve_chunk_output=None):
    """GEMM2 + local top-k sum without the (M, topk, K) cache; see DIRECT_MIN."""
    M, topk = topk_ids.shape
    K = out.shape[1]
    dev = out.device
    x = direct_scratch
    ints = torch.empty((M + M * topk + sorted_ids.numel() + 4,), dtype=torch.int32, device=dev)
    info, inv = ints[:M], ints[M:M + M * topk]
    rows2 = ints[M + M * topk:M + M * topk + sorted_ids.numel()]
    ctr, gate = ints[-4:-3], ints[-3:]
    _ep_direct_info_kernel[(triton.cdiv(M, 1024),)](topk_ids, expert_map, info, inv, ctr, M,
                                                     TOPK=topk, BLOCK=1024, num_warps=4)
    _ep_direct_rows_kernel[(triton.cdiv(sorted_ids.numel(), 1024),)](
        sorted_ids, nvalid, info, rows2, inv, ctr, M, xoff, sorted_ids.numel(), TOPK=topk,
        BIG=_DIRECT_BIG, BLOCK=1024, num_warps=4)
    _ep_direct_gate_kernel[(1,)](nvalid, ctr, gate, x.shape[0], num_warps=1)
    cache3 = None
    specialized = False
    if STEP5_PREFILL_G2 and not use_mls:
        from vllm_hcu.ops.step5.prefill_g2 import try_step5_prefill_g2

        cache3 = _carve(workspace13, (M, topk, K), out.dtype, dev,
                        avoid=preserve_chunk_output if preserve_chunk_output is not None else out)
        specialized = try_step5_prefill_g2(
            act_q, w2, out, cache3, act_s, w2_scale, sorted_ids, rows2,
            expert_ids, gate, num_tokens=M, top_k=topk,
            sort_block_m=cfg2["BLOCK_SIZE_M"], config=cfg2)
    if not specialized:
        # num_valid_tokens only masks rows >= it: make it never mask.
        fake_ids = torch.empty((1, 1), dtype=torch.int32, device=dev).expand(_DIRECT_BIG - 1, 1)
        _gemm(act_q, w2, out, act_s, w2_scale, topk_weights, fake_ids, rows2, sorted_w,
              expert_ids, gate[0:1], 1, compute_type, cfg2, a_sorted=True,
              bottom_a_use_mls_load=use_mls, ck_sorting=False, ck_topk=topk,
              scale_bias_with_routed_weight=True)
        if cache3 is None:
            cache3 = _carve(workspace13, (M, topk, K), out.dtype, dev,
                            avoid=preserve_chunk_output if preserve_chunk_output is not None else out)
        # Overflow fallback exits unless gate[1] > 0.
        _gemm(act_q, w2, cache3, act_s, w2_scale, topk_weights, topk_ids, sorted_ids, sorted_w,
              expert_ids, gate[1:2], 1, compute_type, cfg2, a_sorted=True,
              bottom_a_use_mls_load=use_mls, ck_sorting=True, ck_topk=topk,
              scale_bias_with_routed_weight=True)
    bn = SUM_CFG["BLOCK_N"]
    _ep_direct_fix_kernel[(triton.cdiv(M, DIRECT_TPP), K // bn)](
        out, x, cache3, topk_ids, expert_map, info, inv, gate, M, N=K, TOPK=topk, BLOCK_N=bn,
        TPP=DIRECT_TPP, num_warps=SUM_CFG["num_warps"])


def ep_fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, expert_map, w1_scale, w2_scale,
                     **kw):
    global _announced
    if not _announced:
        _announced = True
        print(f"[ep_fused_moe] active: w1={tuple(w1.shape)} table={EP_TABLE_PATH}", flush=True)
    M = hidden_states.shape[0]
    if M <= CHUNK:
        return _ep_fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, expert_map,
                                 w1_scale, w2_scale, **kw)
    a1 = kw.pop("a1_scale", None)
    out = kw.pop("out", None)
    if out is not None:
        for s in range(0, M, CHUNK):
            _ep_fused_experts(hidden_states[s:s + CHUNK], w1, w2, topk_weights[s:s + CHUNK],
                              topk_ids[s:s + CHUNK], expert_map, w1_scale, w2_scale,
                              a1_scale=None if a1 is None or a1.numel() == 1 else a1[s:s + CHUNK],
                              out=out[s:s + CHUNK], preserve_chunk_output=out, **kw)
        return out
    return torch.cat([
        _ep_fused_experts(hidden_states[s:s + CHUNK], w1, w2, topk_weights[s:s + CHUNK],
                          topk_ids[s:s + CHUNK], expert_map, w1_scale, w2_scale,
                          a1_scale=None if a1 is None or a1.numel() == 1 else a1[s:s + CHUNK], **kw)
        for s in range(0, M, CHUNK)])
