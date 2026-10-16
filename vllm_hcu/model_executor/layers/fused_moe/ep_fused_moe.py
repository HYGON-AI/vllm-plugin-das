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

FP8 = torch.float8_e4m3fn
_announced = False


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
                    BLOCK_R: tl.constexpr):
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
    if x == 0:
        tl.store(nvalid_ptr, tl.sum(padded, axis=0))


# Largest M*topk sorted by _ep_sort_kernel (all routes live in one tile); above
# it moe_sorting_fwd is used. At decode sizes the Triton sort takes 3-6 us vs
# 13-27 us for the CK multi-phase sort plus its workspace memset.
EP_SORT_MAX_ROUTES = int(os.environ.get("EP_FUSED_MOE_SORT_MAX_ROUTES", "2048"))


def ep_sorting(topk_ids, topk_weights, expert_mask, global_E, local_E, bm, expert_map=None):
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
            num_warps=1 if routes <= 256 else 8)
        return sorted_ids, sorted_w, expert_ids, nvalid, bound
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
EP_TABLE_PATH = os.environ.get(
    "EP_FUSED_MOE_TABLE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ep_table.json"))


@functools.lru_cache(maxsize=None)
def _ep_table(key):
    if not os.path.exists(EP_TABLE_PATH):
        return None
    t = json.load(open(EP_TABLE_PATH)).get(key)
    return {int(m): (c1, c2) for m, (c1, c2) in t.items()} if t else None


def pick_configs(w1, w2, topk, M):
    E, N, K = w1.shape
    t = _ep_table(f"E={E},N={N},K={K},topk={topk}")
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
          expert_ids, nvalid, top_k, compute_type, config, **kw):
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
CHUNK = int(os.environ.get("EP_FUSED_MOE_CHUNK", "32768"))


def _carve(ws, shape, dtype, device, *, avoid=None):
    """`shape` of `dtype` at the start of workspace `ws` (any dtype), or fresh if it won't fit."""
    nbytes = dtype.itemsize
    for d in shape:
        nbytes *= d
    # Reduction programs must not overwrite another token's unread routes.
    # Storage identity also catches output slices with a nonzero offset.
    aliases_output = (
        ws is not None and avoid is not None
        and ws.untyped_storage().data_ptr() == avoid.untyped_storage().data_ptr()
    )
    if (ws is None or not ws.is_contiguous()
            or ws.numel() * ws.element_size() < nbytes or aliases_output):
        return torch.empty(shape, dtype=dtype, device=device)
    return ws.view(-1).view(torch.uint8)[:nbytes].view(dtype).view(shape)


def _ep_fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, expert_map,
                     w1_scale, w2_scale, a1_scale=None, global_num_experts=-1,
                     output_dtype=None, gemm1_limit=None, routed_scaling_factor=1.0,
                     configs=None, expert_mask=None, out=None, workspace13=None,
                     workspace2=None):
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

    if hidden_states.dtype in (torch.bfloat16, torch.float16):
        a_q, a_s = per_token_quant_triton(hidden_states, quant_dtype=FP8)
    else:
        a_q, a_s = hidden_states, a1_scale.view(-1, 1) if a1_scale.numel() > 1 else a1_scale

    sorted_ids, sorted_w, expert_ids, nvalid, rows = ep_sorting(
        topk_ids, topk_weights, expert_mask, G, E, bm, expert_map)

    dev = hidden_states.device
    cache1 = _carve(workspace13, (rows, N), out_dtype, dev)
    _gemm(a_q, w1, cache1, a_s, w1_scale, topk_weights, topk_ids, sorted_ids, sorted_w,
          expert_ids, nvalid, topk, compute_type, cfg1, c_sorted=True, ck_sorting=True,
          ck_topk=topk)
    act_q, act_s = ep_swiglustep_quant(cache1, sorted_w, nvalid, rows,
                                       7.0 if gemm1_limit is None else gemm1_limit,
                                       q=_carve(workspace2, (rows, N // 2), FP8, dev))
    del cache1
    cache3 = _carve(workspace13, (M, topk, K), out_dtype, dev, avoid=out)
    _gemm(act_q, w2, cache3, act_s, w2_scale, topk_weights, topk_ids, sorted_ids, sorted_w,
          expert_ids, nvalid, 1, compute_type, cfg2, a_sorted=True,
          bottom_a_use_mls_load=use_mls, ck_sorting=True, ck_topk=topk,
          scale_bias_with_routed_weight=True)
    if out is None:
        out = torch.empty((M, K), dtype=out_dtype, device=dev)
    add = shared_add_provider(out) if shared_add_provider is not None else None
    ep_masked_moe_sum(cache3, out, topk_ids, expert_map, routed_scaling_factor,
                      *(add or ()))
    return out


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
                              out=out[s:s + CHUNK], **kw)
        return out
    return torch.cat([
        _ep_fused_experts(hidden_states[s:s + CHUNK], w1, w2, topk_weights[s:s + CHUNK],
                          topk_ids[s:s + CHUNK], expert_map, w1_scale, w2_scale,
                          a1_scale=None if a1 is None or a1.numel() == 1 else a1[s:s + CHUNK], **kw)
        for s in range(0, M, CHUNK)])
