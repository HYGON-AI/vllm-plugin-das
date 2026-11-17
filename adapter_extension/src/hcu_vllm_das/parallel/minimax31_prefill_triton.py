# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax PCP kernels. Preserve existing FP32 division and FP8 store boundaries."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["SH", "SP", "SQ", "P"])
def _local_pages(S, End, SH, SP, SQ, P, LOCAL: tl.constexpr):
    q, h = tl.program_id(0), tl.program_id(1)
    end = tl.load(End + q).to(tl.int64)
    # Triton signed division truncates toward zero; k_end=0 has no local page.
    last = tl.where(end > 0, (end - 1) // 128, -1)
    for off in range(LOCAL):
        page = last - off
        tl.store(S + h * SH + page * SP + q * SQ, 1.0e30,
                 mask=(page >= 0) & (page < P))


def force_local_pages(score, k_end, local_blocks):
    if (score.ndim != 3 or score.dtype != torch.float32 or not score.is_cuda or
            k_end.shape != (score.shape[2],) or k_end.device != score.device or
            k_end.dtype not in (torch.int32, torch.int64) or not k_end.is_contiguous() or
            type(local_blocks) is not int or not 0 <= local_blocks <= 16):
        raise ValueError("invalid MiniMax score/local-page contract")
    if local_blocks and score.numel():
        _local_pages[(score.shape[2], score.shape[0])](score, k_end, *score.stride(),
                     score.shape[1], local_blocks, num_warps=1)


@triton.jit(do_not_specialize=["N"])
def _quant_q(X, Scale, Out, N, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, other=0).to(tl.float32)
    y = tl.div_rn(x, tl.load(Scale))
    tl.store(Out + i, y.to(Out.dtype.element_ty), i < N)


def quant_q(q, scale):
    if (not q.is_cuda or q.dtype != torch.bfloat16 or not q.is_contiguous() or
            scale.device != q.device or scale.dtype != torch.float32 or
            scale.numel() != 1 or not scale.is_contiguous()):
        raise ValueError("Q quant requires contiguous BF16 and a device FP32 scalar")
    out = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    if q.numel():
        _quant_q[(triton.cdiv(q.numel(), 1024),)](q, scale, out, q.numel(), 1024)
    return out


@triton.jit
def _scale_check(K, V, KS, VS, OK):
    k = tl.load(K); v = tl.load(V)
    valid = ((k > 0) & (k < float("inf")) & (v > 0) & (v < float("inf")) &
             (k == tl.load(KS)) & (v == tl.load(VS)))
    tl.store(OK, valid)


def check_scales(k, v, ks, vs, status):
    _scale_check[(1,)](k, v, ks, vs, status, num_warps=1)
    torch._assert_async(status, "MiniMax KV scales changed or are not finite positive values")


@triton.jit(do_not_specialize=["N", "KP"])
def _cache_store(Raw, MS, IS, KC, VC, IC, KS, VS,
                 RS: tl.constexpr, N, KP,
                 K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
                 I0: tl.constexpr, I1: tl.constexpr,
                 INDEX_FP8: tl.constexpr, MAIN_BYTES: tl.constexpr):
    row = tl.program_id(0)
    ms = tl.load(MS + row).to(tl.int64)
    ix = tl.load(IS + row).to(tl.int64)
    col = tl.arange(0, 512)
    h, d = col // 128, col % 128
    ko = (ms // 128) * K0 + (ms % 128) * K1 + h * K2 + d
    k = tl.load(Raw + row * RS + col).to(tl.float32)
    v = tl.load(Raw + row * RS + 512 + col).to(tl.float32)
    k = tl.div_rn(k, tl.load(KS)); v = tl.div_rn(v, tl.load(VS))
    if MAIN_BYTES:
        # uint8 backing is an FP8 alias, never an integer conversion.
        k = k.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        v = v.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(KC + ko, k, mask=ms >= 0)
    tl.store(VC + ko, v, mask=ms >= 0)
    c = tl.arange(0, 128)
    ik = tl.load(Raw + row * RS + 1024 + c).to(tl.float32)
    if INDEX_FP8:
        ik = tl.minimum(tl.maximum(ik, -448.0), 448.0)
    io = (ix // 128) * I0 + (ix % 128) * I1 + c
    tl.store(IC + io, ik, mask=ix >= 0)


@triton.jit
def _selector_mix(x):
    x = x + 0x9e3779b97f4a7c15
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb
    return x ^ (x >> 31)


@triton.jit(do_not_specialize=["N", "REAL", "LAYER"])
def select_physical_dynamic(Ids, Pos, Base, Counts, Slots, Out,
                            N, REAL, LAYER, B: tl.constexpr):
    """Same uint64 selector; shape and layer ID do not multiply JIT variants."""
    ix = tl.program_id(0) * B + tl.arange(0, B)
    row = ix // 4
    logical = tl.load(Ids + ix, ix < N * 4, other=-1).to(tl.int32)
    live = (ix < N * 4) & (row < REAL) & (logical >= 0) & (logical < 128)
    h = tl.load(Base + row, row < N, other=0).to(tl.uint64)
    pos = tl.load(Pos + row, row < N, other=0).to(tl.uint64)
    h = _selector_mix(h ^ pos)
    h = _selector_mix(h ^ LAYER.to(tl.uint64))
    h = _selector_mix(h ^ logical.to(tl.uint64))
    count = tl.load(Counts + logical, live, other=1).to(tl.uint64)
    replica = (h % count).to(tl.int32)
    physical = tl.load(Slots + logical * 9 + replica, live, other=-1)
    tl.store(Out + ix, tl.where(live, physical, -1), ix < N * 4)


def store_pcp_cache(raw, main_slots, index_slots, keys, values, index_cache, k_scale, v_scale):
    """Same-order BF16 gather payload, separate main/index physical slot maps.

    Shape/layout checks are host-only. Valid slots are owned and bounds-checked
    by the official PCP planner; -1 padding is skipped exactly as stock writers.
    """
    if (not raw.is_cuda or raw.dtype != torch.bfloat16 or raw.ndim != 2 or
            raw.shape[1] != 1152 or raw.stride(1) != 1):
        raise ValueError("fused PCP store requires BF16 K512/V512/index-K128")
    if (keys.ndim != 4 or keys.shape[1:] != (128, 4, 128) or
            values.shape != keys.shape or keys.stride() != values.stride() or
            keys.stride()[1:] != (1024, 256, 1) or keys.stride(0) < 131072 or
            keys.dtype != values.dtype or
            keys.dtype not in (torch.uint8, torch.float8_e4m3fn) or
            index_cache.ndim != 3 or index_cache.shape[1:] != (128, 128) or
            index_cache.stride()[1:] != (128, 1) or index_cache.stride(0) < 16384 or
            index_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn)):
        raise ValueError("fused PCP store requires native token-major cache views")
    for x in (main_slots, index_slots):
        if (x.shape != (raw.shape[0],) or not x.is_contiguous() or
                x.dtype not in (torch.int32, torch.int64)):
            raise ValueError("invalid PCP store slot map")
    for x in (k_scale, v_scale):
        if x.dtype != torch.float32 or x.numel() != 1 or not x.is_contiguous():
            raise ValueError("invalid PCP store scale")
    if any(x.device != raw.device for x in (main_slots,index_slots,keys,values,index_cache,k_scale,v_scale)):
        raise ValueError("PCP store device mismatch")
    if raw.shape[0]:
        _cache_store[(raw.shape[0],)](raw, main_slots, index_slots, keys, values, index_cache,
            k_scale, v_scale, raw.stride(0), raw.shape[0], keys.shape[0],
            *keys.stride()[:3], *index_cache.stride()[:2],
            index_cache.dtype == torch.float8_e4m3fn, keys.dtype == torch.uint8, num_warps=4)
