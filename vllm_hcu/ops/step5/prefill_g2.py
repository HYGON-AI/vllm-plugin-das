# SPDX-License-Identifier: Apache-2.0
"""One channel-FP8 G2 launch with device-side Directsum/overflow destination."""
from __future__ import annotations

import functools
import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_g2_kernel(
    a,
    w,
    out,
    cache3,
    sa,
    sb,
    sids,
    rows2,
    eids,
    gate,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    T: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    count = tl.load(gate) + tl.load(gate + 1)
    nm = tl.cdiv(count, BM)
    nn: tl.constexpr = N // BN
    if pid >= nm * nn:
        return
    pm = pid // nn
    pn = pid % nn
    expert = tl.load(eids + pm)
    if expert < 0:
        return
    tl.assume(expert >= 0)
    rr = pm * BM + tl.arange(0, BM)
    over = tl.load(gate + 2) != 0
    if over:
        enc = tl.load(sids + rr)
        token = enc & 0xFFFFFF
        target = token * T + (enc >> 24)
        valid = token < M
    else:
        target = tl.load(rows2 + rr)
        valid = target < 2147483646
    cc = pn * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    ap = a + rr[:, None].to(tl.int64) * K + kk[None, :]
    bp = w + expert * (N * K) + kk[:, None] + cc[None, :] * K
    acc = tl.full((BM, BN), 0.0, tl.float32)
    for _ in range(K // BK):
        av = tl.load(ap)
        bv = tl.load(bp)
        acc = tl.dot(av, bv, acc=acc)
        ap += BK
        bp += BK
    asc = tl.load(sa + rr, mask=valid, other=0.0)
    bsc = tl.load(sb + expert * N + cc)
    result = (acc * asc[:, None] * bsc[None, :]).to(out.dtype.element_ty)
    offs = target[:, None].to(tl.int64) * N + cc[None, :]
    if over:
        tl.store(cache3 + offs, result, mask=valid[:, None])
    else:
        tl.store(out + offs, result, mask=valid[:, None])


@functools.lru_cache(maxsize=None)
def _is_gfx938(device_index: int) -> bool:
    props = torch.cuda.get_device_properties(device_index)
    return getattr(props, "gcnArchName", "").split(":", 1)[0] == "gfx938"


_announced = False


def try_step5_prefill_g2(
    a: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    cache3: torch.Tensor,
    a_scale: torch.Tensor,
    w_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    direct_rows: torch.Tensor,
    expert_ids: torch.Tensor,
    gate: torch.Tensor,
    *,
    num_tokens: int,
    top_k: int,
    sort_block_m: int,
    config: dict,
) -> bool:
    """Fuse the validated G2 destinations, retaining device-side overflow fallback.

    Checks inspect metadata only. Weights, projection precision, router scaling,
    Directsum row metadata and final accumulation order stay unchanged.
    """
    if num_tokens != 65536 or top_k != 8 or sort_block_m != 256:
        return False
    required = dict(
        BLOCK_SIZE_M=256,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=1,
        COMBINE_SCALE_LOAD=False,
        USE_MLS_LOAD=False,
        schedule_hint="local-prefetch",
        sched_latency="none",
        kpack=1,
        use_block_pingpong=True,
        buffer_cache_swizzle=False,
        num_warps=8,
        num_stages=2,
    )
    if set(config) != set(required) or any(config[k] != v for k, v in required.items()):
        return False
    if config.get("PERSISTENT_PROGRAMS") or config.get("USE_MLS_LOAD", False):
        return False
    if (
        a.ndim != 2
        or a.shape[1] != 1536
        or w.shape not in ((22, 4096, 1536), (23, 4096, 1536))
        or out.shape != (65536, 4096)
        or cache3.shape != (65536, 8, 4096)
        or a.dtype != torch.float8_e4m3fn
        or w.dtype != a.dtype
        or out.dtype != torch.bfloat16
        or cache3.dtype != out.dtype
        or not a.is_cuda
    ):
        return False
    rows = sorted_ids.numel()
    if (
        rows % 256
        or a.shape[0] != rows
        or a_scale.numel() != rows
        or w_scale.numel() != w.shape[0] * 4096
        or a_scale.dtype != torch.float32
        or w_scale.dtype != torch.float32
        or sorted_ids.ndim != 1
        or direct_rows.shape != sorted_ids.shape
        or expert_ids.ndim != 1
        or expert_ids.numel() < rows // 256
        or gate.numel() != 3
        or any(
            t.dtype != torch.int32 for t in (sorted_ids, direct_rows, expert_ids, gate)
        )
    ):
        return False
    tensors = (
        a,
        w,
        out,
        cache3,
        a_scale,
        w_scale,
        sorted_ids,
        direct_rows,
        expert_ids,
        gate,
    )
    if any(t.device != a.device or not t.is_contiguous() for t in tensors):
        return False
    if not _is_gfx938(a.device.index):
        return False
    _prefill_g2_kernel[(triton.cdiv(rows, 256) * 16,)](
        a,
        w,
        out,
        cache3,
        a_scale,
        w_scale,
        sorted_ids,
        direct_rows,
        expert_ids,
        gate,
        65536,
        4096,
        1536,
        8,
        256,
        256,
        64,
        num_warps=8,
        num_stages=2,
        kpack=1,
        use_block_pingpong=True,
        schedule_hint="local-prefetch",
        sched_latency="none",
        buffer_cache_swizzle=False,
    )
    global _announced
    if not _announced:
        _announced = True
        print(
            f"[step5_prefill_g2] active: E{w.shape[0]} M65536 sort256 NK fused-direct-overflow",
            flush=True,
        )
    return True
