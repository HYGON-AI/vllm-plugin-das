# SPDX-License-Identifier: Apache-2.0
"""Bounded channel-FP8 expert grids for Step5's uniform C64 decode shape.

Only the validated E22/M256 contiguous per-token/per-channel layout enters
this implementation. Each expert owns four M workers, retaining parallelism
on hot experts while avoiding a grid sized for unused global route rows.
"""
from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


def can_use_expert_grid(a, a_scale, w1, w2, w1_scale, w2_scale, ids, out_dtype):
    return (
        out_dtype == torch.bfloat16
        and a.is_cuda
        and a.shape == (256, 4096)
        and ids.shape == (256, 8)
        and w1.shape == (22, 3072, 4096)
        and w2.shape == (22, 4096, 1536)
        and a.dtype == w1.dtype == w2.dtype == torch.float8_e4m3fn
        and a_scale is not None
        and a_scale.dtype == torch.float32
        and a_scale.numel() == 256
        and w1_scale is not None and w2_scale is not None
        and w1_scale.shape in ((22, 3072), (22, 3072, 1))
        and w2_scale.shape in ((22, 4096), (22, 4096, 1))
        and w1_scale.dtype == w2_scale.dtype == torch.float32
        and all(t.is_contiguous() for t in (a, a_scale, w1, w2, w1_scale, w2_scale, ids))
        and ids.dtype in (torch.int32, torch.int64)
        and all(t.device == a.device for t in (a_scale, w1, w2, w1_scale, w2_scale, ids))
    )


def expert_grid_configs():
    common = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=256,
                  num_warps=2, num_stages=1)
    return dict(common, kpack=1), dict(common, kpack=2)


@triton.jit
def _expert_grid_gemm(A, B, C, AS, BS, SIDS, LAYOUT,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    MW: tl.constexpr, ASORTED: tl.constexpr, CSORTED: tl.constexpr,
    STAGES: tl.constexpr):
    pid = tl.program_id(0)
    NT: tl.constexpr = (N + BN - 1) // BN
    expert = pid // (NT * MW)
    lane = (pid // NT) % MW
    pn = pid % NT
    start = tl.load(LAYOUT + expert * 2)
    count = tl.load(LAYOUT + expert * 2 + 1)
    if lane >= count:
        return
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    bscale = tl.load(BS + expert * N + rn)
    for mb in range(lane, count, MW):
        rows = (start + mb) * BM + tl.arange(0, BM)
        packed = tl.load(SIDS + rows)
        token = packed & 0xFFFFFF
        slot = packed >> 24
        valid = (token < M) & (slot < TOPK)
        arow = rows if ASORTED else token
        asc = tl.load(AS + arow, mask=valid, other=0.)
        ap = A + arow[:, None] * K + rk[None, :]
        bp = B + expert * (N * K) + rk[:, None] + rn[None, :] * K
        acc = tl.zeros((BM, BN), tl.float32)
        for kb in tl.range(0, tl.cdiv(K, BK), num_stages=STAGES):
            if ASORTED:
                av = tl.load(ap)
            else:
                av = tl.load(ap, mask=valid[:, None], other=0.)
            bv = tl.load(bp)
            acc = tl.dot(av, bv, acc)
            ap += BK
            bp += BK
        out = (acc * asc[:, None] * bscale[None, :]).to(C.dtype.element_ty)
        crow = rows if CSORTED else token * TOPK + slot
        tl.store(C + crow[:, None] * N + rn[None, :], out, mask=valid[:, None])


def invoke_expert_grid(a, b, c, a_scale, b_scale, sorted_ids, layout, topk_ids,
                       config, *, a_sorted=False, c_sorted=False):
    e, n, k = b.shape
    m, topk = topk_ids.shape
    assert e == 22 and m == 256 and topk == 8
    assert a.is_contiguous() and b.is_contiguous() and c.is_contiguous()
    assert a_scale.is_contiguous() and b_scale.is_contiguous()
    assert a.dtype == b.dtype == torch.float8_e4m3fn and c.dtype == torch.bfloat16
    assert n % config['BLOCK_SIZE_N'] == k % config['BLOCK_SIZE_K'] == 0
    _expert_grid_gemm[(e * 4 * triton.cdiv(n, config['BLOCK_SIZE_N']),)](
        a, b, c, a_scale, b_scale, sorted_ids, layout, m, n, k, e, topk,
        config['BLOCK_SIZE_M'], config['BLOCK_SIZE_N'], config['BLOCK_SIZE_K'],
        4, a_sorted, c_sorted, config['num_stages'],
        num_warps=config['num_warps'], num_stages=1, kpack=config['kpack'])
