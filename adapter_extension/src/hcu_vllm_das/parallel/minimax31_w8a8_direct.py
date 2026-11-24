# SPDX-License-Identifier: Apache-2.0
"""Row-major epilogue for the qualified gfx938 M128 W8A8 GEMM."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_direct(
    q, w, q_scale, w_scale, out, K: tl.constexpr, N: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    b_ptr = q + m[None, :] * K + k[:, None]
    a_ptr = w + n[:, None] * K + k[None, :]
    acc = tl.full((BN, BM), 0, tl.int32)
    for _ in range(K // BK):
        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptr += BK
        b_ptr += BK
    qs = tl.load(q_scale + m)
    ws = tl.load(w_scale + n)
    result = acc.to(tl.float32) * (ws[:, None] * qs[None, :])
    tl.store(out + m[None, :] * N + n[:, None], result.to(tl.bfloat16))


def scaled_mm_row_major(q, w, q_scale, w_scale):
    # Keep the original NK weight and MK activation storage. The dot layout
    # remains transposed so that K stays contiguous in both matrix operands.
    m, k = q.shape
    n = w.shape[1]
    out = torch.empty((m, n), device=q.device, dtype=torch.bfloat16)
    _gemm_direct[(m // 64, n // 64)](
        q, w, q_scale, w_scale, out, k, n, 64, 64, 128, num_warps=4,
    )
    return out
