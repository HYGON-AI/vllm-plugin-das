# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skinny bf16 GEMM for the lm_head at decode batch sizes.

For M <= 64 the projection is a pure weight stream (N x K bf16). hipBLASLt picks a
256x256 macro tile there and reaches ~0.93 TB/s on gfx938; this kernel tiles only over
N with a full-M block and reaches 1.4-1.7 TB/s (Step5 lm_head shard 32224 x 4096:
285 -> 153 us at M=16, 190 us at M=64).

Disable with STEP5_SKINNY_LM_HEAD=0.
"""

import os

import torch

from vllm.triton_utils import tl, triton
from vllm_hcu.platforms import envs as henvs

_ENABLED = henvs.optional_custom_op_enabled(
    os.environ.get("STEP5_SKINNY_LM_HEAD", "1") != "0"
)
MAX_M = 64
_BK = 128


@triton.jit
def _skinny_kernel(
    x_ptr,
    w_ptr,
    o_ptr,
    M,
    N,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    pm = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pid * BN + tl.arange(0, BN)
    mmask = rm < M
    nmask = rn < N
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(
            x_ptr + rm[:, None] * K + rk[None, :], mask=mmask[:, None], other=0.0
        )
        wt = tl.load(
            w_ptr + rn[None, :] * K + rk[:, None], mask=nmask[None, :], other=0.0
        )
        acc += tl.dot(x, wt)
    tl.store(
        o_ptr + rm[:, None] * N + rn[None, :],
        acc.to(tl.bfloat16),
        mask=mmask[:, None] & nmask[None, :],
    )


def usable(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        _ENABLED
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.dim() == 2
        and weight.is_contiguous()
        and 0 < x.numel() // x.shape[-1] <= MAX_M
        and x.shape[-1] == weight.shape[1]
        and weight.shape[1] % _BK == 0
    )


def skinny_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x [..., K] @ weight[N, K].T in bf16 (fp32 accumulate); requires usable()."""
    N, K = weight.shape
    x2 = x.reshape(-1, K).contiguous()
    M = x2.shape[0]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    if M <= 16:
        BM, BN, nw = 16, 64, 2
    elif M <= 32:
        BM, BN, nw = 32, 64, 2
    else:
        BM, BN, nw = 64, 128, 4
    _skinny_kernel[(triton.cdiv(N, BN), triton.cdiv(M, BM))](
        x2, weight, out, M, N, K, BM=BM, BN=BN, BK=_BK, num_warps=nw, num_stages=1
    )
    return out.reshape(*x.shape[:-1], N)
