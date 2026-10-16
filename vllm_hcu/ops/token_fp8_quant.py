"""Dynamic per-token fp8 (e4m3) quantization as one Triton program per row.

Bit-identical to torch.ops._C.dynamic_per_token_scaled_fp8_quant without a
scale upper bound (scale = max(amax / 448, 1 / (448 * 512)), q = x / scale),
and faster at decode sizes: 3.1us vs 4.6us for 64 x 4096 bf16 (the Step5 MoE
input, quantized before the DP dispatch). Gated by VLLM_HCU_TRITON_TOKEN_FP8_QUANT.
"""

import os

import torch

from vllm.triton_utils import tl, triton
from vllm_hcu.platforms import envs as henvs

ENABLED = henvs.optional_custom_op_enabled(
    os.environ.get("VLLM_HCU_TRITON_TOKEN_FP8_QUANT", "0") == "1"
)
_MAX_K = 8192


@triton.jit
def _token_fp8_quant_kernel(x_ptr, stride_x, q_ptr, s_ptr, K: tl.constexpr, BK: tl.constexpr):
    t = tl.program_id(0)
    offs = tl.arange(0, BK)
    m = offs < K
    x = tl.load(x_ptr + t.to(tl.int64) * stride_x + offs, mask=m, other=0.0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x), axis=0) / 448.0, 1.0 / (448.0 * 512.0))
    q = tl.minimum(tl.maximum(x / s, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(q_ptr + t.to(tl.int64) * K + offs, q, mask=m)
    tl.store(s_ptr + t, s)


def can_use(x: torch.Tensor) -> bool:
    return (
        ENABLED
        and x.dim() == 2
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.stride(-1) == 1
        and x.shape[-1] <= _MAX_K
        and x.is_cuda
    )


def token_fp8_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [M, K] -> (fp8 [M, K], fp32 scale [M, 1])."""
    M, K = x.shape
    q = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(M, 1, dtype=torch.float32, device=x.device)
    if M > 0:
        _token_fp8_quant_kernel[(M,)](
            x, x.stride(0), q, s, K=K, BK=triton.next_power_of_2(K), num_warps=8
        )
    return q, s
