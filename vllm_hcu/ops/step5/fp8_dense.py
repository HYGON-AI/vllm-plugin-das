# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 dense linears in fp8: bf16 x [M, K] @ W -> bf16 [M, N] as an fp8 GEMM.

The attention and dense-MLP weights are bf16 in the checkpoint. After loading
they are quantized to fp8 e4m3 with one scale per output channel (symmetric
absmax, the scheme of the checkpoint's fp8 MoE experts) and the bf16 copy is
freed. Activations are quantized per token in plain torch ops for large batches,
which Inductor fuses into the producer. Small decode batches may use the exact
Step5 token quantizer under the existing token-quant switch. The GEMM
is LightOp's hipBLASLt channel-wise W8A8 kernel, the HCU scaled_mm backend.

On gfx938 it beats the bf16 hipBLASLt GEMM at every decode and prefill size of
these shapes (M=64: o_proj 29.5 -> 16.2 us, MLP down 39.5 -> 18.0 us; M>=2k:
1.6-1.7x), except that N not divisible by 256 hits a ~70 us slow path, so N is
padded to a multiple of 256 with zero rows and the caller drops the pad columns.
"""

import os

import torch

from vllm.utils.torch_utils import direct_register_custom_op
from vllm_hcu.platforms import envs as henvs

FP8_MAX = 448.0
N_ALIGN = 256
_TUNED_DENSE = henvs.optional_custom_op_enabled(
    os.environ.get("HCU_STEP5_DENSE_TUNING", "0") == "1"
)
_DECODE_TOKEN_QUANT = henvs.optional_custom_op_enabled(
    os.environ.get("VLLM_HCU_TRITON_TOKEN_FP8_QUANT", "0") == "1"
)

if _DECODE_TOKEN_QUANT:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _step5_dense_token_quant_kernel(
        X, Q, S, S0: tl.constexpr, S1: tl.constexpr,
        K: tl.constexpr, BK: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.arange(0, BK)
        x = tl.load(X + row.to(tl.int64) * S0 + col * S1,
                    mask=col < K, other=0.).to(tl.float32)
        # Match torch.amax NaN propagation explicitly, independently of the
        # provider's default max-reduction NaN policy.
        nan_mask = (x != x) & (col < K)
        has_nan = tl.sum(nan_mask.to(tl.int32), 0) > 0
        amax = tl.max(tl.where(nan_mask, 0., tl.abs(x)), 0)
        # Torch's scalar /448 path multiplies by this rounded FP32 constant.
        # Keep the dense floor1e-12: the general token quantizer has a different
        # floor and is not an interchangeable implementation of this formula.
        scale = amax * tl.full((), 1.0 / 448.0, tl.float32)
        scale = tl.maximum(scale, tl.full((), 1.e-12, tl.float32))
        scale = tl.where(has_nan, float('nan'), scale)
        v = tl.div_rn(x, scale)
        clipped = tl.minimum(tl.maximum(v, -448.), 448.)
        clipped = tl.where(v != v, v, clipped)
        tl.store(Q + row.to(tl.int64) * K + col,
                 clipped.to(Q.dtype.element_ty), mask=col < K)
        tl.store(S + row, scale)


@torch.no_grad()
def quantize_weight_fp8(w_kn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 ``[K, N]`` -> (fp8 ``[Np, K]`` contiguous, fp32 scale ``[Np, 1]``).

    Np is N rounded up to N_ALIGN; pad rows are zero with scale 1.
    """
    K, N = w_kn.shape
    n_pad = -N % N_ALIGN
    q = torch.zeros(N + n_pad, K, device=w_kn.device, dtype=torch.float8_e4m3fn)
    scale = torch.ones(N + n_pad, 1, device=w_kn.device, dtype=torch.float32)
    # Chunked over N so the fp32 temporaries stay small.
    for n0 in range(0, N, 1024):
        w = w_kn[:, n0 : n0 + 1024].t().float()
        s = (w.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp_min(1e-12)
        q[n0 : n0 + w.shape[0]] = (w / s).clamp(-FP8_MAX, FP8_MAX).to(q.dtype)
        scale[n0 : n0 + w.shape[0]] = s
    return q, scale


def _quant_per_token_fp8_torch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Original expression, retained for producer fusion and unsupported inputs."""
    xf = x.float()
    s = (xf.abs().amax(dim=-1, keepdim=True) / FP8_MAX).clamp_min(1e-12)
    return (xf / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), s


def _can_use_dense_fp8_quant_op(x: torch.Tensor) -> bool:
    return (
        henvs.optional_custom_op_enabled(_DECODE_TOKEN_QUANT)
        and x.is_cuda
        and x.ndim == 2
        and x.dtype in (torch.bfloat16, torch.float32)
        and 0 < x.shape[1] <= 8192
    )


def _can_use_decode_fp8_quant(x: torch.Tensor) -> bool:
    return _can_use_dense_fp8_quant_op(x) and x.shape[0] <= 64


def _step5_dense_fp8_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # A cached opt-in cannot bypass a subsequently disabled custom-op master.
    if not _can_use_decode_fp8_quant(x):
        return _quant_per_token_fp8_torch(x)
    m, k = x.shape
    q = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((m, 1), device=x.device, dtype=torch.float32)
    if m:
        _step5_dense_token_quant_kernel[(m,)](
            x, q, scale, x.stride(0), x.stride(1), K=k,
            BK=triton.next_power_of_2(k), num_warps=8,
        )
    return q, scale


def _step5_dense_fp8_quant_fake(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn),
        torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32),
    )


direct_register_custom_op(
    op_name="step5_dense_fp8_quant",
    op_func=_step5_dense_fp8_quant,
    mutates_args=[],
    fake_impl=_step5_dense_fp8_quant_fake,
)


def quant_per_token_fp8(
    x: torch.Tensor, dynamic_small_m: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep eager-backend M dispatch inside the op, avoiding shape specialization.

    Other backends retain the Torch producer-fusion path for large batches.
    """
    if _can_use_dense_fp8_quant_op(x) and (dynamic_small_m or x.shape[0] <= 64):
        return torch.ops.vllm.step5_dense_fp8_quant(x)
    return _quant_per_token_fp8_torch(x)


def _step5_fp8_gemm(
    xq: torch.Tensor, xs: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor
) -> torch.Tensor:
    M, K = xq.shape
    N = wq.shape[0]
    if M == 0:
        return torch.empty(0, N, device=xq.device, dtype=torch.bfloat16)
    if henvs.optional_custom_op_enabled(_TUNED_DENSE):
        from .fp8_dense_tuned import tuned_fp8_gemm

        out = tuned_fp8_gemm(xq, xs, wq, ws)
        if out is not None:
            return out
    from lightop.gemm_ops import hipblaslt_w8a8_channelwise_gemm

    status, out = hipblaslt_w8a8_channelwise_gemm(
        xq, wq, xs, ws, M, N, K, "NT", torch.bfloat16, None
    )
    assert status is True, "hipblaslt_w8a8_channelwise_gemm failed"
    return out.view(M, N)


def _step5_fp8_gemm_fake(
    xq: torch.Tensor, xs: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor
) -> torch.Tensor:
    return torch.empty(xq.shape[0], wq.shape[0], device=xq.device, dtype=torch.bfloat16)


direct_register_custom_op(
    op_name="step5_fp8_gemm",
    op_func=_step5_fp8_gemm,
    mutates_args=[],
    fake_impl=_step5_fp8_gemm_fake,
)


def fp8_linear(
    x: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor, n: int | None = None,
    *, dynamic_small_m: bool = False,
) -> torch.Tensor:
    """``x [..., K]`` bf16 @ quantize_weight_fp8 output -> bf16 ``[..., N]``.

    With ``n=None`` the pad columns are kept (for callers that split anyway).
    """
    lead = x.shape[:-1]
    xq, xs = quant_per_token_fp8(
        x.reshape(-1, x.shape[-1]), dynamic_small_m=dynamic_small_m)
    out = torch.ops.vllm.step5_fp8_gemm(xq, xs, wq, ws)
    if n is not None and n != wq.shape[0]:
        out = out[:, :n]
    return (
        out.view(*lead, out.shape[-1])
        if out.is_contiguous()
        else out.reshape(*lead, out.shape[-1])
    )
