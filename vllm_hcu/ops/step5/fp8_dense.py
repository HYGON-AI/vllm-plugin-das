# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 dense linears in fp8: bf16 x [M, K] @ W -> bf16 [M, N] as an fp8 GEMM.

The attention and dense-MLP weights are bf16 in the checkpoint. After loading
they are quantized to fp8 e4m3 with one scale per output channel (symmetric
absmax, the scheme of the checkpoint's fp8 MoE experts) and the bf16 copy is
freed. Activations are quantized per token in plain torch ops, which Inductor
fuses into the producer (the RMSNorm, the attention gate multiply), and the GEMM
is LightOp's hipBLASLt channel-wise W8A8 kernel, the HCU scaled_mm backend.

On gfx938 it beats the bf16 hipBLASLt GEMM at every decode and prefill size of
these shapes (M=64: o_proj 29.5 -> 16.2 us, MLP down 39.5 -> 18.0 us; M>=2k:
1.6-1.7x), except that N not divisible by 256 hits a ~70 us slow path, so N is
padded to a multiple of 256 with zero rows and the caller drops the pad columns.
"""

import torch

from vllm.utils.torch_utils import direct_register_custom_op

FP8_MAX = 448.0
N_ALIGN = 256


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


def quant_per_token_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 ``[M, K]`` -> (fp8 ``[M, K]``, fp32 scale ``[M, 1]``), in torch ops so
    that Inductor fuses it into whatever produced x."""
    xf = x.float()
    s = (xf.abs().amax(dim=-1, keepdim=True) / FP8_MAX).clamp_min(1e-12)
    return (xf / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), s


def _step5_fp8_gemm(
    xq: torch.Tensor, xs: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor
) -> torch.Tensor:
    M, K = xq.shape
    N = wq.shape[0]
    if M == 0:
        return torch.empty(0, N, device=xq.device, dtype=torch.bfloat16)
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
    x: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor, n: int | None = None
) -> torch.Tensor:
    """``x [..., K]`` bf16 @ quantize_weight_fp8 output -> bf16 ``[..., N]``.

    With ``n=None`` the pad columns are kept (for callers that split anyway).
    """
    lead = x.shape[:-1]
    xq, xs = quant_per_token_fp8(x.reshape(-1, x.shape[-1]))
    out = torch.ops.vllm.step5_fp8_gemm(xq, xs, wq, ws)
    if n is not None and n != wq.shape[0]:
        out = out[:, :n]
    return (
        out.view(*lead, out.shape[-1])
        if out.is_contiguous()
        else out.reshape(*lead, out.shape[-1])
    )
