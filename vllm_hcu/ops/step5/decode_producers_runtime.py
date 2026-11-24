# SPDX-License-Identifier: Apache-2.0
"""Step5 decode producers with explicit FP32 residual and BF16 boundaries."""
import os
import torch
from vllm.utils.torch_utils import direct_register_custom_op
from vllm_hcu.platforms import envs as henvs

from .decode_producers_kernel import producer

_ENABLED = os.environ.get('VLLM_HCU_STEP5_DECODE_PRODUCERS', '0') == '1'


def _can_fuse(x, addition, weight):
    return (henvs.optional_custom_op_enabled(_ENABLED)
            and x.is_cuda and x.ndim == 2 and x.dtype == torch.float32
            and x.shape[1] == 4096 and x.shape[0] <= 64 and x.is_contiguous()
            and weight.shape == (4096,) and weight.dtype == torch.float32
            and weight.device == x.device and weight.is_contiguous()
            and (addition is None or (addition.shape == x.shape
                 and addition.dtype in (torch.bfloat16, torch.float32)
                 and addition.device == x.device and addition.is_contiguous())))


def _norm(x, weight, eps):
    if henvs.optional_custom_op_enabled(henvs.VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM):
        out = torch.ops.vllm.hcu_step5_gemma_rms_norm(x, weight, eps)
    else:
        from vllm import ir
        out = ir.ops.rms_norm(x, weight.float() + 1.0, eps)
    return out.to(torch.bfloat16)


def _input(residual: torch.Tensor, pending: torch.Tensor | None,
           weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if _can_fuse(residual, pending, weight):
        return producer(residual, weight, eps, pending, quant=True, warps=8)
    from .fp8_dense import quant_per_token_fp8
    # Each returned tensor owns storage; custom-op outputs may not alias inputs.
    combined = residual.clone() if pending is None else pending.float() + residual
    y = _norm(combined, weight, eps)
    q, scale = quant_per_token_fp8(y, dynamic_small_m=True)
    return combined, y, q, scale


def _post(residual: torch.Tensor, attention: torch.Tensor,
          weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    if _can_fuse(residual, attention, weight):
        return producer(residual, weight, eps, attention, quant=False, warps=8)
    combined = attention.float() + residual
    return combined, _norm(combined, weight, eps)


def _input_fake(residual: torch.Tensor, pending: torch.Tensor | None,
                weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (torch.empty_like(residual, dtype=torch.float32),
            torch.empty_like(residual, dtype=torch.bfloat16),
            torch.empty_like(residual, dtype=torch.float8_e4m3fn),
            torch.empty((residual.shape[0], 1), device=residual.device, dtype=torch.float32))


def _post_fake(residual: torch.Tensor, attention: torch.Tensor,
               weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.empty_like(residual, dtype=torch.float32),
            torch.empty_like(residual, dtype=torch.bfloat16))


direct_register_custom_op(op_name='step5_decode_input_producer_v1', op_func=_input,
                          fake_impl=_input_fake, mutates_args=[])
direct_register_custom_op(op_name='step5_decode_post_producer_v1', op_func=_post,
                          fake_impl=_post_fake, mutates_args=[])
