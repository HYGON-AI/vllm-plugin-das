import os
import torch
from vllm.model_executor.layers.activation import SiluAndMul
from vllm_hcu.platforms import envs as envs
import lightop.op as op


def silu_and_mul_opt_lightop_impl(input: torch.Tensor) -> torch.Tensor:
    d = input.shape[-1] // 2
    output_shape = input.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    op.silu_and_mul_opt(out, input)
    return out

@SiluAndMul.register_oot
class HcuSiluAndMul(SiluAndMul):
    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        if envs.VLLM_HCU_USE_CUSTOM_OPS and  envs.VLLM_HCU_USE_CUSTOM_SILU_AND_MUL:
            return silu_and_mul_opt_lightop_impl(x)
        return super().forward_cuda(x)
