# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import os
import torch

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.custom_op import CustomOp
from vllm.utils.torch_utils import direct_register_custom_op
from vllm_hcu.platforms import envs as henvs

import lightop.op as op


def silu_and_mul_opt_lightop_impl(input: torch.Tensor) -> torch.Tensor:
    d = input.shape[-1] // 2
    output_shape = input.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    op.silu_and_mul_opt(out, input)
    return out


def silu_and_mul_opt_lightop_fake(input: torch.Tensor) -> torch.Tensor:
    d = input.shape[-1] // 2
    output_shape = input.shape[:-1] + (d,)
    return torch.empty(output_shape, dtype=input.dtype, device=input.device)


direct_register_custom_op(
    op_name="silu_and_mul_opt_lightop",
    op_func=silu_and_mul_opt_lightop_impl,
    mutates_args=[],
    fake_impl=silu_and_mul_opt_lightop_fake,
)


@SiluAndMul.register_oot
class HcuSiluAndMul(SiluAndMul):

    def __init__(self, *, compile_native: bool = True) -> None:
        # The upstream constructor eagerly binds torch.ops._C.silu_and_mul on
        # every CUDA-alike platform. That namespace belongs to vLLM's NVIDIA
        # extension and is intentionally absent in a source-tree HCU install.
        CustomOp.__init__(self, compile_native=compile_native)

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        if henvs.VLLM_HCU_USE_CUSTOM_OPS and henvs.VLLM_HCU_USE_CUSTOM_SILU_AND_MUL:
            return torch.ops.vllm.silu_and_mul_opt_lightop(x)

        return self.forward_native(x)

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_hip(x)
