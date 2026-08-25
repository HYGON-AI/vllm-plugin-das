# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import os
import torch

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm_hcu.platforms import envs as henvs

logger = init_logger(__name__)

try:
    from lightop.activation import silu_and_mul_opt
except (ImportError, AttributeError):
    from lightop.op import silu_and_mul_opt

    logger.warning_once(
        "Using deprecated lightop.op activation API because "
        "lightop.activation is unavailable; upgrade LightOp."
    )


def silu_and_mul_opt_lightop_impl(input: torch.Tensor) -> torch.Tensor:
    d = input.shape[-1] // 2
    output_shape = input.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    silu_and_mul_opt(out, input)
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

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        if henvs.VLLM_HCU_USE_CUSTOM_OPS and henvs.VLLM_HCU_USE_CUSTOM_SILU_AND_MUL:
            return torch.ops.vllm.silu_and_mul_opt_lightop(x)

        return super().forward_cuda(x)
