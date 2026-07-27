# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import torch
from vllm.model_executor.layers.layernorm import RMSNorm
from lightop import fused_add_rms_norm
from lightop.op import rmsnorm_forward_autograd
from vllm.utils.torch_utils import direct_register_custom_op
import vllm_hcu.platforms.envs as henvs


def _hcu_rmsnorm_forward_autograd_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    training: bool,
) -> torch.Tensor:
    return rmsnorm_forward_autograd(x, weight, variance_epsilon, training)


def _hcu_rmsnorm_forward_autograd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    training: bool,
) -> torch.Tensor:
    return torch.empty_like(x)


def _hcu_fused_add_rms_norm_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    fused_add_rms_norm(x, residual, weight, variance_epsilon)
    return x, residual


def _hcu_fused_add_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return x, residual


direct_register_custom_op(
    op_name="hcu_rmsnorm_forward_autograd",
    op_func=_hcu_rmsnorm_forward_autograd_impl,
    fake_impl=_hcu_rmsnorm_forward_autograd_fake,
)

direct_register_custom_op(
    op_name="hcu_fused_add_rms_norm",
    op_func=_hcu_fused_add_rms_norm_impl,
    mutates_args=["x", "residual"],
    fake_impl=_hcu_fused_add_rms_norm_fake,
)


@RMSNorm.register_oot
class HcuRMSNorm(RMSNorm):
    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if  henvs.VLLM_HCU_USE_CUSTOM_OPS and henvs.VLLM_HCU_USE_CUSTOM_RMS_NORM:
            if residual is None:
                out = torch.ops.vllm.hcu_rmsnorm_forward_autograd(
                    x, self.weight, self.variance_epsilon, self.training
                )
                return out
            else:
                return torch.ops.vllm.hcu_fused_add_rms_norm(
                    x, residual, self.weight, self.variance_epsilon
                )
        else:
            return self.forward_cuda(x, residual)
