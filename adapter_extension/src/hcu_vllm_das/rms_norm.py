"""vLLM 0.32 OOT LightOp implementation of ``RMSNorm``."""

from __future__ import annotations

import torch
from lightop.norm import fused_add_rms_norm, rmsnorm_forward_autograd

from vllm import envs
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.utils.torch_utils import direct_register_custom_op

from .environment import rms_norm_enabled


_LIGHTOP_RMS_NORM_MIN_WIDTH = 64


def _rms_norm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    training: bool,
) -> torch.Tensor:
    return rmsnorm_forward_autograd(x, weight, variance_epsilon, training)


def _rms_norm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    training: bool,
) -> torch.Tensor:
    del weight, variance_epsilon, training
    return torch.empty_like(x)


def _fused_add_rms_norm_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    fused_add_rms_norm(x, residual, weight, variance_epsilon)
    return x, residual


def _fused_add_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del weight, variance_epsilon
    return x, residual


direct_register_custom_op(
    op_name="hcu_rmsnorm_forward_lightop",
    op_func=_rms_norm_impl,
    mutates_args=[],
    fake_impl=_rms_norm_fake,
)

direct_register_custom_op(
    op_name="hcu_fused_add_rms_norm_lightop",
    op_func=_fused_add_rms_norm_impl,
    mutates_args=["x", "residual"],
    fake_impl=_fused_add_rms_norm_fake,
)


@RMSNorm.register_oot
class HcuRMSNorm(RMSNorm):
    """Select LightOp for compatible RMSNorm calls and otherwise use vLLM."""

    def _can_use_lightop(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> bool:
        if not rms_norm_enabled() or envs.VLLM_BATCH_INVARIANT:
            return False

        pass_weight = self.pass_weight_add if residual is not None else self.pass_weight
        if not pass_weight or self.variance_size_override is not None:
            return False
        if (
            x.shape[-1] < _LIGHTOP_RMS_NORM_MIN_WIDTH
            or x.shape[-1] != self.hidden_size
            or not x.is_contiguous()
        ):
            return False

        weight = self.weight.data
        if not weight.is_contiguous() or weight.dtype != x.dtype:
            return False
        if residual is not None and (
            residual.shape != x.shape
            or residual.dtype != x.dtype
            or not residual.is_contiguous()
        ):
            return False
        return True

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not self._can_use_lightop(x, residual):
            return super().forward_cuda(x, residual)

        weight = self.weight.data
        if residual is None:
            return torch.ops.vllm.hcu_rmsnorm_forward_lightop(
                x,
                weight,
                self.variance_epsilon,
                self.training,
            )
        return torch.ops.vllm.hcu_fused_add_rms_norm_lightop(
            x,
            residual,
            weight,
            self.variance_epsilon,
        )


__all__ = ["HcuRMSNorm"]
