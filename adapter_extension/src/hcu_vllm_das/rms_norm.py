"""Optional eager LightOp RMSNorm; compiled execution stays with vLLM IR."""

from __future__ import annotations

import torch

from vllm import envs
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.utils.torch_utils import direct_register_custom_op

from .environment import rms_norm_enabled
from .extension import get_lightop_norm


_LIGHTOP_RMS_NORM_MIN_WIDTH = 64


def _rms_norm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    training: bool,
) -> torch.Tensor:
    del training  # The vLLM operator is inference-only.
    # vLLM's native IR rounds the normalized value before multiplying weight.
    # The autograd export narrows epsilon to x.dtype. The public inference
    # export retains FP32 epsilon; its default aliases x, so provide an output.
    normalized = get_lightop_norm().rms_norm(
        x, torch.ones_like(weight), variance_epsilon, torch.empty_like(x)
    )
    return normalized * weight


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
) -> None:
    get_lightop_norm().fused_add_rms_norm(
        x, residual, torch.ones_like(weight), variance_epsilon
    )
    x.mul_(weight)


def _fused_add_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> None:
    del x, residual, weight, variance_epsilon


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

    @classmethod
    def enabled(cls) -> bool:
        """Expose an explicit opt-in without hiding official enablement."""

        return rms_norm_enabled() or super().enabled()

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
            x.dtype not in {torch.bfloat16, torch.float16}
            or x.shape[-1] < _LIGHTOP_RMS_NORM_MIN_WIDTH
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
        # Core owns compiler precision and fusion across residual chains.
        # Opaque eager vendor ops would force intermediate low-precision stores.
        if torch.compiler.is_compiling():
            return super().forward_cuda(x, residual)

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
        torch.ops.vllm.hcu_fused_add_rms_norm_lightop(
            x,
            residual,
            weight,
            self.variance_epsilon,
        )
        return x, residual


HcuRMSNorm.name = RMSNorm.name


__all__ = ["HcuRMSNorm"]
