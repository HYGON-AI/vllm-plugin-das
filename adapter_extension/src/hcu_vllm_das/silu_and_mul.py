"""vLLM 0.32 OOT LightOp implementation of ``SiluAndMul``."""

from __future__ import annotations

import torch

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.utils.torch_utils import direct_register_custom_op

from .environment import silu_and_mul_enabled


def _output(input_value: torch.Tensor) -> torch.Tensor:
    output_shape = input_value.shape[:-1] + (input_value.shape[-1] // 2,)
    return torch.empty(
        output_shape,
        dtype=input_value.dtype,
        device=input_value.device,
    )


def _lightop_impl(input_value: torch.Tensor) -> torch.Tensor:
    from lightop.activation import silu_and_mul_opt

    output = _output(input_value)
    silu_and_mul_opt(output, input_value)
    return output


def _fake_impl(input_value: torch.Tensor) -> torch.Tensor:
    return _output(input_value)


direct_register_custom_op(
    op_name="hcu_silu_and_mul_lightop",
    op_func=_lightop_impl,
    mutates_args=[],
    fake_impl=_fake_impl,
)


@SiluAndMul.register_oot
class HcuSiluAndMul(SiluAndMul):
    """Select LightOp on HCU with two explicit rollback switches."""

    @classmethod
    def enabled(cls) -> bool:
        """Expose an explicit opt-in without hiding official enablement."""

        return silu_and_mul_enabled() or super().enabled()

    def forward_hip(self, input_value: torch.Tensor) -> torch.Tensor:
        if silu_and_mul_enabled():
            return torch.ops.vllm.hcu_silu_and_mul_lightop(input_value)
        return super().forward_cuda(input_value)


__all__ = ["HcuSiluAndMul"]
