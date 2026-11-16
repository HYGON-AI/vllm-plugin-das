# SPDX-License-Identifier: Apache-2.0
# Ported from vllm-hcu-main d095594: registers
# torch.ops.vllm.hcu_rms_norm_gated_strided_z (the fused gated RMSNorm that
# consumes the [L, num_v, head_v] strided z view directly). In 0.18 the
# registration lived in vllm_hcu/ops/rms_norm_gated.py; in the 0.25 plugin the
# rms_norm_gated module is the LightOp implementation, so the registration is
# standalone here.

import torch

from vllm.utils.torch_utils import direct_register_custom_op

from vllm_hcu.ops.hcu_gated_rmsnorm_strided_z import (
    hcu_gated_rmsnorm_strided_z as _strided_z_impl,
)


def _hcu_rms_norm_gated_strided_z_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    z_view: torch.Tensor,
    eps: float,
    norm_before_gate: bool,
    activation: str,
) -> torch.Tensor:
    return _strided_z_impl(
        x,
        weight,
        z_view,
        eps=eps,
        norm_before_gate=norm_before_gate,
        activation=activation,
    )


def _hcu_rms_norm_gated_strided_z_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    z_view: torch.Tensor,
    eps: float,
    norm_before_gate: bool,
    activation: str,
) -> torch.Tensor:
    return torch.empty_like(x)


try:
    direct_register_custom_op(
        op_name="hcu_rms_norm_gated_strided_z",
        op_func=_hcu_rms_norm_gated_strided_z_impl,
        fake_impl=_hcu_rms_norm_gated_strided_z_fake,
    )
except RuntimeError:
    # Already registered by a previous load attempt
    pass
