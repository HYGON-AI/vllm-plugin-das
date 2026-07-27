# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from typing import Optional

import torch
import torch.nn as nn

from vllm.utils.torch_utils import direct_register_custom_op

class FusedRMSNormAndQuant(nn.Module):
    """Fuse Root mean square normalization and int8 quant.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        weight_dtype = dtype or torch.get_default_dtype()
        self.has_weight = has_weight
        self.weight = torch.ones(hidden_size, dtype=weight_dtype)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        quant_dtype: torch.dtype = torch.int8,
        update_input: Optional[bool] = True
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        i_q, i_s = torch.ops.vllm.fused_rmsquant_customer_impl(input=x,
                                                 weight=self.weight,
                                                 epsilon=self.variance_epsilon,
                                                 quant_dtype=quant_dtype,
                                                 residual=residual,
                                                 update_input=update_input)

        return i_q, i_s, residual
    

def fused_rmsquant_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    residual: Optional[torch.Tensor] = None,
    update_input: Optional[bool] = True
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input, device=input.device, dtype=quant_dtype)
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                        device=input.device,
                        dtype=torch.float32)
    
    from lightop.op import rms_norm_dynamic_per_token_quant as ligtop_rms_norm_dynamic_per_token_quant
    ligtop_rms_norm_dynamic_per_token_quant(output, input, weight,
                                               scales, epsilon,
                                               residual, update_input)
    return output, scales

def fused_rmsquant_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    residual: Optional[torch.Tensor] = None,
    update_input: Optional[bool] = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.compile"""
    output = torch.empty_like(input, dtype=quant_dtype)
    scales = torch.empty((input.numel() // input.shape[-1], 1),
                        device=input.device,
                        dtype=torch.float32)
    return output, scales

direct_register_custom_op(
    op_name="fused_rmsquant_customer_impl",
    op_func=fused_rmsquant_impl,
    mutates_args=["input", "residual"],
    fake_impl=fused_rmsquant_fake,
)
