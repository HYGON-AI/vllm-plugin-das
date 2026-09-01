# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import torch
import torch.nn as nn

from vllm.utils.torch_utils import direct_register_custom_op

class FusedSiluAndMulAndQuant(nn.Module):
    """Fuse silu and mul and int8 quant.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                quant_dtype: torch.dtype = torch.int8,) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.fuse_silu_mul_quant(x, quant_dtype)

def fuse_silu_mul_quant_real(input: torch.Tensor,
                             quant_dtype: torch.dtype
                                   ) -> tuple[torch.Tensor, torch.Tensor]:
    from lightop.activation import (
        fuse_silu_mul_per_token_quant as fuse_silu_mul_quant_lightop,
    )
    output = torch.empty(input.shape[0], input.shape[-1] // 2, dtype=quant_dtype, device=input.device)
    scales = torch.empty((input.shape[0], 1),
                        device=input.device,
                        dtype=torch.float32)
    fuse_silu_mul_quant_lightop(input, output=output, scales=scales)
    return output, scales

def fuse_silu_mul_quant_fake(input: torch.Tensor,
                             quant_dtype: torch.dtype
                                   ) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty(input.shape[0], input.shape[-1] // 2, dtype=quant_dtype, device=input.device)
    scales = torch.empty((input.shape[0], 1),
                        device=input.device,
                        dtype=torch.float32)
    return output, scales

direct_register_custom_op(
    op_name="fuse_silu_mul_quant",
    op_func=fuse_silu_mul_quant_real,
    mutates_args=[],
    fake_impl=fuse_silu_mul_quant_fake,
)
