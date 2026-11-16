# SPDX-License-Identifier: Apache-2.0
"""Dispatcher boundary for the categorized public LightOp AutoAWQ API."""

from __future__ import annotations

import torch
from vllm.utils.torch_utils import direct_register_custom_op


def _gemm(a: torch.Tensor, b: torch.Tensor, scales_zeros: torch.Tensor) -> torch.Tensor:
    from lightop.gemm_ops import gemm_awq_w4a16_marlin

    return gemm_awq_w4a16_marlin(a, b, scales_zeros)


def _fake(a: torch.Tensor, b: torch.Tensor, scales_zeros: torch.Tensor) -> torch.Tensor:
    return a.new_empty((a.shape[0], scales_zeros.shape[0]))


direct_register_custom_op(
    op_name="hcu_lightop_autoawq",
    op_func=_gemm,
    mutates_args=[],
    fake_impl=_fake,
)
