# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Numerically equivalent HCU-native dynamic FP8 quantization fallbacks."""

from __future__ import annotations

import torch
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform


def dynamic_per_token_quant_fp8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize each token using vLLM's reference contract."""

    if x.ndim != 2:
        raise ValueError(
            f"HCU dynamic per-token FP8 quantization requires rank 2, got {x.ndim}"
        )
    fp8_min, fp8_max = get_fp8_min_max()
    min_scale = 1.0 / (fp8_max * 512.0)
    scales = (x.abs().amax(dim=-1, keepdim=True).float() / fp8_max).clamp(
        min=min_scale
    )
    quantized = (
        x.float() * scales.reciprocal()
    ).clamp(fp8_min, fp8_max).to(current_platform.fp8_dtype())
    return quantized, scales


__all__ = ["dynamic_per_token_quant_fp8"]
