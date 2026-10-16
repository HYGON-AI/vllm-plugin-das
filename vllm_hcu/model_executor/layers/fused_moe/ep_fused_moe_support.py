# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Shared metadata-only contract for the optional channel-FP8 EP kernels."""
from __future__ import annotations

import torch


def supports_channel_ep_fused(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    *,
    block_shape,
    activation,
    expert_map,
    apply_router_weight_on_input: bool,
) -> bool:
    # Both weight preparation and execution must agree before keeping 3-D
    # channel weights unpacked. Block-FP8 has the same weight rank, but a
    # completely different scale interpretation.
    if (
        block_shape is not None
        or getattr(activation, "value", activation) != "swiglustep"
        or expert_map is None
        or apply_router_weight_on_input
        or w1.ndim != 3
        or w2.ndim != 3
        or w1.dtype != torch.float8_e4m3fn
        or w2.dtype != w1.dtype
        or w2.device != w1.device
        or not w1.is_contiguous()
        or not w2.is_contiguous()
    ):
        return False
    experts, twice_intermediate, hidden = w1.shape
    if (
        experts <= 0
        or hidden <= 0
        or hidden % 4096
        or twice_intermediate <= 0
        or twice_intermediate % 6
        or w2.shape != (experts, hidden, twice_intermediate // 2)
    ):
        return False
    # The activation kernel splits the intermediate width into three equal
    # power-of-two chunks; the reduction kernel uses 4096-wide blocks.
    chunk = twice_intermediate // 6
    if chunk & (chunk - 1):
        return False
    for scale, width in ((w1_scale, twice_intermediate), (w2_scale, hidden)):
        if (
            scale is None
            or scale.shape not in ((experts, width), (experts, width, 1))
            or not scale.is_floating_point()
            or scale.device != w1.device
        ):
            return False
    return True
