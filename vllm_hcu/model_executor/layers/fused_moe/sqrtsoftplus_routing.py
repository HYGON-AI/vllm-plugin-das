# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""LightOp integration for vLLM's non-hash sqrt-softplus MoE routing."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger


_SUPPORTED_EXPERT_COUNTS = frozenset((256, 384))
_MAX_TOPK = 16
logger = init_logger(__name__)


def _is_contiguous(value: Any) -> bool:
    check = getattr(value, "is_contiguous", None)
    return bool(callable(check) and check())


def can_use_lightop_sqrtsoftplus(
    gating_output: Any,
    correction_bias: Any,
    *,
    topk: int,
    input_tokens: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
) -> bool:
    """Return whether LightOp can preserve this vLLM routing contract."""

    del input_tokens  # It is only semantically relevant with a hash table.
    logits_shape = getattr(gating_output, "shape", ())
    bias_shape = getattr(correction_bias, "shape", ())
    logits_device = getattr(gating_output, "device", None)
    bias_device = getattr(correction_bias, "device", None)
    if hash_indices_table is not None:
        return False
    if len(logits_shape) != 2 or len(bias_shape) != 1:
        return False
    num_experts = int(logits_shape[-1])
    return bool(
        getattr(logits_device, "type", None) == "cuda"
        and bias_device == logits_device
        and getattr(gating_output, "dtype", None) == torch.float32
        and getattr(correction_bias, "dtype", None) == torch.float32
        and _is_contiguous(gating_output)
        and _is_contiguous(correction_bias)
        and int(bias_shape[0]) == num_experts
        and num_experts in _SUPPORTED_EXPERT_COUNTS
        and 0 < int(topk) <= _MAX_TOPK
    )


def run_lightop_sqrtsoftplus(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    indices_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the LightOp route and normalize its index dtype for vLLM."""

    try:
        from lightop.moe import moe_fused_gate_sqrtsoftplus
    except ImportError as exc:
        raise RuntimeError(
            "VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_GATE requires "
            "lightop.moe.moe_fused_gate_sqrtsoftplus"
        ) from exc
    if not callable(moe_fused_gate_sqrtsoftplus):
        raise RuntimeError(
            "VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_GATE requires callable "
            "lightop.moe.moe_fused_gate_sqrtsoftplus"
        )
    logger.warning_once("Using LightOp sqrt-softplus MoE routing.")
    topk_weights, topk_ids = moe_fused_gate_sqrtsoftplus(
        gating_output,
        correction_bias,
        int(topk),
        0,
        bool(renormalize),
        float(routed_scaling_factor),
        True,
    )
    if topk_ids.dtype != indices_dtype:
        topk_ids = topk_ids.to(dtype=indices_dtype)
    return topk_weights, topk_ids


__all__ = [
    "can_use_lightop_sqrtsoftplus",
    "run_lightop_sqrtsoftplus",
]
