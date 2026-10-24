# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""LightOp integration for vLLM's non-hash sqrt-softplus MoE routing."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger


_MEASURED_ROUTE_SHAPES = frozenset(
    {
        (256, 6, 1024),
        (256, 8, 256),
        (256, 8, 1024),
        (384, 6, 1024),
        (384, 8, 256),
        (384, 8, 1024),
    }
)
logger = init_logger(__name__)


def _is_contiguous(value: Any) -> bool:
    check = getattr(value, "is_contiguous", None)
    return bool(callable(check) and check())


def _load_lightop_sqrtsoftplus() -> Any | None:
    try:
        from lightop.moe import moe_fused_gate_sqrtsoftplus
    except ImportError:
        return None
    if not callable(moe_fused_gate_sqrtsoftplus):
        return None
    return moe_fused_gate_sqrtsoftplus


def _load_lightop_sqrtsoftplus_v41_fused() -> Any | None:
    """Load the installed V4.1 gate+padding wrapper without eager import."""
    try:
        from lightop.moe import moe_fused_gate_sqrtsoftplus_v41_hcu_fused
    except ImportError:
        return None
    if not callable(moe_fused_gate_sqrtsoftplus_v41_hcu_fused):
        return None
    return moe_fused_gate_sqrtsoftplus_v41_hcu_fused


def is_lightop_sqrtsoftplus_available() -> bool:
    """Return whether the optional categorized LightOp export is usable."""

    return _load_lightop_sqrtsoftplus() is not None


def is_lightop_sqrtsoftplus_v41_fused_available() -> bool:
    """Return whether the exact V4.1 gate+padding export is callable."""
    return _load_lightop_sqrtsoftplus_v41_fused() is not None


def is_lightop_sqrtsoftplus_shape_supported(
    tokens: int,
    experts: int,
    topk: int,
) -> bool:
    """Return whether production-route performance passed for this shape."""

    return (int(experts), int(topk), int(tokens)) in _MEASURED_ROUTE_SHAPES


def is_lightop_sqrtsoftplus_v41_shape_supported(
    tokens: int,
    experts: int,
    topk: int,
) -> bool:
    """Return the source-proven V4.1 correctness domain.

    The aa4a343 kernel dispatches every row count, top-k 1..16, and 256/384
    experts. The narrower measured table above remains only for the older
    generic optional route and must not gate V4.1 correctness.
    """
    del tokens
    return int(experts) in (256, 384) and 1 <= int(topk) <= 16


def can_use_lightop_sqrtsoftplus(
    gating_output: Any,
    correction_bias: Any,
    *,
    topk: int,
    input_tokens: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
    bias_vl: torch.Tensor | None = None,
    image_sentinel_lo: int = 0,
) -> bool:
    """Return whether LightOp can preserve this vLLM routing contract."""

    del input_tokens  # It is only semantically relevant with a hash table.
    logits_shape = getattr(gating_output, "shape", ())
    bias_shape = getattr(correction_bias, "shape", ())
    logits_device = getattr(gating_output, "device", None)
    bias_device = getattr(correction_bias, "device", None)
    if (
        hash_indices_table is not None
        or bias_vl is not None
        or image_sentinel_lo != 0
    ):
        return False
    if len(logits_shape) != 2 or len(bias_shape) != 1:
        return False
    num_tokens = int(logits_shape[0])
    num_experts = int(logits_shape[-1])
    return bool(
        getattr(logits_device, "type", None) == "cuda"
        and bias_device == logits_device
        and getattr(gating_output, "dtype", None) == torch.float32
        and getattr(correction_bias, "dtype", None) == torch.float32
        and _is_contiguous(gating_output)
        and _is_contiguous(correction_bias)
        and int(bias_shape[0]) == num_experts
        and is_lightop_sqrtsoftplus_shape_supported(
            num_tokens, num_experts, topk
        )
    )


def can_use_lightop_sqrtsoftplus_v41_fused(
    gating_output: Any,
    correction_bias: Any,
    *,
    topk: int,
    indices_dtype: torch.dtype,
    input_tokens: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
    bias_vl: torch.Tensor | None = None,
    image_sentinel_lo: int = 0,
) -> bool:
    """Check the installed V4.1 gate+padding contract before calling it.

    The V4.1 export has no input-token/hash/VL arguments and returns int64
    expert IDs. VL/sentinel routing therefore remains on the generic route.
    """
    if (
        hash_indices_table is not None
        or bias_vl is not None
        or image_sentinel_lo != 0
        or indices_dtype not in (
        torch.int32,
        torch.int64,
        )
    ):
        return False
    logits_shape = getattr(gating_output, "shape", ())
    bias_shape = getattr(correction_bias, "shape", ())
    logits_device = getattr(gating_output, "device", None)
    bias_device = getattr(correction_bias, "device", None)
    if len(logits_shape) != 2 or len(bias_shape) not in (1, 2):
        return False
    num_tokens = int(logits_shape[0])
    num_experts = int(logits_shape[-1])
    if image_sentinel_lo != 0 and (
        input_tokens is None or len(bias_shape) != 2
    ):
        return False
    return bool(
        getattr(logits_device, "type", None) == "cuda"
        and bias_device == logits_device
        and getattr(gating_output, "dtype", None) == torch.float32
        and getattr(correction_bias, "dtype", None) == torch.float32
        and _is_contiguous(gating_output)
        and _is_contiguous(correction_bias)
        and (
            (len(bias_shape) == 1 and int(bias_shape[0]) == num_experts)
            or (
                len(bias_shape) == 2
                and tuple(int(x) for x in bias_shape) == (num_tokens, num_experts)
            )
        )
        and is_lightop_sqrtsoftplus_v41_shape_supported(
            num_tokens, num_experts, topk
        )
    )


def make_v41_bias(
    correction_bias: torch.Tensor,
    input_tokens: torch.Tensor | None,
    bias_vl: torch.Tensor | None,
    image_sentinel_lo: int,
    rows: int,
) -> torch.Tensor | None:
    """Build the source-proven 2D VL/text bias without changing token IDs."""
    if bias_vl is None:
        return correction_bias
    if (
        correction_bias is None
        or input_tokens is None
        or image_sentinel_lo <= 0
    ):
        return None
    if input_tokens.ndim != 1 or int(input_tokens.shape[0]) != rows:
        return None
    image = (
        (input_tokens >= image_sentinel_lo)
        & (input_tokens < image_sentinel_lo + 5)
    ).unsqueeze(1)
    return torch.where(image, bias_vl.view(1, -1), correction_bias.view(1, -1))


def get_v41_padding_mask() -> torch.Tensor | None:
    """Return vLLM's device padding mask without inspecting its values.

    The LightOp V4.1 export has no bool-mask argument. The mask is therefore
    applied to the returned tensors with device-side operations, preserving
    graph capture and the official ``(-1, 0)`` padded-row contract.
    """
    try:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
    except ImportError:
        return None
    if not is_forward_context_available():
        return None
    return get_forward_context().is_padding


def is_v41_padding_free_forward() -> bool:
    """Return whether no padding mask was published for this forward."""

    return get_v41_padding_mask() is None


def apply_v41_padding_contract(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Restore vLLM's padded-row outputs using device-side mask operations."""

    if padding_mask is None:
        return topk_weights, topk_ids
    if (
        padding_mask.dtype != torch.bool
        or padding_mask.ndim != 1
        or int(padding_mask.shape[0]) != int(topk_weights.shape[0])
        or padding_mask.device != topk_weights.device
    ):
        return None
    padded = padding_mask.unsqueeze(1)
    topk_weights = torch.where(
        padded, torch.zeros_like(topk_weights), topk_weights
    )
    topk_ids = torch.where(
        padded, torch.full_like(topk_ids, -1), topk_ids
    )
    return topk_weights, topk_ids


def is_v41_padding_mask_compatible(
    padding_mask: torch.Tensor | None,
    *,
    rows: int,
    device: Any,
) -> bool:
    """Check the mask ABI without reading device values or raising."""

    return padding_mask is None or bool(
        padding_mask.dtype == torch.bool
        and padding_mask.ndim == 1
        and int(padding_mask.shape[0]) == int(rows)
        and padding_mask.device == device
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

    operator = _load_lightop_sqrtsoftplus()
    if operator is None:
        raise RuntimeError(
            "VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_GATE requires callable "
            "lightop.moe.moe_fused_gate_sqrtsoftplus"
        )
    topk_weights, topk_ids = operator(
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
    logger.warning_once("Using LightOp sqrt-softplus MoE routing.")
    return topk_weights, topk_ids


def run_lightop_sqrtsoftplus_v41_fused(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    indices_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the installed V4.1 gate-and-bias ABI without padding fusion.

    The export itself returns int64 expert IDs. The vLLM BaseRouter performs
    the final requested dtype conversion after this return, so int32 and int64
    requests are both valid without changing the downstream contract here.
    Padding rows are rejected by the caller's explicit predicate; this adapter
    intentionally does not pass the optional native padding-count argument.
    """
    if indices_dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "LightOp V4.1 fused sqrt-softplus gate requires int32/int64 "
            "expert-ID consumer contract"
        )
    operator = _load_lightop_sqrtsoftplus_v41_fused()
    if operator is None:
        raise RuntimeError(
            "VLLM_HCU_USE_LIGHTOP_SQRTSOFTPLUS_V41_FUSED_GATE requires "
            "callable lightop.moe.moe_fused_gate_sqrtsoftplus_v41_hcu_fused"
        )
    topk_weights, topk_ids = operator(
        gating_output,
        correction_bias,
        int(topk),
        0,
        bool(renormalize),
        float(routed_scaling_factor),
        True,
    )
    if topk_ids.dtype != torch.int64:
        raise RuntimeError(
            "LightOp V4.1 fused sqrt-softplus gate returned non-int64 expert IDs"
        )
    if topk_ids.dtype != indices_dtype:
        topk_ids = topk_ids.to(dtype=indices_dtype)
    logger.warning_once("Using LightOp V4.1 fused sqrt-softplus MoE routing.")
    return topk_weights, topk_ids


__all__ = [
    "can_use_lightop_sqrtsoftplus",
    "can_use_lightop_sqrtsoftplus_v41_fused",
    "is_lightop_sqrtsoftplus_available",
    "is_lightop_sqrtsoftplus_v41_fused_available",
    "is_lightop_sqrtsoftplus_shape_supported",
    "is_lightop_sqrtsoftplus_v41_shape_supported",
    "make_v41_bias",
    "get_v41_padding_mask",
    "apply_v41_padding_contract",
    "is_v41_padding_mask_compatible",
    "run_lightop_sqrtsoftplus",
    "run_lightop_sqrtsoftplus_v41_fused",
    "is_v41_padding_free_forward",
]
