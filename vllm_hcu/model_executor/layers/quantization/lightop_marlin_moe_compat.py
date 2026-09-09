# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Compatibility fixes for LightOp's vendored SlimQuant Marlin runners."""

from __future__ import annotations

import functools
import importlib
from collections.abc import Callable

import torch
import triton
import triton.language as tl
from vllm import _custom_ops as ops


_IMPLEMENTATION_PREFIX = "lightop."
_PATCH_MARKER = "_vllm_hcu_safe_alignment_installed"


def is_lightop_marlin_moe_supported(moe: object) -> bool:
    """Check LightOp's two-stage Marlin config before weights are packed."""
    try:
        from lightop import envs as lightop_envs
        from lightop.moe import get_moe_cuda_marlin_config

        in_dtype = getattr(moe, "in_dtype")
        compute_dtype = {
            torch.bfloat16: tl.bfloat16,
            torch.float16: tl.float16,
            torch.float32: tl.float32,
        }.get(in_dtype)
        if compute_dtype is None:
            return False

        device = torch.device(getattr(moe, "device"))
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        intermediate = int(getattr(moe, "intermediate_size_per_partition"))
        hidden = int(getattr(moe, "hidden_dim"))
        w13_num_shards = int(getattr(moe, "w13_num_shards"))
        result = get_moe_cuda_marlin_config(
            int(getattr(moe, "num_experts")),
            1,
            w13_num_shards * intermediate,
            hidden,
            hidden,
            intermediate,
            int(getattr(moe, "experts_per_token")),
            lightop_envs.LMSLIM_GPU_NAME,
            properties.multi_processor_count,
            compute_dtype,
        )
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return False

    return bool(
        isinstance(result, tuple)
        and len(result) == 3
        and isinstance(result[0], dict)
        and isinstance(result[1], dict)
        and result[0]
        and result[1]
        and result[2]
    )


def _safe_remap_expert_ids(
    expert_ids: torch.Tensor,
    expert_map: torch.Tensor,
) -> torch.Tensor:
    valid = (expert_ids >= 0) & (expert_ids < expert_map.numel())
    safe_ids = expert_ids.clamp(min=0, max=expert_map.numel() - 1).long()
    mapped = expert_map[safe_ids]
    return torch.where(valid, mapped, torch.full_like(mapped, -1))


def ensure_safe_marlin_moe_alignment(fused_experts: Callable[..., object]) -> None:
    """Install stable vLLM alignment in a LightOp vendored Marlin runner.

    LightOp 0.6's out-parameter alignment kernel only writes routed token
    positions. Its vendored SlimQuant runners allocate ``sorted_ids`` with
    ``torch.empty``, leaving padding positions as arbitrary token ids, and its
    multi-token ordering differs from the order expected by Marlin. Patch the
    runner-local helper to use vLLM's stable alignment operation.  The native
    operation initializes every padding position with the sentinel expected by
    the kernels, so its output buffer does not need a separate fill.
    """

    module_name = getattr(fused_experts, "__module__", "")
    if not module_name.startswith(_IMPLEMENTATION_PREFIX):
        return

    implementation = importlib.import_module(module_name)
    original = getattr(implementation, "moe_align_block_size", None)
    if not callable(original):
        raise RuntimeError(
            f"LightOp Marlin implementation {module_name!r} has no callable "
            "moe_align_block_size"
        )
    if getattr(original, _PATCH_MARKER, False):
        return

    @functools.wraps(original)
    def safe_alignment(
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
        expert_map: torch.Tensor | None = None,
        pad_sorted_ids: bool = False,
        ignore_invalid_experts: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_num_tokens_padded = topk_ids.numel() + num_experts * (
            block_size - 1
        )
        if pad_sorted_ids:
            max_num_tokens_padded = triton.cdiv(
                max_num_tokens_padded,
                block_size,
            ) * block_size
        if topk_ids.numel() < num_experts:
            max_num_tokens_padded = min(
                topk_ids.numel() * block_size,
                max_num_tokens_padded,
            )

        sorted_ids = torch.empty(
            (max_num_tokens_padded,),
            dtype=torch.int32,
            device=topk_ids.device,
        )
        expert_ids = torch.empty(
            (triton.cdiv(max_num_tokens_padded, block_size),),
            dtype=torch.int32,
            device=topk_ids.device,
        )
        num_tokens_post_pad = torch.empty(
            (1,),
            dtype=torch.int32,
            device=topk_ids.device,
        )
        ops.moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            expert_map if ignore_invalid_experts else None,
        )
        if expert_map is not None and not ignore_invalid_experts:
            expert_ids = _safe_remap_expert_ids(expert_ids, expert_map)
        return sorted_ids, expert_ids, num_tokens_post_pad

    implementation._vllm_hcu_original_moe_align_block_size = original
    setattr(safe_alignment, _PATCH_MARKER, True)
    implementation.moe_align_block_size = safe_alignment


__all__ = [
    "ensure_safe_marlin_moe_alignment",
    "is_lightop_marlin_moe_supported",
]
