# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Use the optional LightOP MoE alignment kernel on HCU."""

from __future__ import annotations

import functools
import sys
from types import ModuleType

from ._common import load_exact_module, require_callable, require_parameter_names

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.moe_align_block_size"
PATCH_ID = "worker.op_opt.moe.align_block_size"
TARGETS = (f"{TARGET_MODULE}.moe_align_block_size",)
_MARKER = "_vllm_hcu_moe_align_applied"
_CAPTURED_BINDING_MODULES = (
    "vllm.model_executor.layers.fused_moe.fused_moe",
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_wna16_rdna3",
    "vllm.model_executor.layers.fused_moe.experts.mxfp8_native_moe",
    "vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe",
    "vllm.model_executor.layers.fused_moe.experts.fused_humming_moe",
    "vllm.model_executor.layers.fused_moe.experts.marlin_moe",
    "vllm.model_executor.layers.fused_moe.experts.triton_moe",
)


def _rebind_loaded_consumers(original, replacement) -> None:
    """Update exact modules that imported the canonical function by value."""

    for module_name in _CAPTURED_BINDING_MODULES:
        consumer = sys.modules.get(module_name)
        if consumer is None:
            continue
        if getattr(consumer, "moe_align_block_size", None) is original:
            consumer.moe_align_block_size = replacement


def _safe_remap_expert_ids(torch_module, expert_ids, expert_map):
    """Map initialized expert ids and turn unused buffer slots into -1."""

    valid = (expert_ids >= 0) & (expert_ids < expert_map.numel())
    safe_ids = expert_ids.clamp(min=0, max=expert_map.numel() - 1).to(
        dtype=torch_module.long
    )
    mapped = expert_map[safe_ids]
    return torch_module.where(
        valid,
        mapped,
        torch_module.full_like(mapped, -1),
    )


def _has_compiled_moe_align(torch_module) -> bool:
    torch_ops = getattr(torch_module, "ops", None)
    moe_ops = getattr(torch_ops, "_moe_C", None)
    return callable(getattr(moe_ops, "moe_align_block_size", None))


def _torch_moe_align_block_size(
    torch_module,
    topk_ids,
    block_size,
    num_experts,
    expert_map,
    pad_sorted_ids,
    ignore_invalid_experts,
    round_up,
):
    num_tokens = topk_ids.numel()
    max_num_tokens_padded = num_tokens + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if num_tokens < num_experts:
        max_num_tokens_padded = min(
            num_tokens * block_size,
            max_num_tokens_padded,
        )

    sorted_ids_with_sentinel = torch_module.full(
        (max_num_tokens_padded + 1,),
        fill_value=num_tokens,
        dtype=torch_module.int32,
        device=topk_ids.device,
    )
    max_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch_module.full(
        (max_blocks,),
        fill_value=-1,
        dtype=torch_module.int32,
        device=topk_ids.device,
    )

    flat_ids = topk_ids.reshape(-1).to(dtype=torch_module.long)
    valid = (flat_ids >= 0) & (flat_ids < num_experts)
    alignment_ids = flat_ids
    if expert_map is not None and ignore_invalid_experts:
        safe_ids = flat_ids.clamp(min=0, max=expert_map.numel() - 1)
        alignment_ids = expert_map[safe_ids].to(dtype=torch_module.long)
        valid = valid & (alignment_ids >= 0) & (alignment_ids < num_experts)

    sort_keys = torch_module.where(
        valid,
        alignment_ids,
        torch_module.full_like(alignment_ids, num_experts),
    )
    order = torch_module.argsort(sort_keys, stable=True)
    count_ids = torch_module.where(
        valid,
        alignment_ids,
        torch_module.full_like(alignment_ids, num_experts),
    )
    counts_with_invalid = torch_module.zeros(
        (num_experts + 1,),
        dtype=torch_module.long,
        device=topk_ids.device,
    )
    counts_with_invalid.scatter_add_(
        0,
        count_ids,
        torch_module.ones_like(count_ids),
    )
    counts = counts_with_invalid[:num_experts]
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    actual_starts = torch_module.cumsum(counts, dim=0) - counts
    padded_starts = torch_module.cumsum(padded_counts, dim=0) - padded_counts

    ordered_keys = sort_keys[order]
    ordered_valid = ordered_keys < num_experts
    safe_ordered_keys = ordered_keys.clamp(min=0, max=num_experts - 1)
    ranks = torch_module.arange(
        num_tokens,
        device=topk_ids.device,
        dtype=torch_module.long,
    ) - actual_starts[safe_ordered_keys]
    destinations = padded_starts[safe_ordered_keys] + ranks
    safe_destinations = torch_module.where(
        ordered_valid,
        destinations,
        torch_module.full_like(destinations, max_num_tokens_padded),
    )
    sorted_ids_with_sentinel.scatter_(
        0,
        safe_destinations,
        torch_module.where(
            ordered_valid,
            order,
            torch_module.full_like(order, num_tokens),
        ).to(dtype=torch_module.int32),
    )
    sorted_ids = sorted_ids_with_sentinel[:max_num_tokens_padded]

    blocks_per_expert = padded_counts // block_size
    block_positions = torch_module.arange(
        max_blocks,
        device=topk_ids.device,
        dtype=torch_module.long,
    )
    cumulative_blocks = torch_module.cumsum(blocks_per_expert, dim=0)
    block_experts = torch_module.searchsorted(
        cumulative_blocks,
        block_positions,
        right=True,
    ).clamp(max=num_experts - 1)
    valid_blocks = block_positions < cumulative_blocks[-1]
    output_experts = block_experts
    if expert_map is not None and not ignore_invalid_experts:
        output_experts = expert_map[block_experts].to(dtype=torch_module.long)
    expert_ids.copy_(
        torch_module.where(
            valid_blocks,
            output_experts,
            torch_module.full_like(output_experts, -1),
        ).to(dtype=torch_module.int32)
    )
    num_tokens_post_pad = padded_counts.sum().to(dtype=torch_module.int32).reshape(1)
    return sorted_ids, expert_ids, num_tokens_post_pad


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        _rebind_loaded_consumers(
            target._vllm_hcu_original_moe_align_block_size,
            target.moe_align_block_size,
        )
        return False
    original = require_callable(target, "moe_align_block_size", TARGETS[0])
    require_parameter_names(
        original,
        TARGETS[0],
        (
            "topk_ids",
            "block_size",
            "num_experts",
            "expert_map",
            "pad_sorted_ids",
            "ignore_invalid_experts",
        ),
    )

    @functools.wraps(original)
    def hcu_moe_align_block_size(
        topk_ids,
        block_size,
        num_experts,
        expert_map=None,
        pad_sorted_ids=False,
        ignore_invalid_experts=False,
    ):
        from vllm_hcu.platforms import envs as henvs

        enabled = bool(
            henvs.VLLM_HCU_USE_CUSTOM_OPS
            and henvs.VLLM_HCU_USE_LIGHTOP_MOE_ALIGN
        )
        compiled_available = _has_compiled_moe_align(target.torch)
        if not enabled and not compiled_available:
            return _torch_moe_align_block_size(
                target.torch,
                topk_ids,
                block_size,
                num_experts,
                expert_map,
                pad_sorted_ids,
                ignore_invalid_experts,
                target.round_up,
            )
        needs_safe_native_remap = (
            not enabled
            and expert_map is not None
            and not ignore_invalid_experts
        )
        if not enabled and not needs_safe_native_remap:
            return original(
                topk_ids,
                block_size,
                num_experts,
                expert_map,
                pad_sorted_ids,
                ignore_invalid_experts,
            )
        max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
        if pad_sorted_ids:
            max_num_tokens_padded = target.round_up(
                max_num_tokens_padded,
                block_size,
            )
        if topk_ids.numel() < num_experts:
            max_num_tokens_padded = min(
                topk_ids.numel() * block_size,
                max_num_tokens_padded,
            )
        if enabled:
            # Triton treats ``topk_ids.numel()`` as the padding token.  The
            # LightOP 0.6 out-parameter kernel only writes routed tokens even
            # when its fused-fill argument is enabled, so an empty buffer can
            # leave arbitrary token ids inside the valid padded range.
            sorted_ids = target.torch.full(
                (max_num_tokens_padded,),
                fill_value=topk_ids.numel(),
                dtype=target.torch.int32,
                device=topk_ids.device,
            )
        else:
            sorted_ids = target.torch.empty(
                (max_num_tokens_padded,),
                dtype=target.torch.int32,
                device=topk_ids.device,
            )
        max_blocks = target.triton.cdiv(max_num_tokens_padded, block_size)
        expert_ids = target.torch.empty(
            (max_blocks,), dtype=target.torch.int32, device=topk_ids.device
        )
        num_tokens_post_pad = target.torch.empty(
            (1,), dtype=target.torch.int32, device=topk_ids.device
        )
        if enabled:
            try:
                from lightop.moe import moe_align_block_size_out
            except (ImportError, AttributeError) as exc:
                raise RuntimeError(
                    "VLLM_HCU_USE_LIGHTOP_MOE_ALIGN requires "
                    "lightop.moe.moe_align_block_size_out; upgrade LightOp"
                ) from exc
            moe_align_block_size_out(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
                expert_map if ignore_invalid_experts else None,
                None,
                None,
                is_ep=False,
                is_fuse_fill=False,
            )
        else:
            target.ops.moe_align_block_size(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
                None,
            )
        if expert_map is not None and not ignore_invalid_experts:
            expert_ids = _safe_remap_expert_ids(
                target.torch,
                expert_ids,
                expert_map,
            )
        return sorted_ids, expert_ids, num_tokens_post_pad

    target._vllm_hcu_original_moe_align_block_size = original
    target.moe_align_block_size = hcu_moe_align_block_size
    _rebind_loaded_consumers(original, hcu_moe_align_block_size)
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
