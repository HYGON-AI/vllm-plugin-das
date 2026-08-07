# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Use the optional LightOP MoE alignment kernel on HCU."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import load_exact_module, require_callable, require_parameter_names

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.moe_align_block_size"
PATCH_ID = "worker.op_opt.moe.align_block_size"
TARGETS = (f"{TARGET_MODULE}.moe_align_block_size",)
_MARKER = "_vllm_hcu_moe_align_applied"


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


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
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
                from lightop import op as lightop
            except (ImportError, AttributeError) as exc:
                raise RuntimeError(
                    "VLLM_HCU_USE_LIGHTOP_MOE_ALIGN is enabled, but "
                    "lightop.op is unavailable"
                ) from exc
            try:
                lightop.moe_align_block_size(
                    topk_ids,
                    num_experts,
                    block_size,
                    sorted_ids,
                    expert_ids,
                    num_tokens_post_pad,
                    expert_map if ignore_invalid_experts else None,
                    None,  # expert_mask
                    None,  # num_local_tokens
                    False,  # Is_EP
                    False,  # Is_fuse_fill; padding was initialized above
                )
            except (TypeError, AttributeError) as exc:
                raise RuntimeError(
                    "installed LightOP lacks the required HCU MoE align API"
                ) from exc
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
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
