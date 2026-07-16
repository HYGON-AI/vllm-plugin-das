# SPDX-License-Identifier: Apache-2.0
"""Use the optional LightOP MoE alignment kernel on HCU."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import load_exact_module, require_callable, require_parameter_names

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.moe_align_block_size"
PATCH_ID = "worker.op_opt.moe.align_block_size"
TARGETS = (f"{TARGET_MODULE}.moe_align_block_size",)
_MARKER = "_vllm_hcu_moe_align_applied"


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
            henvs.VLLM_HCU_USE_CUSTOM_OPS and henvs.VLLM_HCU_USE_LIGHTOP_MOE_ALIGN
        )
        if not enabled:
            return original(
                topk_ids,
                block_size,
                num_experts,
                expert_map,
                pad_sorted_ids,
                ignore_invalid_experts,
            )
        try:
            from lightop import op as lightop
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "VLLM_HCU_USE_LIGHTOP_MOE_ALIGN is enabled, but lightop.op is unavailable"
            ) from exc
        max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
        if pad_sorted_ids:
            max_num_tokens_padded = target.round_up(max_num_tokens_padded, block_size)
        if topk_ids.numel() < num_experts:
            max_num_tokens_padded = min(topk_ids.numel() * block_size, max_num_tokens_padded)
        sorted_ids = target.torch.empty(
            (max_num_tokens_padded,), dtype=target.torch.int32, device=topk_ids.device
        )
        max_blocks = target.triton.cdiv(max_num_tokens_padded, block_size)
        expert_ids = target.torch.empty(
            (max_blocks,), dtype=target.torch.int32, device=topk_ids.device
        )
        num_tokens_post_pad = target.torch.empty(
            (1,), dtype=target.torch.int32, device=topk_ids.device
        )
        try:
            lightop.moe_align_block_size(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
                expert_map=None,
            )
        except (TypeError, AttributeError) as exc:
            raise RuntimeError("installed LightOP lacks the required HCU MoE align API") from exc
        if expert_map is not None and not ignore_invalid_experts:
            expert_ids = expert_map[expert_ids]
        return sorted_ids, expert_ids, num_tokens_post_pad

    target._vllm_hcu_original_moe_align_block_size = original
    target.moe_align_block_size = hcu_moe_align_block_size
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
