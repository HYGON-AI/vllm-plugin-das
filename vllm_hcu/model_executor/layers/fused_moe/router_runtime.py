# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU router implementations used by the v0.25.1 runtime adapters."""

from __future__ import annotations

def eplb_map_to_physical_and_record(
    module,
    original,
    topk_ids,
    expert_load_view,
    logical_to_physical_map,
    logical_replica_count,
    record_enabled,
    num_unpadded_tokens=None,
):
    from vllm_hcu.platforms import envs as henvs

    if not henvs.VLLM_HCU_USE_TORCH_EPLB_MAP_RECORD:
        return original(
            topk_ids,
            expert_load_view,
            logical_to_physical_map,
            logical_replica_count,
            record_enabled,
            num_unpadded_tokens,
        )
    topk_shape = topk_ids.shape
    flat = topk_ids.reshape(-1)
    if flat.numel() == 0:
        return topk_ids
    num_active_experts = topk_shape[-1]
    valid_expert = (flat >= 0) & (flat < logical_replica_count.shape[0])
    safe_expert = module.torch.where(valid_expert, flat, module.torch.zeros_like(flat)).long()
    replica_count = logical_replica_count[safe_expert].clamp_min(1).long()
    token_idx = (
        module.torch.arange(
            flat.numel(), device=flat.device, dtype=module.torch.long
        )
        // num_active_experts
    )
    replica_idx = ((token_idx * 2654435769) % (1 << 32)) % replica_count
    mapped = logical_to_physical_map[safe_expert, replica_idx].to(flat.dtype)
    physical = module.torch.where(valid_expert, mapped, module.torch.full_like(flat, -1))
    valid_physical = (physical >= 0) & (physical < expert_load_view.shape[0])
    safe_physical = module.torch.where(
        valid_physical,
        physical,
        module.torch.zeros_like(physical),
    ).long()
    increments = valid_physical.to(expert_load_view.dtype) * record_enabled.to(
        expert_load_view.dtype
    )
    if num_unpadded_tokens is not None:
        increments = increments * (
            token_idx < num_unpadded_tokens.to(token_idx.device)
        ).to(expert_load_view.dtype)
    expert_load_view.scatter_add_(0, safe_physical, increments)
    return physical.reshape(topk_shape)


def make_hcu_grouped_topk_router(base_class):
    class HcuGroupedTopKRouter(base_class):
        def _compute_routing(
            self,
            hidden_states,
            router_logits,
            indices_type,
            *,
            input_ids=None,
        ):
            from vllm_hcu.platforms import envs as henvs

            num_experts = router_logits.shape[-1]
            valid_grouping = (
                num_experts > self.num_expert_group
                and num_experts % self.num_expert_group == 0
            )
            enabled = bool(
                valid_grouping
                and self.e_score_correction_bias is not None
                and henvs.VLLM_HCU_USE_CUSTOM_OPS
                and henvs.VLLM_HCU_USE_FUSE_MOE_GATE
            )
            if not enabled:
                return super()._compute_routing(
                    hidden_states,
                    router_logits,
                    indices_type,
                    input_ids=input_ids,
                )
            from lightop.moe import moe_fused_gate
            topk_weights, topk_ids = moe_fused_gate(
                router_logits,
                self.e_score_correction_bias,
                self.num_expert_group,
                self.topk_group,
                self.top_k,
                0,
                self.routed_scaling_factor,
            )
            if indices_type is not None and topk_ids.dtype != indices_type:
                topk_ids = topk_ids.to(indices_type)
            return topk_weights, topk_ids

    HcuGroupedTopKRouter.__name__ = "HcuGroupedTopKRouter"
    HcuGroupedTopKRouter.__qualname__ = "HcuGroupedTopKRouter"
    HcuGroupedTopKRouter.__module__ = __name__
    return HcuGroupedTopKRouter


__all__ = ["eplb_map_to_physical_and_record", "make_hcu_grouped_topk_router"]
