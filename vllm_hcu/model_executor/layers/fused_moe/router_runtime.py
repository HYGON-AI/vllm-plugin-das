# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU router implementations used by the v0.25.1 runtime adapters."""

from __future__ import annotations

import math
from collections.abc import Callable

from .lightop_routing import lightop_moe_gate_kwargs


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
            # Import the module first so a newer LightOp can advertise extra
            # scoring/normalization modes without a vLLM condition change.
            # If the optional backend is unavailable, retain the historical
            # fail-fast behavior for the legacy mode but let unsupported modes
            # use the official router.
            scoring_func = getattr(self, "scoring_func", None)
            renormalize = getattr(self, "renormalize", None)
            try:
                import lightop.moe as lightop_moe
            except ImportError:
                if scoring_func != "sigmoid" or not bool(renormalize):
                    return super()._compute_routing(
                        hidden_states,
                        router_logits,
                        indices_type,
                        input_ids=input_ids,
                    )
                raise
            gate_kwargs = lightop_moe_gate_kwargs(
                lightop_moe,
                scoring_func,
                renormalize,
            )
            if gate_kwargs is None:
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
                # FusedMoE passes 1.0 to the router when MoERunner owns
                # output scaling. Otherwise this is the effective router
                # scale and LightOp must apply it to the routing weights.
                self.routed_scaling_factor != 1.0,
                **gate_kwargs,
            )
            if indices_type is not None and topk_ids.dtype != indices_type:
                topk_ids = topk_ids.to(indices_type)
            return topk_weights, topk_ids

    HcuGroupedTopKRouter.__name__ = "HcuGroupedTopKRouter"
    HcuGroupedTopKRouter.__qualname__ = "HcuGroupedTopKRouter"
    HcuGroupedTopKRouter.__module__ = __name__
    return HcuGroupedTopKRouter


def _ep_world_size() -> int:
    from vllm.distributed.parallel_state import get_ep_group

    return get_ep_group().world_size


def _nearest_max_rank_load(
    num_assignments: int,
    ep_size: int,
    target: float,
) -> int:
    """Choose the integer max load whose mean/max ratio is nearest target."""
    minimum_load = math.ceil(num_assignments / ep_size)
    ideal_load = num_assignments / (ep_size * target)
    candidates = {
        max(minimum_load, min(num_assignments, math.floor(ideal_load))),
        max(minimum_load, min(num_assignments, math.ceil(ideal_load))),
    }
    return min(
        candidates,
        key=lambda load: (
            abs(num_assignments / (ep_size * load) - target),
            load,
        ),
    )


def make_hcu_eplb_balancedness_strategy(base_class):
    class HcuEplbBalancednessRouting(base_class):
        def __init__(
            self,
            ep_size_getter: Callable[[], int] = _ep_world_size,
        ) -> None:
            self._ep_size_getter = ep_size_getter

        def route_tokens(
            self,
            hidden_states,
            router_logits,
            top_k,
            indices_type=None,
        ):
            import torch

            from vllm_hcu.platforms import envs as henvs

            ep_size = int(self._ep_size_getter())
            num_tokens = hidden_states.shape[0]
            num_experts = router_logits.shape[-1]
            target = float(
                henvs.VLLM_HCU_MOE_ROUTING_SIMULATION_BALANCEDNESS
            )

            if ep_size < 1:
                raise ValueError(f"EP size must be positive, got {ep_size}")
            if num_experts < ep_size:
                raise ValueError(
                    f"Routing simulation needs at least one expert per EP rank; "
                    f"got {num_experts} experts for EP size {ep_size}"
                )
            minimum = 1.0 / ep_size
            if not math.isfinite(target) or target < minimum:
                raise ValueError(
                    "VLLM_HCU_MOE_ROUTING_SIMULATION_BALANCEDNESS must be "
                    f"at least the EP-size minimum {minimum:g}, got {target}"
                )
            if target > 1.0:
                raise ValueError(
                    "VLLM_HCU_MOE_ROUTING_SIMULATION_BALANCEDNESS must be "
                    f"at most 1.0, got {target}"
                )
            if top_k < 1:
                raise ValueError(f"top_k must be positive, got {top_k}")
            minimum_experts_per_rank = num_experts // ep_size
            if top_k > minimum_experts_per_rank:
                raise ValueError(
                    "HCU EPLB routing simulation requires top_k to be no "
                    "larger than the smallest EP-rank expert partition so "
                    "each token receives distinct experts; "
                    f"got top_k={top_k}, minimum experts per rank="
                    f"{minimum_experts_per_rank}"
                )

            output_shape = (num_tokens, top_k)
            weights = torch.ones(
                output_shape,
                dtype=torch.float32,
                device=hidden_states.device,
            )
            num_assignments = num_tokens * top_k
            if num_assignments == 0:
                dtype = indices_type if indices_type is not None else torch.long
                return weights, torch.empty(
                    output_shape,
                    dtype=dtype,
                    device=hidden_states.device,
                )

            assignment_ids = torch.arange(
                num_assignments,
                dtype=torch.long,
                device=hidden_states.device,
            )
            if ep_size == 1:
                rank_ids = torch.zeros_like(assignment_ids)
            else:
                hot_count = _nearest_max_rank_load(
                    num_assignments,
                    ep_size,
                    target,
                )
                cool_ids = 1 + (assignment_ids - hot_count).clamp_min(0) % (
                    ep_size - 1
                )
                rank_ids = torch.where(
                    assignment_ids < hot_count,
                    torch.zeros_like(assignment_ids),
                    cool_ids,
                )

            experts_per_rank = num_experts // ep_size
            extra_experts = num_experts % ep_size
            rank_sizes = experts_per_rank + (rank_ids < extra_experts).long()
            rank_starts = (
                rank_ids * experts_per_rank
                + torch.minimum(
                    rank_ids,
                    torch.full_like(rank_ids, extra_experts),
                )
            )
            expert_ids = rank_starts + assignment_ids % rank_sizes
            if indices_type is None:
                indices_type = torch.long
            return weights, expert_ids.reshape(output_shape).to(indices_type)

    HcuEplbBalancednessRouting.__name__ = "HcuEplbBalancednessRouting"
    HcuEplbBalancednessRouting.__qualname__ = "HcuEplbBalancednessRouting"
    HcuEplbBalancednessRouting.__module__ = __name__
    return HcuEplbBalancednessRouting


__all__ = [
    "eplb_map_to_physical_and_record",
    "make_hcu_eplb_balancedness_strategy",
    "make_hcu_grouped_topk_router",
]
