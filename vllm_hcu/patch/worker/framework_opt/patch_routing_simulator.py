# SPDX-License-Identifier: Apache-2.0
"""Deterministic rank-load simulation registered with the current router."""
import math
import os

from ._common import load_exact_module, PatchCompatibilityError

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.router.routing_simulator_router"
PATCH_ID = "worker.framework_opt.eplb.routing_simulator"
TARGETS = (f"{TARGET_MODULE}.RoutingSimulator.register_strategy",)
STRATEGY_NAME = "hcu_eplb_balancedness"
_MARKER = "_vllm_hcu_eplb_simulation"


def get_ep_world_size():
    from vllm.distributed import get_ep_group
    return get_ep_group().world_size


def apply_to_module(module):
    target = load_exact_module(TARGET_MODULE, module)
    simulator = target.RoutingSimulator
    if getattr(target, _MARKER, False):
        if STRATEGY_NAME not in simulator.get_available_strategies():
            raise PatchCompatibilityError("EPLB simulator marker is stale")
        return False

    class BalancedRouting(target.RoutingStrategy):
        def route_tokens(self, hidden_states, router_logits, top_k, indices_type=None):
            import torch
            size = get_ep_world_size()
            experts = router_logits.shape[-1]
            balance = float(os.environ.get("VLLM_HCU_MOE_ROUTING_SIMULATION_BALANCEDNESS", "1.0"))
            if size < 1 or not math.isfinite(balance) or not 1.0 / size <= balance <= 1.0:
                raise ValueError("Routing balancedness must be in [1 / EP size, 1]")
            if not 1 <= top_k <= experts // size:
                raise ValueError("Routing simulation needs top_k distinct experts per rank")
            count = hidden_states.shape[0] * top_k
            index = torch.arange(count, device=hidden_states.device)
            hot = round(count / size / balance)
            ranks = torch.zeros_like(index) if size == 1 else torch.where(
                index < hot, 0, 1 + (index - hot).clamp_min(0) % (size - 1))
            quotient, remainder = divmod(experts, size)
            sizes = quotient + (ranks < remainder).long()
            starts = ranks * quotient + ranks.clamp_max(remainder)
            ids = (starts + index % sizes).reshape(-1, top_k)
            weights = torch.ones(ids.shape, device=hidden_states.device, dtype=torch.float32)
            return weights, ids.to(indices_type or torch.long)

    simulator.register_strategy(STRATEGY_NAME, BalancedRouting())
    setattr(target, _MARKER, True)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
