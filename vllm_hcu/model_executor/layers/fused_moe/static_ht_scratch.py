# SPDX-License-Identifier: Apache-2.0
"""Private validation for the opt-in static HT W4A8 scratch contract."""

import torch

ALIGNMENT = 256


def is_linear_map(expert_map, global_experts, local_experts, offset):
    if (
        not isinstance(expert_map, torch.Tensor)
        or expert_map.ndim != 1
        or expert_map.dtype != torch.int32
        or expert_map.numel() != global_experts
        or local_experts <= 0
        or offset < 0
        or offset + local_experts > global_experts
    ):
        return False
    actual = expert_map.detach().cpu()
    expected = torch.full((global_experts,), -1, dtype=actual.dtype)
    expected[offset : offset + local_experts] = torch.arange(
        local_experts, dtype=actual.dtype
    )
    return bool(torch.equal(actual, expected))


def make_plan(counts, local_experts, received):
    if (
        not isinstance(counts, list)
        or len(counts) != local_experts
        or received <= 0
        or any(type(n) is not int or n < 0 for n in counts)
        or not any(counts)
    ):
        return None
    return tuple(((n + 255) // 256) * 256 for n in counts)


def check_recount(plan, actual):
    values = actual.detach().cpu().tolist()  # Explicit diagnostic GPU sync.
    if len(values) != len(plan) or any(
        n < 0 or n > bound for n, bound in zip(values, plan)
    ):
        raise RuntimeError("static HT scratch: post-map expert count exceeds bound")


def map_signature(expert_map):
    try:
        version = expert_map._version
    except RuntimeError:
        version = None
    return (
        expert_map.data_ptr(),
        tuple(expert_map.shape),
        tuple(expert_map.stride()),
        expert_map.dtype,
        expert_map.device,
        version,
    )


def _map_current(plan):
    if plan.expert_map is None:
        return True
    if map_signature(plan.expert_map) != plan.signature:
        return False
    if plan.owner_ref is not None:
        owner = plan.owner_ref()
        manager = getattr(owner, "expert_map_manager", None)
        return getattr(manager, "expert_map", None) is plan.expert_map
    return True


class _Plan:
    def __init__(self, bounds, diagnostic, expert_map, version, owner_ref):
        self.bounds = bounds
        self.diagnostic = diagnostic
        self.expert_map = expert_map
        self.version = version
        self.signature = map_signature(expert_map) if expert_map is not None else None
        self.owner_ref = owner_ref


def attach_plan(
    metadata,
    counts,
    local_experts,
    received,
    diagnostic,
    expert_map=None,
    version=None,
    owner_ref=None,
):
    bounds = make_plan(counts, local_experts, received)
    if bounds is not None:
        metadata._hcu_static_ht_scratch = _Plan(
            bounds, diagnostic, expert_map, version, owner_ref
        )


def get_plan(metadata, local_experts, alignment):
    plan = getattr(metadata, "_hcu_static_ht_scratch", None)
    if (
        isinstance(plan, _Plan)
        and len(plan.bounds) == local_experts
        and alignment == ALIGNMENT
        and _map_current(plan)
    ):
        return plan
    return None


def check_scatter_map(metadata, expert_map):
    plan = getattr(metadata, "_hcu_static_ht_scratch", None)
    if (
        isinstance(plan, _Plan)
        and plan.expert_map is not None
        and (plan.expert_map is not expert_map or not _map_current(plan))
    ):
        raise RuntimeError("static HT scratch: expert map changed before scatter")
