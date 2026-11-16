# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Rank-local replica ordering for vLLM EPLB routing maps."""

from __future__ import annotations

import random

import torch


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _strict_integer(value: object, name: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _choose_least_assigned(
    candidates: list[int],
    assignment_counts: dict[int, int],
    rng: random.Random,
) -> int:
    shuffled = candidates.copy()
    rng.shuffle(shuffled)
    return min(shuffled, key=assignment_counts.__getitem__)


def _validated_replica_map(
    logical_to_physical_map: torch.Tensor,
    *,
    ep_rank: int,
    ep_size: int,
    num_nodes: int,
    num_physical_experts: int,
) -> torch.Tensor:
    if not isinstance(logical_to_physical_map, torch.Tensor):
        raise TypeError("logical_to_physical_map must be a torch.Tensor")
    if logical_to_physical_map.ndim != 3:
        raise ValueError("logical_to_physical_map must be three-dimensional")
    if logical_to_physical_map.dtype not in _INTEGER_DTYPES:
        raise TypeError("logical_to_physical_map must use an integer dtype")

    _strict_integer(ep_size, "ep_size", minimum=1)
    _strict_integer(ep_rank, "ep_rank", minimum=0)
    if ep_rank >= ep_size:
        raise ValueError(f"ep_rank must be in [0, {ep_size}), got {ep_rank}")
    _strict_integer(num_nodes, "num_nodes", minimum=1)
    if ep_size % num_nodes != 0:
        raise ValueError(
            "ep_size must be divisible by num_nodes, got "
            f"{ep_size} and {num_nodes}"
        )
    _strict_integer(
        num_physical_experts,
        "num_physical_experts",
        minimum=1,
    )
    if num_physical_experts % ep_size != 0:
        raise ValueError(
            "num_physical_experts must be divisible by ep_size, got "
            f"{num_physical_experts} and {ep_size}"
        )

    result = logical_to_physical_map.detach().to(device="cpu").clone()
    invalid_ids = (result < -1) | (result >= num_physical_experts)
    if torch.any(invalid_ids):
        raise ValueError(
            "logical_to_physical_map contains an invalid physical expert ID"
        )

    for layer_id, layer in enumerate(result):
        for logical_expert_id, row in enumerate(layer):
            candidates = [
                int(value) for value in row.tolist() if int(value) >= 0
            ]
            if not candidates:
                raise ValueError(
                    f"logical expert {logical_expert_id} in layer {layer_id} "
                    "has no physical replicas"
                )
            if len(candidates) != len(set(candidates)):
                raise ValueError(
                    f"logical expert {logical_expert_id} in layer {layer_id} "
                    "has duplicate physical replicas"
                )
    return result


def build_nearest_replica_order(
    logical_to_physical_map: torch.Tensor,
    *,
    ep_rank: int,
    ep_size: int,
    num_nodes: int,
    num_physical_experts: int,
) -> torch.Tensor:
    """Order replicas by same GPU, same node, then remote physical ID."""

    result = _validated_replica_map(
        logical_to_physical_map,
        ep_rank=ep_rank,
        ep_size=ep_size,
        num_nodes=num_nodes,
        num_physical_experts=num_physical_experts,
    )
    num_local_experts = num_physical_experts // ep_size
    ranks_per_node = ep_size // num_nodes

    def locality(physical_id: int) -> tuple[int, int]:
        rank = physical_id // num_local_experts
        if rank == ep_rank:
            distance = 0
        elif rank // ranks_per_node == ep_rank // ranks_per_node:
            distance = 1
        else:
            distance = 2
        return distance, physical_id

    for layer in result:
        for row in layer:
            candidates = [
                int(value) for value in row.tolist() if int(value) >= 0
            ]
            ordered = sorted(candidates, key=locality)
            row.fill_(-1)
            row[: len(ordered)] = torch.tensor(ordered, dtype=row.dtype)
    return result.to(device=logical_to_physical_map.device)


def build_locality_fair_replica_order(
    logical_to_physical_map: torch.Tensor,
    *,
    ep_rank: int,
    ep_size: int,
    num_nodes: int,
    num_physical_experts: int,
    seed: int = 42,
    layer_offset: int = 0,
) -> torch.Tensor:
    """Choose a deterministic, locality-aware primary for one source rank.

    Every source rank prefers a same-GPU replica, then a same-node replica,
    and finally a remote replica. Choices within each locality tier are spread
    evenly across candidates. Remaining columns are a cyclic permutation of
    the original replica list, preserving each logical expert's replica set.
    """

    if type(seed) is not int:
        raise ValueError(f"seed must be an integer, got {seed!r}")
    _strict_integer(layer_offset, "layer_offset", minimum=0)
    result = _validated_replica_map(
        logical_to_physical_map,
        ep_rank=ep_rank,
        ep_size=ep_size,
        num_nodes=num_nodes,
        num_physical_experts=num_physical_experts,
    )
    num_local_experts = num_physical_experts // ep_size
    num_gpus_per_node = ep_size // num_nodes
    num_node_experts = num_local_experts * num_gpus_per_node

    replica_counts = (result >= 0).sum(dim=-1)
    redundant_rows = torch.nonzero(
        replica_counts > 1,
        as_tuple=False,
    ).tolist()
    for layer_id, logical_expert_id in redundant_rows:
        rng = random.Random(
            seed
            + (layer_offset + layer_id) * 1_000_003
            + logical_expert_id * 9_176
        )
        row = result[layer_id, logical_expert_id]
        candidates = [
            int(value) for value in row.tolist() if int(value) >= 0
        ]
        assignments = [-1] * ep_size
        assignment_counts = {candidate: 0 for candidate in candidates}

        candidates_by_gpu = [[] for _ in range(ep_size)]
        candidates_by_node = [[] for _ in range(num_nodes)]
        for candidate in candidates:
            candidates_by_gpu[candidate // num_local_experts].append(candidate)
            candidates_by_node[candidate // num_node_experts].append(candidate)

        for rank, local_candidates in enumerate(candidates_by_gpu):
            if local_candidates:
                choice = _choose_least_assigned(
                    local_candidates,
                    assignment_counts,
                    rng,
                )
                assignments[rank] = choice
                assignment_counts[choice] += 1

        for node_id, node_candidates in enumerate(candidates_by_node):
            if not node_candidates:
                continue
            first_rank = node_id * num_gpus_per_node
            remaining_ranks = [
                rank
                for rank in range(first_rank, first_rank + num_gpus_per_node)
                if assignments[rank] == -1
            ]
            rng.shuffle(remaining_ranks)
            for rank in remaining_ranks:
                choice = _choose_least_assigned(
                    node_candidates,
                    assignment_counts,
                    rng,
                )
                assignments[rank] = choice
                assignment_counts[choice] += 1

        remaining_ranks = [
            rank
            for rank, assignment in enumerate(assignments)
            if assignment == -1
        ]
        rng.shuffle(remaining_ranks)
        for rank in remaining_ranks:
            choice = _choose_least_assigned(
                candidates,
                assignment_counts,
                rng,
            )
            assignments[rank] = choice
            assignment_counts[choice] += 1

        primary_index = candidates.index(assignments[ep_rank])
        ordered = candidates[primary_index:] + candidates[:primary_index]
        row.fill_(-1)
        row[: len(ordered)] = torch.tensor(ordered, dtype=row.dtype)

    return result.to(device=logical_to_physical_map.device)


__all__ = [
    "build_locality_fair_replica_order",
    "build_nearest_replica_order",
]
