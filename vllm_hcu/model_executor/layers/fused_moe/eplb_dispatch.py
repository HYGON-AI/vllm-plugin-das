# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Rank-local replica ordering for vLLM EPLB routing maps."""

from __future__ import annotations

import random

import torch


def _choose_least_assigned(
    candidates: list[int],
    assignment_counts: dict[int, int],
    rng: random.Random,
) -> int:
    shuffled = candidates.copy()
    rng.shuffle(shuffled)
    return min(shuffled, key=assignment_counts.__getitem__)


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
    """Put a locality-fair primary replica first for one source EP rank.

    vLLM hashes token positions into replica columns. Each EP rank receives a
    different cyclic ordering, so every hash column remains balanced across
    source ranks while column zero preserves GPU/node locality for decode.
    The three-pass assignment follows SGLang-DAS commit 1cc92e0d.
    """

    if not isinstance(logical_to_physical_map, torch.Tensor):
        raise TypeError("logical_to_physical_map must be a torch.Tensor")
    if logical_to_physical_map.ndim != 3:
        raise ValueError("logical_to_physical_map must be three-dimensional")
    if logical_to_physical_map.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("logical_to_physical_map must use an integer dtype")
    if ep_size <= 0:
        raise ValueError(f"ep_size must be positive, got {ep_size}")
    if not 0 <= ep_rank < ep_size:
        raise ValueError(f"ep_rank must be in [0, {ep_size}), got {ep_rank}")
    if num_nodes <= 0 or ep_size % num_nodes != 0:
        raise ValueError(
            "ep_size must be divisible by a positive num_nodes, got "
            f"{ep_size} and {num_nodes}"
        )
    if num_physical_experts <= 0:
        raise ValueError(
            "num_physical_experts must be positive, got "
            f"{num_physical_experts}"
        )
    if num_physical_experts % ep_size != 0:
        raise ValueError(
            "num_physical_experts must be divisible by ep_size, got "
            f"{num_physical_experts} and {ep_size}"
        )
    source_device = logical_to_physical_map.device
    result = logical_to_physical_map.detach().to(device="cpu").clone()
    invalid_ids = (result < -1) | (result >= num_physical_experts)
    if torch.any(invalid_ids):
        raise ValueError(
            "logical_to_physical_map contains an invalid physical expert id"
        )
    replica_counts = (result >= 0).sum(dim=-1)
    empty_rows = torch.nonzero(replica_counts == 0, as_tuple=False)
    if empty_rows.numel() > 0:
        layer_id, logical_expert_id = empty_rows[0].tolist()
        raise ValueError(
            f"logical expert {logical_expert_id} in layer {layer_id} "
            "has no physical replicas"
        )
    num_local_experts = num_physical_experts // ep_size
    num_gpus_per_node = ep_size // num_nodes
    num_node_experts = num_local_experts * num_gpus_per_node

    redundant_rows = torch.nonzero(replica_counts > 1, as_tuple=False).tolist()
    for layer_id, logical_expert_id in redundant_rows:
        rng = random.Random(
            seed
            + (layer_offset + layer_id) * 1_000_003
            + logical_expert_id * 9_176
        )
        row = result[layer_id, logical_expert_id]
        candidates = [int(value) for value in row.tolist() if value >= 0]
        if len(candidates) != len(set(candidates)):
            raise ValueError(
                f"logical expert {logical_expert_id} has duplicate physical replicas"
            )
        assignments = [-1] * ep_size
        assignment_counts = {candidate: 0 for candidate in candidates}

        candidates_by_gpu = [[] for _ in range(ep_size)]
        candidates_by_node = [[] for _ in range(num_nodes)]
        for candidate in candidates:
            candidates_by_gpu[candidate // num_local_experts].append(candidate)
            candidates_by_node[candidate // num_node_experts].append(candidate)

        # Strongest preference: a source rank always starts with a same-GPU
        # replica when one exists.
        for rank, local_candidates in enumerate(candidates_by_gpu):
            if local_candidates:
                choice = _choose_least_assigned(
                    local_candidates,
                    assignment_counts,
                    rng,
                )
                assignments[rank] = choice
                assignment_counts[choice] += 1

        # Then balance remaining source ranks over replicas on their node.
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

        # Only ranks whose node has no replica reach the global fallback.
        remaining_ranks = [
            rank for rank, assignment in enumerate(assignments) if assignment == -1
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

    return result.to(device=source_device)


__all__ = ["build_locality_fair_replica_order"]
