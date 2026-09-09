# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from collections import Counter

import pytest
import torch


def test_locality_fair_removes_same_node_first_replica_bias() -> None:
    from vllm_hcu.model_executor.layers.fused_moe.eplb_dispatch import (
        build_locality_fair_replica_order,
    )

    logical_to_physical_map = torch.tensor([[[0, 1]]], dtype=torch.int64)

    rank_maps = [
        build_locality_fair_replica_order(
            logical_to_physical_map,
            ep_rank=ep_rank,
            ep_size=8,
            num_nodes=1,
            num_physical_experts=8,
            seed=42,
        )
        for ep_rank in range(8)
    ]

    assert rank_maps[0].tolist() == [[[0, 1]]]
    assert rank_maps[1].tolist() == [[[1, 0]]]
    assert Counter(rank_map[0, 0, 0].item() for rank_map in rank_maps) == {
        0: 4,
        1: 4,
    }
    assert all(sorted(rank_map[0, 0].tolist()) == [0, 1] for rank_map in rank_maps)


def test_locality_fair_rejects_non_divisible_expert_topology() -> None:
    from vllm_hcu.model_executor.layers.fused_moe.eplb_dispatch import (
        build_locality_fair_replica_order,
    )

    with pytest.raises(ValueError, match="num_physical_experts.*ep_size"):
        build_locality_fair_replica_order(
            torch.tensor([[[0, 1]]], dtype=torch.int64),
            ep_rank=0,
            ep_size=8,
            num_nodes=1,
            num_physical_experts=10,
        )


@pytest.mark.parametrize(
    ("mapping", "kwargs", "message"),
    [
        (
            torch.tensor([[0, 1]], dtype=torch.int64),
            {"ep_rank": 0, "ep_size": 2, "num_nodes": 1, "num_physical_experts": 2},
            "three-dimensional",
        ),
        (
            torch.tensor([[[0, 1]]], dtype=torch.float32),
            {"ep_rank": 0, "ep_size": 2, "num_nodes": 1, "num_physical_experts": 2},
            "integer dtype",
        ),
        (
            torch.tensor([[[0, 1]]], dtype=torch.int64),
            {"ep_rank": 2, "ep_size": 2, "num_nodes": 1, "num_physical_experts": 2},
            "ep_rank",
        ),
        (
            torch.tensor([[[0, 1]]], dtype=torch.int64),
            {"ep_rank": 0, "ep_size": 2, "num_nodes": 3, "num_physical_experts": 2},
            "ep_size.*num_nodes",
        ),
        (
            torch.tensor([[[0, 2]]], dtype=torch.int64),
            {"ep_rank": 0, "ep_size": 2, "num_nodes": 1, "num_physical_experts": 2},
            "physical expert id",
        ),
        (
            torch.tensor([[[-1, -1]]], dtype=torch.int64),
            {"ep_rank": 0, "ep_size": 2, "num_nodes": 1, "num_physical_experts": 2},
            "no physical replicas",
        ),
        (
            torch.tensor([[[0, 0]]], dtype=torch.int64),
            {"ep_rank": 0, "ep_size": 2, "num_nodes": 1, "num_physical_experts": 2},
            "duplicate physical replicas",
        ),
    ],
)
def test_locality_fair_rejects_invalid_mapping_contract(
    mapping: torch.Tensor,
    kwargs: dict[str, int],
    message: str,
) -> None:
    from vllm_hcu.model_executor.layers.fused_moe.eplb_dispatch import (
        build_locality_fair_replica_order,
    )

    with pytest.raises((TypeError, ValueError), match=message):
        build_locality_fair_replica_order(mapping, **kwargs)


def test_locality_fair_layer_commit_matches_full_commit_order() -> None:
    from vllm_hcu.model_executor.layers.fused_moe.eplb_dispatch import (
        build_locality_fair_replica_order,
    )

    mapping = torch.tensor(
        [
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            [[0, 2, 4, 6], [1, 3, 5, 7]],
        ],
        dtype=torch.int64,
    )
    kwargs = {
        "ep_rank": 5,
        "ep_size": 8,
        "num_nodes": 2,
        "num_physical_experts": 8,
        "seed": 42,
    }

    full = build_locality_fair_replica_order(mapping, **kwargs)
    second_layer = build_locality_fair_replica_order(
        mapping[1:2],
        layer_offset=1,
        **kwargs,
    )

    torch.testing.assert_close(second_layer, full[1:2], rtol=0, atol=0)


def test_locality_fair_preserves_node_locality_before_global_fairness() -> None:
    from vllm_hcu.model_executor.layers.fused_moe.eplb_dispatch import (
        build_locality_fair_replica_order,
    )

    mapping = torch.tensor([[[0, 1, 4]]], dtype=torch.int64)
    primaries = []
    for ep_rank in range(8):
        rank_map = build_locality_fair_replica_order(
            mapping,
            ep_rank=ep_rank,
            ep_size=8,
            num_nodes=2,
            num_physical_experts=8,
        )
        primary = rank_map[0, 0, 0].item()
        primaries.append(primary)
        assert primary // 4 == ep_rank // 4

    assert Counter(primaries) == {0: 2, 1: 2, 4: 4}
