# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe.eplb_dispatch import (
    build_locality_fair_replica_order,
    build_nearest_replica_order,
)


def _topology(**updates):
    values = {
        "ep_rank": 0,
        "ep_size": 4,
        "num_nodes": 2,
        "num_physical_experts": 8,
    }
    values.update(updates)
    return values


def test_nearest_orders_same_gpu_then_same_node_then_remote() -> None:
    mapping = torch.tensor(
        [[[0, 4, 7, -1], [0, 4, 5, -1], [0, 1, -1, -1]]]
    )
    original = mapping.clone()

    result = build_nearest_replica_order(
        mapping,
        ep_rank=3,
        ep_size=4,
        num_nodes=2,
        num_physical_experts=8,
    )

    assert result.tolist() == [
        [[7, 4, 0, -1], [4, 5, 0, -1], [0, 1, -1, -1]]
    ]
    assert torch.equal(mapping, original)
    assert torch.equal(
        result,
        build_nearest_replica_order(
            result,
            ep_rank=3,
            ep_size=4,
            num_nodes=2,
            num_physical_experts=8,
        ),
    )


def test_locality_fair_is_deterministic_balanced_and_rank_local() -> None:
    mapping = torch.tensor([[[0, 4, -1], [1, 2, -1], [3, 5, -1]]])
    original = mapping.clone()
    outputs = [
        build_locality_fair_replica_order(
            mapping,
            ep_rank=rank,
            ep_size=8,
            num_nodes=2,
            num_physical_experts=8,
        )
        for rank in range(8)
    ]

    assert [output[0, 0, 0].item() for output in outputs] == [
        0,
        0,
        0,
        0,
        4,
        4,
        4,
        4,
    ]
    assert outputs[1][0, 1, 0] == 1
    assert outputs[2][0, 1, 0] == 2
    assert torch.equal(mapping, original)
    assert not torch.equal(outputs[1], outputs[2])

    for rank, result in enumerate(outputs):
        repeated = build_locality_fair_replica_order(
            mapping,
            ep_rank=rank,
            ep_size=8,
            num_nodes=2,
            num_physical_experts=8,
        )
        assert torch.equal(result, repeated)
        assert torch.equal(result.sort(-1).values, mapping.sort(-1).values)

    primaries = torch.tensor([output[0, 1, 0] for output in outputs])
    counts = torch.bincount(primaries, minlength=8)
    assert counts[1] == counts[2] == 4


def test_layer_offset_preserves_chunked_locality_fair_order() -> None:
    mapping = torch.tensor([[[0, 2, 4, 6]], [[1, 3, 5, 7]]])
    topology = {
        "ep_rank": 7,
        "ep_size": 8,
        "num_nodes": 2,
        "num_physical_experts": 8,
    }

    assert torch.equal(
        build_locality_fair_replica_order(mapping, **topology)[1:],
        build_locality_fair_replica_order(
            mapping[1:], layer_offset=1, **topology
        ),
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"ep_rank": -1},
        {"ep_rank": 4},
        {"ep_rank": True},
        {"ep_size": 0},
        {"ep_size": True},
        {"num_nodes": 0},
        {"num_nodes": 3},
        {"num_nodes": True},
        {"num_physical_experts": 7},
        {"num_physical_experts": True},
    ],
)
@pytest.mark.parametrize(
    "builder", [build_nearest_replica_order, build_locality_fair_replica_order]
)
def test_invalid_topology_is_rejected(builder, updates) -> None:
    with pytest.raises(ValueError):
        builder(torch.tensor([[[0, 1]]]), **_topology(**updates))


@pytest.mark.parametrize(
    "mapping",
    [
        torch.tensor([[0, 1]]),
        torch.tensor([[[[0, 1]]]]),
        torch.tensor([[[0.0, 1.0]]]),
        torch.tensor([[[0, 0]]]),
        torch.tensor([[[-1, -1]]]),
        torch.tensor([[[0, 8]]]),
        torch.tensor([[[-2, 0]]]),
    ],
)
@pytest.mark.parametrize(
    "builder", [build_nearest_replica_order, build_locality_fair_replica_order]
)
def test_invalid_replica_maps_are_rejected(builder, mapping) -> None:
    expected_error = TypeError if mapping.dtype.is_floating_point else ValueError
    with pytest.raises(expected_error):
        builder(mapping, **_topology())


def test_seed_and_layer_offset_are_deterministic_inputs() -> None:
    mapping = torch.tensor([[[0, 2, 4, 6]]])
    topology = {
        "ep_rank": 7,
        "ep_size": 8,
        "num_nodes": 2,
        "num_physical_experts": 8,
    }

    first = build_locality_fair_replica_order(
        mapping,
        seed=7,
        layer_offset=3,
        **topology,
    )
    repeated = build_locality_fair_replica_order(
        mapping,
        seed=7,
        layer_offset=3,
        **topology,
    )
    assert torch.equal(first, repeated)

    for invalid in (True, 1.5):
        with pytest.raises(ValueError):
            build_locality_fair_replica_order(
                mapping,
                seed=invalid,
                **topology,
            )
        with pytest.raises(ValueError):
            build_locality_fair_replica_order(
                mapping,
                layer_offset=invalid,
                **topology,
            )
