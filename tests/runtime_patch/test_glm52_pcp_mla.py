# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts for PCP MLA and sparse-indexer cache-input gathers."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch


class _FakePCPGroup:
    def __init__(self, expected_calls, *, world_size=2, rank=0):
        self.world_size = world_size
        self.rank_in_group = rank
        self._expected_calls = list(expected_calls)
        self.calls: list[str] = []

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        assert dim == 0
        name, expected_local, gathered = self._expected_calls.pop(0)
        torch.testing.assert_close(tensor, expected_local)
        self.calls.append(name)
        return gathered

    def assert_exhausted(self) -> None:
        assert self._expected_calls == []


def _pcp_module():
    return importlib.import_module(
        "vllm_hcu.model_executor.layers.attention.pcp"
    )


def test_mla_prefill_gathers_uneven_rank_inputs_in_collective_order(
    monkeypatch: pytest.MonkeyPatch,
):
    pcp = _pcp_module()
    local_kv = torch.tensor([[10.0], [11.0], [12.0]])
    local_rope = torch.tensor([[110.0], [111.0], [112.0]])
    local_slots = torch.tensor([100, 101, 102], dtype=torch.int64)
    gathered_kv = torch.tensor([[10.0], [11.0], [12.0], [20.0], [21.0]])
    gathered_rope = torch.tensor(
        [[110.0], [111.0], [112.0], [120.0], [121.0]]
    )
    gathered_slots = torch.tensor([100, 101, 102, 200, 201], dtype=torch.int64)
    group = _FakePCPGroup(
        [
            ("kv", local_kv, gathered_kv),
            ("rope_k", local_rope, gathered_rope),
            ("slot_mapping", local_slots, gathered_slots),
        ]
    )
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: group)
    metadata = SimpleNamespace(
        pcp_world_size=2,
        pcp_token_counts=(3, 2),
        num_decode_tokens=0,
        num_prefills=2,
    )

    actual_kv, actual_rope, actual_slots = (
        pcp.maybe_gather_mla_latent_cache_inputs(
            local_kv,
            local_rope,
            gathered_slots,
            metadata,
        )
    )

    assert actual_kv.shape[0] == sum(metadata.pcp_token_counts)
    assert actual_rope.shape[0] == actual_kv.shape[0]
    assert actual_slots.shape[0] == actual_kv.shape[0]
    torch.testing.assert_close(actual_kv, gathered_kv)
    torch.testing.assert_close(actual_rope, gathered_rope)
    torch.testing.assert_close(actual_slots, gathered_slots)
    assert group.calls == ["kv", "rope_k", "slot_mapping"]
    group.assert_exhausted()


def test_mla_mixed_batch_writes_decode_once_then_rank_ordered_prefills(
    monkeypatch: pytest.MonkeyPatch,
):
    pcp = _pcp_module()
    local_kv = torch.tensor([[9.0], [10.0], [11.0]])
    local_rope = torch.tensor([[109.0], [110.0], [111.0]])
    expanded_slots = torch.tensor(
        [900, 100, 101, -1, 200], dtype=torch.int64
    )
    group = _FakePCPGroup(
        [
            (
                "kv",
                torch.tensor([[10.0], [11.0]]),
                torch.tensor([[10.0], [11.0], [20.0]]),
            ),
            (
                "rope_k",
                torch.tensor([[110.0], [111.0]]),
                torch.tensor([[110.0], [111.0], [120.0]]),
            ),
            (
                "slot_mapping",
                torch.tensor([100, 101], dtype=torch.int64),
                torch.tensor([100, 101, 200], dtype=torch.int64),
            ),
        ]
    )
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: group)
    metadata = SimpleNamespace(
        pcp_world_size=2,
        pcp_token_counts=(3, 2),
        num_decode_tokens=1,
        num_prefills=2,
    )

    actual_kv, actual_rope, actual_slots = (
        pcp.maybe_gather_mla_latent_cache_inputs(
            local_kv,
            local_rope,
            expanded_slots,
            metadata,
        )
    )

    torch.testing.assert_close(
        actual_kv, torch.tensor([[9.0], [10.0], [11.0], [20.0]])
    )
    torch.testing.assert_close(
        actual_rope,
        torch.tensor([[109.0], [110.0], [111.0], [120.0]]),
    )
    torch.testing.assert_close(
        actual_slots, torch.tensor([900, 100, 101, 200], dtype=torch.int64)
    )
    assert group.calls == ["kv", "rope_k", "slot_mapping"]
    group.assert_exhausted()


def test_indexer_prefill_gathers_k_and_matching_slots_in_rank_order(
    monkeypatch: pytest.MonkeyPatch,
):
    pcp = _pcp_module()
    local_k = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expanded_slots = torch.tensor([10, 11, 20], dtype=torch.int64)
    gathered_k = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    )
    group = _FakePCPGroup(
        [
            ("indexer_k", local_k, gathered_k),
            (
                "slot_mapping",
                torch.tensor([10, 11], dtype=torch.int64),
                expanded_slots,
            ),
        ]
    )
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: group)
    metadata = SimpleNamespace(
        pcp_world_size=2,
        pcp_token_counts=(2, 1),
        num_decode_tokens=0,
        num_prefills=2,
    )

    actual_k, actual_slots = pcp.maybe_gather_indexer_k(
        local_k,
        expanded_slots,
        metadata,
    )

    torch.testing.assert_close(actual_k, gathered_k)
    torch.testing.assert_close(actual_slots, expanded_slots)
    assert group.calls == ["indexer_k", "slot_mapping"]
    group.assert_exhausted()


@pytest.mark.parametrize("kind", ["mla", "indexer"])
def test_empty_shard_pure_prefill_still_joins_cache_collectives(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
):
    pcp = _pcp_module()
    expanded_slots = torch.tensor([-1, 200], dtype=torch.int64)
    metadata = SimpleNamespace(
        pcp_world_size=2,
        num_decode_tokens=0,
        # This rank retains only an empty virtual prefill row. The peer owns
        # a real prefill token, and Task 2 pads this rank to the same width.
        num_prefills=0,
    )
    if kind == "mla":
        local_k = torch.tensor([[-99.0]])
        local_rope = torch.tensor([[[-199.0]]])
        gathered_k = torch.tensor([[-99.0], [20.0]])
        gathered_rope = torch.tensor([[[-199.0]], [[120.0]]])
        group = _FakePCPGroup(
            [
                ("kv", local_k, gathered_k),
                ("rope_k", local_rope.reshape(1, -1), gathered_rope.reshape(2, -1)),
                (
                    "slot_mapping",
                    torch.tensor([-1], dtype=torch.int64),
                    expanded_slots,
                ),
            ]
        )
        monkeypatch.setattr(pcp, "get_pcp_group", lambda: group)

        actual_k, actual_rope, actual_slots = (
            pcp.maybe_gather_mla_latent_cache_inputs(
                local_k,
                local_rope,
                expanded_slots,
                metadata,
            )
        )

        torch.testing.assert_close(actual_rope, gathered_rope)
        expected_calls = ["kv", "rope_k", "slot_mapping"]
    else:
        local_k = torch.tensor([[-99.0, -98.0]])
        gathered_k = torch.tensor([[-99.0, -98.0], [20.0, 21.0]])
        group = _FakePCPGroup(
            [
                ("indexer_k", local_k, gathered_k),
                (
                    "slot_mapping",
                    torch.tensor([-1], dtype=torch.int64),
                    expanded_slots,
                ),
            ]
        )
        monkeypatch.setattr(pcp, "get_pcp_group", lambda: group)

        actual_k, actual_slots = pcp.maybe_gather_indexer_k(
            local_k,
            expanded_slots,
            metadata,
        )

        expected_calls = ["indexer_k", "slot_mapping"]

    torch.testing.assert_close(actual_k, gathered_k)
    torch.testing.assert_close(actual_slots, expanded_slots)
    assert group.calls == expected_calls
    group.assert_exhausted()


@pytest.mark.parametrize("kind", ["mla", "indexer"])
def test_empty_shard_mixed_batch_still_joins_prefill_collectives(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
):
    pcp = _pcp_module()
    expanded_slots = torch.tensor([900, -1, -1, 200], dtype=torch.int64)
    gathered_prefill_slots = torch.tensor([-1, 200], dtype=torch.int64)
    metadata = SimpleNamespace(
        pcp_world_size=2,
        num_decode_tokens=1,
        # The decode row is replicated, while only the peer owns a nonempty
        # prefill shard. The second local tensor row is Task 2 padding.
        num_prefills=0,
    )
    if kind == "mla":
        local_k = torch.tensor([[9.0], [-99.0]])
        local_rope = torch.tensor([[[109.0]], [[-199.0]]])
        gathered_prefill_k = torch.tensor([[-99.0], [20.0]])
        gathered_prefill_rope = torch.tensor([[[-199.0]], [[120.0]]])
        group = _FakePCPGroup(
            [
                ("kv", local_k[1:], gathered_prefill_k),
                (
                    "rope_k",
                    local_rope[1:].reshape(1, -1),
                    gathered_prefill_rope.reshape(2, -1),
                ),
                (
                    "slot_mapping",
                    torch.tensor([-1], dtype=torch.int64),
                    gathered_prefill_slots,
                ),
            ]
        )
        monkeypatch.setattr(pcp, "get_pcp_group", lambda: group)

        actual_k, actual_rope, actual_slots = (
            pcp.maybe_gather_mla_latent_cache_inputs(
                local_k,
                local_rope,
                expanded_slots,
                metadata,
            )
        )

        torch.testing.assert_close(
            actual_rope,
            torch.tensor([[[109.0]], [[-199.0]], [[120.0]]]),
        )
        expected_calls = ["kv", "rope_k", "slot_mapping"]
        expected_k = torch.tensor([[9.0], [-99.0], [20.0]])
    else:
        local_k = torch.tensor([[9.0, 10.0], [-99.0, -98.0]])
        gathered_prefill_k = torch.tensor(
            [[-99.0, -98.0], [20.0, 21.0]]
        )
        group = _FakePCPGroup(
            [
                ("indexer_k", local_k[1:], gathered_prefill_k),
                (
                    "slot_mapping",
                    torch.tensor([-1], dtype=torch.int64),
                    gathered_prefill_slots,
                ),
            ]
        )
        monkeypatch.setattr(pcp, "get_pcp_group", lambda: group)

        actual_k, actual_slots = pcp.maybe_gather_indexer_k(
            local_k,
            expanded_slots,
            metadata,
        )

        expected_calls = ["indexer_k", "slot_mapping"]
        expected_k = torch.tensor(
            [[9.0, 10.0], [-99.0, -98.0], [20.0, 21.0]]
        )

    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(
        actual_slots,
        torch.tensor([900, -1, 200], dtype=torch.int64),
    )
    assert group.calls == expected_calls
    group.assert_exhausted()


@pytest.mark.parametrize(
    ("pcp_world_size", "num_prefills", "num_decode_tokens"),
    [(1, 1, 0), (2, 0, 3)],
)
def test_pcp_one_and_decode_only_avoid_collectives_with_aligned_slots(
    monkeypatch: pytest.MonkeyPatch,
    pcp_world_size: int,
    num_prefills: int,
    num_decode_tokens: int,
):
    pcp = _pcp_module()
    monkeypatch.setattr(
        pcp,
        "get_pcp_group",
        lambda: pytest.fail("identity branch entered a PCP collective"),
    )
    kv = torch.randn(3, 2)
    rope = torch.randn(3, 1, 2)
    indexer_k = torch.randn(3, 4)
    slots = (
        torch.arange(3, dtype=torch.int64)
        if pcp_world_size == 1
        else torch.tensor([10, 11, 12, -1, -1, -1], dtype=torch.int64)
    )
    metadata = SimpleNamespace(
        pcp_world_size=pcp_world_size,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
    )

    actual_kv, actual_rope, actual_slots = (
        pcp.maybe_gather_mla_latent_cache_inputs(
            kv,
            rope,
            slots,
            metadata,
        )
    )
    actual_indexer_k, actual_indexer_slots = pcp.maybe_gather_indexer_k(
        indexer_k,
        slots,
        metadata,
    )

    assert actual_kv is kv
    assert actual_rope is rope
    assert actual_indexer_k is indexer_k
    if pcp_world_size == 1:
        assert actual_slots is slots
        assert actual_indexer_slots is slots
    else:
        expected_slots = slots[:num_decode_tokens]
        assert actual_slots is not slots
        assert actual_indexer_slots is not slots
        torch.testing.assert_close(actual_slots, expected_slots)
        torch.testing.assert_close(actual_indexer_slots, expected_slots)
        assert actual_slots.shape[0] == actual_kv.shape[0]
        assert actual_indexer_slots.shape[0] == actual_indexer_k.shape[0]


def test_mla_gather_rejects_mismatched_latent_and_rope_token_shapes(
    monkeypatch: pytest.MonkeyPatch,
):
    pcp = _pcp_module()
    monkeypatch.setattr(
        pcp,
        "get_pcp_group",
        lambda: pytest.fail("shape validation must precede collectives"),
    )
    metadata = SimpleNamespace(
        pcp_world_size=2,
        pcp_token_counts=(3, 2),
        num_decode_tokens=0,
        num_prefills=2,
    )

    with pytest.raises(AssertionError, match="same token dimension"):
        pcp.maybe_gather_mla_latent_cache_inputs(
            torch.ones(3, 2),
            torch.ones(2, 1, 2),
            torch.arange(5),
            metadata,
        )
