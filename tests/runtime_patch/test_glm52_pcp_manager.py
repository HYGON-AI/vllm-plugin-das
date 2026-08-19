# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU-only contracts for the GLM-5.2 virtual-batch PCP manager."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm_hcu.patch.config import HcuFeatureConfig
from vllm_hcu.patch.platform.core_fix._common import PatchCompatibilityError
from vllm_hcu.v1.pcp_manager import (
    HcuPCPManager,
    RankSegment,
    maybe_build_pcp_manager,
    maybe_partition_pcp_batch,
    maybe_restore_pcp_for_sampling,
)


class _InMemoryPCPGroup:
    """Deterministic all-gather double; no accelerator process group needed."""

    def __init__(self, rank: int, world_size: int) -> None:
        self.rank_in_group = rank
        self.world_size = world_size
        self.gathered: list[torch.Tensor] | None = None
        self.collective_calls = 0

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        self.collective_calls += 1
        assert dim == 0
        assert self.gathered is not None
        assert torch.equal(tensor, self.gathered[self.rank_in_group])
        return torch.cat(self.gathered, dim=dim)


class _InMemoryBlockTables:
    """Block/slot table double that performs real tensor indexing on CPU."""

    def __init__(self) -> None:
        canonical = torch.arange(32 * 4, dtype=torch.int32).reshape(32, 4)
        self.block_tables = [SimpleNamespace(gpu=canonical)]
        self.input_block_tables = [torch.full_like(canonical, -77)]
        self.num_kv_cache_groups = 1
        self.slot_mappings = torch.empty((1, 128), dtype=torch.int64)

    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
        num_reqs: int,
    ) -> tuple[torch.Tensor, ...]:
        raise AssertionError(
            "PCP must not gather into runner-owned v0.25.1 input block tables"
        )

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        for req_idx in range(idx_mapping.numel()):
            start = int(query_start_loc[req_idx])
            stop = int(query_start_loc[req_idx + 1])
            self.slot_mappings[0, start:stop] = (
                idx_mapping[req_idx] * 100 + positions[start:stop]
            )
        return self.slot_mappings[:, :num_tokens]


def _make_config(pcp_size: int = 2, **overrides: object) -> object:
    values = {
        "architecture": "GlmMoeDsaForCausalLM",
        "use_v2": True,
        "use_mla": True,
        "pp": 1,
        "dcp": 1,
        "dp": 1,
        "enable_expert_parallel": True,
        "enforce_eager": True,
    }
    values.update(overrides)
    return SimpleNamespace(
        use_v2_model_runner=values["use_v2"],
        model_config=SimpleNamespace(
            architectures=[values["architecture"]],
            use_mla=values["use_mla"],
            enforce_eager=values["enforce_eager"],
            is_multimodal_model=False,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            prefill_context_parallel_size=pcp_size,
            pipeline_parallel_size=values["pp"],
            decode_context_parallel_size=values["dcp"],
            data_parallel_size=values["dp"],
            enable_expert_parallel=values["enable_expert_parallel"],
            cp_kv_cache_interleave_size=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=16,
            max_num_batched_tokens=128,
        ),
        speculative_config=None,
        lora_config=None,
        cache_config=SimpleNamespace(kv_offloading_size=None),
        kv_transfer_config=None,
        additional_config={"hcu": HcuFeatureConfig().to_dict()},
    )


def _make_batch(
    requests: list[tuple[str, list[int], int, bool]],
    *,
    pad_to: int | None = None,
) -> InputBatch:
    """Build a literal MRV2 step batch on CPU.

    Each request tuple is ``(req_id, scheduled_token_ids, seq_len,
    is_prefilling)``. Token IDs are intentionally globally unique so restored
    order is independently visible in hidden states.
    """

    num_reqs = len(requests)
    num_scheduled_tokens = np.asarray(
        [len(tokens) for _, tokens, _, _ in requests], dtype=np.int32
    )
    num_tokens = int(num_scheduled_tokens.sum())
    num_tokens_after_padding = pad_to or num_tokens
    assert num_tokens_after_padding >= num_tokens

    query_start_loc_np = np.empty(num_reqs + 1, dtype=np.int32)
    query_start_loc_np[0] = 0
    np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])
    query_start_loc = torch.from_numpy(query_start_loc_np.copy())

    input_ids = torch.full((num_tokens_after_padding,), -99, dtype=torch.int32)
    positions = torch.full((num_tokens_after_padding,), -99, dtype=torch.int64)
    is_padding = torch.ones(num_tokens_after_padding, dtype=torch.bool)
    num_computed_tokens_np = np.empty(num_reqs, dtype=np.int32)
    for req_idx, (_, tokens, seq_len, _) in enumerate(requests):
        start = int(query_start_loc_np[req_idx])
        stop = int(query_start_loc_np[req_idx + 1])
        computed = seq_len - len(tokens)
        input_ids[start:stop] = torch.tensor(tokens, dtype=torch.int32)
        positions[start:stop] = torch.arange(computed, seq_len)
        is_padding[start:stop] = False
        num_computed_tokens_np[req_idx] = computed

    idx_mapping_np = np.arange(10, 10 + num_reqs, dtype=np.int32)
    idx_mapping = torch.from_numpy(idx_mapping_np.copy())
    cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
    return InputBatch(
        req_ids=[req_id for req_id, _, _, _ in requests],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        idx_mapping=idx_mapping,
        idx_mapping_np=idx_mapping_np,
        expanded_idx_mapping=idx_mapping.clone(),
        expanded_local_pos=torch.zeros(num_reqs, dtype=torch.int32),
        num_scheduled_tokens=num_scheduled_tokens,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens_after_padding,
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        query_start_loc=query_start_loc,
        query_start_loc_np=query_start_loc_np,
        seq_lens=torch.tensor(
            [seq_len for _, _, seq_len, _ in requests], dtype=torch.int32
        ),
        seq_lens_cpu_upper_bound=torch.tensor(
            [seq_len for _, _, seq_len, _ in requests], dtype=torch.int32
        ),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=num_computed_tokens_np,
        prefill_len_np=np.asarray(
            [
                seq_len if is_prefilling else 0
                for _, _, seq_len, is_prefilling in requests
            ],
            dtype=np.int32,
        ),
        num_computed_prefill_tokens_np=num_computed_tokens_np.copy(),
        is_prefilling_np=np.asarray(
            [is_prefilling for _, _, _, is_prefilling in requests], dtype=np.bool_
        ),
        max_seq_len_np=None,
        input_ids=input_ids,
        positions=positions,
        is_padding=is_padding,
        logits_indices=query_start_loc[1:] - 1,
        cu_num_logits=torch.from_numpy(cu_num_logits_np.copy()),
        cu_num_logits_np=cu_num_logits_np,
        has_structured_output_reqs=False,
        prompt_lens=None,
    )


def _snapshot_batch(batch: InputBatch) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for name, value in vars(batch).items():
        if isinstance(value, torch.Tensor):
            snapshot[name] = value.clone()
        elif isinstance(value, np.ndarray):
            snapshot[name] = value.copy()
        else:
            snapshot[name] = deepcopy(value)
    return snapshot


def _assert_batch_matches_snapshot(
    batch: InputBatch, snapshot: dict[str, object]
) -> None:
    for name, expected in snapshot.items():
        actual = getattr(batch, name)
        if isinstance(expected, torch.Tensor):
            assert torch.equal(actual, expected), name
        elif isinstance(expected, np.ndarray):
            assert np.array_equal(actual, expected), name
        else:
            assert actual == expected, name


def _make_managers(
    pcp_size: int = 2,
    *,
    block_tables: object | None = None,
) -> tuple[list[HcuPCPManager], list[_InMemoryPCPGroup]]:
    groups = [_InMemoryPCPGroup(rank, pcp_size) for rank in range(pcp_size)]
    managers = [
        HcuPCPManager(
            _make_config(pcp_size),
            torch.device("cpu"),
            req_states=SimpleNamespace(),
            block_tables=[] if block_tables is None else block_tables,
            pcp_group=groups[rank],
        )
        for rank in range(pcp_size)
    ]
    return managers, groups


@pytest.mark.parametrize("length", [1, 2, 7, 8, 9, 31])
def test_dual_chunk_swap_covers_each_prefill_token_once(length: int) -> None:
    """A wrong chunk boundary must drop or duplicate a global prompt token."""

    segments = [
        HcuPCPManager.rank_segments(length, pcp_rank=rank, pcp_size=2)
        for rank in range(2)
    ]
    covered = sorted(
        token
        for rank_segments in segments
        for segment in rank_segments
        for token in range(segment.start, segment.stop)
    )
    assert covered == list(range(length))
    assert all(len(rank_segments) == 2 for rank_segments in segments)


def test_dual_chunk_swap_uses_symmetric_segments_and_keeps_empty_rows() -> None:
    """Collapsing either virtual row must break the two-row attention layout."""

    assert HcuPCPManager.rank_segments(8, pcp_rank=0, pcp_size=2) == (
        RankSegment(0, 2),
        RankSegment(6, 8),
    )
    assert HcuPCPManager.rank_segments(8, pcp_rank=1, pcp_size=2) == (
        RankSegment(2, 4),
        RankSegment(4, 6),
    )
    segments = HcuPCPManager.rank_segments(1, pcp_rank=1, pcp_size=2)
    assert len(segments) == 2
    assert any(segment.start == segment.stop for segment in segments)


def test_partition_creates_two_prefill_rows_and_replicates_decode() -> None:
    """Treating decode like prefill must desynchronize decode request rows."""

    global_batch = _make_batch(
        [("prefill", list(range(100, 108)), 8, True), ("decode", [900], 17, False)]
    )
    snapshot = _snapshot_batch(global_batch)
    managers, _ = _make_managers()
    local_batches = [manager.partition_batch(global_batch) for manager in managers]

    for local in local_batches:
        assert local.req_ids.count("prefill") == 2
        assert local.req_ids.count("decode") == 1
        assert local.num_reqs == 3
        assert sorted(local.idx_mapping_np.tolist()) == [10, 10, 11]
        decode_row = local.req_ids.index("decode")
        decode_start = int(local.query_start_loc_np[decode_row])
        decode_stop = int(local.query_start_loc_np[decode_row + 1])
        assert local.input_ids[decode_start:decode_stop].tolist() == [900]

    prefill_tokens = sorted(
        token
        for local in local_batches
        for row, req_id in enumerate(local.req_ids)
        if req_id == "prefill"
        for token in local.input_ids[
            int(local.query_start_loc_np[row]) : int(local.query_start_loc_np[row + 1])
        ].tolist()
    )
    assert prefill_tokens == list(range(100, 108))
    _assert_batch_matches_snapshot(global_batch, snapshot)


def test_partition_pads_each_rank_to_equal_width_and_remaps_logits() -> None:
    """Unequal rank widths would make the hidden-state collective invalid."""

    global_batch = _make_batch(
        [("short", [10], 1, True), ("uneven", list(range(20, 29)), 9, True)],
        pad_to=12,
    )
    managers, _ = _make_managers()
    local_batches = [manager.partition_batch(global_batch) for manager in managers]

    assert len({local.num_tokens_after_padding for local in local_batches}) == 1
    for local in local_batches:
        assert local.num_tokens == int(local.num_scheduled_tokens.sum())
        assert local.input_ids.shape == local.positions.shape == local.is_padding.shape
        assert local.input_ids.numel() == local.num_tokens_after_padding
        nonempty_stops = [
            int(local.query_start_loc_np[row + 1] - 1)
            for row, count in enumerate(local.num_scheduled_tokens)
            if count > 0
        ]
        assert local.logits_indices.tolist() == nonempty_stops
        assert local.expanded_local_pos.dtype == torch.int32
        assert local.expanded_local_pos.tolist() == [0] * len(nonempty_stops)
        assert local.cu_num_logits_np.tolist() == np.cumsum(
            [0]
            + [int(count > 0) for count in local.num_scheduled_tokens]
        ).tolist()


def test_restore_returns_global_token_and_request_order() -> None:
    """Rank-order concatenation without the global mapping must scramble output."""

    global_batch = _make_batch(
        [
            ("a", list(range(100, 107)), 7, True),
            ("decode", [500], 23, False),
            ("b", list(range(200, 209)), 9, True),
        ]
    )
    managers, groups = _make_managers()
    local_batches = [manager.partition_batch(global_batch) for manager in managers]
    local_hidden = [
        local.input_ids.to(torch.float32).unsqueeze(1)
        for local in local_batches
    ]
    for group in groups:
        group.gathered = local_hidden

    expected = global_batch.input_ids[: global_batch.num_tokens].to(torch.float32)
    for manager, local in zip(managers, local_hidden):
        restored = manager.restore_hidden_states(local)
        assert restored[:, 0].tolist() == expected.tolist()
        sampled_hidden, sampled_batch = manager.restore_for_sampling(local)
        assert sampled_hidden[:, 0].tolist() == expected.tolist()
        assert sampled_batch is global_batch


def test_dummy_slots_are_invalid_without_touching_real_block_tables() -> None:
    """A zero dummy slot could accidentally write a real KV cache block."""

    manager, _ = _make_managers(block_tables=_InMemoryBlockTables())
    slots = manager[0].get_dummy_slot_mappings(5)
    assert slots.dtype == torch.int64
    assert slots.device.type == "cpu"
    assert slots.tolist() == [[-1] * 10]


def test_prepare_attn_uses_local_rows_but_global_kv_slot_ownership() -> None:
    """Replicated decode slots must be writable on exactly one PCP rank."""

    block_tables = _InMemoryBlockTables()
    global_batch = _make_batch(
        [("prefill", list(range(100, 108)), 8, True), ("decode", [900], 17, False)]
    )
    managers, _ = _make_managers(block_tables=block_tables)
    local_batches = [manager.partition_batch(global_batch) for manager in managers]
    runner_owned_tables = [table.clone() for table in block_tables.input_block_tables]

    for manager, local in zip(managers, local_batches):
        local_tables, gathered_slots = manager.prepare_attn(local)
        assert torch.equal(
            local_tables[0],
            block_tables.block_tables[0].gpu[local.idx_mapping.to(torch.int64)],
        )
        written_slots = gathered_slots[gathered_slots != -1].tolist()
        assert sorted(written_slots) == list(range(1000, 1008)) + [1116]
    for actual, expected in zip(block_tables.input_block_tables, runner_owned_tables):
        assert torch.equal(actual, expected)


def test_prepare_attn_rejects_missing_canonical_block_storage() -> None:
    """Falling back to gathered scratch rows could consume stale block IDs."""

    block_tables = _InMemoryBlockTables()
    managers, _ = _make_managers(block_tables=block_tables)
    local_batch = managers[0].partition_batch(
        _make_batch([("prefill", list(range(100, 108)), 8, True)])
    )
    del block_tables.block_tables

    with pytest.raises(PatchCompatibilityError, match="canonical block_tables"):
        managers[0].prepare_attn(local_batch)


def test_prepare_attn_rejects_kv_group_count_mismatch() -> None:
    """Silently truncating a KV group would leave its local table unprepared."""

    block_tables = _InMemoryBlockTables()
    block_tables.block_tables.append(
        SimpleNamespace(gpu=block_tables.block_tables[0].gpu.clone())
    )
    managers, _ = _make_managers(block_tables=block_tables)
    local_batch = managers[0].partition_batch(
        _make_batch([("prefill", list(range(100, 108)), 8, True)])
    )

    with pytest.raises(PatchCompatibilityError, match="KV-group count"):
        managers[0].prepare_attn(local_batch)


def test_optional_helpers_are_true_noops_without_a_manager() -> None:
    """PCP=1 must preserve object identity and allocate no manager or collective."""

    batch = _make_batch([("decode", [42], 3, False)])
    hidden = torch.tensor([[42.0]])
    assert maybe_build_pcp_manager(
        _make_config(1), torch.device("cpu"), SimpleNamespace(), []
    ) is None
    assert maybe_partition_pcp_batch(None, batch) is batch
    restored_hidden, restored_batch = maybe_restore_pcp_for_sampling(
        None, hidden, batch
    )
    assert restored_hidden is hidden
    assert restored_batch is batch


def test_build_helper_enforces_the_approved_runtime_contract() -> None:
    """Constructing a manager after bypassing Task 1 scope validation is a bug."""

    with pytest.raises((AssertionError, ValueError), match="pipeline parallel"):
        maybe_build_pcp_manager(
            _make_config(2, pp=2), torch.device("cpu"), SimpleNamespace(), []
        )
