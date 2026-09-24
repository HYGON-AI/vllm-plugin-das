# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contract for the PCP row layouts DeepSeek-V4.1 cache writes depend on.

The V4.1 cache writers only work if three things agree: the manager's expanded
row order, the batch's global token order, and the slot mapping that names each
row's cache location.  This test drives the manager's real
``_build_batch_layout`` for small batches and then proves, against the helpers
the adapter calls, that

* gathering rank-local rows restores the global token order exactly,
* a rank-local slot mapping gathered the same way names each global token's own
  slot, so every rank writes the same cache locations, and
* padding rows never carry a real token's data.

It runs on CPU without a process group, so it guards the layout contract
independently of the HCU runtime.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_hcu.v1.pcp_manager import HcuPCPManager


class _FakeGroup:
    """A PCP group whose collectives run locally over every rank's rows."""

    def __init__(self, rank: int, payloads: list[torch.Tensor]) -> None:
        self.rank_in_group = rank
        self.world_size = len(payloads)
        self._payloads = payloads

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        assert dim == 0, "the PCP layout only gathers rows"
        return torch.cat(self._payloads, dim=0)


def _manager(pcp_size: int, use_mla: bool = True) -> HcuPCPManager:
    manager = object.__new__(HcuPCPManager)
    manager.pcp_size = pcp_size
    manager.pcp_rank = 0
    manager._use_mla = use_mla
    manager.device = torch.device("cpu")
    return manager


def _input_batch(
    query_lens: list[int],
    is_prefilling: list[bool],
) -> SimpleNamespace:
    query_start = np.concatenate(([0], np.cumsum(query_lens))).astype(np.int32)
    return SimpleNamespace(
        num_scheduled_tokens=np.asarray(query_lens, dtype=np.int32),
        is_prefilling_np=np.asarray(is_prefilling, dtype=np.bool_),
        query_start_loc_np=query_start,
        num_computed_tokens_np=np.zeros(len(query_lens), dtype=np.int32),
        num_tokens=int(query_start[-1]),
    )


def _local_rows(
    segments_by_rank: list[list],
    global_values: torch.Tensor,
    *,
    padded: int,
    fill: float,
) -> list[torch.Tensor]:
    """Place each rank's owned global rows into its padded local row space."""

    per_rank: list[torch.Tensor] = []
    for segments in segments_by_rank:
        rows = torch.full((padded,), fill, dtype=global_values.dtype)
        for segment in segments:
            if segment.num_actual_tokens == 0:
                continue
            rows[segment.local_slice] = global_values[segment.global_slice]
        per_rank.append(rows)
    return per_rank


def _case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    query_lens: list[int],
    is_prefilling: list[bool],
    pcp_size: int,
) -> None:
    from vllm_hcu.model_executor.layers.attention import pcp

    manager = _manager(pcp_size)
    input_batch = _input_batch(query_lens, is_prefilling)
    segments_by_rank, _ = manager._build_batch_layout(input_batch)
    padded = manager._padded_num_tokens
    global_num_tokens = input_batch.num_tokens
    expanded_width = padded * pcp_size

    assert manager._padded_gather_idx is not None
    assert manager._hidden_restore_idx is not None
    assert manager._padded_gather_idx.numel() == expanded_width
    assert manager._hidden_restore_idx.numel() == global_num_tokens

    global_values = torch.arange(global_num_tokens, dtype=torch.float32) + 1.0
    global_slots = torch.arange(global_num_tokens, dtype=torch.int64) * 10 + 7

    # Real rows only: the local row space is [0, local_real) with padding after.
    local_values = _local_rows(
        segments_by_rank, global_values, padded=padded, fill=0.0
    )
    local_slots: list[torch.Tensor] = []
    for rank, segments in enumerate(segments_by_rank):
        real = sum(segment.num_actual_tokens for segment in segments)
        slots = torch.full((real,), -1, dtype=torch.int64)
        for segment in segments:
            if segment.num_actual_tokens == 0:
                continue
            slots[segment.local_slice] = global_slots[segment.global_slice]
        local_slots.append(slots)
        # Padding sits after the real rows and never holds a token.
        assert torch.all(local_values[rank][real:] == 0)

    class Layout:
        pcp_world_size = pcp_size
        pcp_has_global_prefill = True
        pcp_local_num_tokens = padded
        pcp_global_num_tokens = global_num_tokens
        pcp_restore_idx = manager._hidden_restore_idx
        pcp_padded_gather_idx = manager._padded_gather_idx

    # Every rank's local rows restore the same global token order.
    for rank in range(pcp_size):
        monkeypatch.setattr(
            pcp, "get_pcp_group", lambda r=rank: _FakeGroup(r, local_values)
        )
        restored = pcp.restore_pcp_rows_to_global(local_values[rank], Layout)
        torch.testing.assert_close(restored, global_values)

    # A rank-local slot mapping gathered the same way names each global token's
    # own slot, so the complete cache write is identical on every rank.
    padded_slots = [
        torch.cat(
            (
                slots,
                torch.full((padded - slots.numel(),), -1, dtype=torch.int64),
            )
        )
        for slots in local_slots
    ]
    for rank, slots in enumerate(local_slots):
        monkeypatch.setattr(
            pcp, "get_pcp_group", lambda r=rank: _FakeGroup(r, padded_slots)
        )
        gathered = pcp.globalize_pcp_slot_mapping(slots, Layout)
        torch.testing.assert_close(gathered, global_slots)

    # An expanded mapping is only reordered, not re-gathered.
    monkeypatch.setattr(
        pcp, "get_pcp_group", lambda: _FakeGroup(0, padded_slots)
    )
    expanded_slots = torch.full((expanded_width,), -1, dtype=torch.int64)
    for rank, segments in enumerate(segments_by_rank):
        rank_offset = rank * padded
        for segment in segments:
            if segment.num_actual_tokens == 0:
                continue
            start = rank_offset + segment.local_slice.start
            stop = rank_offset + segment.local_slice.stop
            expanded_slots[start:stop] = global_slots[segment.global_slice]
    torch.testing.assert_close(
        pcp.globalize_pcp_slot_mapping(expanded_slots, Layout), global_slots
    )


def test_pcp_rows_restore_for_a_single_prefill(monkeypatch: pytest.MonkeyPatch) -> None:
    """One prefill request splits across ranks and restores exactly."""

    _case(monkeypatch, query_lens=[7], is_prefilling=[True], pcp_size=2)


def test_pcp_rows_restore_for_uneven_multi_request_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uneven lengths and a decode row sharing one batch still round-trip."""

    _case(
        monkeypatch,
        query_lens=[5, 13, 1],
        is_prefilling=[True, True, False],
        pcp_size=2,
    )


def test_pcp_rows_restore_with_four_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wider PCP group still round-trips every global row."""

    _case(monkeypatch, query_lens=[9, 4], is_prefilling=[True, True], pcp_size=4)


def test_pcp_rows_restore_handles_an_empty_shard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request shorter than the rank count leaves ranks with no real rows."""

    _case(monkeypatch, query_lens=[1, 6], is_prefilling=[True, True], pcp_size=4)
