# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Behavioural tests for the SlotMappingMode backport adapter."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import ModuleType

import pytest

from vllm_hcu.patch.worker.framework_opt import patch_slot_mapping_modes as patch


class _FakeBlockTable:
    def __init__(self) -> None:
        self.calls = 0

    def compute_slot_mapping(self, num_reqs, query_start_loc, positions) -> None:
        self.calls += 1


class _FakeMultiGroupBlockTable:
    """Stands in for the paired-vLLM type the adapter patches."""

    def __init__(self, groups: int) -> None:
        self.block_tables = [_FakeBlockTable() for _ in range(groups)]

    def compute_slot_mapping(self, num_reqs, query_start_loc, positions) -> None:
        for block_table in self.block_tables:
            block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)


@pytest.fixture(autouse=True)
def _patched_class():
    """Patch a fresh module, restoring the fake class afterwards."""
    original = _FakeMultiGroupBlockTable.compute_slot_mapping
    saved = patch._ORIGINAL
    module = ModuleType(patch.TARGET_MODULE)
    module.MultiGroupBlockTable = _FakeMultiGroupBlockTable
    assert patch.apply_to_module(module) is True
    yield module
    _FakeMultiGroupBlockTable.compute_slot_mapping = original
    patch._ORIGINAL = saved


def _calls(table: _FakeMultiGroupBlockTable) -> list[int]:
    return [group.calls for group in table.block_tables]


def test_apply_is_idempotent_on_the_same_module(_patched_class):
    assert patch.apply_to_module(_patched_class) is False
    assert getattr(
        _FakeMultiGroupBlockTable.compute_slot_mapping,
        "_vllm_hcu_slot_mapping_wrapper",
    )


def test_applying_twice_does_not_double_wrap():
    module = ModuleType(patch.TARGET_MODULE)
    module.MultiGroupBlockTable = _FakeMultiGroupBlockTable
    patch.apply_to_module(module)
    wrapped_once = _FakeMultiGroupBlockTable.compute_slot_mapping
    # A second module object re-runs apply over the same class; the method must
    # not be wrapped a second time (that would double the skip loop).
    other = ModuleType(patch.TARGET_MODULE)
    other.MultiGroupBlockTable = _FakeMultiGroupBlockTable
    assert patch.apply_to_module(other) is False
    assert _FakeMultiGroupBlockTable.compute_slot_mapping is wrapped_once


def test_without_a_report_every_group_runs():
    table = _FakeMultiGroupBlockTable(5)
    table.compute_slot_mapping(1, None, None)
    assert _calls(table) == [1, 1, 1, 1, 1]


def test_reported_groups_are_skipped_across_steps():
    table = _FakeMultiGroupBlockTable(5)
    patch.set_slot_mapping_groups_without_slots(table, [1, 2, 3, 4])
    table.compute_slot_mapping(1, None, None)
    table.compute_slot_mapping(1, None, None)
    # Only the MLA group (index 0) is addressed per token.
    assert _calls(table) == [2, 0, 0, 0, 0]


def test_empty_report_keeps_stock_behaviour():
    table = _FakeMultiGroupBlockTable(3)
    patch.set_slot_mapping_groups_without_slots(table, [])
    table.compute_slot_mapping(1, None, None)
    assert _calls(table) == [1, 1, 1]


def test_out_of_range_group_index_is_rejected():
    table = _FakeMultiGroupBlockTable(5)
    with pytest.raises(ValueError):
        patch.set_slot_mapping_groups_without_slots(table, [5])
    with pytest.raises(ValueError):
        patch.set_slot_mapping_groups_without_slots(table, [-1])
    table.compute_slot_mapping(1, None, None)
    assert _calls(table) == [1, 1, 1, 1, 1]


def _runner_source() -> str:
    """Read the runner source without importing it (it is worker-process only)."""
    import vllm_hcu.v1 as runner_package

    return (Path(runner_package.__file__).parent / "hcu_model_runner.py").read_text(
        encoding="utf-8"
    )


def test_runner_derives_skipped_groups_from_mamba_specs():
    source = _runner_source()
    assert "get_kv_cache_spec_kind(" in source
    assert "KVCacheSpecKind.MAMBA" in source
    assert "set_slot_mapping_groups_without_slots(" in source


def test_runner_reports_groups_right_after_the_rebuild():
    source = _runner_source()
    assert source.count("set_slot_mapping_groups_without_slots(") == 1
    call_at = source.index("set_slot_mapping_groups_without_slots(")
    # The single call site must follow the InputBatch rebuild it reports for,
    # not the placeholder construction in __init__.
    rebuild_at = source.rindex("self.input_batch = InputBatch(")
    assert call_at > rebuild_at
