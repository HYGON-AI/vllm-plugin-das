# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

import pytest

from vllm_hcu.patch.runtime_state import (
    LatchedPatchError,
    PatchRegistry,
    PatchStatus,
    ProcessRole,
    run_patch,
)


def test_report_contains_process_identity_targets_and_feature_state() -> None:
    registry = PatchRegistry()
    registry.set_process_role(ProcessRole.WORKER)
    registry.declare("worker.attention", ("vllm.attention.Backend", "vllm.attention.impl"))
    assert registry.begin("worker.attention")
    registry.mark_applied("worker.attention", feature_enabled=True)

    report = registry.report()
    assert isinstance(report["pid"], int)
    assert report["process_role"] == "Worker"
    patch = report["patches"]["worker.attention"]
    assert patch["pid"] == report["pid"]
    assert patch["process_role"] == "Worker"
    assert patch["status"] == "applied"
    assert patch["targets"] == ["vllm.attention.Backend", "vllm.attention.impl"]
    assert patch["failure_reason"] is None
    assert patch["feature_enabled"] is True


def test_run_patch_is_idempotent_and_failure_is_latched() -> None:
    registry = PatchRegistry()
    calls = 0

    def fail() -> None:
        nonlocal calls
        calls += 1
        raise ValueError("bad target")

    with pytest.raises(ValueError, match="bad target"):
        run_patch("broken", "vllm.target", fail, registry=registry)
    with pytest.raises(LatchedPatchError, match="previously failed"):
        run_patch("broken", "vllm.target", fail, registry=registry)
    assert calls == 1
    record = registry.get("broken")
    assert record is not None
    assert record.status is PatchStatus.FAILED
    assert record.failure_reason == "ValueError: bad target"
    assert record.feature_enabled is False


def test_concurrent_callers_execute_patch_once() -> None:
    registry = PatchRegistry()
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    results: list[object] = []

    def callback() -> str:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=5)
        return "owner-result"

    def invoke() -> None:
        results.append(run_patch("once", "vllm.once", callback, registry=registry))

    owner = threading.Thread(target=invoke)
    waiter = threading.Thread(target=invoke)
    owner.start()
    assert entered.wait(timeout=5)
    waiter.start()
    release.set()
    owner.join(timeout=5)
    waiter.join(timeout=5)

    assert not owner.is_alive() and not waiter.is_alive()
    assert calls == 1
    assert sorted(results, key=lambda item: item is None) == ["owner-result", None]


def test_reset_for_tests_clears_latched_state_and_role() -> None:
    registry = PatchRegistry()
    registry.set_process_role("EngineCore")
    registry.declare("one", "vllm.one")
    registry.mark_skipped("one", "feature disabled")
    registry.reset_for_tests()

    assert registry.snapshot() == ()
    assert registry.report()["process_role"] == "Main"


def test_armed_and_capability_skipped_records_preserve_requested_feature() -> None:
    registry = PatchRegistry()
    registry.declare("optional", "vllm.optional")
    registry.set_feature_enabled("optional", True)
    assert registry.get("optional").feature_enabled is True
    registry.mark_skipped("optional", "RCCL symbol unavailable")
    record = registry.get("optional")
    assert record.status is PatchStatus.SKIPPED
    assert record.feature_enabled is True
    assert record.failure_reason == "RCCL symbol unavailable"


def test_declaration_rejects_id_collision() -> None:
    registry = PatchRegistry()
    registry.declare("same-id", "vllm.first")
    with pytest.raises(ValueError, match="already bound"):
        registry.declare("same-id", "vllm.second")
