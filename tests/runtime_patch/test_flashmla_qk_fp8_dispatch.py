# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Calibration must select on costs without caching input validity."""

import importlib.util
from pathlib import Path

import pytest


def load_policy():
    path = Path(__file__).resolve().parents[2] / (
        "vllm_hcu/v1/attention/ops/flashmla_qk_fp8_dispatch.py"
    )
    spec = importlib.util.spec_from_file_location("dispatch_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Unsupported(ValueError):
    pass


class Clock:
    def __init__(self, ready=True):
        self.time = 0
        self.ready = ready
        self.events = []
        self.calls = []

    def event(self):
        clock = self

        class Event:
            def record(self):
                self.time = clock.time

            def query(self):
                return clock.ready

            def elapsed_time(self, end):
                assert clock.ready
                return end.time - self.time

            def synchronize(self):
                pytest.fail("policy must not wait for a GPU event")

        event = Event()
        self.events.append(event)
        return event

    def operation(self, name, cost):
        def run():
            self.calls.append(name)
            self.time += cost
            return name

        return run


@pytest.mark.parametrize(
    "fp8_cost,chosen", [(0.5, "fp8"), (1.2, "bf16"), (0.95, "bf16")]
)
def test_three_actual_pairs_require_a_clear_fp8_gain(fp8_cost, chosen):
    module = load_policy()
    clock = Clock()
    policy = module.DispatchPolicy(event_factory=clock.event)
    fp8 = clock.operation("fp8", fp8_cost)
    bf16 = clock.operation("bf16", 1)
    for _ in range(3):
        assert policy.run("shape", fp8, bf16, Unsupported) == "bf16"
    assert clock.calls == ["bf16", "fp8", "fp8", "bf16", "bf16", "fp8"]
    clock.calls.clear()
    for _ in range(5):
        assert policy.run("shape", fp8, bf16, Unsupported) == chosen
    assert clock.calls == [chosen] * 5
    assert len(clock.events) == 12


def test_pending_samples_do_not_block_or_keep_probing():
    module = load_policy()
    clock = Clock(ready=False)
    policy = module.DispatchPolicy(event_factory=clock.event)
    fp8 = clock.operation("fp8", 0.5)
    bf16 = clock.operation("bf16", 1)
    for _ in range(20):
        assert policy.run("shape", fp8, bf16, Unsupported) == "bf16"
    assert clock.calls.count("fp8") == 3
    clock.ready = True
    assert policy.run("shape", fp8, bf16, Unsupported) == "fp8"


def test_cached_fast_choice_still_executes_current_checked_callback():
    module = load_policy()
    clock = Clock()
    policy = module.DispatchPolicy(event_factory=clock.event)
    fp8 = clock.operation("fp8", 0.5)
    bf16 = clock.operation("bf16", 1)
    for _ in range(4):
        policy.run("shape", fp8, bf16, Unsupported)

    def invalid_current_input():
        raise Unsupported("indices changed in place")

    assert policy.run("shape", invalid_current_input, bf16, Unsupported) == "bf16"
    # Rejecting FP8 is a safe cost decision; no invalid tensor is cached.
    assert policy.run("shape", fp8, bf16, Unsupported) == "bf16"


@pytest.mark.parametrize("fail_on_call", [0, 1])
def test_unsupported_during_calibration_calls_bf16_once(fail_on_call):
    module = load_policy()
    clock = Clock()
    policy = module.DispatchPolicy(event_factory=clock.event)
    good = clock.operation("fp8", 0.5)
    bf16 = clock.operation("bf16", 1)
    for _ in range(fail_on_call):
        policy.run("shape", good, bf16, Unsupported)
    clock.calls.clear()

    def unsupported():
        raise Unsupported("short count")

    assert policy.run("shape", unsupported, bf16, Unsupported) == "bf16"
    assert clock.calls == ["bf16"]


def test_native_runtime_errors_are_not_hidden():
    module = load_policy()
    clock = Clock()
    policy = module.DispatchPolicy(event_factory=clock.event)

    def broken():
        raise RuntimeError("native kernel failed")

    with pytest.raises(RuntimeError, match="native kernel failed"):
        policy.run("shape", broken, clock.operation("bf16", 1), Unsupported)


def test_cache_is_bounded_and_recalibrates_evicted_shape():
    module = load_policy()
    clock = Clock()
    policy = module.DispatchPolicy(event_factory=clock.event, max_entries=2)
    fp8 = clock.operation("fp8", 0.5)
    bf16 = clock.operation("bf16", 1)
    for key in ["a", "b", "c", "a"]:
        policy.run(key, fp8, bf16, Unsupported)
    assert clock.calls.count("fp8") == 4
    assert len(policy.entries) == 2


def test_noisy_pair_does_not_override_median_choice():
    module = load_policy()
    clock = Clock()
    policy = module.DispatchPolicy(event_factory=clock.event)
    bf16 = clock.operation("bf16", 1)
    for cost in [0.5, 10, 0.5]:
        policy.run("shape", clock.operation("fp8", cost), bf16, Unsupported)
    assert policy.run("shape", clock.operation("fp8", 0.5), bf16, Unsupported) == "fp8"


@pytest.mark.parametrize("initial,later", [(0.5, 2), (2, 0.5)])
def test_decision_is_reconsidered_when_locality_changes(initial, later):
    module = load_policy()
    clock = Clock()
    policy = module.DispatchPolicy(event_factory=clock.event, refresh_calls=4)
    bf16 = clock.operation("bf16", 1)
    for _ in range(7):
        policy.run("shape", clock.operation("fp8", initial), bf16, Unsupported)
    # Four selected calls expire the choice; the next three gather fresh pairs.
    for _ in range(3):
        policy.run("shape", clock.operation("fp8", later), bf16, Unsupported)
    assert policy.run("shape", clock.operation("fp8", later), bf16, Unsupported) == (
        "fp8" if later < 0.9 else "bf16"
    )
