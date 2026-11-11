# SPDX-License-Identifier: Apache-2.0
"""GPU resource waiting must be bounded and leave model execution separate."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tests.integration import gpu_memory as memory


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(memory.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(memory.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    return now


def _devices(free):
    return [{"device": index, "free": amount, "total": 100}
            for index, amount in enumerate(free)]


def test_memory_wait_is_opt_in_outside_ci(monkeypatch):
    monkeypatch.delenv(memory.WAIT_TIMEOUT_ENV, raising=False)
    monkeypatch.delenv("VLLM_HCU_IS_IN_CI", raising=False)
    monkeypatch.setattr(memory, "_read_gpu_memory", lambda *a, **k: pytest.fail("unexpected probe"))
    assert memory.configured_gpu_wait_timeout() == 0
    memory.wait_for_gpu_memory(required_cards=1, gpu_memory_utilization=0.6)
    assert memory.configured_gpu_wait_timeout({"VLLM_HCU_IS_IN_CI": "1"}) == 600
    assert memory.configured_gpu_wait_timeout({memory.WAIT_TIMEOUT_ENV: "0"}) == 0
    assert memory.configured_gpu_wait_timeout({memory.WAIT_TIMEOUT_ENV: "900"}) == 900


def test_wait_checks_all_required_cards_and_resumes(clock, monkeypatch):
    snapshots = iter([_devices([80, 19]), _devices([80, 60])])
    environments = []
    def read(cards, environment, timeout):
        assert cards == 2
        environments.append(environment)
        return next(snapshots)
    monkeypatch.setattr(memory, "_read_gpu_memory", read)
    messages = []
    environment = {memory.WAIT_TIMEOUT_ENV: "30", memory.POLL_INTERVAL_ENV: "10",
                   "CUDA_VISIBLE_DEVICES": "2,3"}
    memory.wait_for_gpu_memory(
        required_cards=2, gpu_memory_utilization=0.6, env=environment, report=messages.append,
    )
    assert clock[0] == 10
    assert len(environments) == 2
    assert all(e["CUDA_VISIBLE_DEVICES"] == "2,3" for e in environments)
    assert "waiting" in messages[0]
    assert "ready" in messages[-1]


def test_memory_wait_times_out_with_resource_error(clock, monkeypatch):
    monkeypatch.setattr(memory, "_read_gpu_memory", lambda *a: _devices([19]))
    with pytest.raises(memory.GpuMemoryUnavailableError, match="unavailable after 30s"):
        memory.wait_for_gpu_memory(
            required_cards=1, gpu_memory_utilization=0.6,
            env={memory.WAIT_TIMEOUT_ENV: "30", memory.POLL_INTERVAL_ENV: "10"},
            report=lambda message: None,
        )
    assert clock[0] == 30


def test_probe_error_waits_for_recovery(clock, monkeypatch):
    calls = []
    def read(*args):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("driver temporarily unavailable")
        return _devices([90])
    monkeypatch.setattr(memory, "_read_gpu_memory", read)
    messages = []
    memory.wait_for_gpu_memory(
        required_cards=1, gpu_memory_utilization=0.6,
        env={memory.WAIT_TIMEOUT_ENV: "30", memory.POLL_INTERVAL_ENV: "10"},
        report=messages.append,
    )
    assert len(calls) == 2
    assert "probe unavailable" in messages[0]


@pytest.mark.parametrize("environment", [
    {memory.WAIT_TIMEOUT_ENV: "-1"},
    {memory.WAIT_TIMEOUT_ENV: "10", memory.POLL_INTERVAL_ENV: "0"},
])
def test_invalid_wait_settings_fail_early(environment):
    with pytest.raises(ValueError):
        memory.wait_for_gpu_memory(required_cards=1, gpu_memory_utilization=0.6, env=environment)


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, float("nan")])
def test_invalid_memory_fraction_fails_early(fraction):
    with pytest.raises(ValueError):
        memory.wait_for_gpu_memory(
            required_cards=1, gpu_memory_utilization=fraction,
            env={memory.WAIT_TIMEOUT_ENV: "30"},
        )


def test_memory_probe_runs_in_isolated_process(monkeypatch):
    def run(command, **kwargs):
        assert command[0] == sys.executable
        assert command[-2:] == ["--probe", "2"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2,3"
        assert kwargs["timeout"] == 12
        return subprocess.CompletedProcess(command, 0, json.dumps(_devices([80, 80])), "")
    monkeypatch.setattr(memory.subprocess, "run", run)
    assert memory._read_gpu_memory(2, {"CUDA_VISIBLE_DEVICES": "2,3"}, 12) == _devices([80, 80])


@pytest.mark.parametrize("arguments,expected", [
    (["vllm", "serve", "--gpu-memory-utilization", "0.4"], 0.4),
    (["vllm", "serve", "--gpu-memory-utilization=0.6"], 0.6),
    (["vllm", "serve"], None),
])
def test_memory_fraction_uses_actual_server_arguments(arguments, expected):
    assert memory.gpu_memory_utilization_from_args(arguments) == expected

def test_wait_resolves_runtime_default_when_no_fraction_is_given(clock, monkeypatch):
    defaults = []
    def default(environment, timeout):
        defaults.append(environment)
        return 0.75
    monkeypatch.setattr(memory, "_default_gpu_memory_utilization", default)
    snapshots = iter([_devices([60]), _devices([80])])
    monkeypatch.setattr(memory, "_read_gpu_memory", lambda *a: next(snapshots))
    memory.wait_for_gpu_memory(
        required_cards=1, gpu_memory_utilization=None,
        env={memory.WAIT_TIMEOUT_ENV: "30", memory.POLL_INTERVAL_ENV: "10"},
        report=lambda message: None,
    )
    assert len(defaults) == 1
    assert clock[0] == 10


def test_default_fraction_is_read_from_runtime_in_child(monkeypatch):
    def run(command, **kwargs):
        assert command[-1] == "--default-utilization"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3"
        return subprocess.CompletedProcess(command, 0, "runtime log\n0.9\n", "")
    monkeypatch.setattr(memory.subprocess, "run", run)
    assert memory._default_gpu_memory_utilization({"CUDA_VISIBLE_DEVICES": "3"}, 10) == 0.9
