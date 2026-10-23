# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""GPU wait integration without loading models or creating a device context."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.integration import model_runtime as runtime


@pytest.fixture
def constructors(monkeypatch):
    events = []
    fake_vllm = ModuleType("vllm")

    def construct(**kwargs):
        events.append(("model", kwargs))
        return SimpleNamespace(kwargs=kwargs)

    fake_vllm.LLM = construct
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(runtime, "configured_gpu_wait_timeout", lambda env=None: 600)
    monkeypatch.setattr(
        runtime, "wait_for_gpu_memory",
        lambda **kwargs: events.append(("wait", kwargs)),
    )
    return events


@pytest.mark.parametrize(
    ("parallel", "cards"),
    [
        ({}, 1),
        ({"tensor_parallel_size": 8}, 8),
        ({"tensor_parallel_size": 2, "pipeline_parallel_size": 2,
          "prefill_context_parallel_size": 2}, 8),
        ({"tensor_parallel_size": 2, "data_parallel_size": 4,
          "data_parallel_size_local": 2}, 4),
    ],
)
def test_model_wait_uses_actual_topology_and_memory_budget(constructors, parallel, cards):
    kwargs = runtime._llm_kwargs(
        Path("/models/test"), enforce_eager=True,
        gpu_memory_utilization=0.6, **parallel,
    )
    result = runtime._create_llm(**kwargs)

    assert [event[0] for event in constructors] == ["wait", "model"]
    assert constructors[0][1]["required_cards"] == cards
    assert constructors[0][1]["gpu_memory_utilization"] == 0.6
    assert result.kwargs == kwargs


def test_memory_timeout_does_not_start_model(monkeypatch, constructors):
    def unavailable(**kwargs):
        raise RuntimeError("required GPU memory is unavailable")

    monkeypatch.setattr(runtime, "wait_for_gpu_memory", unavailable)
    with pytest.raises(RuntimeError, match="GPU memory is unavailable"):
        runtime._create_llm(gpu_memory_utilization=0.6)
    assert constructors == []


@pytest.mark.parametrize("first_fails", [False, True])
def test_cli_case_shares_one_wait_and_resets_scope(monkeypatch, constructors, first_fails):
    def two_models(_path):
        runtime._create_llm(gpu_memory_utilization=0.6)
        if first_fails:
            raise RuntimeError("case failed")
        runtime._create_llm(gpu_memory_utilization=0.6)
        return {"ok": True}

    monkeypatch.setattr(runtime, "_case_smoke", two_models)
    argv = ["smoke", "--model", "/models/test"]
    if first_fails:
        with pytest.raises(RuntimeError, match="case failed"):
            runtime._main(argv)
    else:
        runtime._main(argv)
    assert sum(event[0] == "wait" for event in constructors) == 1

    monkeypatch.setattr(
        runtime, "_case_smoke",
        lambda path: (runtime._create_llm(gpu_memory_utilization=0.6), {})[-1],
    )
    runtime._main(argv)
    assert sum(event[0] == "wait" for event in constructors) == 2


def test_disabled_wait_does_not_leak_state_to_next_case(monkeypatch, constructors):
    monkeypatch.setattr(runtime, "configured_gpu_wait_timeout", lambda env=None: 0)
    with runtime._case_gpu_memory_wait_scope():
        runtime._create_llm(gpu_memory_utilization=0.6)
    assert all(event[0] == "model" for event in constructors)

    monkeypatch.setattr(runtime, "configured_gpu_wait_timeout", lambda env=None: 600)
    with runtime._case_gpu_memory_wait_scope():
        runtime._create_llm(gpu_memory_utilization=0.6)
    # Calls outside an explicit CLI case do not retain readiness across calls.
    runtime._create_llm(gpu_memory_utilization=0.6)
    runtime._create_llm(gpu_memory_utilization=0.6)
    assert sum(event[0] == "wait" for event in constructors) == 3


@pytest.mark.parametrize(
    ("extra_env", "budget"),
    [
        ({}, 600),
        ({"HCU_CI_GPU_MEMORY_WAIT_TIMEOUT_S": "0"}, 0),
        ({"HCU_CI_GPU_MEMORY_WAIT_TIMEOUT_S": "90"}, 90),
    ],
)
def test_parent_timeout_reserves_final_child_wait_budget(monkeypatch, tmp_path, extra_env, budget):
    monkeypatch.setenv("VLLM_HCU_IS_IN_CI", "1")
    monkeypatch.delenv("HCU_CI_GPU_MEMORY_WAIT_TIMEOUT_S", raising=False)
    monkeypatch.setenv("VLLM_HCU_INTEGRATION_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_HCU_RELEASE_WHEEL", raising=False)
    timeouts = []

    class Process:
        def __init__(self, command, **kwargs):
            kwargs["stdout"].write(runtime.RESULT_PREFIX + json.dumps({"ok": True}) + "\n")

        def wait(self, *, timeout):
            timeouts.append(timeout)
            return 0

    monkeypatch.setattr(runtime.subprocess, "Popen", Process)
    result = runtime.run_vllm_case(
        "smoke", Path("/models/test"), timeout_s=77, extra_env=extra_env,
    )
    assert result["ok"]
    assert timeouts == [77 + budget]


def test_dp_coordinator_checks_all_cards_before_creating_ranks(monkeypatch, constructors):
    events = []
    monkeypatch.setattr(
        runtime, "wait_for_gpu_memory",
        lambda **kwargs: events.append(("wait", kwargs)),
    )

    def stop_before_spawning(method):
        events.append(("context", method))
        raise RuntimeError("stop before creating ranks")

    monkeypatch.setattr(runtime.multiprocessing, "get_context", stop_before_spawning)
    with pytest.raises(RuntimeError, match="stop before creating ranks"):
        runtime._case_tp_ep_smoke_data_parallel(
            Path("/models/test"), tensor_parallel_size=2, data_parallel_size=4,
            gpu_memory_utilization=0.6, all2all_backend=None, moe_backend="auto",
        )
    assert [event[0] for event in events] == ["wait", "context"]
    assert events[0][1]["required_cards"] == 8
    assert events[0][1]["gpu_memory_utilization"] == 0.6


def test_dp_rank_uses_coordinator_result_and_restores_environment(monkeypatch):
    monkeypatch.setenv("HCU_CI_GPU_MEMORY_WAIT_TIMEOUT_S", "42")
    monkeypatch.setattr(runtime.signal, "signal", lambda *args: None)
    monkeypatch.setattr(runtime.os, "setpgid", lambda *args: None)
    monkeypatch.setattr(runtime.os, "getpid", lambda: 101)
    results = []

    def rank_case(*args, **kwargs):
        assert os.environ["HCU_CI_GPU_MEMORY_WAIT_TIMEOUT_S"] == "0"
        return {"ok": True}

    monkeypatch.setattr(runtime, "_case_tp_ep_smoke_rank", rank_case)
    runtime._tp_ep_data_parallel_rank(
        0, 4, "127.0.0.1", 12345, Path("/models/test"), 2, 0.6, None, "auto",
        SimpleNamespace(put=results.append), SimpleNamespace(value=0),
        nullcontext(), SimpleNamespace(set=lambda: None),
        SimpleNamespace(wait=lambda: True),
    )
    assert results == [(0, {"ok": True})]
    assert os.environ["HCU_CI_GPU_MEMORY_WAIT_TIMEOUT_S"] == "42"


@pytest.mark.parametrize("shadow_tests_package", [False, True])
def test_release_wheel_direct_script_imports_wait_helper_outside_checkout(
    tmp_path, shadow_tests_package,
):
    environment = os.environ.copy()
    environment["VLLM_HCU_RELEASE_WHEEL"] = "1"
    environment["PYTHONPATH"] = ""
    if shadow_tests_package:
        foreign_root = tmp_path / "foreign-packages"
        foreign_tests = foreign_root / "tests"
        foreign_tests.mkdir(parents=True)
        (foreign_tests / "__init__.py").write_text(
            "raise RuntimeError('unrelated installed tests package was imported')\n"
        )
        environment["PYTHONPATH"] = str(foreign_root)
    result = subprocess.run(
        [sys.executable, str(Path(runtime.__file__).resolve()), "--help"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--gpu-memory-utilization" in result.stdout
