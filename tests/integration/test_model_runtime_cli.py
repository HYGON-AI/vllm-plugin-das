# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU-safe CLI coverage for the model runtime subprocess harness."""

from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

from tests.integration import model_runtime


class _FakeEvent:
    def __init__(self):
        self._is_set = False

    def set(self):
        self._is_set = True

    def is_set(self):
        return self._is_set

    def wait(self, timeout=None):
        del timeout
        return self._is_set


class _FakeResultQueue:
    def __init__(self, error=None):
        self.error = error

    def get(self, timeout=None):
        del timeout
        if self.error is not None:
            raise self.error
        raise AssertionError("unexpected queue read")


class _FakeRankProcess:
    def __init__(self, pid, exitcode, args, join_error=None):
        self.pid = pid
        self.exitcode = exitcode
        self.args = args
        self.join_error = join_error

    def start(self):
        for value in self.args:
            if isinstance(value, _FakeEvent):
                value.set()

    def join(self, timeout=None):
        del timeout
        if self.join_error is not None:
            raise self.join_error

    def is_alive(self):
        return self.exitcode is None

    def terminate(self):
        self.exitcode = -signal.SIGTERM

    def kill(self):
        self.exitcode = -signal.SIGKILL


class _FakeMultiprocessingContext:
    def __init__(self, exitcodes, *, join_error=None, queue_error=None):
        self.exitcodes = list(exitcodes)
        self.join_error = join_error
        self.queue = _FakeResultQueue(queue_error)
        self.processes = []

    def Event(self):
        return _FakeEvent()

    def Lock(self):
        return self

    def Value(self, typecode, value):
        del typecode
        return SimpleNamespace(value=value)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback

    def Queue(self):
        return self.queue

    def Process(self, *, target, args):
        del target
        index = len(self.processes)
        process = _FakeRankProcess(
            1000 + index,
            self.exitcodes[index],
            args,
            join_error=self.join_error,
        )
        self.processes.append(process)
        return process


def _run_fake_dp_case(monkeypatch, context):
    cleaned = []
    monkeypatch.setattr(
        model_runtime.multiprocessing,
        "get_context",
        lambda method: context,
    )
    monkeypatch.setattr(
        model_runtime,
        "_terminate_data_parallel_process_groups",
        lambda processes, ready_events, process_group_id: cleaned.append(
            ([process.pid for process in processes], list(ready_events))
        ),
        raising=False,
    )
    kwargs = {
        "tensor_parallel_size": 1,
        "data_parallel_size": len(context.exitcodes),
        "gpu_memory_utilization": 0.9,
        "all2all_backend": "deepep_high_throughput",
        "moe_backend": "dpsk_deep_gemm",
    }
    return cleaned, kwargs


def test_tp_ep_dp_uses_explicit_multiprocess_launcher(monkeypatch):
    captured = {}
    expected = {"parallel_config": {"data_parallel_size": 8}, "output": []}

    def fail_single_process_llm(**kwargs):
        raise AssertionError(f"single-process LLM was constructed: {kwargs}")

    def fake_data_parallel_case(model_path, **kwargs):
        captured["model_path"] = model_path
        captured.update(kwargs)
        return expected

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=fail_single_process_llm))
    monkeypatch.setattr(
        model_runtime,
        "_case_tp_ep_smoke_data_parallel",
        fake_data_parallel_case,
        raising=False,
    )

    result = model_runtime._case_tp_ep_smoke(
        Path("/models/fake"),
        tensor_parallel_size=1,
        data_parallel_size=8,
        gpu_memory_utilization=0.9,
        all2all_backend="deepep_high_throughput",
        moe_backend="dpsk_deep_gemm",
    )

    assert result is expected
    assert captured == {
        "model_path": Path("/models/fake"),
        "tensor_parallel_size": 1,
        "data_parallel_size": 8,
        "gpu_memory_utilization": 0.9,
        "all2all_backend": "deepep_high_throughput",
        "moe_backend": "dpsk_deep_gemm",
    }


def test_tp_ep_cli_forwards_data_parallel_and_all2all(monkeypatch, capsys):
    captured = {}

    def fake_case(model_path, **kwargs):
        captured["model_path"] = model_path
        captured.update(kwargs)
        return {"output": []}

    monkeypatch.setattr(model_runtime, "_case_tp_ep_smoke", fake_case)
    assert model_runtime._main(
        [
            "tp-ep-smoke",
            "--model",
            "/models/fake",
            "--tensor-parallel-size",
            "1",
            "--data-parallel-size",
            "8",
            "--all2all-backend",
            "deepep_low_latency",
            "--moe-backend",
            "dpsk_deep_gemm",
        ]
    ) == 0
    assert captured == {
        "model_path": Path("/models/fake"),
        "tensor_parallel_size": 1,
        "data_parallel_size": 8,
        "gpu_memory_utilization": 0.6,
        "all2all_backend": "deepep_low_latency",
        "moe_backend": "dpsk_deep_gemm",
    }
    assert "VLLM_HCU_RESULT=" in capsys.readouterr().out


def test_tp_ep_ll_uses_supported_deepep_token_capacity(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.llm_engine = SimpleNamespace(vllm_config=None)

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM))
    monkeypatch.setattr(model_runtime, "_generate_with_llm", lambda *args, **kwargs: [])
    monkeypatch.setattr(model_runtime, "_shutdown_llm", lambda llm: None)

    model_runtime._case_tp_ep_smoke_rank(
        Path("/models/fake"),
        tensor_parallel_size=1,
        data_parallel_size=8,
        gpu_memory_utilization=0.9,
        all2all_backend="deepep_low_latency",
        moe_backend="dpsk_deep_gemm",
    )

    assert captured["max_num_batched_tokens"] == 256


def test_tp_ep_dp_cleans_every_rank_group_after_rank_failure(monkeypatch):
    context = _FakeMultiprocessingContext([1, None])
    cleaned, kwargs = _run_fake_dp_case(monkeypatch, context)

    with pytest.raises(RuntimeError, match="rank process .* failed"):
        model_runtime._case_tp_ep_smoke_data_parallel(Path("/models/fake"), **kwargs)

    assert [entry[0] for entry in cleaned] == [[1000, 1001]]


def test_tp_ep_dp_installs_sigterm_cleanup_handler(monkeypatch):
    context = _FakeMultiprocessingContext([1, None])
    cleaned, kwargs = _run_fake_dp_case(monkeypatch, context)
    handlers = []
    monkeypatch.setattr(
        model_runtime.signal,
        "signal",
        lambda sig, handler: handlers.append((sig, handler)) or signal.SIG_DFL,
    )

    with pytest.raises(RuntimeError, match="rank process .* failed"):
        model_runtime._case_tp_ep_smoke_data_parallel(Path("/models/fake"), **kwargs)

    assert cleaned
    assert handlers[0][0] == signal.SIGTERM
    with pytest.raises(
        model_runtime._DataParallelTermination,
        match="received signal 15",
    ):
        handlers[0][1](signal.SIGTERM, None)
    assert handlers[-1] == (signal.SIGTERM, signal.SIG_DFL)


def test_tp_ep_dp_cleans_every_rank_group_after_interruption(monkeypatch):
    context = _FakeMultiprocessingContext(
        [None, None],
        join_error=KeyboardInterrupt(),
    )
    cleaned, kwargs = _run_fake_dp_case(monkeypatch, context)

    with pytest.raises(KeyboardInterrupt):
        model_runtime._case_tp_ep_smoke_data_parallel(Path("/models/fake"), **kwargs)

    assert [entry[0] for entry in cleaned] == [[1000, 1001]]


def test_tp_ep_dp_cleans_every_rank_group_after_queue_failure(monkeypatch):
    context = _FakeMultiprocessingContext(
        [0, 0],
        queue_error=RuntimeError("queue failed"),
    )
    cleaned, kwargs = _run_fake_dp_case(monkeypatch, context)

    with pytest.raises(RuntimeError, match="queue failed"):
        model_runtime._case_tp_ep_smoke_data_parallel(Path("/models/fake"), **kwargs)

    assert [entry[0] for entry in cleaned] == [[1000, 1001]]


def test_owned_process_group_cleanup_escalates_and_verifies(monkeypatch):
    signals = []
    waits = iter([{1001}, set()])
    monkeypatch.setattr(
        model_runtime.os,
        "killpg",
        lambda pgid, sig: signals.append((pgid, sig)),
    )
    monkeypatch.setattr(
        model_runtime,
        "_wait_for_process_groups",
        lambda pgids, timeout_s: next(waits),
        raising=False,
    )

    model_runtime._terminate_owned_process_groups(
        [1000, 1001],
        term_timeout_s=1,
        kill_timeout_s=1,
    )

    assert signals == [
        (1000, signal.SIGTERM),
        (1001, signal.SIGTERM),
        (1001, signal.SIGKILL),
    ]


def test_owned_process_group_cleanup_reaps_leaders_while_waiting(monkeypatch):
    polled = []
    proc = SimpleNamespace(poll=lambda: polled.append(True))
    monkeypatch.setattr(model_runtime.os, "killpg", lambda pgid, sig: None)

    def fake_wait(pgids, timeout_s, *, process_leaders=()):
        del pgids, timeout_s
        for leader in process_leaders:
            leader.poll()
        return set()

    monkeypatch.setattr(model_runtime, "_wait_for_process_groups", fake_wait)

    model_runtime._terminate_owned_process_groups(
        [1000],
        process_leaders=[proc],
        term_timeout_s=1,
        kill_timeout_s=1,
    )

    assert polled == [True]


def test_case_group_cleanup_does_not_skip_an_exited_leader(monkeypatch):
    cleaned = []
    proc = SimpleNamespace(pid=2000, poll=lambda: 1, wait=lambda timeout: 1)
    monkeypatch.setattr(
        model_runtime,
        "_terminate_owned_process_groups",
        lambda pgids, **kwargs: cleaned.append(list(pgids)),
        raising=False,
    )

    model_runtime._terminate_case_process_group(proc)

    assert cleaned == [[2000]]


def test_dp_rank_joins_the_shared_owned_process_group(monkeypatch):
    calls = []
    process_group_id = SimpleNamespace(value=1234)
    ready = _FakeEvent()
    start_gate = _FakeEvent()
    start_gate.set()
    result_queue = SimpleNamespace(put=lambda result: calls.append(("put", result)))
    monkeypatch.setattr(
        model_runtime.os,
        "setpgid",
        lambda pid, pgid: calls.append((pid, pgid)),
    )
    monkeypatch.setattr(model_runtime.signal, "signal", lambda sig, handler: None)
    monkeypatch.setattr(
        model_runtime,
        "_case_tp_ep_smoke_rank",
        lambda *args, **kwargs: {"output": []},
    )

    model_runtime._tp_ep_data_parallel_rank(
        1,
        2,
        "127.0.0.1",
        12345,
        Path("/models/fake"),
        1,
        0.9,
        "deepep_high_throughput",
        "dpsk_deep_gemm",
        result_queue,
        process_group_id,
        _FakeMultiprocessingContext([]),
        ready,
        start_gate,
    )

    assert calls[0] == (0, 1234)
    assert ready.is_set()
