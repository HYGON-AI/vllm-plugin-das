# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Regression tests for stale HCU multiprocess RPC deadlines."""

from __future__ import annotations

import os
from collections import deque
from types import ModuleType

import pytest

os.environ.setdefault("VLLM_PLUGINS", "__disabled__")


class _FakeClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now


class _FakeBroadcastQueue:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def enqueue(self, request: object) -> None:
        self.requests.append(request)


class _FakeResponseQueue:
    def __init__(self, status: object) -> None:
        self.status = status
        self.timeouts: list[float | None] = []

    def dequeue(self, timeout: float | None = None) -> tuple[object, str]:
        assert timeout is None or timeout >= 0.0, (
            f"dequeue received negative timeout: {timeout}"
        )
        self.timeouts.append(timeout)
        return self.status, "response"


def _make_executor(hcu_executor):
    executor = object.__new__(hcu_executor.HcuMultiprocExecutor)
    executor.rpc_broadcast_mq = _FakeBroadcastQueue()
    executor.response_mqs = [
        _FakeResponseQueue(hcu_executor._upstream.WorkerProc.ResponseStatus.SUCCESS)
    ]
    executor.futures_queue = deque()
    executor.is_failed = False
    return executor


def test_hcu_collective_rpc_clamps_stale_deadline(monkeypatch) -> None:
    from vllm_hcu.v1.executor import multiproc_executor as hcu_executor

    clock = _FakeClock(100.0)
    monkeypatch.setattr(hcu_executor._upstream.time, "monotonic", clock.monotonic)

    executor = _make_executor(hcu_executor)

    future = executor.collective_rpc(
        "sample_tokens",
        timeout=1.0,
        non_block=True,
    )
    clock.now += 5.0

    assert future.result() == ["response"]
    assert executor.response_mqs[0].timeouts == [0.0]


@pytest.mark.parametrize(
    ("timeout", "elapsed", "expected"),
    [
        (None, 20.0, None),
        (5.0, 2.0, 3.0),
    ],
)
def test_hcu_collective_rpc_preserves_valid_timeout(
    monkeypatch,
    timeout: float | None,
    elapsed: float,
    expected: float | None,
) -> None:
    from vllm_hcu.v1.executor import multiproc_executor as hcu_executor

    clock = _FakeClock(100.0)
    monkeypatch.setattr(hcu_executor._upstream.time, "monotonic", clock.monotonic)
    executor = _make_executor(hcu_executor)

    future = executor.collective_rpc(
        "sample_tokens",
        timeout=timeout,
        non_block=True,
    )
    clock.now += elapsed

    assert future.result() == ["response"]
    assert executor.response_mqs[0].timeouts == [expected]


def test_multiproc_patch_clamps_negative_zmq_poll_timeout() -> None:
    from vllm.distributed.device_communicators.shm_broadcast import (
        MessageQueue as UpstreamMessageQueue,
    )
    from vllm_hcu.patch.platform.framework_opt import patch_multiproc_executor

    class MessageQueue(UpstreamMessageQueue):
        pass

    class MultiprocExecutor:
        def _init_executor(self) -> None:
            pass

    module = ModuleType(patch_multiproc_executor.TARGET_MODULE)
    module.MessageQueue = MessageQueue
    module.MultiprocExecutor = MultiprocExecutor
    assert patch_multiproc_executor.apply_to_module(module) is True
    assert patch_multiproc_executor.apply_to_module(module) is False

    class Socket:
        def __init__(self) -> None:
            self.poll_timeouts: list[int | None] = []

        def poll(self, timeout: int | None) -> bool:
            self.poll_timeouts.append(timeout)
            return False

    socket = Socket()
    for timeout in (-1.0, None, 0.001, 2.5):
        with pytest.raises(TimeoutError):
            MessageQueue.recv(socket, timeout)
    assert socket.poll_timeouts == [0, None, 1, 2500]
