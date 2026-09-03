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


def _make_patch_target_module(patch_multiproc_executor):
    from vllm.distributed.device_communicators.shm_broadcast import (
        MessageQueue as UpstreamMessageQueue,
    )
    from vllm.v1.executor import multiproc_executor as upstream

    class MessageQueue(UpstreamMessageQueue):
        pass

    class MultiprocExecutor:
        def _init_executor(self) -> None:
            pass

        def collective_rpc(
            self,
            method,
            timeout=None,
            args=(),
            kwargs=None,
            non_block=False,
            unique_reply_rank=None,
            kv_output_aggregator=None,
        ):
            pass

    module = ModuleType(patch_multiproc_executor.TARGET_MODULE)
    module.FutureWrapper = upstream.FutureWrapper
    module.MessageQueue = MessageQueue
    module.MultiprocExecutor = MultiprocExecutor
    module.WorkerProc = upstream.WorkerProc
    return module, MessageQueue


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
    from vllm_hcu.patch.platform.framework_opt import patch_multiproc_executor

    module, message_queue = _make_patch_target_module(patch_multiproc_executor)
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
            message_queue.recv(socket, timeout)
    assert socket.poll_timeouts == [0, None, 1, 2500]


def test_multiproc_patch_rejects_stale_recv_wrapper() -> None:
    from vllm_hcu.patch.platform.framework_opt import patch_multiproc_executor

    module, message_queue = _make_patch_target_module(patch_multiproc_executor)
    assert patch_multiproc_executor.apply_to_module(module) is True

    message_queue.recv = staticmethod(lambda socket, timeout: None)

    with pytest.raises(
        patch_multiproc_executor.PatchCompatibilityError,
        match="marker.*stale",
    ):
        patch_multiproc_executor.apply_to_module(module)


def test_multiproc_patch_rejects_orphan_recv_wrapper() -> None:
    from vllm_hcu.patch.platform.framework_opt import patch_multiproc_executor

    module, message_queue = _make_patch_target_module(patch_multiproc_executor)

    def orphan_recv(socket, timeout):
        pass

    setattr(orphan_recv, patch_multiproc_executor._RECV_MARKER, True)
    message_queue.recv = staticmethod(orphan_recv)

    with pytest.raises(
        patch_multiproc_executor.PatchCompatibilityError,
        match="wrapped without its owner marker",
    ):
        patch_multiproc_executor.apply_to_module(module)


def test_multiproc_patch_rejects_collective_rpc_signature_drift() -> None:
    from vllm_hcu.patch.platform.framework_opt import patch_multiproc_executor

    module, _ = _make_patch_target_module(patch_multiproc_executor)

    def incompatible_collective_rpc(self, operation, timeout=None):
        pass

    module.MultiprocExecutor.collective_rpc = incompatible_collective_rpc

    with pytest.raises(
        patch_multiproc_executor.PatchCompatibilityError,
        match=patch_multiproc_executor.TARGETS[1],
    ):
        patch_multiproc_executor.apply_to_module(module)


@pytest.mark.parametrize(
    ("attribute", "target"),
    [
        ("FutureWrapper", "vllm.v1.executor.multiproc_executor.FutureWrapper"),
        ("WorkerProc", "vllm.v1.executor.multiproc_executor.WorkerProc"),
    ],
)
def test_multiproc_patch_rejects_missing_rpc_dependency(
    attribute: str,
    target: str,
) -> None:
    from vllm_hcu.patch.platform.framework_opt import patch_multiproc_executor

    module, _ = _make_patch_target_module(patch_multiproc_executor)
    delattr(module, attribute)

    with pytest.raises(
        patch_multiproc_executor.PatchCompatibilityError,
        match=target,
    ):
        patch_multiproc_executor.apply_to_module(module)


def test_multiproc_patch_rejects_future_wrapper_signature_drift() -> None:
    from vllm_hcu.patch.platform.framework_opt import patch_multiproc_executor

    module, _ = _make_patch_target_module(patch_multiproc_executor)

    class FutureWrapper:
        def __init__(self, pending):
            pass

    module.FutureWrapper = FutureWrapper

    with pytest.raises(
        patch_multiproc_executor.PatchCompatibilityError,
        match=patch_multiproc_executor.TARGETS[3],
    ):
        patch_multiproc_executor.apply_to_module(module)


@pytest.mark.parametrize("include_response_status", [False, True])
def test_multiproc_patch_rejects_incomplete_worker_status_contract(
    include_response_status: bool,
) -> None:
    from vllm_hcu.patch.platform.framework_opt import patch_multiproc_executor

    module, _ = _make_patch_target_module(patch_multiproc_executor)

    class WorkerProc:
        pass

    if include_response_status:
        class ResponseStatus:
            pass

        WorkerProc.ResponseStatus = ResponseStatus
    module.WorkerProc = WorkerProc

    with pytest.raises(
        patch_multiproc_executor.PatchCompatibilityError,
        match=patch_multiproc_executor.TARGETS[5],
    ):
        patch_multiproc_executor.apply_to_module(module)
