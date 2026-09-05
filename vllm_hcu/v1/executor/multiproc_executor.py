# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""HCU multiprocess executor selected through vLLM's executor backend API."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from vllm.v1.executor import multiproc_executor as _upstream


_FORK_ORIGINAL_MESSAGE_QUEUE: object | None = None
_FORK_PROXY_MESSAGE_QUEUE: object | None = None


class _MessageQueueConstructorProxy:
    """Constructor proxy that preserves ``MessageQueue`` class attributes."""

    def __init__(self, original_message_queue: object, max_chunks: int) -> None:
        self._original_message_queue = original_message_queue
        self._max_chunks = max_chunks
        self.__name__ = getattr(original_message_queue, "__name__", type(self).__name__)
        self.__qualname__ = getattr(
            original_message_queue, "__qualname__", type(self).__qualname__
        )
        self.__module__ = getattr(original_message_queue, "__module__", __name__)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # The legacy change only resized the scheduler-output broadcast
        # queue, whose construction supplies max_chunk_bytes.  Worker response
        # queues and queues reconstructed from handles must retain their
        # official size/semantics.
        if "max_chunk_bytes" in kwargs:
            requested = kwargs.get("max_chunks", 0)
            if not isinstance(requested, int) or isinstance(requested, bool):
                raise TypeError(
                    "MessageQueue max_chunks must be an integer, got "
                    f"{type(requested).__name__}"
                )
            kwargs["max_chunks"] = max(requested, self._max_chunks)
        return self._original_message_queue(*args, **kwargs)  # type: ignore[operator]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original_message_queue, name)


def _restore_message_queue_after_fork() -> None:
    """Do not leak the parent's temporary constructor into forked Workers."""

    global _FORK_ORIGINAL_MESSAGE_QUEUE, _FORK_PROXY_MESSAGE_QUEUE
    if (
        _FORK_PROXY_MESSAGE_QUEUE is not None
        and _upstream.MessageQueue is _FORK_PROXY_MESSAGE_QUEUE
    ):
        _upstream.MessageQueue = _FORK_ORIGINAL_MESSAGE_QUEUE  # type: ignore[assignment]
    _FORK_ORIGINAL_MESSAGE_QUEUE = None
    _FORK_PROXY_MESSAGE_QUEUE = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_restore_message_queue_after_fork)


class HcuMultiprocExecutor(_upstream.MultiprocExecutor):
    """Size the scheduler-output MQ for all concurrently in-flight PP batches.

    vLLM v0.25.1 owns batch concurrency on ``VllmConfig`` and creates the queue
    inside a monolithic ``_init_executor``.  The subclass supplies a constructor
    proxy only while that parent initializer runs and restores the module binding
    in ``finally``.  This keeps the official lifecycle, worker startup, and
    failure cleanup intact.
    """

    _mq_constructor_lock = threading.RLock()

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: Any | None = None,
        ec_output_aggregator: Any | None = None,
    ) -> Any:
        """Call every worker without allowing an expired deadline to go negative."""
        assert self.rpc_broadcast_mq is not None, (
            "collective_rpc should not be called on follower node"
        )
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        aggregators = [
            aggregator
            for aggregator in (kv_output_aggregator, ec_output_aggregator)
            if aggregator is not None
        ]
        if aggregators:
            output_rank = None

            def aggregate(outputs: Any) -> Any:
                rank = unique_reply_rank or 0
                result = outputs[rank]
                for aggregator in aggregators:
                    result = aggregator.aggregate(outputs, output_rank=rank)
                return result
        else:
            output_rank = unique_reply_rank
            aggregate = lambda value: value

        if isinstance(method, str):
            send_method = method
        else:
            send_method = _upstream.cloudpickle.dumps(
                method,
                protocol=_upstream.pickle.HIGHEST_PROTOCOL,
            )
        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))

        response_mqs: Sequence[_upstream.MessageQueue] = self.response_mqs
        if output_rank is not None:
            response_mqs = (response_mqs[output_rank],)

        def get_response() -> Any:
            responses = []
            for mq in response_mqs:
                dequeue_timeout = (
                    None
                    if deadline is None
                    else max(0.0, deadline - time.monotonic())
                )
                try:
                    status, result = mq.dequeue(timeout=dequeue_timeout)
                except TimeoutError as exc:
                    raise TimeoutError(f"RPC call to {method} timed out.") from exc
                if status != _upstream.WorkerProc.ResponseStatus.SUCCESS:
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause"
                    )
                responses.append(result)
            return responses[0] if output_rank is not None else responses

        future = _upstream.FutureWrapper(
            self.futures_queue,
            get_response=get_response,
            aggregate=aggregate,
        )
        return future if non_block else future.result()

    def _init_executor(self) -> None:
        global _FORK_ORIGINAL_MESSAGE_QUEUE, _FORK_PROXY_MESSAGE_QUEUE
        original_message_queue = _upstream.MessageQueue
        max_chunks = max(10, 4 * self.vllm_config.max_concurrent_batches)
        hcu_message_queue = _MessageQueueConstructorProxy(
            original_message_queue, max_chunks
        )

        with self._mq_constructor_lock:
            if _upstream.MessageQueue is not original_message_queue:
                raise RuntimeError(
                    "MultiprocExecutor.MessageQueue changed during HCU initialization"
                )
            if (
                _FORK_ORIGINAL_MESSAGE_QUEUE is not None
                or _FORK_PROXY_MESSAGE_QUEUE is not None
            ):
                raise RuntimeError("nested HCU MessageQueue constructor override")
            _FORK_ORIGINAL_MESSAGE_QUEUE = original_message_queue
            _FORK_PROXY_MESSAGE_QUEUE = hcu_message_queue
            _upstream.MessageQueue = hcu_message_queue
            try:
                super()._init_executor()
            finally:
                try:
                    if _upstream.MessageQueue is not hcu_message_queue:
                        raise RuntimeError(
                            "MultiprocExecutor.MessageQueue was replaced concurrently"
                        )
                    _upstream.MessageQueue = original_message_queue
                finally:
                    _FORK_ORIGINAL_MESSAGE_QUEUE = None
                    _FORK_PROXY_MESSAGE_QUEUE = None


__all__ = ["HcuMultiprocExecutor"]
