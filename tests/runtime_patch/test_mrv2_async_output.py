# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch


class _Clock:
    def __init__(self) -> None:
        self.now = 10.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.now += duration


def test_hcu_async_output_event_wait_times_out_instead_of_hanging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.v1 import async_utils

    clock = _Clock()

    class Event:
        def __init__(self) -> None:
            self.queries = 0

        def query(self) -> bool:
            self.queries += 1
            return False

        def synchronize(self) -> None:
            raise AssertionError("a bounded wait must not block in synchronize()")

    event = Event()
    monkeypatch.setattr(async_utils.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(async_utils.time, "sleep", clock.sleep)

    with pytest.raises(TimeoutError, match="copy sampled-token output"):
        async_utils.wait_for_hcu_event(
            event,
            operation="copy sampled-token output",
            timeout_s=0.025,
        )

    assert event.queries > 1


@pytest.mark.parametrize("timeout_s", [0.0, -1.0])
def test_hcu_async_output_event_wait_rejects_unbounded_timeout(
    timeout_s: float,
) -> None:
    from vllm_hcu.v1 import async_utils

    class Event:
        def query(self) -> bool:
            raise AssertionError("invalid timeouts must fail before querying")

        def synchronize(self) -> None:
            raise AssertionError("invalid timeouts must never block")

    with pytest.raises(ValueError, match="timeout must be positive"):
        async_utils.wait_for_hcu_event(
            Event(),
            operation="copy sampled-token output",
            timeout_s=timeout_s,
        )


def test_hcu_async_output_uses_generic_event_and_returns_copied_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.v1 import async_utils

    output_type = async_utils.HcuAsyncOutput
    calls: list[tuple[str, object]] = []

    class Event:
        def record(self, stream: object) -> None:
            calls.append(("record", stream))

        def query(self) -> bool:
            calls.append(("query", self))
            return True

        def synchronize(self) -> None:
            calls.append(("synchronize", self))

    class CopyStream:
        def wait_stream(self, stream: object) -> None:
            calls.append(("wait_stream", stream))

    event = Event()
    main_stream = object()
    copy_stream = CopyStream()
    monkeypatch.setattr(async_utils.torch, "Event", lambda: event)
    monkeypatch.setattr(
        async_utils.torch.cuda,
        "Event",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("HCU async output must not create a CUDA-only event")
        ),
    )
    monkeypatch.setattr(
        async_utils,
        "stream",
        lambda to_stream, from_stream: contextlib.nullcontext(),
    )

    model_runner_output = SimpleNamespace(
        req_ids=["request-0"],
        sampled_token_ids=None,
        prompt_logprobs_dict={},
    )
    sampler_output = SimpleNamespace(
        sampled_token_ids=torch.tensor([[17, 23]], dtype=torch.int64),
        logprobs_tensors=None,
        num_nans=None,
    )
    output = output_type(
        model_runner_output=model_runner_output,
        sampler_output=sampler_output,
        num_sampled_tokens=torch.tensor([1], dtype=torch.int64),
        main_stream=main_stream,
        copy_stream=copy_stream,
    )

    assert output.get_output() is model_runner_output
    assert model_runner_output.sampled_token_ids == [[17]]
    assert calls == [
        ("wait_stream", main_stream),
        ("record", copy_stream),
        ("query", event),
        ("synchronize", event),
    ]


def test_hcu_mrv2_runner_routes_async_output_before_upstream_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.worker.gpu import model_runner as upstream_model_runner
    from vllm_hcu.v1 import async_utils, hcu_model_runner_v2

    observed_output_types: list[object] = []
    monkeypatch.setattr(upstream_model_runner, "AsyncOutput", object())

    def upstream_init(self, vllm_config, device) -> None:
        observed_output_types.append(upstream_model_runner.AsyncOutput)

    monkeypatch.setattr(
        hcu_model_runner_v2.GPUModelRunner,
        "__init__",
        upstream_init,
    )

    runner = hcu_model_runner_v2.HcuGPUModelRunnerV2(object(), object())

    assert runner.pcp_manager is None
    assert observed_output_types == [async_utils.HcuAsyncOutput]


def test_worker_converts_async_output_timeout_to_failure_response() -> None:
    from vllm.v1.executor.multiproc_executor import WorkerProc
    from vllm.v1.outputs import AsyncModelRunnerOutput

    responses: list[object] = []

    class ResponseQueue:
        def enqueue(self, response: object) -> None:
            responses.append(response)

    class TimedOutOutput(AsyncModelRunnerOutput):
        def get_output(self):
            raise TimeoutError("copy sampled-token output timed out")

    worker = object.__new__(WorkerProc)
    worker.worker_response_mq = ResponseQueue()

    worker.enqueue_output(TimedOutOutput())

    assert responses == [
        (
            WorkerProc.ResponseStatus.FAILURE,
            "copy sampled-token output timed out",
        )
    ]
