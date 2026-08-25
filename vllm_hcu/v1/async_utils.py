# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU-specific asynchronous output helpers for Model Runner V2."""

from __future__ import annotations

import time
from typing import Protocol

import torch

from vllm import envs
from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import (
    AsyncOutput,
    async_copy_to_np,
    stream,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput


class _QueryEvent(Protocol):
    def query(self) -> bool: ...

    def synchronize(self) -> None: ...


def wait_for_hcu_event(
    event: _QueryEvent,
    *,
    operation: str,
    timeout_s: float | None = None,
) -> None:
    """Wait for an HCU event without allowing an infinite worker stall."""

    if timeout_s is None:
        timeout_s = float(envs.VLLM_ENGINE_ITERATION_TIMEOUT_S)
    if timeout_s <= 0:
        raise ValueError(
            "HCU asynchronous output event timeout must be positive, "
            f"got {timeout_s:g}s."
        )

    deadline = time.monotonic() + timeout_s
    sleep_s = 0.0001
    while not event.query():
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise TimeoutError(
                f"HCU event timed out after {timeout_s:g}s while attempting to "
                f"{operation}."
            )
        time.sleep(min(sleep_s, remaining_s))
        sleep_s = min(sleep_s * 2, 0.01)


class HcuAsyncOutput(AsyncOutput):
    """MRV2 async output using the accelerator-generic HCU event API."""

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
    ) -> None:
        # Retain the GPU tensors while their non-blocking copies run on the
        # dedicated output stream, matching upstream AsyncOutput's lifetime.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.copy_event = torch.Event()

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)
            self.sampled_token_ids = async_copy_to_np(
                sampler_output.sampled_token_ids
            )
            self.logprobs_tensors: LogprobsTensors | None = None
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = (
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
            self.num_nans = None
            if sampler_output.num_nans is not None:
                self.num_nans = async_copy_to_np(sampler_output.num_nans)
            self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)
            self.prompt_logprobs_dict = {
                key: value.to_cpu_nonblocking() if value is not None else None
                for key, value in model_runner_output.prompt_logprobs_dict.items()
            }
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        wait_for_hcu_event(
            self.copy_event,
            operation="copy sampled-token output",
        )
        return super().get_output()


__all__ = ["HcuAsyncOutput", "wait_for_hcu_event"]
