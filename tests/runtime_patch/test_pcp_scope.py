# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts for the replicated-MTP PCP scope."""

from __future__ import annotations

import asyncio
import threading

import pytest
import torch
from torch._dynamo.testing import CompileCounter

from vllm_hcu.model_executor.layers.attention import pcp


def test_effective_pcp_world_size_is_fullgraph_compilable_across_scope() -> None:
    """Reading replicated-MTP state inside a full graph must stay traceable."""

    torch._dynamo.reset()
    compile_counter = CompileCounter()

    def add_pcp_width(value: torch.Tensor) -> torch.Tensor:
        return value + pcp.effective_pcp_world_size(2)

    compiled_add_pcp_width = torch.compile(
        add_pcp_width,
        backend=compile_counter,
        fullgraph=True,
    )

    value = torch.tensor(0)
    assert compiled_add_pcp_width(value).item() == 2
    assert compile_counter.frame_count == 1
    with pcp.replicated_mtp_batch_scope():
        assert compiled_add_pcp_width(value).item() == 1
    assert compiled_add_pcp_width(value).item() == 2
    assert compile_counter.frame_count >= 2
    assert compile_counter.op_count >= 2


def test_replicated_mtp_scope_restores_nested_and_exception_state() -> None:
    """Nested or failed sampling must not leak replicated-MTP state."""

    assert pcp.in_replicated_mtp_batch() is False
    assert pcp.effective_pcp_world_size(2) == 2
    with pcp.replicated_mtp_batch_scope():
        assert pcp.in_replicated_mtp_batch() is True
        assert pcp.effective_pcp_world_size(2) == 1
        with pcp.replicated_mtp_batch_scope():
            assert pcp.in_replicated_mtp_batch() is True
            assert pcp.effective_pcp_world_size(2) == 1
        assert pcp.in_replicated_mtp_batch() is True

    with pytest.raises(RuntimeError, match="sampling failed"):
        with pcp.replicated_mtp_batch_scope():
            raise RuntimeError("sampling failed")

    assert pcp.in_replicated_mtp_batch() is False
    assert pcp.effective_pcp_world_size(2) == 2


def test_compiled_replicated_mtp_state_is_thread_local() -> None:
    """A sampling scope in one worker thread must not affect another."""

    torch._dynamo.reset()
    compile_counter = CompileCounter()

    def add_pcp_width(value: torch.Tensor) -> torch.Tensor:
        return value + pcp.effective_pcp_world_size(2)

    compiled_add_pcp_width = torch.compile(
        add_pcp_width,
        backend=compile_counter,
        fullgraph=True,
    )
    value = torch.tensor(0)
    assert compiled_add_pcp_width(value).item() == 2
    assert compile_counter.frame_count == 1
    results: list[int] = []

    def run_in_worker_thread() -> None:
        results.append(compiled_add_pcp_width(value).item())
        with pcp.replicated_mtp_batch_scope():
            results.append(compiled_add_pcp_width(value).item())
        results.append(compiled_add_pcp_width(value).item())

    with pcp.replicated_mtp_batch_scope():
        assert compiled_add_pcp_width(value).item() == 1
        worker = threading.Thread(target=run_in_worker_thread)
        worker.start()
        worker.join()
        assert compiled_add_pcp_width(value).item() == 1

    assert results == [2, 1, 2]
    assert compiled_add_pcp_width(value).item() == 2
    assert compile_counter.frame_count >= 2
    assert compile_counter.op_count >= 2


def test_eager_replicated_mtp_state_remains_task_local() -> None:
    """Keeping graph-safe state must not weaken eager asyncio isolation."""

    async def exercise_overlapping_tasks() -> tuple[list[int], list[int]]:
        entered_scope = asyncio.Event()
        release_scope = asyncio.Event()

        async def scoped_task() -> list[int]:
            with pcp.replicated_mtp_batch_scope():
                before = pcp.effective_pcp_world_size(2)
                entered_scope.set()
                await release_scope.wait()
                after = pcp.effective_pcp_world_size(2)
                return [before, after]

        async def normal_task() -> list[int]:
            await entered_scope.wait()
            values = [pcp.effective_pcp_world_size(2)]
            release_scope.set()
            values.append(pcp.effective_pcp_world_size(2))
            return values

        return await asyncio.gather(scoped_task(), normal_task())

    scoped_values, normal_values = asyncio.run(exercise_overlapping_tasks())

    assert scoped_values == [1, 1]
    assert normal_values == [2, 2]
    assert pcp.effective_pcp_world_size(2) == 2
