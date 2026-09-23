# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts for the replicated-MTP PCP scope."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
import torch
from torch._dynamo.testing import CompileCounter

from vllm_hcu.model_executor.layers.attention import pcp


class _FakePCPGroup:
    world_size = 2
    rank_in_group = 1

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        assert dim == 0
        return torch.cat((tensor, tensor + 100), dim=dim)


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


def test_generic_cache_gather_preserves_decode_and_gathers_prefill(
    monkeypatch,
) -> None:
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: _FakePCPGroup())
    metadata = SimpleNamespace(
        pcp_world_size=2,
        num_decode_tokens=1,
        pcp_has_global_prefill=True,
    )
    values = torch.tensor([[1], [2], [3]], dtype=torch.int64)
    positions = torch.tensor([10, 11, 12], dtype=torch.int64)
    slots = torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int64)

    (gathered_values, gathered_positions), gathered_slots = (
        pcp.maybe_gather_cache_inputs(
            (values, positions),
            slots,
            metadata,
        )
    )

    assert gathered_values.flatten().tolist() == [1, 2, 3, 102, 103]
    assert gathered_positions.tolist() == [10, 11, 12, 111, 112]
    assert gathered_slots.tolist() == [10, 21, 22, 121, 122]
    assert pcp.local_pcp_slot_mapping(slots, 3, metadata).tolist() == [20, 21, 22]


def test_generic_cache_inputs_trim_decode_only_padding() -> None:
    metadata = SimpleNamespace(
        pcp_world_size=1,
        num_actual_tokens=2,
        pcp_has_global_prefill=False,
    )
    values = torch.arange(4)
    slots = torch.tensor([7, 8, -1, -1], dtype=torch.int64)

    (trimmed,), trimmed_slots = pcp.maybe_gather_cache_inputs(
        (values,),
        slots,
        metadata,
    )

    assert trimmed.tolist() == [0, 1]
    assert trimmed_slots.tolist() == [7, 8]
    assert pcp.local_pcp_slot_mapping(slots, 4, metadata).tolist() == [7, 8]
