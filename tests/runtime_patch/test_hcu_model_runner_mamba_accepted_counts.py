# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU regression for async aligned-Mamba accepted-count staging.

The test executes the runner producer, previous-position builder, and
accepted-count consumer from the source AST. It controls completion of the
D2H copy without importing vLLM or exercising HCU/GPU integration.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm_hcu"
    / "v1"
    / "hcu_model_runner.py"
)


def _runner_nodes() -> tuple[ast.FunctionDef, ast.FunctionDef, ast.If]:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GPUModelRunner"
    )
    methods = {
        node.name: node
        for node in runner.body
        if isinstance(node, ast.FunctionDef)
    }
    producer = methods["_update_states_after_model_execute"]
    compute_prev_positions = methods["_compute_prev_positions"]
    prepare_inputs = methods["_prepare_inputs"]
    accepted_count_sync = next(
        node
        for node in prepare_inputs.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self.num_accepted_tokens_event is not None"
    )
    return producer, compute_prev_positions, accepted_count_sync


class _PendingD2H:
    def __init__(self) -> None:
        self.target: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.completed = False

    def schedule(self, target: torch.Tensor, values: torch.Tensor) -> None:
        self.target = target
        self.values = values.clone()
        self.completed = False

    def complete(self) -> None:
        assert self.target is not None and self.values is not None
        self.target[: self.values.numel()].copy_(self.values)
        self.completed = True


class _Event:
    def __init__(self, pending_d2h: _PendingD2H) -> None:
        self.pending_d2h = pending_d2h
        self.recorded = False
        self.synchronize_calls = 0

    def record(self) -> None:
        self.recorded = True

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        if not self.pending_d2h.completed and self.pending_d2h.target is not None:
            self.pending_d2h.complete()


class _CpuGpuBuffer:
    def __init__(self, size: int) -> None:
        self.cpu = torch.ones(size, dtype=torch.int32)
        self.np = self.cpu.numpy()
        self.gpu = torch.ones(size, dtype=torch.int32)

    def copy_to_gpu(self, num_reqs: int | None = None) -> None:
        count = self.cpu.numel() if num_reqs is None else num_reqs
        self.gpu[:count].copy_(self.cpu[:count])


class _InputBatch:
    def __init__(self) -> None:
        self.req_ids = ["A", "B"]
        self.prev_req_id_to_index = {"A": 0, "B": 1}
        self.num_accepted_tokens_cpu_tensor = torch.ones(4, dtype=torch.int32)
        self.num_accepted_tokens_cpu = self.num_accepted_tokens_cpu_tensor.numpy()


class _Runner:
    def __init__(
        self,
        pending_d2h: _PendingD2H,
        *,
        async_scheduling: bool = True,
        mamba_cache_mode: str = "align",
    ) -> None:
        self.speculative_config = object()
        self.model_config = SimpleNamespace(is_hybrid=True)
        self.cache_config = SimpleNamespace(mamba_cache_mode=mamba_cache_mode)
        self.kv_cache_config = object()
        self.compilation_config = SimpleNamespace(static_forward_context={})
        self.input_batch = _InputBatch()
        self.num_accepted_tokens = _CpuGpuBuffer(4)
        self.num_accepted_tokens_event = _Event(pending_d2h)
        self.use_async_scheduling = async_scheduling
        self.prev_positions = _CpuGpuBuffer(4)
        self.model = SimpleNamespace(get_mamba_state_copy_func=lambda: ())
        self._mamba_bufs = object()

    def _get_mamba_bufs(self) -> object:
        return self._mamba_bufs


def _production_functions(runner: _Runner, pending_d2h: _PendingD2H):
    producer_node, prev_positions_node, consumer_node = _runner_nodes()

    def postprocess_mamba_align_gpu(**kwargs: object) -> None:
        target = kwargs["num_accepted_tokens_cpu_tensor"]
        values = kwargs["num_accepted_tokens_gpu"]
        count = kwargs["num_reqs"]
        assert isinstance(target, torch.Tensor)
        assert isinstance(values, torch.Tensor)
        assert isinstance(count, int)
        pending_d2h.schedule(target, values[:count])

    namespace = {
        "torch": torch,
        "mamba_utils": SimpleNamespace(
            postprocess_mamba_align_gpu=postprocess_mamba_align_gpu
        ),
    }
    future_annotations = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.Module(
        body=[future_annotations, producer_node, prev_positions_node],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)

    args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg="self"), ast.arg(arg="num_reqs"),
              ast.arg(arg="prev_req_id_to_index")],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )
    consumer_function = ast.FunctionDef(
        name="_consume_accepted_counts",
        args=args,
        body=[consumer_node],
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    consumer_module = ast.Module(
        body=[future_annotations, consumer_function], type_ignores=[]
    )
    ast.fix_missing_locations(consumer_module)
    consumer_namespace = {"np": np}
    exec(compile(consumer_module, str(SOURCE_PATH), "exec"), consumer_namespace)

    return (
        namespace["_update_states_after_model_execute"],
        namespace["_compute_prev_positions"],
        consumer_namespace["_consume_accepted_counts"],
    )


def _run_producer(runner: _Runner, pending_d2h: _PendingD2H):
    producer, compute_prev_positions, consumer = _production_functions(
        runner, pending_d2h
    )
    producer(
        runner,
        torch.tensor([[11, 12, 13, 14], [21, -1, -1, -1]]),
        None,
    )
    assert runner.num_accepted_tokens_event.recorded
    return compute_prev_positions, consumer


def _run_consumer(
    runner: _Runner, compute_prev_positions, consumer, num_reqs: int
) -> None:
    compute_prev_positions(runner, num_reqs)
    consumer(
        runner,
        num_reqs,
        runner.input_batch.prev_req_id_to_index,
    )


@pytest.mark.parametrize("completion_timing", ["before", "during", "after"])
def test_async_align_counts_use_old_batch_order_after_reorder(
    completion_timing: str,
) -> None:
    pending_d2h = _PendingD2H()
    runner = _Runner(pending_d2h)
    compute_prev_positions, consumer = _run_producer(runner, pending_d2h)

    if completion_timing == "before":
        pending_d2h.complete()

    # A exits and C reuses its row with the default accepted count of one.
    runner.input_batch.req_ids = ["C", "B"]
    runner.input_batch.num_accepted_tokens_cpu[0] = 1

    if completion_timing == "during":
        pending_d2h.complete()

    # The current batch changes from [C, B] to [B, C].
    runner.input_batch.req_ids[:] = ["B", "C"]
    runner.input_batch.num_accepted_tokens_cpu[:2] = (
        runner.input_batch.num_accepted_tokens_cpu[:2][::-1].copy()
    )

    _run_consumer(runner, compute_prev_positions, consumer, num_reqs=2)

    assert runner.prev_positions.np[:2].tolist() == [1, -1]
    assert runner.num_accepted_tokens.np[:2].tolist() == [1, 1]
    assert runner.num_accepted_tokens.gpu[:2].tolist() == [1, 1]
    assert runner.input_batch.num_accepted_tokens_cpu[:2].tolist() == [1, 1]
    assert runner.num_accepted_tokens_event.synchronize_calls == 1
    assert pending_d2h.completed


def test_async_align_producer_uses_independent_accepted_count_storage() -> None:
    pending_d2h = _PendingD2H()
    runner = _Runner(pending_d2h)
    _run_producer(runner, pending_d2h)

    assert pending_d2h.target is runner.num_accepted_tokens.cpu
    assert pending_d2h.target.data_ptr() != (
        runner.input_batch.num_accepted_tokens_cpu_tensor.data_ptr()
    )
    assert not np.shares_memory(
        pending_d2h.target.numpy(),
        runner.input_batch.num_accepted_tokens_cpu,
    )


def test_first_batch_without_previous_map_uses_default_count() -> None:
    pending_d2h = _PendingD2H()
    runner = _Runner(pending_d2h)
    _producer, compute_prev_positions, consumer = _production_functions(
        runner, pending_d2h
    )
    runner.input_batch.req_ids = ["C"]
    runner.input_batch.prev_req_id_to_index = {}

    _run_consumer(runner, compute_prev_positions, consumer, num_reqs=1)

    assert runner.prev_positions.np[:1].tolist() == [-1]
    assert runner.num_accepted_tokens.np[:1].tolist() == [1]


def test_sync_align_keeps_input_batch_as_d2h_target() -> None:
    pending_d2h = _PendingD2H()
    runner = _Runner(
        pending_d2h,
        async_scheduling=False,
        mamba_cache_mode="align",
    )
    compute_prev_positions, consumer = _run_producer(runner, pending_d2h)

    assert pending_d2h.target is runner.input_batch.num_accepted_tokens_cpu_tensor
    pending_d2h.complete()
    _run_consumer(runner, compute_prev_positions, consumer, num_reqs=2)

    assert runner.num_accepted_tokens.np[:2].tolist() == [4, 1]


def test_async_nonalign_keeps_input_batch_copy_and_consumer_source() -> None:
    pending_d2h = _PendingD2H()
    runner = _Runner(
        pending_d2h,
        async_scheduling=True,
        mamba_cache_mode="default",
    )
    compute_prev_positions, consumer = _run_producer(runner, pending_d2h)

    # The non-align producer still performs its original direct CPU copy.
    assert pending_d2h.target is None
    assert runner.input_batch.num_accepted_tokens_cpu[:2].tolist() == [4, 1]
    runner.num_accepted_tokens.np[:2] = [9, 9]

    _run_consumer(runner, compute_prev_positions, consumer, num_reqs=2)

    assert runner.num_accepted_tokens.np[:2].tolist() == [4, 1]
