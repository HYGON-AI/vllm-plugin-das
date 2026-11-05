# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check compact allocation and lifetime without reserving a physical GPU pool."""

import inspect
import weakref
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.v1.attention.ops import dsv4_compact_indexer as compact
from vllm_hcu.v1.attention.ops import rocm_aiter_mla_sparse as sparse


class _RecordingKernel:
    def __init__(self, kernel):
        names = getattr(kernel, "arg_names", None)
        self.names = names if names is not None else tuple(inspect.signature(kernel).parameters)
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            # Recording launches must not itself keep retired buffers alive.
            self.calls.append({name: value for name, value in zip(self.names, args)
                               if not isinstance(value, torch.Tensor)})
        return launch


@pytest.fixture(params=["runtime", "function"])
def workspace_runtime(monkeypatch, request):
    monkeypatch.setattr(compact, "_WORKSPACES", {})
    # vLLM supplies a minimal Triton stub when no accelerator is available.
    # Allocation tests need only this host-side integer helper from Triton.
    if not hasattr(compact.triton, "next_power_of_2"):
        monkeypatch.setattr(compact.triton, "next_power_of_2",
                            lambda value: 1 << (int(value) - 1).bit_length(),
                            raising=False)
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda device: SimpleNamespace(cuda_stream=7))
    capture = SimpleNamespace(active=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capture.active)
    kernels = []
    for name in ("_stage_pages", "_mask_missing"):
        kernel = getattr(compact, name)
        if request.param == "function":
            kernel = getattr(kernel, "fn", kernel)
        recording = _RecordingKernel(kernel)
        monkeypatch.setattr(compact, name, recording)
        kernels.append(recording)
    shapes = []

    def logits(q, cache, weights, lengths, table, **kwargs):
        shapes.append(cache.shape)
        return torch.empty((q.shape[0], kwargs["max_context_len"]), device="meta")

    monkeypatch.setattr(sparse, "_get_lightop_attention",
                        lambda: SimpleNamespace(paged_mqa_logits=logits))

    def run(rows, pages=131072, width=2048):
        # Meta tensors exercise the real allocator and shape logic on CPU;
        # native kernels and CUDA state queries are the substituted boundaries.
        q = torch.empty((rows, 1, 64, 128), device="meta")
        cache = torch.empty((pages, 64, 1, 132), device="meta", dtype=torch.uint8)
        weights = torch.empty((rows, 64), device="meta")
        lengths = torch.empty(rows, device="meta", dtype=torch.int32)
        table = torch.empty((rows, width), device="meta", dtype=torch.int32)
        compact.compact_indexer_logits(q, cache, weights, lengths, table, 64)
        state, = compact._WORKSPACES.values()
        return state

    return SimpleNamespace(run=run, capture=capture, shapes=shapes, stage=kernels[0])


@pytest.mark.parametrize("pages,rows,width", [
    (131072, 6, 1),  # Review case: a 1056-MiB physical pool, six live pages.
    (10, 3, 3),     # Power-of-two growth must not exceed the physical pool.
    (10, 4, 3),     # Batch slots cover the pool: use physical page IDs.
])
def test_compact_workspace_is_bounded_by_batch_and_physical_pool(
    workspace_runtime, pages, rows, width,
):
    runtime = workspace_runtime
    state = runtime.run(rows, pages, width)
    capacity = state["page_capacity"]
    assert min(rows * width, pages) <= capacity
    assert capacity <= pages
    assert capacity <= compact.triton.next_power_of_2(rows * width)
    assert runtime.shapes[0] == (capacity + 1, 64, 1, 132)
    if pages == 131072:
        assert state["compact"].numel() * state["compact"].element_size() < 1024 * 1024
    assert runtime.stage.calls[0]["COMPACT_PAGES"] == capacity
    assert runtime.stage.calls[0]["USE_PHYSICAL"] == (rows * width >= pages)


def test_eager_growth_releases_all_retired_buffers(workspace_runtime):
    retired = []
    for rows in (6, 12, 24, 48):
        state = workspace_runtime.run(rows)
        assert state["old"] == []
        assert all(ref() is None for ref in retired)
        retired = [weakref.ref(state[name]) for name in ("compact", "mapped", "new_lengths")]
    assert state["page_capacity"] == 131072


def test_growth_retains_only_captured_workspaces(workspace_runtime):
    runtime = workspace_runtime
    state = runtime.run(6)  # Eager warmup allocates the buffers used by capture.
    runtime.capture.active = True
    runtime.run(6)
    captured = [weakref.ref(state[name]) for name in ("compact", "mapped", "new_lengths")]
    runtime.capture.active = False
    runtime.run(12)
    assert len(state["old"]) == 1
    eager = [weakref.ref(state[name]) for name in ("compact", "mapped", "new_lengths")]
    runtime.run(24)
    assert all(ref() is None for ref in eager)
    assert all(ref() is not None for ref in captured)
    assert len(state["old"]) == 1


def test_eager_growth_releases_old_storage_before_allocating(workspace_runtime, monkeypatch):
    state = workspace_runtime.run(24)
    old = weakref.ref(state["compact"])
    empty = torch.empty

    def allocate(*args, **kwargs):
        if args and args[0] == (131073, 64, 1, 132):
            assert old() is None, "old compact still live when allocating its replacement"
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    workspace_runtime.run(48)
