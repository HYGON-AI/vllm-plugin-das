# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check compact workspace allocation without reserving a physical GPU pool."""

from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.v1.attention.ops import dsv4_compact_indexer as compact
from vllm_hcu.v1.attention.ops import rocm_aiter_mla_sparse as sparse


class _RecordingKernel:
    def __init__(self, names):
        self.names = names
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append(dict(zip(self.names, args)))
        return launch


@pytest.mark.parametrize("pages,rows,width", [
    (131072, 6, 1),  # Review case: a 1056-MiB physical pool, six live pages.
    (10, 3, 3),     # Power-of-two growth must not exceed the physical pool.
    (10, 4, 3),     # Batch slots cover the pool: use physical page IDs.
])
def test_compact_workspace_is_bounded_by_batch_and_physical_pool(
    monkeypatch, pages, rows, width,
):
    monkeypatch.setattr(compact, "_WORKSPACES", {})
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda device: SimpleNamespace(cuda_stream=7))
    stage = _RecordingKernel(compact._stage_pages.arg_names)
    mask = _RecordingKernel(compact._mask_missing.arg_names)
    monkeypatch.setattr(compact, "_stage_pages", stage)
    monkeypatch.setattr(compact, "_mask_missing", mask)
    seen = []

    def logits(q, cache, weights, lengths, table, **kwargs):
        seen.append(cache)
        return torch.empty((q.shape[0], kwargs["max_context_len"]), device="meta")

    monkeypatch.setattr(sparse, "_get_lightop_attention",
                        lambda: SimpleNamespace(paged_mqa_logits=logits))
    # Meta tensors exercise the actual allocator and shape logic on CPU;
    # native kernels and the CUDA stream are the only substituted boundaries.
    q = torch.empty((rows, 1, 64, 128), device="meta")
    cache = torch.empty((pages, 64, 1, 132), device="meta", dtype=torch.uint8)
    weights = torch.empty((rows, 64), device="meta")
    lengths = torch.empty(rows, device="meta", dtype=torch.int32)
    table = torch.empty((rows, width), device="meta", dtype=torch.int32)
    compact.compact_indexer_logits(q, cache, weights, lengths, table, 64)

    state, = compact._WORKSPACES.values()
    capacity = state["page_capacity"]
    assert min(rows * width, pages) <= capacity
    assert capacity <= pages
    assert capacity <= compact.triton.next_power_of_2(rows * width)
    assert seen[0].shape == (capacity + 1, 64, 1, 132)
    if pages == 131072:
        assert seen[0].numel() * seen[0].element_size() < 1024 * 1024
    assert stage.calls[0]["COMPACT_PAGES"] == capacity
    assert stage.calls[0]["USE_PHYSICAL"] == (rows * width >= pages)
