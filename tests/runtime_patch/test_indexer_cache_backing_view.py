# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts for sparse-indexer KV-cache backing views."""

import torch

from vllm_hcu.v1.attention.ops import rocm_aiter_mla_sparse as sparse
from vllm_hcu.v1.attention.ops.rocm_aiter_mla_sparse import (
    _indexer_cache_as_bf16_nhd_view,
    _indexer_cache_as_hipc_view,
)


class _FakeTritonKernel:
    def __init__(self):
        self.grid = None
        self.args = None

    def __getitem__(self, grid):
        self.grid = grid

        def launch(*args):
            self.args = args

        return launch


def test_collapsed_indexer_cache_restores_physical_page_axis():
    backing = torch.empty(3, 64, 132, dtype=torch.uint8)
    collapsed = backing.as_strided(
        size=(3, 1, 132),
        stride=(64 * 132, 132, 1),
    )

    restored = _indexer_cache_as_hipc_view(collapsed)

    assert restored.shape == (3, 64, 132)
    assert restored.stride() == (64 * 132, 132, 1)
    assert restored.data_ptr() == collapsed.data_ptr()
    assert collapsed.shape == (3, 1, 132)


def test_regular_indexer_cache_is_not_reinterpreted():
    cache = torch.empty(3, 64, 132, dtype=torch.uint8)

    assert _indexer_cache_as_hipc_view(cache) is cache


def test_noncollapsed_padded_indexer_cache_is_not_reinterpreted():
    backing = torch.empty(3, 33, 132, dtype=torch.uint8)
    padded = backing.as_strided(
        size=(3, 32, 132),
        stride=(33 * 132, 132, 1),
    )

    assert _indexer_cache_as_hipc_view(padded) is padded


def test_logical_lbhnc_indexer_cache_has_zero_copy_bf16_nhd_view():
    cache = torch.empty(3, 1, 64, 128, dtype=torch.bfloat16)

    normalized = _indexer_cache_as_bf16_nhd_view(cache, head_dim=128)

    assert normalized.shape == (3, 64, 128)
    assert normalized.stride() == (64 * 128, 128, 1)
    assert normalized.data_ptr() == cache.data_ptr()


def test_logical_lbnhc_indexer_cache_has_zero_copy_bf16_nhd_view():
    backing = torch.empty(3 * 64 * 128, dtype=torch.bfloat16)
    cache = backing.as_strided(
        size=(3, 1, 64, 128),
        stride=(64 * 128, 128, 128, 1),
    )

    normalized = _indexer_cache_as_bf16_nhd_view(cache, head_dim=128)

    assert normalized.shape == (3, 64, 128)
    assert normalized.stride() == (64 * 128, 128, 1)
    assert normalized.data_ptr() == cache.data_ptr()


def test_collapsed_bf16_indexer_cache_restores_page_before_normalizing():
    backing = torch.empty(3, 64, 128, dtype=torch.bfloat16)
    collapsed = backing.as_strided(
        size=(3, 1, 128),
        stride=(64 * 128, 128, 1),
    )

    normalized = _indexer_cache_as_bf16_nhd_view(collapsed, head_dim=128)

    assert normalized.shape == (3, 64, 128)
    assert normalized.stride() == (64 * 128, 128, 1)
    assert normalized.data_ptr() == collapsed.data_ptr()


def test_bf16_writer_passes_logical_4d_cache_as_nhd_to_kernel(monkeypatch):
    kernel = _FakeTritonKernel()
    monkeypatch.setattr(sparse, "_indexer_k_bf16_cache_kernel", kernel)
    cache = torch.empty(3, 1, 64, 128, dtype=torch.bfloat16)

    sparse.indexer_k_bf16_cache_triton(
        torch.empty(2, 128, dtype=torch.bfloat16),
        cache,
        torch.tensor([0, 65], dtype=torch.int32),
    )

    assert kernel.grid == (2,)
    assert kernel.args[1].shape == (3, 64 * 128)
    assert kernel.args[1].stride(0) == 64 * 128
    assert kernel.args[4:7] == (64, 2, 128)


def test_bf16_gather_passes_logical_4d_cache_as_nhd_to_kernel(monkeypatch):
    kernel = _FakeTritonKernel()
    monkeypatch.setattr(
        sparse, "_cp_gather_indexer_k_bf16_cache_kernel", kernel
    )
    cache = torch.empty(3, 1, 64, 128, dtype=torch.bfloat16)

    sparse.cp_gather_indexer_k_bf16_cache_triton(
        cache,
        torch.empty(2, 128, dtype=torch.bfloat16),
        torch.zeros(1, 2, dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
    )

    assert kernel.grid == (2,)
    assert kernel.args[0].shape == (3, 64 * 128)
    assert kernel.args[0].stride(0) == 64 * 128
    assert kernel.args[4:10] == (64, 1, 2, 64 * 128, 128, 2)
