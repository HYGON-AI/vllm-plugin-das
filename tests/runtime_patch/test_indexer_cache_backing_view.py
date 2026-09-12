# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts for sparse-indexer KV-cache backing views."""

import torch

from vllm_hcu.v1.attention.ops.rocm_aiter_mla_sparse import (
    _indexer_cache_as_hipc_view,
)


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
