# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Contracts for target-owned KV page sizing and HCU stride capabilities."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache
from vllm_hcu.model_executor.layers.kv_cache_utils import (
    has_mixed_kv_cache_block_dims,
)


@dataclass(frozen=True)
class _FakeSpec:
    num_kv_heads: int = 4
    head_size: int = 128


class _FakeBackend:
    def __init__(self, block_dim: int) -> None:
        self.block_dim = block_dim
        self.calls: list[tuple[object, ...]] = []

    def get_kv_cache_block_dim(self, *args: object, **kwargs: object) -> int:
        self.calls.append((*args, kwargs))
        return self.block_dim


@dataclass(frozen=True)
class _FakeGroup:
    kv_cache_group_id: int
    backend: _FakeBackend
    kv_cache_spec: _FakeSpec = _FakeSpec()


def test_mixed_kv_block_dim_detection_skips_runner_only_group() -> None:
    groups = [
        _FakeGroup(0, _FakeBackend(0)),
        _FakeGroup(1, _FakeBackend(1)),
        _FakeGroup(2, _FakeBackend(1)),
    ]

    assert has_mixed_kv_cache_block_dims(groups, [64, 64], "auto") is True
    assert groups[0].backend.calls == [(64, 4, 128, {"cache_dtype_str": "auto"})]
    assert groups[2].backend.calls == []


def test_uniform_kv_block_dim_detection_is_false() -> None:
    groups = [
        _FakeGroup(0, _FakeBackend(0)),
        _FakeGroup(1, _FakeBackend(0)),
    ]

    assert has_mixed_kv_cache_block_dims(groups, [64, 64], "auto") is False


def test_target_kv_page_policy_uses_backend_capability_for_padding() -> None:
    def spec(block_size: int, *, indexes: bool) -> AttentionSpec:
        return AttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float16,
            indexes_kv_by_block_stride=indexes,
        )

    padded = unify_kv_cache_spec_page_size(
        {"small": spec(3, indexes=True), "large": spec(5, indexes=True)}
    )
    assert padded["small"].block_size == 3
    assert padded["small"].page_size_padded == 20
    assert padded["large"].page_size_bytes == 20

    with pytest.raises(NotImplementedError, match="cannot be padded"):
        unify_kv_cache_spec_page_size(
            {"small": spec(3, indexes=False), "large": spec(5, indexes=True)}
        )


def test_target_padded_attention_view_uses_physical_page_stride() -> None:
    spec = AttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float16,
        page_size_padded=32,
        indexes_kv_by_block_stride=True,
    )
    raw = torch.zeros(64, dtype=torch.int8)
    shape = (2, 2, 2, 1, 2)

    cache = _reshape_attention_kv_cache(
        raw,
        spec,
        shape,
        (0, 1, 2, 3, 4),
        2,
        None,
    )

    assert cache.shape == shape
    assert cache.stride(0) == 16
    cache[1].fill_(7)
    assert torch.count_nonzero(raw[:32]) == 0
    assert torch.count_nonzero(raw[32:48]) > 0
    assert torch.count_nonzero(raw[48:]) == 0


def test_hcu_flashmla_declares_block_stride_indexing_contract() -> None:
    from vllm_hcu.v1.attention.backends.mla.flashmla import HcuFlashMLABackend

    assert HcuFlashMLABackend.get_kv_cache_stride_order() == (0, 1, 2)
    assert HcuFlashMLABackend.get_kv_cache_stride_order(
        include_num_layers_dimension=True
    ) == (1, 0, 2, 3)
    assert HcuFlashMLABackend.indexes_kv_by_block_stride() is True
