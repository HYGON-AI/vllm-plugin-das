# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Packed KV-cache contracts for the HCU V1 runner."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm.v1.attention.backend as target_attention_backend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    KVQuantMode,
    MLAAttentionSpec,
    TQFullAttentionSpec,
)
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler

pytestmark = pytest.mark.hcu


@pytest.fixture(scope="module")
def runner_module():
    patch = pytest.MonkeyPatch()
    # The installed target wheel predates this HCU-only metadata alias.
    patch.setattr(
        target_attention_backend,
        "CpCommonAttentionMetadata",
        object,
        raising=False,
    )
    module = importlib.import_module("vllm_hcu.v1.hcu_model_runner")
    yield module
    patch.undo()


class _Backend:
    def __init__(self, *, mla: bool = False) -> None:
        self.mla = mla
        self.shape_calls: list[tuple[int, int, int, int, str]] = []

    def get_kv_cache_shape(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str,
    ) -> tuple[int, ...]:
        self.shape_calls.append(
            (
                num_blocks,
                block_size,
                num_kv_heads,
                head_size,
                cache_dtype_str,
            )
        )
        if self.mla:
            return (num_blocks, block_size, head_size)
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    def get_kv_cache_stride_order(self) -> tuple[int, ...]:
        return (0, 1, 2) if self.mla else (0, 1, 2, 3, 4)

    def get_kv_cache_block_dim(self, *args: object, **kwargs: object) -> int:
        return 0


def _group(
    group_id: int,
    layer_names: list[str],
    spec: AttentionSpec,
    backend: _Backend,
) -> SimpleNamespace:
    return SimpleNamespace(
        kv_cache_group_id=group_id,
        layer_names=layer_names,
        kv_cache_spec=spec,
        backend=backend,
    )


def _runner(
    runner_module: Any,
    config: KVCacheConfig,
    groups: list[SimpleNamespace],
    *,
    cache_dtype: str = "fp8",
    use_mla: bool = False,
):
    runner = object.__new__(runner_module.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.runner_only_attn_layers = set()
    runner.kv_cache_config = config
    runner.attn_groups = [groups]
    runner.cache_config = SimpleNamespace(cache_dtype=cache_dtype)
    runner.vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(backend=None),
        model_config=SimpleNamespace(use_mla=use_mla),
    )
    return runner


def _plain_spec(*, quant_mode: KVQuantMode = KVQuantMode.NONE) -> AttentionSpec:
    return AttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float16,
        kv_quant_mode=quant_mode,
    )


def test_runner_propagates_backend_stride_capability_to_attention_spec(
    runner_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _plain_spec()
    config = SimpleNamespace(name="config")
    entered_configs: list[object] = []
    backend_calls: list[None] = []

    class _CurrentConfig:
        def __init__(self, current: object) -> None:
            self.current = current

        def __enter__(self) -> None:
            entered_configs.append(self.current)

        def __exit__(self, *args: object) -> None:
            return None

    class _StrideBackend:
        @staticmethod
        def indexes_kv_by_block_stride() -> bool:
            backend_calls.append(None)
            return True

    layer = SimpleNamespace(
        get_kv_cache_spec=lambda vllm_config: spec,
        get_attn_backend=lambda: _StrideBackend,
    )
    monkeypatch.setattr(runner_module, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(
        runner_module,
        "get_layers_from_vllm_config",
        lambda vllm_config, layer_type: {"layer": layer},
    )
    monkeypatch.setattr(
        runner_module,
        "set_current_vllm_config",
        _CurrentConfig,
    )
    runner = object.__new__(runner_module.GPUModelRunner)
    runner.vllm_config = config
    runner.shared_kv_cache_layers = {}

    result = runner.get_kv_cache_spec()

    assert result["layer"].indexes_kv_by_block_stride is True
    assert entered_configs == [config]
    assert backend_calls == [None]


def test_runner_allocates_one_packed_backing_and_independent_plain_tensors(
    runner_module,
) -> None:
    spec = _plain_spec()
    layers = ["packed.0", "packed.1", "plain.0", "plain.1"]
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(64, ["packed.0"], offset=0, block_stride=32),
            KVCacheTensor(64, ["packed.1"], offset=16, block_stride=32),
            KVCacheTensor(16, ["plain.0"]),
            KVCacheTensor(16, ["plain.1"]),
        ],
        kv_cache_groups=[KVCacheGroupSpec(layers, spec)],
    )
    runner = _runner(runner_module, config, [])

    tensors = runner._allocate_kv_cache_tensors(config)

    assert tensors["packed.0"].untyped_storage().data_ptr() == tensors[
        "packed.1"
    ].untyped_storage().data_ptr()
    assert tensors["plain.0"].untyped_storage().data_ptr() != tensors[
        "plain.1"
    ].untyped_storage().data_ptr()
    assert tensors["packed.0"].untyped_storage().data_ptr() != tensors[
        "plain.0"
    ].untyped_storage().data_ptr()


def test_runner_uses_per_layer_dtype_and_target_nonpacked_reshape(
    runner_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_module, "get_hcu_flash_attn_mode", lambda: "classic")
    plain_spec = _plain_spec()
    quant_spec = _plain_spec(quant_mode=KVQuantMode.FP8_PER_TENSOR)
    tq_spec = TQFullAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float16,
        tq_slot_size=8,
    )
    plain_backend = _Backend()
    quant_backend = _Backend()
    tq_backend = _Backend()
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(2 * plain_spec.page_size_bytes, ["plain"]),
            KVCacheTensor(2 * quant_spec.page_size_bytes, ["quant"]),
            KVCacheTensor(2 * tq_spec.page_size_bytes, ["tq"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["plain"], plain_spec),
            KVCacheGroupSpec(["quant"], quant_spec),
            KVCacheGroupSpec(["tq"], tq_spec),
        ],
    )
    groups = [
        _group(0, ["plain"], plain_spec, plain_backend),
        _group(1, ["quant"], quant_spec, quant_backend),
        _group(2, ["tq"], tq_spec, tq_backend),
    ]
    runner = _runner(runner_module, config, groups, cache_dtype="fp8")
    raw = runner._allocate_kv_cache_tensors(config)

    caches = runner._reshape_kv_cache_tensors(raw, [2, 2, 2])

    assert plain_backend.shape_calls[-1][-1] == "auto"
    assert quant_backend.shape_calls[-1][-1] == "fp8"
    assert tq_backend.shape_calls[-1][-1] == "fp8"
    assert {cache.shape for cache in caches.values()} == {(2, 2, 2, 1, 2)}
    assert caches["plain"].is_contiguous()
    assert caches["quant"].is_contiguous()
    assert caches["tq"].is_contiguous()


def test_runner_uses_storage_block_size_for_compressed_mla(
    runner_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_module, "get_hcu_flash_attn_mode", lambda: "classic")
    spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float16,
        compress_ratio=128,
    )
    backend = _Backend(mla=True)
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[KVCacheTensor(2 * spec.page_size_bytes, ["mla"])],
        kv_cache_groups=[KVCacheGroupSpec(["mla"], spec)],
    )
    runner = _runner(
        runner_module,
        config,
        [_group(0, ["mla"], spec, backend)],
        use_mla=True,
    )
    raw = runner._allocate_kv_cache_tensors(config)

    caches = runner._reshape_kv_cache_tensors(raw, [64])

    assert spec.storage_block_size == 1
    assert backend.shape_calls == [(2, 1, 1, 2, "auto")]
    assert caches["mla"].shape == (2, 1, 2)

    invalid = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float16,
        compress_ratio=128,
    )
    invalid_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[KVCacheTensor(0, ["invalid"])],
        kv_cache_groups=[KVCacheGroupSpec(["invalid"], invalid)],
    )
    invalid_runner = _runner(
        runner_module,
        invalid_config,
        [_group(0, ["invalid"], invalid, _Backend(mla=True))],
        use_mla=True,
    )
    with pytest.raises(ValueError, match="page size must be positive"):
        invalid_runner._reshape_kv_cache_tensors(
            invalid_runner._allocate_kv_cache_tensors(invalid_config),
            [64],
        )


def test_runner_packed_views_preserve_block_stride_and_layer_offset(
    runner_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_module, "get_hcu_flash_attn_mode", lambda: "classic")
    spec = _plain_spec()
    num_blocks = 2
    block_stride = 2 * spec.page_size_bytes
    total_size = num_blocks * block_stride
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                total_size,
                ["layer.0"],
                offset=0,
                block_stride=block_stride,
            ),
            KVCacheTensor(
                total_size,
                ["layer.1"],
                offset=spec.page_size_bytes,
                block_stride=block_stride,
            ),
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0", "layer.1"], spec)],
    )
    backend = _Backend()
    runner = _runner(
        runner_module,
        config,
        [_group(0, ["layer.0", "layer.1"], spec, backend)],
    )
    raw = runner._allocate_kv_cache_tensors(config)

    caches = runner._reshape_kv_cache_tensors(raw, [2])
    first = caches["layer.0"]
    second = caches["layer.1"]

    assert first.shape == second.shape == (2, 2, 2, 1, 2)
    assert first.stride(0) == second.stride(0) == block_stride // 2
    assert second.data_ptr() - first.data_ptr() == spec.page_size_bytes
    first[1].fill_(3)
    second[1].fill_(5)
    packed = raw["layer.0"].view(num_blocks, block_stride)
    assert torch.all(packed[1, : spec.page_size_bytes].view(torch.float16) == 3)
    assert torch.all(packed[1, spec.page_size_bytes :].view(torch.float16) == 5)
    assert torch.count_nonzero(packed[0]) == 0


def test_target_simple_offload_preserves_packed_descriptors() -> None:
    spec = _plain_spec()
    gpu_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(128, ["layer.0"], offset=0, block_stride=32),
            KVCacheTensor(128, ["layer.1"], offset=16, block_stride=32),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer.0"], spec),
            KVCacheGroupSpec(["layer.1"], spec),
        ],
    )

    cpu_config = SimpleCPUOffloadScheduler._derive_cpu_config(
        gpu_config,
        cpu_capacity_bytes=256,
    )

    assert cpu_config.num_blocks == 8
    assert [tensor.size for tensor in cpu_config.kv_cache_tensors] == [256, 256]
    assert [tensor.offset for tensor in cpu_config.kv_cache_tensors] == [0, 16]
    assert [tensor.block_stride for tensor in cpu_config.kv_cache_tensors] == [32, 32]
