# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts for the packed KV helpers used by the HCU MRV2 adapter."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    AttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor,
    KVQuantMode,
)
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler

from vllm_hcu.v1.dsa_cache_layer_split import make_packed_kv_tensor
from vllm_hcu.v1.kv_cache import (
    _allocate_modern_raw_kv_cache, _reshape_modern_packed_cache,
)

# These methods operate on real CPU tensors. Retain the registered hardware
# tier while testing the maintained adapter, without importing the old MRV1.
pytestmark = pytest.mark.hcu


class _Backend:
    def __init__(self, *, mla=False, stride_order=None):
        self.mla = mla
        self.stride_order = stride_order
        self.shape_calls = []

    def get_kv_cache_shape(self, num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str):
        self.shape_calls.append((num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str))
        if self.mla:
            return (num_blocks, block_size, head_size)
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    def get_kv_cache_stride_order(self):
        return self.stride_order or ((0, 1, 2) if self.mla else (0, 1, 2, 3, 4))


def _plain_spec(*, quant_mode=KVQuantMode.NONE):
    return AttentionSpec(
        block_size=2, num_kv_heads=1, head_size=2,
        dtype=torch.float16, kv_quant_mode=quant_mode,
    )


def _packed_tensor(size, name, *, page_size, offset, block_stride, backing_id=("test-packed",)):
    return make_packed_kv_tensor(
        KVCacheTensor, size=size, layer_names=[name], page_size=page_size,
        offset=offset, block_stride=block_stride, backing_id=backing_id,
    )


def test_packed_cache_observes_backend_physical_stride_order():
    spec = AttentionSpec(block_size=2, num_kv_heads=3, head_size=4, dtype=torch.float16)
    descriptor = _packed_tensor(256, "layer", page_size=spec.page_size_bytes, offset=0, block_stride=128)
    raw, _ = _allocate_modern_raw_kv_cache(SimpleNamespace(kv_cache_tensors=[descriptor]), torch.device("cpu"))
    backend = _Backend(stride_order=(0, 1, 3, 2, 4))

    view = _reshape_modern_packed_cache(raw["layer"], descriptor, spec, backend, 2, "auto")

    assert view.shape == (2, 2, 2, 3, 4)
    assert view.stride() == (64, 24, 4, 8, 1)
    view[1, 0, 1, 2, 3] = 7
    flat = raw["layer"].view(torch.float16)
    assert flat[64 + 4 + 16 + 3] == 7
    assert torch.count_nonzero(flat[:64]) == 0


def test_runner_allocates_one_packed_backing_and_independent_plain_tensors():
    spec = _plain_spec()
    config = SimpleNamespace(kv_cache_tensors=[
        _packed_tensor(64, "packed.0", page_size=spec.page_size_bytes, offset=0, block_stride=32),
        _packed_tensor(64, "packed.1", page_size=spec.page_size_bytes, offset=16, block_stride=32),
        _packed_tensor(16, "plain.0", page_size=16, offset=0, block_stride=0, backing_id=None),
        _packed_tensor(16, "plain.1", page_size=16, offset=0, block_stride=0, backing_id=None),
    ])

    tensors, _ = _allocate_modern_raw_kv_cache(config, torch.device("cpu"))

    pointer = lambda name: tensors[name].untyped_storage().data_ptr()
    assert pointer("packed.0") == pointer("packed.1")
    assert pointer("plain.0") != pointer("plain.1")
    assert pointer("packed.0") != pointer("plain.0")


def test_modern_allocator_uses_explicit_pool_identity():
    spec = _plain_spec()
    descriptors = [
        _packed_tensor(64, "shared.0", page_size=spec.page_size_bytes, offset=0, block_stride=32, backing_id=("a",)),
        _packed_tensor(64, "shared.1", page_size=spec.page_size_bytes, offset=16, block_stride=32, backing_id=("a",)),
        _packed_tensor(64, "independent", page_size=spec.page_size_bytes, offset=0, block_stride=32, backing_id=("b",)),
    ]
    raw, _ = _allocate_modern_raw_kv_cache(SimpleNamespace(kv_cache_tensors=descriptors), torch.device("cpu"))
    assert raw["shared.0"].untyped_storage().data_ptr() == raw["shared.1"].untyped_storage().data_ptr()
    assert raw["shared.0"].untyped_storage().data_ptr() != raw["independent"].untyped_storage().data_ptr()


def test_modern_allocator_rejects_ambiguous_legacy_packed_pools():
    spec = _plain_spec()
    descriptors = [
        _packed_tensor(64, name, page_size=spec.page_size_bytes, offset=0, block_stride=32, backing_id=None)
        for name in ("a", "b")
    ]
    with pytest.raises(ValueError, match="explicit backing_id"):
        _allocate_modern_raw_kv_cache(SimpleNamespace(kv_cache_tensors=descriptors), torch.device("cpu"))


@pytest.mark.parametrize("quant_mode,expected", [(KVQuantMode.NONE, "auto"), (KVQuantMode.FP8_PER_TENSOR, "fp8")])
def test_packed_cache_uses_per_layer_cache_dtype(quant_mode, expected):
    spec = _plain_spec(quant_mode=quant_mode)
    descriptor = _packed_tensor(64, "layer", page_size=spec.page_size_bytes, offset=0, block_stride=32)
    backend = _Backend()
    raw, _ = _allocate_modern_raw_kv_cache(SimpleNamespace(kv_cache_tensors=[descriptor]), torch.device("cpu"))

    view = _reshape_modern_packed_cache(raw["layer"], descriptor, spec, backend, 2, "fp8")

    assert backend.shape_calls == [(2, 2, 1, 2, expected)]
    assert view.dtype == torch.float16
    assert view.shape == (2, 2, 2, 1, 2)


def test_packed_cache_uses_storage_block_size_for_compressed_mla():
    # The adapter consumes this spec protocol across target versions; older
    # target dataclasses do not expose compress_ratio/storage_block_size.
    spec = SimpleNamespace(
        block_size=128, storage_block_size=1, num_kv_heads=1, head_size=2,
        dtype=torch.float16, kv_quant_mode=KVQuantMode.NONE, page_size_bytes=4,
    )
    descriptor = _packed_tensor(2 * spec.page_size_bytes, "mla", page_size=spec.page_size_bytes, offset=0, block_stride=spec.page_size_bytes)
    backend = _Backend(mla=True)
    raw, _ = _allocate_modern_raw_kv_cache(SimpleNamespace(kv_cache_tensors=[descriptor]), torch.device("cpu"))

    view = _reshape_modern_packed_cache(raw["mla"], descriptor, spec, backend, 2, "fp8")

    assert spec.storage_block_size == 1
    assert backend.shape_calls == [(2, 1, 1, 2, "auto")]
    assert view.shape == (2, 1, 2)
    invalid = SimpleNamespace(**{**vars(spec), "storage_block_size": 0})
    with pytest.raises(ValueError, match="positive"):
        _reshape_modern_packed_cache(raw["mla"], descriptor, invalid, backend, 2, "fp8")


def test_runner_packed_views_preserve_block_stride_and_layer_offset():
    spec = _plain_spec()
    stride = 2 * spec.page_size_bytes
    descriptors = [
        _packed_tensor(2 * stride, name, page_size=spec.page_size_bytes, offset=offset, block_stride=stride)
        for name, offset in (("layer.0", 0), ("layer.1", spec.page_size_bytes))
    ]
    raw, by_layer = _allocate_modern_raw_kv_cache(SimpleNamespace(kv_cache_tensors=descriptors), torch.device("cpu"))
    caches = {
        name: _reshape_modern_packed_cache(raw[name], by_layer[name], spec, _Backend(), 2, "auto")
        for name in raw
    }
    first, second = caches["layer.0"], caches["layer.1"]
    assert first.shape == second.shape == (2, 2, 2, 1, 2)
    assert first.stride(0) == second.stride(0) == stride // 2
    assert second.data_ptr() - first.data_ptr() == spec.page_size_bytes
    first[1].fill_(3)
    second[1].fill_(5)
    packed = raw["layer.0"].view(2, stride)
    assert torch.all(packed[1, :spec.page_size_bytes].view(torch.float16) == 3)
    assert torch.all(packed[1, spec.page_size_bytes:].view(torch.float16) == 5)
    assert torch.count_nonzero(packed[0]) == 0


def test_target_simple_offload_preserves_packed_descriptors():
    from vllm_hcu.patch.platform.framework_opt import patch_packed_kv_offload

    patch_packed_kv_offload.apply_to_module(importlib.import_module("vllm.v1.simple_kv_offload.manager"))
    spec = _plain_spec()
    config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            _packed_tensor(128, "a", page_size=spec.page_size_bytes, offset=0, block_stride=32, backing_id=("offload",)),
            _packed_tensor(128, "b", page_size=spec.page_size_bytes, offset=16, block_stride=32, backing_id=("offload",)),
        ],
        kv_cache_groups=[KVCacheGroupSpec(["a"], spec), KVCacheGroupSpec(["b"], spec)],
    )
    cpu = SimpleCPUOffloadScheduler._derive_cpu_config(config, cpu_capacity_bytes=256)
    assert cpu.num_blocks == 8
    assert [tensor.size for tensor in cpu.kv_cache_tensors] == [256, 256]
    assert [tensor.offset for tensor in cpu.kv_cache_tensors] == [0, 16]
    assert [tensor.block_stride for tensor in cpu.kv_cache_tensors] == [32, 32]
