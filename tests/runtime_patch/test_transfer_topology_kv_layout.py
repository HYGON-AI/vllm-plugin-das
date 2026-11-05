# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts for TransferTopology KV registration layouts."""

from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.platform.framework_opt import patch_kv_connector_utils as adapter
from vllm_hcu.patch.platform.framework_opt._common import PatchCompatibilityError


def _module() -> ModuleType:
    module = ModuleType(adapter.TARGET_MODULE)
    module.logger = SimpleNamespace(debug=lambda *args: None)

    class TransferTopology:
        def __init__(
            self, backends=(), *, tensor_shape=None, is_mla=False, is_mamba=False
        ):
            self.attn_backends = backends
            self.tensor_shape = tensor_shape
            self.is_mla = is_mla
            self.is_mamba = is_mamba
            self.initializations = 0
            self.__post_init__()

        def __post_init__(self):
            self.initializations += 1

    module.TransferTopology = TransferTopology
    return module


def _backend(shape):
    probes = []

    def get_shape(**kwargs):
        probes.append(kwargs)
        return shape

    return SimpleNamespace(get_kv_cache_shape=get_shape), probes


@pytest.fixture
def mamba_spec_type(monkeypatch: pytest.MonkeyPatch):
    interface = ModuleType("vllm.v1.kv_cache_interface")

    class MambaSpec:
        pass

    interface.MambaSpec = MambaSpec
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    return MambaSpec


@pytest.mark.parametrize(
    ("probe_shape", "cache_shape", "is_mla", "expected"),
    [
        ((2, 1, 16, 1, 1), (2, 3, 4, 2, 2), False, (False, False, False, True)),
        ((1, 2, 16, 1, 1), (3, 2, 4, 2, 2), False, (True, False, True, False)),
        ((1, 16, 1), (3, 4, 8), True, (False, False, False, False)),
    ],
)
def test_layout_controls_separate_kv_registration_regions(
    mamba_spec_type, probe_shape, cache_shape, is_mla, expected
):
    module = _module()
    backend, probes = _backend(probe_shape)
    assert adapter.apply_to_module(module)
    topology = module.TransferTopology([backend], is_mla=is_mla)
    assert topology.initializations == 1
    assert (
        topology.is_kv_layout_blocks_first,
        topology.cross_layers_blocks,
        topology.virtually_split_kv_in_blocks,
        topology.split_k_and_v,
    ) == expected
    assert probes == [
        {"num_blocks": 1, "block_size": 16, "num_kv_heads": 1, "head_size": 1}
    ]
    cache = torch.arange(math.prod(cache_shape)).reshape(cache_shape)
    regions = topology.get_transfer_cache_regions(cache, object())
    if expected[-1]:
        assert regions is cache
        assert regions[0].data_ptr() != regions[1].data_ptr()
    else:
        assert len(regions) == 1
        assert regions[0] is cache


@pytest.mark.parametrize("stride_api", ["present", "absent", "unimplemented"])
@pytest.mark.parametrize("probe_shape", [(1, 2, 16, 1, 1), (2, 1, 16, 1, 1)])
def test_cross_layer_caches_keep_one_registration_region(
    mamba_spec_type, stride_api, probe_shape
):
    module = _module()
    backend, _ = _backend(probe_shape)
    stride_calls = []
    if stride_api != "absent":
        def get_stride(*, include_num_layers_dimension):
            stride_calls.append(include_num_layers_dimension)
            if stride_api == "unimplemented":
                raise NotImplementedError
            return (0, 2, 1, 3, 4, 5)

        backend.get_kv_cache_stride_order = get_stride
    adapter.apply_to_module(module)
    cache = torch.zeros(80, 3, 2, 4, 2, 2)
    topology = module.TransferTopology([backend], tensor_shape=cache.shape)
    assert topology.cross_layers_blocks
    assert not topology.virtually_split_kv_in_blocks
    assert not topology.split_k_and_v
    regions = topology.get_transfer_cache_regions(cache, object())
    assert len(regions) == 1
    assert regions[0] is cache
    assert stride_calls == ([] if stride_api == "absent" else [True])


def test_shape_probe_skips_a_dsa_indexer_without_shape_api():
    module = _module()
    backend, probes = _backend((1, 2, 16, 1, 1))
    adapter.apply_to_module(module)
    topology = module.TransferTopology([SimpleNamespace(), backend])
    assert topology.is_kv_layout_blocks_first
    assert topology.virtually_split_kv_in_blocks
    assert len(probes) == 1


def test_mla_indexer_only_topology_does_not_require_a_shape_api(mamba_spec_type):
    module = _module()
    adapter.apply_to_module(module)
    topology = module.TransferTopology([SimpleNamespace()], is_mla=True)
    assert not topology.is_kv_layout_blocks_first
    assert not topology.cross_layers_blocks
    assert not topology.split_k_and_v
    cache = torch.zeros(3, 4, 8)
    assert topology.get_transfer_cache_regions(cache, object())[0] is cache


def test_mamba_registers_convolution_cache_without_ssm_state(mamba_spec_type):
    module = _module()
    adapter.apply_to_module(module)
    topology = module.TransferTopology(is_mamba=True)
    conv, ssm = torch.zeros(3, 4, 8), torch.ones(3, 4, 8)
    regions = topology.get_transfer_cache_regions((conv, ssm), mamba_spec_type())
    assert len(regions) == 1
    assert regions[0] is conv


def test_hybrid_attention_regions_use_a_transposed_view(mamba_spec_type):
    module = _module()
    adapter.apply_to_module(module)
    topology = module.TransferTopology(is_mamba=True)
    cache = torch.arange(48).reshape(2, 3, 8)
    regions = topology.get_transfer_cache_regions(cache, object())
    assert len(regions) == 1
    assert regions[0].shape == (3, 2, 8)
    assert regions[0].stride() == cache.transpose(0, 1).stride()
    assert regions[0].data_ptr() == cache.data_ptr()
    assert torch.equal(regions[0], cache.transpose(0, 1))


def test_repeated_application_does_not_wrap_initialization_twice():
    module = _module()
    assert adapter.apply(module)
    post_init = module.TransferTopology.__post_init__
    assert adapter.apply_to_module(module) is False
    assert module.TransferTopology.__post_init__ is post_init
    backend, _ = _backend((1, 2, 16, 1, 1))
    assert module.TransferTopology([backend]).initializations == 1


def test_existing_upstream_layout_apis_remain_intact():
    module = _module()
    topology_class = module.TransferTopology
    original_init = topology_class.__post_init__
    def original_regions(self, cache, spec):
        return ["upstream-region"]

    topology_class.get_transfer_cache_regions = original_regions
    for name in (
        "virtually_split_kv_in_blocks", "split_k_and_v",
        "is_kv_layout_blocks_first", "cross_layers_blocks",
    ):
        setattr(topology_class, name, property(lambda self: False))
    assert adapter.apply_to_module(module) is False
    assert adapter.apply_to_module(module) is False
    assert topology_class.__post_init__ is original_init
    assert topology_class.get_transfer_cache_regions is original_regions
    topology = topology_class()
    assert topology.initializations == 1
    assert topology.get_transfer_cache_regions(None, None) == ["upstream-region"]


def test_stale_marker_fails_before_mutating_topology():
    module = _module()
    original = module.TransferTopology.__post_init__
    setattr(module, adapter._MARKER, True)
    with pytest.raises(PatchCompatibilityError, match="stale"):
        adapter.apply_to_module(module)
    assert module.TransferTopology.__post_init__ is original
    assert not hasattr(module.TransferTopology, "get_transfer_cache_regions")


@pytest.mark.parametrize("broken", ["wrong_module", "missing_class", "missing_init"])
def test_incompatible_upstream_targets_fail_closed(broken):
    module = _module()
    if broken == "wrong_module":
        module.__name__ = "unexpected.module"
    elif broken == "missing_class":
        del module.TransferTopology
    else:
        del module.TransferTopology.__post_init__
    with pytest.raises(PatchCompatibilityError):
        adapter.apply_to_module(module)
