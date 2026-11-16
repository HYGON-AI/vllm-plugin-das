# SPDX-License-Identifier: Apache-2.0
"""Direct contracts for the Mooncake transfer-layout adapter."""
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.platform.framework_opt import patch_kv_connector_utils as adapter


@pytest.mark.parametrize('shape,is_mla,blocks_first,split', [
    ((1, 2, 16, 1, 1), False, True, False),
    ((2, 1, 16, 1, 1), False, False, True),
    (None, True, False, False),
])
def test_transfer_layout_registration_and_idempotence(shape, is_mla, blocks_first, split):
    target = ModuleType(adapter.TARGET_MODULE)
    target.logger = SimpleNamespace(debug=lambda *args: None)

    class Topology:
        def __init__(self):
            self.is_mla = is_mla
            self.is_mamba = False
            self.tensor_shape = None
            self.attn_backends = ([] if shape is None else [SimpleNamespace(
                get_kv_cache_shape=lambda **kwargs: shape)])
            self.__post_init__()

        def __post_init__(self):
            self.base_initialized = True

    target.TransferTopology = Topology
    assert adapter.apply_to_module(target)
    assert not adapter.apply_to_module(target)
    topology = Topology()
    assert topology.base_initialized
    assert topology.is_kv_layout_blocks_first is blocks_first
    assert topology.split_k_and_v is split
    assert topology.virtually_split_kv_in_blocks is blocks_first
    cache = torch.arange(8).reshape(2, 4)
    regions = topology.get_transfer_cache_regions(cache, SimpleNamespace())
    assert (regions is cache) if split else (len(regions) == 1 and regions[0] is cache)
