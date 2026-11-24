# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Run explicitly on HCU: real Core routing kernel, doubled model/EP owners."""
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from hcu_vllm_das import static_eplb_runtime as runtime
from replay_v41_static_eplb import (
    EP_SIZE, LOGICAL, PHYSICAL, ROWS, ModelDouble, cfg, load, write_map,
)

pytestmark = pytest.mark.hcu


@pytest.fixture(autouse=True)
def require_hcu():
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip('real HCU required')


@pytest.mark.parametrize('model_type', ['hy_v4', 'hy_v4_mtp'])
@pytest.mark.parametrize('policy', ['nearest', 'locality_fair'])
@pytest.mark.parametrize('rank', [0, 1])
def test_installed_core_hashes_all_hy4_replicas(tmp_path, model_type, policy, rank):
    from vllm.model_executor.layers.fused_moe.router.base_router import (
        eplb_map_to_physical_and_record,
    )
    assert torch.version.hip and torch.cuda.is_available()
    path = write_map(tmp_path)
    model = ModelDouble(load(path))
    weights = torch.zeros(1, device='cuda')
    for layer in model.layers:
        layer.routed_experts.get_expert_weights = lambda: [weights]
    config = cfg(expert_map_path=path, static_dispatch_policy=policy)
    config.model_config.hf_config.model_type = model_type
    group = NS(device_group=NS(rank=lambda: rank, size=lambda: EP_SIZE))
    with (patch('vllm.distributed.get_ep_group', return_value=group),
          patch('vllm.distributed.get_node_count', return_value=1),
          patch('vllm.model_executor.models.interfaces.get_mixture_of_experts_model',
                return_value=model)):
        runtime.install_routing(model, config)
    ids = torch.empty((2056, 8), dtype=torch.int64, device='cuda')
    address = ids.data_ptr()
    for step in range(3):
        logical = (torch.arange(ids.numel()).reshape(ids.shape) + step) % LOGICAL
        logical[-1, -2:] = torch.tensor([-1, LOGICAL])
        ids.copy_(logical)
        for layer, row in zip(model.layers, ROWS):
            state = layer.eplb_state
            counts = state.logical_replica_count.cpu()
            assert counts.tolist() == torch.bincount(torch.tensor(row), minlength=LOGICAL).tolist()
            out = eplb_map_to_physical_and_record(ids, state.expert_load_view,
                state.logical_to_physical_map, state.logical_replica_count,
                state.should_record_tensor)
            token = torch.arange(ids.shape[0])[:, None]
            valid = (logical >= 0) & (logical < LOGICAL)
            safe = logical.clamp(0, LOGICAL - 1)
            column = ((token * 2654435769) & 0xFFFFFFFF) % counts[safe]
            expected = state.logical_to_physical_map.cpu()[safe, column]
            expected[~valid] = -1
            torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
            assert state.expert_load_view.shape == (PHYSICAL,)
            assert torch.count_nonzero(state.expert_load_view).item() == 0
        assert ids.data_ptr() == address
