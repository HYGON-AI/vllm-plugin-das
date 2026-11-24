# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HY4 static publication retains all loaded replicas for Core's token hash.

Distributed membership and model allocation are CPU doubles. The production
publication and replica ordering execute without replacing their functions.
"""
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from hcu_vllm_das import static_eplb_runtime as runtime
from replay_v41_static_eplb import (
    EP_SIZE, LOGICAL, ROWS, ModelDouble, cfg, load, write_map,
)


@pytest.mark.parametrize('model_type', ['hy_v4', 'hy_v4_mtp'])
@pytest.mark.parametrize('policy', ['nearest', 'locality_fair'])
@pytest.mark.parametrize('rank', [0, 1])
def test_hy4_routes_tokens_across_every_loaded_replica(tmp_path, model_type, policy, rank):
    path = write_map(tmp_path)
    plan = load(path)
    model = ModelDouble(plan)
    config = cfg(expert_map_path=path, static_dispatch_policy=policy)
    config.model_config.hf_config.model_type = model_type
    group = NS(device_group=NS(rank=lambda: rank, size=lambda: EP_SIZE))
    with (patch('vllm.distributed.get_ep_group', return_value=group),
          patch('vllm.distributed.get_node_count', return_value=1),
          patch('vllm.model_executor.models.interfaces.get_mixture_of_experts_model',
                return_value=model)):
        runtime.install_routing(model, config)
        published = model.layers[0].eplb_state.logical_to_physical_map
        runtime.install_routing(model, config)
        assert model.layers[0].eplb_state.logical_to_physical_map is published
    for layer, row in zip(model.layers, ROWS):
        state = layer.eplb_state
        expected = torch.bincount(torch.tensor(row), minlength=LOGICAL)
        assert torch.equal(state.logical_replica_count, expected)
        assert not bool(state.should_record_tensor.item())
        # Core uses Knuth's 32-bit token hash. Both loaded replicas of hot
        # experts must be reachable; primary-only publication concentrates load.
        token = torch.arange(2056, dtype=torch.long)
        for logical in range(LOGICAL):
            copies = [i for i, value in enumerate(row) if value == logical]
            column = ((token * 2654435769) & 0xFFFFFFFF) % state.logical_replica_count[logical]
            physical = state.logical_to_physical_map[logical, column]
            assert set(physical.tolist()) == set(copies)
            assert all(row[i] == logical for i in physical.tolist())
            if len(copies) == 2:
                assert (physical == copies[0]).sum() == 1028
                assert (physical == copies[1]).sum() == 1028
