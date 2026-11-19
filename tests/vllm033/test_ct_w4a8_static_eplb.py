# SPDX-License-Identifier: Apache-2.0
"""CPU native CT loader integration. HIPC conversion's device packer is doubled."""
import json
from types import SimpleNamespace as NS

import pytest
import torch

from test_fp4_full_chain import chain
from test_v41_static_eplb import static_config
from replay_v41_static_eplb import cfg as placement_config
from hcu_vllm_das import static_eplb_runtime as runtime
from hcu_vllm_das.quantization.ct_w4a8_method import HcuCTW4A8MoEMethod
from hcu_vllm_das.quantization.ct_w4a8_weights import PackedCTWeights
from vllm.model_executor.layers.fused_moe import RoutedExperts


def test_ct_admits_only_owned_static_eplb(chain, monkeypatch):
    monkeypatch.setenv('HCU_VLLM_PCP_IMPL', 'native')
    method = HcuCTW4A8MoEMethod.__new__(HcuCTW4A8MoEMethod)
    method._config = chain.cfg
    assert not method.supports_eplb
    static_config(chain.cfg)
    assert method.supports_eplb
    chain.cfg.parallel_config.enable_eplb = True
    with pytest.raises(ValueError, match='both cannot be active'):
        _ = method.supports_eplb


def test_ct_static_eplb_rejects_ep_weight_filter(chain, monkeypatch):
    monkeypatch.setenv('HCU_VLLM_PCP_IMPL', 'native')
    method = HcuCTW4A8MoEMethod.__new__(HcuCTW4A8MoEMethod)
    method._config = static_config(chain.cfg)
    chain.cfg.parallel_config.enable_ep_weight_filter = True
    with pytest.raises(runtime.StaticEplbError, match='--enable-ep-weight-filter'):
        _ = method.supports_eplb


def make_experts(rank):
    experts = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(experts)
    experts.quant_config = NS(get_name=lambda: 'hcu_dsv41_ct_w4a8')
    method = HcuCTW4A8MoEMethod.__new__(HcuCTW4A8MoEMethod)
    method.moe_kernel = None
    experts.quant_method = method
    experts.local_num_experts = 3
    experts.moe_config = NS(
        num_logical_experts=4, num_experts=6, num_local_experts=3,
        tp_rank=0, is_act_and_mul=True, tp_shard_with_padding=False,
        moe_parallel_config=NS(tp_size=1))
    experts.expert_map_manager = NS(
        num_fused_shared_experts=0,
        map_global_to_local=lambda e: e - rank * 3 if rank * 3 <= e < (rank + 1) * 3 else -1)
    experts._fused_shared_expert_quantizer = None
    owner = PackedCTWeights(experts, 3, 512, 256, experts.weight_loader)
    method._packed = owner
    return experts, owner


@pytest.mark.parametrize('rank', [0, 1])
def test_native_ct_fanout_scales_shapes_and_reload_guard(tmp_path, chain, monkeypatch, rank):
    row = [2, 0, 3, 1, 0, 1]
    config = placement_config(width=2, num_redundant_experts=2)
    config.model_config.hf_config.n_routed_experts = 4
    path = tmp_path / 'map.json'
    document = {'version': 2, 'format': 'vllm_offline_eplb_physical_to_logical_by_model',
        'model_maps': {'CTModel': {
            'record_kind': 'proposal', 'model_class': 'CTModel',
            'model_name': config.model_config.model, 'num_moe_layers': 1,
            'num_logical_experts': 4, 'num_physical_experts': 6,
            'num_redundant_experts': 2, 'physical_to_logical_map': [row]}}}
    path.write_text(json.dumps(document))
    config.additional_config['hcu_static_eplb'].update(
        expert_map_path=str(path), num_redundant_experts=2)
    experts, owner = make_experts(rank)
    model = NS(moe_mlp_layers=[NS(experts=NS(routed_experts=experts))],
        num_moe_layers=1, num_redundant_experts=2,
        num_physical_experts=6, num_logical_experts=4)
    monkeypatch.setattr(runtime, 'resolve_model_key', lambda model: 'CTModel')
    monkeypatch.setattr('vllm.model_executor.models.interfaces.get_mixture_of_experts_model',
                        lambda model: model)
    runtime.bind_placement(model, config)
    first_loader = experts.w13_weight_packed.weight_loader
    runtime.bind_placement(model, config)
    assert experts.w13_weight_packed.weight_loader is first_loader

    for logical in range(4):
        for shard, rows, columns in [('w1', 256, 512), ('w3', 256, 512), ('w2', 512, 256)]:
            stem = 'w2' if shard == 'w2' else 'w13'
            # Eight packed signed-INT4 values, all equal to logical-8.
            packed = logical * 0x11111111
            for suffix, data in [
                ('weight_packed', torch.full((rows, columns // 8), packed, dtype=torch.int32)),
                ('weight_scale', torch.full((rows, 1), float(logical + 1))),
                ('weight_shape', torch.tensor([rows, columns]))]:
                name = stem + '_' + suffix
                parameter = getattr(experts, name)
                actual = parameter.weight_loader(parameter, data, name,
                    shard_id=shard, expert_id=logical, return_success=True)
                assert bool(actual) == (logical in row[rank * 3:(rank + 1) * 3])

    for local, logical in enumerate(row[rank * 3:(rank + 1) * 3]):
        for stem in ('w13', 'w2'):
            assert torch.all(getattr(experts, stem + '_weight_packed')[local] == logical * 0x11111111)
            assert torch.all(getattr(experts, stem + '_weight_scale')[local] == logical + 1)
        assert torch.equal(experts.w13_weight_shape[local], torch.tensor([256, 512]))
        assert torch.equal(experts.w2_weight_shape[local], torch.tensor([512, 256]))

    owner.pack(lambda tensor: tensor)
    before = experts.w13_weight_packed.clone()
    with pytest.raises(RuntimeError, match='reload after packing'):
        first_loader(experts.w13_weight_packed, torch.zeros(256, 64, dtype=torch.int32),
                     'w13_weight_packed', shard_id='w1', expert_id=0)
    assert torch.equal(experts.w13_weight_packed, before)


def test_ct_rejects_loader_from_other_expert_owner():
    experts, _ = make_experts(0)
    foreign, _ = make_experts(1)
    with pytest.raises(runtime.StaticEplbError, match='unsupported loader owner'):
        runtime._loader_function(foreign.w13_weight_packed.weight_loader, experts, 'w13_weight_packed')
