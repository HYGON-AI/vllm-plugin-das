"""Opt-in synchronous HT EPLB wiring; no backend support by inheritance."""
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_hcu.models.kimi_k3.amd import linear
from vllm_hcu.models.kimi_k3.amd.ops.eplb import kimi_eplb_options
from vllm_hcu.model_executor.layers.quantization.kimi_k3_w4a8 import KimiK3W4A8MoEMethod


def config():
    return NS(model_config=NS(enforce_eager=True), parallel_config=NS(
        enable_eplb=True, enable_expert_parallel=True,
        all2all_backend='deepep_high_throughput',
        eplb_config=NS(use_async=False, num_redundant_experts=16)))


@pytest.mark.parametrize('master,leaf,ht,expected', [
    ('1','1',True,True), ('true','true',True,True),
    ('0','1',True,False), ('1','0',True,False),
    ('false','true',True,False), ('1','1',False,False),
    ('1',None,True,False)])
def test_support_requires_master_leaf_and_ht(monkeypatch, master, leaf, ht, expected):
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', master)
    if leaf is None:
        monkeypatch.delenv('VLLM_HCU_USE_KIMI_HT_EPLB', raising=False)
    else:
        monkeypatch.setenv('VLLM_HCU_USE_KIMI_HT_EPLB', leaf)
    method = KimiK3W4A8MoEMethod(NS(activation=NS(value='situ'), swiglu_beta=None,
        moe_parallel_config=NS(use_deepep_ht_kernels=ht, use_deepep_ll_kernels=False)))
    assert method.supports_eplb is expected


@pytest.mark.parametrize('bad', ['disabled', 'async', 'graph', 'backend', 'no_ep'])
def test_unsupported_modes_fail_before_allocations(monkeypatch, bad):
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '1')
    monkeypatch.setenv('VLLM_HCU_USE_KIMI_HT_EPLB', '1')
    cfg = config()
    if bad == 'disabled':
        monkeypatch.setenv('VLLM_HCU_USE_KIMI_HT_EPLB', '0')
    elif bad == 'async':
        cfg.parallel_config.eplb_config.use_async = True
    elif bad == 'graph':
        cfg.model_config.enforce_eager = False
    elif bad == 'backend':
        cfg.parallel_config.all2all_backend = 'deepep_low_latency'
    else:
        cfg.parallel_config.enable_expert_parallel = False
    with pytest.raises(ValueError, match='Kimi'):
        kimi_eplb_options(cfg)


def test_factory_receives_eplb_and_redundant_slots(monkeypatch):
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '1')
    monkeypatch.setenv('VLLM_HCU_USE_KIMI_HT_EPLB', '1')
    monkeypatch.setattr(linear, 'get_current_vllm_config', config)
    monkeypatch.setattr(linear, 'get_tensor_model_parallel_world_size', lambda: 1)
    monkeypatch.setattr(linear, 'GateLinear', lambda **kw: torch.nn.Module())
    seen = {}
    def factory(**kwargs):
        seen.update(kwargs)
        return torch.nn.Module()
    monkeypatch.setattr(linear, 'FusedMoE', factory)
    cfg = NS(hidden_size=32, moe_intermediate_size=32, num_experts=32,
        num_experts_per_token=2, moe_renormalize=True, routed_expert_hidden_size=None,
        latent_moe_use_norm=False, routed_scaling_factor=1., num_shared_experts=None,
        hidden_act='situ', activation_situ_beta=1., activation_situ_linear_beta=2.,
        use_grouped_topk=False, num_expert_group=1, topk_group=1,
        moe_router_activation_func='sigmoid')
    linear.KimiMoE(cfg)
    assert seen['enable_eplb'] is True
    assert seen['num_redundant_experts'] == 16
    assert seen['num_experts'] == 32  # Gate remains logical, storage adds slots.


def test_loader_mapping_includes_redundant_experts(monkeypatch):
    model = linear.KimiLinearModel.__new__(linear.KimiLinearModel)
    torch.nn.Module.__init__(model)
    model.config = NS(linear_attn_config=None, is_moe=True, num_experts=32)
    model.num_redundant_experts = 16
    actual = linear.fused_moe_make_expert_params_mapping
    mappings = []
    def capture(*args, **kwargs):
        result = actual(*args, **kwargs)
        mappings.extend(result)
        return result
    monkeypatch.setattr(linear, 'fused_moe_make_expert_params_mapping', capture)
    model.load_weights([])
    assert {row[2] for row in mappings} == set(range(48))
    assert any(row[2] >= 32 and 'experts.0.' in row[1] for row in mappings)


def test_loader_visits_every_replica_for_weights_and_scales():
    model = linear.KimiLinearModel.__new__(linear.KimiLinearModel)
    torch.nn.Module.__init__(model)
    model.config = NS(linear_attn_config=None, is_moe=True, num_experts=2,
                      is_linear_attn=False)
    model.num_redundant_experts = 2
    model.mlp = torch.nn.Module()
    model.mlp.experts = torch.nn.Module()
    calls = []
    def loader(param, value, name, *, expert_id, shard_id):
        calls.append((name, expert_id))
        with torch.no_grad():
            param[expert_id].copy_(value)
    for name, dtype in [('w13_weight', torch.int8), ('w13_weight_scale', torch.float32)]:
        param = torch.nn.Parameter(torch.zeros(4, 1, dtype=dtype), requires_grad=False)
        param.weight_loader = loader
        model.mlp.experts.register_parameter(name, param)
    model.load_weights([
        ('mlp.experts.0.w1.weight', torch.tensor([7], dtype=torch.int8)),
        ('mlp.experts.0.w1.weight_scale', torch.tensor([0.125])),
    ])
    assert len(calls) == 4
    assert model.mlp.experts.w13_weight[:, 0].tolist() == [7, 0, 7, 0]
    assert model.mlp.experts.w13_weight_scale[:, 0].tolist() == [.125, 0, .125, 0]
