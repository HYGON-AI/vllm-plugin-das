# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Real Core constructor/loader methods; distributed/GEMM leaves doubled."""
from types import SimpleNamespace as NS

import torch

from hcu_vllm_das.models import hy4_static as owned
from test_hy4_static_runtime import cfg


def test_core_forwards_and_compile_wrapper_are_retained():
    assert owned.HcuHY4StaticForCausalLM.forward is owned.native.HYV4ForCausalLM.forward
    assert owned.HcuHY4StaticBackbone.forward is owned.native.HYV4Model.forward
    assert owned.HcuHY4StaticMTP.forward is owned.mtp.HYV4MTP.forward
    assert owned.HcuHY4StaticMoE.forward is owned.moe.HYV4MoEFused.forward
    assert owned.HcuHY4StaticBackbone.__init__.__code__ is owned.native.HYV4Model.__init__.__code__
    assert owned.native.HYV4DecoderLayer.__init__.__globals__['HYV4MoEFused'] is owned.moe.HYV4MoEFused
    assert owned.mtp.HYV4MultiTokenPredictorLayer.__init__.__globals__['HYV4DecoderLayer'] is owned.nvidia.HYV4DecoderLayer


def test_physical_capacity_is_requested_before_expert_allocation(cfg, monkeypatch):
    from hcu_adapter.bridges import vllm as host
    monkeypatch.setattr(host, 'require_extension_healthy', lambda name: None)
    monkeypatch.setattr(owned, 'get_current_vllm_config', lambda: cfg)
    namespace = owned._MOE_INIT.__globals__
    monkeypatch.setitem(namespace, 'get_current_vllm_config', lambda: cfg)
    monkeypatch.setitem(namespace, 'get_tensor_model_parallel_world_size', lambda: 1)
    monkeypatch.setitem(namespace, 'get_ep_group', lambda: NS(
        device_group=NS(size=lambda: 16), rank_in_group=3, size=lambda: 16))
    monkeypatch.setitem(namespace, 'GateLinear', lambda *args, **kwargs: torch.nn.Identity())
    allocated = []

    def factory(**kwargs):
        count = kwargs['num_experts'] + kwargs['num_redundant_experts']
        allocated.append((count, kwargs['enable_eplb']))
        # The distributed factory/GEMM is a leaf double, not device proof.
        runner = torch.nn.Module()
        runner.routed_experts = NS(local_num_experts=count // 16,
            moe_config=NS(num_experts=count, moe_parallel_config=NS(ep_size=16)))
        runner.eplb_state = NS()
        runner.register_parameter('probe', torch.nn.Parameter(torch.empty(count // 16, 1)))
        return runner

    monkeypatch.setattr(owned.moe, 'FusedMoEFactory', factory)
    hf = NS(num_experts=256, hidden_size=8, expert_hidden_dim=8,
        num_experts_per_tok=6, num_shared_experts=0,
        route_norm=True, router_scaling_factor=1., swiglu_limit=10.)
    layer = owned.HcuHY4StaticMoE(hf, prefix='model.layers.78.mlp')
    assert allocated == [(272, True)]
    assert layer.n_redundant_experts == 16
    assert layer.n_physical_experts == 272
    assert layer.n_local_physical_experts == 17
    assert (layer.physical_expert_start, layer.physical_expert_end) == (51, 68)


def test_named_native_loader_and_postprocessing_order_are_preserved(cfg, monkeypatch):
    model = owned.HcuHY4StaticForCausalLM.__new__(owned.HcuHY4StaticForCausalLM)
    torch.nn.Module.__init__(model)
    model._hcu_static_cfg = cfg
    model._hcu_static_expected_plan = NS(source_sha256='snapshot')
    calls = []
    monkeypatch.setattr(owned.runtime, 'bind_placement',
        lambda model, cfg, **kwargs: calls.append(('bind', kwargs)) or NS(source_sha256='snapshot'))
    monkeypatch.setattr(owned.runtime, 'install_routing', lambda model, cfg: calls.append('route'))
    monkeypatch.setattr(owned.native.HYV4ForCausalLM, 'load_weights',
        lambda self, weights: calls.append(('native-load', weights)) or {'loaded'})
    monkeypatch.setattr(owned.native.HYV4ForCausalLM, 'process_weights_after_loading',
        lambda self: calls.append('native-post'))
    weights = iter(())
    assert model.load_weights(weights) == {'loaded'}
    assert calls == [('bind', {'bind_weight_loaders': True}), ('native-load', weights)]
    calls.clear()
    model.process_weights_after_loading()
    assert calls == ['native-post', ('bind', {'bind_weight_loaders': False}), 'route']
