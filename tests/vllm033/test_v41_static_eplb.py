# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for static EPLB composed with the official FP4/SWA owner.

Device kernels, distributed membership and heavyweight outer allocation are
explicit doubles. Native constructor namespaces and admission run as source.
"""
from types import SimpleNamespace as NS

import pytest
import torch

from replay_v41_static_eplb import CASES
from test_fp4_full_chain import chain
from test_dsv41_megamoe_cpu import mega_config
from hcu_vllm_das import fp4_indexer
from hcu_vllm_das import static_eplb_runtime as runtime
from hcu_vllm_das.models import v41_pcp as models
from hcu_vllm_das.moe import dsv41_megamoe as mega
from hcu_vllm_das.moe.dsv41_deepgemm import HcuDSV41DeepGemmMethod


@pytest.mark.parametrize('name,case', CASES, ids=[name for name, _ in CASES])
def test_existing_static_eplb_contracts(name, case):
    case()


def static_config(cfg, *, recording=False):
    cfg.additional_config = {'hcu_static_eplb': {
        'num_redundant_experts': 32,
        'static_dispatch_policy': 'locality_fair',
        'expert_map_record_path' if recording else 'expert_map_path': '/tmp/expert_map.json',
    }}
    cfg.model_config.hf_config.n_routed_experts = 384
    cfg.parallel_config.prefill_context_parallel_size = 8
    cfg.parallel_config.enable_expert_parallel = True
    cfg.parallel_config.enable_elastic_ep = False
    cfg.parallel_config.all2all_backend = 'deepep_high_throughput'
    cfg.parallel_config.use_ubatching = False
    cfg.parallel_config.worker_cls = 'auto'
    cfg.parallel_config.expert_placement_strategy = 'linear'
    cfg.attention_config.backend = None
    cfg.kernel_config.moe_backend = 'deep_gemm'
    cfg.cache_config.swa_bounded_replay = True
    return cfg


@pytest.mark.parametrize('recording', [False, True])
@pytest.mark.parametrize('policy', ['nearest', 'locality_fair'])
def test_static_eplb_rejects_round_robin_placement_before_moe_construction(
        chain, recording, policy):
    cfg = static_config(chain.cfg, recording=recording)
    cfg.parallel_config.expert_placement_strategy = 'round_robin'
    cfg.additional_config['hcu_static_eplb']['static_dispatch_policy'] = policy
    with pytest.raises(runtime.StaticEplbError, match='--expert-placement-strategy linear'):
        runtime.moe_eplb_kwargs(cfg, 'model.layers.0.ffn.experts')


@pytest.mark.parametrize('recording', [False, True], ids=['deploy', 'calibrate'])
@pytest.mark.parametrize('filter_weights', [False, True], ids=['unfiltered', 'filtered'])
def test_static_eplb_ep_weight_filter(chain, monkeypatch, recording, filter_weights):
    cfg = static_config(chain.cfg, recording=recording)
    cfg.parallel_config.enable_ep_weight_filter = filter_weights
    monkeypatch.setenv('HCU_VLLM_PCP_IMPL', 'native')
    if filter_weights and not recording:
        with pytest.raises(runtime.StaticEplbError, match='--enable-ep-weight-filter'):
            runtime.moe_eplb_kwargs(cfg, 'model.layers.0.ffn.experts')
    else:
        assert runtime.moe_eplb_kwargs(cfg, 'model.layers.0.ffn.experts') == {
            'enable_eplb': True,
            'num_redundant_experts': 0 if recording else 32,
        }


@pytest.mark.parametrize('recording,replay,skip_padding,allowed', [
    (True, True, None, True),
    (True, True, '1', True),
    (True, True, '0', False),
    (True, False, '0', True),
    (False, True, '0', True),
])
def test_static_eplb_swa_replay_calibration_padding(
        chain, monkeypatch, recording, replay, skip_padding, allowed):
    cfg = static_config(chain.cfg, recording=recording)
    cfg.cache_config.swa_bounded_replay = replay
    monkeypatch.setenv('HCU_VLLM_PCP_IMPL', 'native')
    if skip_padding is None:
        monkeypatch.delenv('VLLM_MOE_SKIP_PADDING', raising=False)
    else:
        monkeypatch.setenv('VLLM_MOE_SKIP_PADDING', skip_padding)
    if not allowed:
        with pytest.raises(runtime.StaticEplbError, match='VLLM_MOE_SKIP_PADDING=1'):
            runtime.validate_config(cfg)
    else:
        assert runtime.validate_config(cfg)


@pytest.mark.parametrize('fp4', [False, True])
@pytest.mark.parametrize('static', [False, True])
def test_outer_selects_composed_constructor(chain, monkeypatch, fp4, static):
    cfg = static_config(chain.cfg)
    if not static:
        cfg.additional_config = {}
    monkeypatch.setenv('HCU_VLLM_PCP_IMPL', 'native')
    monkeypatch.setenv('VLLM_HCU_DSV41_PCP_ATTENTION', '1')
    if not fp4:
        cfg.attention_config.resolve_indexer_kv_dtype = lambda _: 'fp8'
    from hcu_adapter.kernels import pcp_fork
    from hcu_vllm_das import inverse_rope_pcp, v41_pcp_activation
    monkeypatch.setattr(v41_pcp_activation, 'check_health', lambda: None)
    monkeypatch.setattr(inverse_rope_pcp, '_state', object())
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 3)
    attention_cls = fp4_indexer.Fp4PcpAttention if fp4 else pcp_fork.build_classes()['attention']
    calls = []
    device_calls = []

    def qualify(device):
        assert len(calls) == 1
        device_calls.append(device)

    monkeypatch.setattr(inverse_rope_pcp, 'prepare_worker_device', qualify)

    def constructor(name):
        def init(self, *, vllm_config, prefix):
            torch.nn.Module.__init__(self)
            self.attn = attention_cls.__new__(attention_cls)
            torch.nn.Module.__init__(self.attn)
            calls.append((name, vllm_config, prefix))
        return init

    for name in ('_OUTER_INIT', '_EPLB_OUTER_INIT', '_FP4_OUTER_INIT', '_FP4_EPLB_OUTER_INIT'):
        monkeypatch.setattr(models, name, constructor(name))
    outer = models.HcuV41PcpForCausalLM(vllm_config=cfg, prefix='target')
    expected = ('_FP4_EPLB_OUTER_INIT' if static else '_FP4_OUTER_INIT') if fp4 else (
        '_EPLB_OUTER_INIT' if static else '_OUTER_INIT')
    assert calls == [(expected, cfg, 'target')]
    assert device_calls == [torch.device('cuda', 3)]
    assert outer._hcu_static_eplb_config is (cfg if static else None)


@pytest.mark.parametrize('mode,admitted', [('stock', True), ('native', False)])
@pytest.mark.parametrize('provider_bound', [False, True])
def test_outer_skips_inverse_rope_on_core_attention(chain, monkeypatch, mode, admitted,
                                                   provider_bound):
    from hcu_adapter.kernels import pcp_fork
    from hcu_vllm_das import inverse_rope_pcp, inverse_rope_policy, v41_pcp_activation

    cfg = static_config(chain.cfg)
    cfg.additional_config = {}
    cfg.attention_config.resolve_indexer_kv_dtype = lambda _: 'fp8'
    monkeypatch.setenv('VLLM_HCU_DSV41_PCP_ATTENTION', '1')
    monkeypatch.setattr(v41_pcp_activation, 'check_health', lambda: None)
    monkeypatch.setattr(inverse_rope_policy, 'selected_mode', lambda: 'hcu')
    monkeypatch.setattr(inverse_rope_pcp, '_state', object() if provider_bound else None)
    monkeypatch.setattr(pcp_fork, 'selected_mode', lambda: mode)
    monkeypatch.setattr(models, 'validate_config', lambda _: admitted)
    monkeypatch.setattr(models, 'use_host_target', lambda _: False)
    monkeypatch.setattr(models, 'check_native_contract', lambda: None)
    monkeypatch.setattr(models.native, '_select_dsv4_attn_cls', lambda _: torch.nn.Identity)
    monkeypatch.setattr(torch.cuda, 'current_device',
                        lambda: pytest.fail('Core attention should not query a GPU for inverse RoPE'))

    def construct(self, *, vllm_config, prefix):
        torch.nn.Module.__init__(self)
        self.attn = models.select_attention(vllm_config)()

    monkeypatch.setattr(models, '_OUTER_INIT', construct)
    monkeypatch.setattr(models.native_vl.DeepseekV41ForCausalLM, '__init__', construct)
    outer = models.HcuV41PcpForCausalLM(vllm_config=cfg)
    assert isinstance(outer.attn, torch.nn.Identity)


def test_fp4_static_namespaces_preserve_native_and_swa(chain, monkeypatch):
    cfg = static_config(chain.cfg)
    monkeypatch.setenv('HCU_VLLM_PCP_IMPL', 'native')
    assert runtime.validate_config(cfg)
    namespace = models._FP4_EPLB_DECODER_INIT.__globals__
    assert namespace['DeepseekV4MoE'] is models.HcuV41EplbMoE
    assert namespace['Engram'] is models.HcuV41PcpEngram
    assert namespace['_select_dsv4_attn_cls'](cfg) is fp4_indexer.Fp4PcpAttention
    assert models._FP4_EPLB_BACKBONE_INIT.__globals__['DeepseekV4DecoderLayer'] is (
        models.HcuV41Fp4EplbDecoderLayer)
    assert models.HcuV41Fp4EplbLLM.model_cls is models.HcuV41Fp4EplbBackbone
    assert models._FP4_EPLB_OUTER_INIT.__globals__['DeepseekV41LLMForCausalLM'] is (
        models.HcuV41Fp4EplbLLM)
    assert models.HcuV41Fp4EplbBackbone._decoder_replay_supported is (
        models.HcuV41PcpBackbone._decoder_replay_supported)
    backbone = NS(start_layer=0, end_layer=40, use_sequence_parallel=False,
                  config=NS(num_hidden_layers=40, engram_layer_ids=[]))
    assert models.HcuV41Fp4EplbBackbone._decoder_replay_supported(backbone, cfg, 36)
    assert models.native.DeepseekV4DecoderLayer.__init__.__globals__['DeepseekV4MoE'] is (
        models.native.DeepseekV4MoE)
    assert models._BASE_MOE_NATIVE.__globals__['FusedMoEFactory'] is (
        models.native_v4.FusedMoEFactory)


def test_fp4_static_keeps_swa_refusals(chain, monkeypatch):
    cfg = static_config(chain.cfg)
    monkeypatch.setenv('HCU_VLLM_PCP_IMPL', 'native')
    backbone = NS(start_layer=0, end_layer=40, use_sequence_parallel=True,
                  config=NS(num_hidden_layers=40, engram_layer_ids=[]))
    assert not models.HcuV41Fp4EplbBackbone._decoder_replay_supported(backbone, cfg, 36)


@pytest.mark.parametrize('field,value', [
    ('speculative_config', NS(method='dspark', num_speculative_tokens=5,
                              enable_adaptive_verification=False)),
    ('kv_transfer_config', object()),
])
def test_composed_scope_rejects_unported_features(chain, monkeypatch, field, value):
    cfg = static_config(chain.cfg)
    monkeypatch.setenv('HCU_VLLM_PCP_IMPL', 'native')
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        runtime.validate_config(cfg)


@pytest.mark.parametrize('recording,physical', [(False, 416), (True, 384)])
def test_deepgemm_accepts_static_physical_capacity(chain, monkeypatch, recording, physical):
    cfg = static_config(chain.cfg, recording=recording)
    _, moe = mega_config()
    moe.max_num_tokens = 32768
    moe.num_experts = physical
    moe.num_local_experts = physical // 8
    monkeypatch.setenv('HCU_VLLM_PCP_IMPL', 'native')
    monkeypatch.setattr(mega, 'get_current_vllm_config', lambda: cfg)
    method = HcuDSV41DeepGemmMethod(None, None, moe, 'model.layers.0.ffn.experts')
    assert method.supports_eplb
    assert method.moe.num_experts == physical
    # MegaMoE keeps its original narrow identity-layout domain.
    with pytest.raises(ValueError):
        mega.HcuDSV41MegaMoEMethod(None, None, moe, 'model.layers.0.ffn.experts')


@pytest.mark.parametrize('physical,global_eplb', [(384, False), (408, False), (416, True)])
def test_deepgemm_rejects_mismatched_or_dynamic_placement(chain, monkeypatch, physical, global_eplb):
    cfg = static_config(chain.cfg)
    _, moe = mega_config()
    moe.num_experts = physical
    cfg.parallel_config.enable_eplb = global_eplb
    monkeypatch.setattr(mega, 'get_current_vllm_config', lambda: cfg)
    with pytest.raises(ValueError):
        HcuDSV41DeepGemmMethod(None, None, moe, 'model.layers.0.ffn.experts')


def test_weight_fanout_precedes_native_loading(monkeypatch):
    outer = models.HcuV41PcpForCausalLM.__new__(models.HcuV41PcpForCausalLM)
    torch.nn.Module.__init__(outer)
    cfg = object()
    outer._hcu_static_eplb_config = cfg
    calls = []
    monkeypatch.setattr(runtime, 'bind_placement', lambda model, config: calls.append('bind'))

    def native_load(self, weights):
        calls.append(('load', weights))
        return {'loaded'}

    monkeypatch.setattr(models.native_vl.DeepseekV41ForCausalLM, 'load_weights', native_load)
    assert outer.load_weights('checkpoint') == {'loaded'}
    assert calls == ['bind', ('load', 'checkpoint')]


def test_factory_context_resets_after_constructor_failure(chain, monkeypatch):
    cfg = static_config(chain.cfg)

    def fail(self, *args, **kwargs):
        assert models._MOE_CONFIG.get() is cfg
        raise RuntimeError('allocation failed')

    monkeypatch.setattr(models.native.DeepseekV4MoE, '__init__', fail)
    with pytest.raises(RuntimeError, match='allocation failed'):
        models.HcuV41EplbMoE(cfg, prefix='model.layers.0.ffn')
    assert models._MOE_CONFIG.get() is None
    with pytest.raises(RuntimeError, match='outside owned construction'):
        models._eplb_factory()


def test_factory_adds_capacity_without_replacing_native_globals(chain, monkeypatch):
    cfg = static_config(chain.cfg)
    monkeypatch.setenv('HCU_VLLM_PCP_IMPL', 'native')
    calls = []
    original = models._BASE_MOE_NATIVE.__globals__['FusedMoEFactory']
    monkeypatch.setattr(models.native_v4, 'FusedMoEFactory',
                        lambda **kwargs: calls.append(kwargs))
    token = models._MOE_CONFIG.set(cfg)
    try:
        models._eplb_factory(num_experts=384, prefix='model.layers.0.ffn.experts')
    finally:
        models._MOE_CONFIG.reset(token)
    assert calls == [{'num_experts': 384, 'prefix': 'model.layers.0.ffn.experts',
                      'enable_eplb': True, 'num_redundant_experts': 32}]
    assert models._EPLB_BASE_MOE_INIT.__globals__['FusedMoEFactory'] is models._eplb_factory
    assert models.HcuV41EplbMoE.__mro__[1:3] == (
        models.native.DeepseekV4MoE, models._HcuEplbBaseMoE)
    # The only global substitution here is pytest's final factory double.
    # The production constructor retains its original private namespace.
    assert original is not models._eplb_factory


@pytest.mark.parametrize('recording,physical,local', [(False, 416, 52), (True, 384, 48)])
def test_model_protocol_publishes_ep_physical_counts(chain, recording, physical, local):
    cfg = static_config(chain.cfg, recording=recording)
    routed = NS(local_num_experts=local, moe_config=NS(
        num_experts=physical, moe_parallel_config=NS(ep_size=8)))
    moe = NS(n_logical_experts=384, prefix='model.layers.0.ffn',
             experts=NS(routed_experts=routed, eplb_state=object()))
    runtime.publish_physical_metadata(moe, cfg, moe.prefix)
    assert moe.n_physical_experts == physical
    assert moe.n_local_physical_experts == local
    assert moe.n_redundant_experts == physical - 384
    routed.moe_config.num_experts += 8
    with pytest.raises(runtime.StaticEplbError, match='layout mismatch'):
        runtime.publish_physical_metadata(moe, cfg, moe.prefix)
