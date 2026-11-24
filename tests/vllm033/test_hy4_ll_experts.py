# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HY4-only LL policy; native helper and expert apply execute on CPU doubles."""
from importlib import import_module, util
from types import SimpleNamespace as NS

import pytest
import torch


def module():
    name = 'hcu_vllm_das.hy4_ll_experts'
    assert util.find_spec(name) is not None, 'Owned HY4 LL tuning policy is missing'
    return import_module(name)


def fixture_config():
    from vllm.config import CUDAGraphMode
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    cfg = NS(model_config=NS(dtype=torch.bfloat16, enforce_eager=False,
        is_multimodal_model=False, hf_config=NS(model_type='hy_v4')),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
            prefill_context_parallel_size=1, decode_context_parallel_size=1,
            data_parallel_size=16, enable_expert_parallel=True,
            enable_eplb=False, enable_elastic_ep=False), lora_config=None,
        compilation_config=NS(mode=0, cudagraph_mode=CUDAGraphMode.FULL))
    moe = NS(num_experts=256, num_logical_experts=256, num_local_experts=16,
        experts_per_token=8, hidden_dim=6144, intermediate_size_per_partition=2048,
        in_dtype=torch.bfloat16, activation=MoEActivation.SILU,
        has_bias=False, is_lora_enabled=False, apply_router_weight_on_input=False,
        moe_parallel_config=NS(tp_size=1, dp_size=16, pcp_size=1, ep_size=16,
            use_ep=True, all2all_backend='deepep_low_latency'))
    return cfg, moe, NS(gemm1_clamp_limit=10.), NS(
        max_num_tokens_per_rank=lambda: 32, num_dispatchers=lambda: 16)


@pytest.mark.parametrize('master,child,want', [
    (None, None, False), ('0', '1', False), ('1', None, False),
    ('1', '0', False), ('1', '1', True), ('true', 'on', True)])
def test_optional_switches_preserve_default(monkeypatch, master, child, want):
    owned = module()
    for key, value in [('VLLM_HCU_USE_CUSTOM_OPS', master), ('HCU_VLLM_HY4_LL_TUNING', child)]:
        if value is None: monkeypatch.delenv(key, raising=False)
        else: monkeypatch.setenv(key, value)
    monkeypatch.setattr(owned, '_supported', lambda *args: True)
    assert owned.enabled(*fixture_config()) is want


def test_master_rollback_does_not_import_provider_or_query_hardware(monkeypatch):
    owned = module()
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '0')
    monkeypatch.setenv('HCU_VLLM_HY4_LL_TUNING', '1')
    monkeypatch.setattr(owned, '_supported', lambda *a: pytest.fail('queried disabled scope'))
    monkeypatch.setattr(owned, 'adapter_module', lambda *a: pytest.fail('imported disabled kernel'))
    assert not owned.enabled(*fixture_config())


@pytest.mark.parametrize('change', ['arch', 'platform', 'model', 'multimodal', 'dtype',
    'eager', 'compile', 'graph', 'tp', 'pp', 'pcp', 'dcp', 'dp', 'eplb', 'elastic',
    'lora', 'backend', 'actual_ep', 'actual_dp', 'hidden', 'intermediate', 'experts',
    'local_experts', 'topk', 'capacity', 'dispatchers', 'clamp', 'bias', 'input_weight'])
def test_unmeasured_scope_retains_native(monkeypatch, change):
    owned = module()
    from vllm.config import CUDAGraphMode
    cfg, moe, quant, prepare = fixture_config()
    monkeypatch.setattr(owned, '_device_arch', lambda: None if change == 'platform' else
        'gfx936' if change == 'arch' else 'gfx938')
    changes = {
        'model': (cfg.model_config.hf_config, 'model_type', 'deepseek_v4'),
        'multimodal': (cfg.model_config, 'is_multimodal_model', True),
        'dtype': (cfg.model_config, 'dtype', torch.float32),
        'eager': (cfg.model_config, 'enforce_eager', True),
        'compile': (cfg.compilation_config, 'mode', 3),
        'graph': (cfg.compilation_config, 'cudagraph_mode', CUDAGraphMode.NONE),
        'tp': (cfg.parallel_config, 'tensor_parallel_size', 2),
        'pp': (cfg.parallel_config, 'pipeline_parallel_size', 2),
        'pcp': (cfg.parallel_config, 'prefill_context_parallel_size', 16),
        'dcp': (cfg.parallel_config, 'decode_context_parallel_size', 2),
        'dp': (cfg.parallel_config, 'data_parallel_size', 8),
        'eplb': (cfg.parallel_config, 'enable_eplb', True),
        'elastic': (cfg.parallel_config, 'enable_elastic_ep', True),
        'lora': (cfg, 'lora_config', object()),
        'backend': (moe.moe_parallel_config, 'all2all_backend', 'deepep_high_throughput'),
        'actual_ep': (moe.moe_parallel_config, 'ep_size', 8),
        'actual_dp': (moe.moe_parallel_config, 'dp_size', 8),
        'hidden': (moe, 'hidden_dim', 7168),
        'intermediate': (moe, 'intermediate_size_per_partition', 4096),
        'experts': (moe, 'num_experts', 272),
        'local_experts': (moe, 'num_local_experts', 17),
        'topk': (moe, 'experts_per_token', 4),
        'capacity': (prepare, 'max_num_tokens_per_rank', lambda: 64),
        'dispatchers': (prepare, 'num_dispatchers', lambda: 8),
        'clamp': (quant, 'gemm1_clamp_limit', None),
        'bias': (moe, 'has_bias', True),
        'input_weight': (moe, 'apply_router_weight_on_input', True),
    }
    if change in changes:
        obj, key, value = changes[change]; setattr(obj, key, value)
    assert not owned._supported(cfg, moe, quant, prepare)


@pytest.mark.parametrize('model_type', ['hy_v4', 'hy_v4_mtp'])
def test_backbone_and_mtp_can_use_same_quant_math(monkeypatch, model_type):
    owned = module(); args = fixture_config()
    args[0].model_config.hf_config.model_type = model_type
    monkeypatch.setattr(owned, '_device_arch', lambda: 'gfx938')
    assert owned._supported(*args)


@pytest.mark.parametrize('vector,want', [(None, 512), ([0]*16, 32), ([4]*16, 32),
    ([24]*16, 32), ([32]*16, 32), ([64]*16, 32),
    ([512]*16, 256), ([2048]*16, 512)])
def test_native_cpu_dp_hint_keeps_missing_metadata_fallback(vector, want):
    owned = module()
    from vllm.forward_context import DPMetadata, ForwardContext, override_forward_context
    context = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={},
        dp_metadata=None if vector is None else DPMetadata(torch.tensor(vector, dtype=torch.int32)))
    instance = owned.TunedMaskedExperts.__new__(owned.TunedMaskedExperts)
    with override_forward_context(context):
        assert instance.expected_m(256, 512, 8) == want


@pytest.mark.parametrize('experts,capacity,topk', [(272,512,8), (256,256,8), (256,512,4)])
def test_other_expert_geometry_keeps_capacity_fallback(experts, capacity, topk):
    owned = module()
    instance = owned.TunedMaskedExperts.__new__(owned.TunedMaskedExperts)
    assert instance.expected_m(experts, capacity, topk) == capacity


def test_private_ops_keep_shared_provider_and_fixed_clamp_abi(monkeypatch):
    owned = module(); calls = []
    gemm = object(); vendor = object()
    original = NS(gemm_masked=gemm, clamp_quant_ep=vendor)
    def provider(value, counts, *, limit, parallel_tokens):
        calls.append((value, counts, limit, parallel_tokens)); return 'q', 's'
    monkeypatch.setattr(owned, 'adapter_module', lambda name:
        NS(persistent_masked_m_silu_mul_clamp_int8_quant=provider))
    private = owned.tuned_ops(original)
    assert private is not original and original.clamp_quant_ep is vendor
    assert private.gemm_masked is gemm
    x, counts = object(), object()
    assert private.clamp_quant_ep(x, limit=10., mask_m=counts, expect_m=512) == ('q', 's')
    assert calls == [(x, counts, 10., 64)]


def test_actual_expert_apply_preserves_counts_capacity_and_clamp(monkeypatch):
    owned = module()
    from hcu_vllm_das.moe import ct_w4a8_runtime as runtime
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    calls = []; counts = torch.tensor([2], dtype=torch.int32)
    def gemm(a, b, output, count_arg, hint):
        assert count_arg is counts
        calls.append(('gemm', hint, tuple(output.shape))); output.zero_()
    def clamp(value, *, limit, mask_m, expect_m):
        assert mask_m is counts
        calls.append(('clamp', limit, expect_m))
        return torch.zeros(1, 512, 2, dtype=torch.int8), torch.ones(1, 512, 1)
    instance = NS(max_num_tokens=32, num_dispatchers=16, weights=('w1', 'w2'),
        w1_scale=torch.ones(1, 4, 1), w2_scale=torch.ones(1, 2, 1),
        quant_config=NS(gemm1_clamp_limit=10.),
        ops=NS(gemm_masked=gemm, clamp_quant_ep=clamp),
        _small_batch_hint=False, _live_row_activation=False,
        _check=lambda *a: None, moe_problem_size=lambda *a: (1, 512, 4, 2, 8),
        expected_m=lambda *a: 128)
    instance._masked_gemm=lambda *args:runtime.HcuCTMaskedExperts._masked_gemm(instance,*args)
    runtime.HcuCTMaskedExperts.apply(instance, torch.empty(1, 512, 2),
        torch.zeros(1, 512, 2, dtype=torch.int8), None, None,
        torch.ones(1, 8), torch.zeros(1, 8, dtype=torch.int64), MoEActivation.SILU,
        256, None, torch.ones(1, 512, 1), None,
        torch.empty(1, 512, 4), torch.empty(1, 512, 2),
        NS(expert_num_tokens=counts), False)
    assert calls == [('gemm', 128, (1, 512, 4)), ('clamp', 10., 512),
                     ('gemm', 128, (1, 512, 2))]


def test_disabled_factory_preserves_original_ops_and_expert_class(monkeypatch):
    owned = module()
    from hcu_vllm_das.moe import ct_w4a8_runtime as runtime
    cfg, moe, quant, prepare = fixture_config()
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '0')
    monkeypatch.setattr(runtime.HcuCTMaskedExperts, '__init__', lambda self, *a:
        setattr(self, 'received', a))
    weights, ops = object(), object()
    result = owned.make_ll_experts(moe, quant, weights, ops, prepare)
    assert type(result) is runtime.HcuCTMaskedExperts
    assert result.received == (moe, quant, weights, ops, 32, 16)


def test_enabled_factory_resolves_real_adapter_without_mutating_shared_ops(monkeypatch):
    owned = module()
    cfg, moe, quant, prepare = fixture_config()
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '1')
    monkeypatch.setenv('HCU_VLLM_HY4_LL_TUNING', '1')
    monkeypatch.setattr(owned, '_device_arch', lambda: 'gfx938')
    monkeypatch.setattr(owned, 'get_current_vllm_config', lambda: cfg)
    monkeypatch.setattr(owned.TunedMaskedExperts, '__init__', lambda self, *a:
        setattr(self, 'received', a))
    original = NS(gemm_masked=object(), clamp_quant_ep=object())
    weights = object()
    result = owned.make_ll_experts(moe, quant, weights, original, prepare)
    assert type(result) is owned.TunedMaskedExperts
    private = result.received[3]
    assert private is not original and private.gemm_masked is original.gemm_masked
    assert private.clamp_quant_ep is not original.clamp_quant_ep
    assert result.received[2] is weights
    # This calls the real Adapter owner, with a CPU input rejected before any
    # kernel launch. It catches an unresolved/misrouted cross-package ABI.
    with pytest.raises(ValueError, match='CUDA|cuda|device'):
        private.clamp_quant_ep(torch.zeros(1, 2, 4, dtype=torch.bfloat16),
            limit=10., mask_m=torch.zeros(1, dtype=torch.int32), expect_m=512)
