# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Optional HY4 HT wire policy and collective-call contracts; CPU only."""
from importlib import import_module
from types import SimpleNamespace as NS

import pytest
import torch


def owned():
    return import_module('hcu_vllm_das.hy4_ht_int8')


def scope():
    cfg = NS(model_config=NS(dtype=torch.bfloat16, enforce_eager=True,
        is_multimodal_model=False, hf_config=NS(model_type='hy_v4')),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
            prefill_context_parallel_size=16, decode_context_parallel_size=1,
            data_parallel_size=1, enable_expert_parallel=True,
            enable_eplb=False, enable_elastic_ep=False, enable_dbo=False, ubatch_size=0),
        scheduler_config=NS(async_scheduling=False),
        use_v2_model_runner=True, lora_config=None, compilation_config=NS(mode=0))
    moe = NS(hidden_dim=6144, in_dtype=torch.bfloat16, experts_per_token=8,
        has_bias=False, is_lora_enabled=False, apply_router_weight_on_input=False,
        moe_parallel_config=NS(tp_size=1, dp_size=1, pcp_size=16, ep_size=16,
            use_ep=True, all2all_backend='deepep_high_throughput'))
    prepare = NS(num_dispatchers=lambda: 16, dp_size=1)
    return cfg, moe, prepare


@pytest.mark.parametrize('master,child,want', [(None,None,False),
    (None,'1',False),('0','1',False),('1',None,False),('1','0',False),('1','1',True)])
def test_opt_in_and_master_rollback(monkeypatch, master, child, want):
    m = owned()
    for name, value in [('VLLM_HCU_USE_CUSTOM_OPS', master),
                        ('HCU_VLLM_HY4_HT_INT8_DISPATCH', child)]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(m, '_supported', lambda *args: True)
    assert m.enabled(*scope()) is want


def test_disabled_does_not_query_scope(monkeypatch):
    m = owned()
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '0')
    monkeypatch.setenv('HCU_VLLM_HY4_HT_INT8_DISPATCH', '1')
    monkeypatch.setattr(m, '_supported', lambda *a: pytest.fail('queried disabled hardware/scope'))
    assert not m.enabled(*scope())


@pytest.mark.parametrize('change', ['hardware','model','dtype','multimodal','eager',
    'runner','compile','lora','tp','pp','pcp','dcp','dp','ep_disabled','eplb','elastic',
    'actual_ep','backend','hidden','input_weight','dbo','ubatching','async'])
def test_unverified_scope_retains_existing_prepare(monkeypatch, change):
    m = owned()
    cfg, moe, prepare = scope()
    monkeypatch.setattr(m, '_device_arch', lambda: 'gfx936' if change == 'hardware' else 'gfx938')
    changes = {
        'model': (cfg.model_config.hf_config,'model_type','other'),
        'dtype': (cfg.model_config,'dtype',torch.float16),
        'multimodal': (cfg.model_config,'is_multimodal_model',True),
        'eager': (cfg.model_config,'enforce_eager',False),
        'runner': (cfg,'use_v2_model_runner',False),
        'compile': (cfg.compilation_config,'mode',3),
        'lora': (cfg,'lora_config',object()),
        'tp': (cfg.parallel_config,'tensor_parallel_size',2),
        'pp': (cfg.parallel_config,'pipeline_parallel_size',2),
        'pcp': (cfg.parallel_config,'prefill_context_parallel_size',8),
        'dcp': (cfg.parallel_config,'decode_context_parallel_size',2),
        'dp': (cfg.parallel_config,'data_parallel_size',2),
        'ep_disabled': (cfg.parallel_config,'enable_expert_parallel',False),
        'eplb': (cfg.parallel_config,'enable_eplb',True),
        'elastic': (cfg.parallel_config,'enable_elastic_ep',True),
        'actual_ep': (moe.moe_parallel_config,'ep_size',8),
        'backend': (moe.moe_parallel_config,'all2all_backend','deepep_low_latency'),
        'hidden': (moe,'hidden_dim',4096),
        'input_weight': (moe,'apply_router_weight_on_input',True),
        'dbo': (cfg.parallel_config,'enable_dbo',True),
        'ubatching': (cfg.parallel_config,'ubatch_size',2),
        'async': (cfg.scheduler_config,'async_scheduling',True),
    }
    if change in changes:
        obj, key, value = changes[change]
        setattr(obj, key, value)
    assert not m._supported(cfg, moe, prepare)


@pytest.mark.parametrize('ubatch_size', [0, 1])
def test_supported_scope_and_actual_selector(monkeypatch, ubatch_size):
    m = owned()
    cfg, moe, prepare = scope()
    cfg.parallel_config.ubatch_size = ubatch_size
    monkeypatch.setattr(m, '_device_arch', lambda: 'gfx938')
    assert m._supported(cfg, moe, prepare)
    from hcu_vllm_das.moe.ct_w4a8_prepare import HcuCTHTPrepareAndFinalize
    native = HcuCTHTPrepareAndFinalize(object(), 16, 1, 0)
    monkeypatch.setattr(m, 'enabled', lambda *args: True)
    selected = m.select_prepare(native, cfg, moe)
    assert type(selected) is m.HY4HTInt8PrepareAndFinalize
    assert selected.buffer is native.buffer and selected.rank_expert_offset == native.rank_expert_offset
    assert selected.num_dispatchers() == 16 and selected.dp_size == 1


@pytest.fixture
def active(monkeypatch):
    m = owned()
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '1')
    monkeypatch.setenv('HCU_VLLM_HY4_HT_INT8_DISPATCH', '1')
    from hcu_vllm_das.moe.ct_w4a8_prepare import HcuCTHTPrepareAndFinalize
    p = m.HY4HTInt8PrepareAndFinalize(object(), 16, 1, 0)
    config = NS(quant_dtype=torch.int8, per_act_token_quant=True, block_shape=None)
    return m, p, HcuCTHTPrepareAndFinalize, config


def inputs(rows):
    x = torch.zeros((rows, 6144), dtype=torch.bfloat16)
    return x, torch.ones((rows, 8), dtype=torch.float32), torch.zeros((rows, 8), dtype=torch.int64)


@pytest.mark.parametrize('rows,want_int8', [
    (378, False), (755, False), (756, True), (1134, True), (1512, True), (1890, True),
])
def test_wire_admits_measured_prefill_sizes(active, monkeypatch, rows, want_int8):
    m, p, base, config = active
    x, weights, ids = inputs(rows)
    monkeypatch.setattr(base, 'prepare_async', lambda *a, **k: 'bf16')
    monkeypatch.setattr(m, 'moe_kernel_quantize_input', lambda *a, **k:
        (torch.empty_like(x, dtype=torch.int8), torch.ones((rows, 1), dtype=torch.float32)))
    monkeypatch.setattr(p, '_do_dispatch', lambda **k: 'int8')
    actual = p.prepare_async(x, weights, ids, 272, None, False, config)
    assert actual == ('int8' if want_int8 else 'bf16')


@pytest.mark.parametrize('offset', [-1,0,1])
def test_wire_boundary_and_native_receiver_identity(active, monkeypatch, offset):
    m, p, base, config = active
    x, weights, ids = inputs(m.MIN_TOKENS + offset)
    receiver = object()
    calls = []
    q, scale = torch.empty_like(x, dtype=torch.int8), torch.ones((x.shape[0],1), dtype=torch.float32)
    def quant(value, supplied_scale, **kwargs):
        assert value is x and supplied_scale is None
        assert kwargs == dict(quant_dtype=torch.int8, per_act_token_quant=True, block_shape=None)
        calls.append('quant')
        return q, scale
    def dispatch(**kwargs):
        assert kwargs['tokens'] is q and kwargs['token_scales'] is scale
        assert kwargs['rank_topk_ids'] is ids and kwargs['rank_topk_weights'] is weights
        assert kwargs['defer_input_quant'] is True and kwargs['a1_scale'] is None
        calls.append('dispatch')
        return receiver
    monkeypatch.setattr(m, 'moe_kernel_quantize_input', quant)
    monkeypatch.setattr(p, '_do_dispatch', dispatch)
    monkeypatch.setattr(base, 'prepare_async', lambda *a, **k: calls.append('native') or receiver)
    assert p.prepare_async(x, weights, ids, 272, None, False, config) is receiver
    assert calls == (['native'] if offset < 0 else ['quant','dispatch'])


def test_master_runtime_rollback_uses_original_without_touching_tensor(active, monkeypatch):
    m, p, base, config = active
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '0')
    result = object()
    monkeypatch.setattr(base, 'prepare_async', lambda *a, **k: result)
    monkeypatch.setattr(m, 'moe_kernel_quantize_input', lambda *a, **k: pytest.fail('quantized after rollback'))
    assert p.prepare_async(object(), object(), object(), 272, None, False, config) is result


def test_collective_exception_is_preserved_without_retry(active, monkeypatch):
    m, p, base, config = active
    x, weights, ids = inputs(m.MIN_TOKENS)
    error = RuntimeError('dispatch failed')
    calls = []
    monkeypatch.setattr(m, 'moe_kernel_quantize_input', lambda *a, **k:
        (torch.empty_like(x,dtype=torch.int8), torch.ones((x.shape[0],1),dtype=torch.float32)))
    def dispatch(**kwargs):
        calls.append('dispatch')
        raise error
    monkeypatch.setattr(p, '_do_dispatch', dispatch)
    monkeypatch.setattr(base, 'prepare_async', lambda *a, **k: pytest.fail('retried a collective'))
    with pytest.raises(RuntimeError) as caught:
        p.prepare_async(x, weights, ids, 272, None, False, config)
    assert caught.value is error and calls == ['dispatch']


def test_selector_preserves_disabled_and_foreign_objects(monkeypatch):
    m = owned()
    native = object()
    monkeypatch.setattr(m, 'enabled', lambda *a: False)
    cfg, moe, _ = scope()
    assert m.select_prepare(native, cfg, moe) is native


@pytest.mark.parametrize('kind', ['empty','fp16','stride','hidden','weighted','ndim'])
def test_runtime_fallback_preserves_native_arguments(active, monkeypatch, kind):
    m, p, base, config = active
    x, weights, ids = inputs(0 if kind == 'empty' else m.MIN_TOKENS)
    if kind == 'fp16':
        x = x.to(torch.float16)
    elif kind == 'stride':
        x = torch.empty((m.MIN_TOKENS, 12288), dtype=torch.bfloat16)[:, ::2]
    elif kind == 'hidden':
        x = x[:, :512].contiguous()
    elif kind == 'ndim':
        x = x.reshape(-1)
    result = object()
    def native(*args, **kwargs):
        assert args[1] is x and args[2] is weights and args[3] is ids
        assert args[6] is (kind == 'weighted')
        assert kwargs['defer_input_quant'] is False
        return result
    monkeypatch.setattr(base, 'prepare_async', native)
    monkeypatch.setattr(m, 'moe_kernel_quantize_input', lambda *a, **k: pytest.fail('quantized fallback'))
    assert p.prepare_async(x, weights, ids, 272, None, kind == 'weighted', config) is result


@pytest.mark.parametrize('kind', ['q_dtype','scale_dtype','scale_shape','q_shape','device'])
def test_invalid_quantized_wire_never_enters_collective(active, monkeypatch, kind):
    m, p, base, config = active
    x, weights, ids = inputs(m.MIN_TOKENS)
    q = torch.empty_like(x, dtype=torch.bfloat16 if kind == 'q_dtype' else torch.int8)
    scale = torch.ones((x.shape[0], 2 if kind == 'scale_shape' else 1),
                       dtype=torch.float16 if kind == 'scale_dtype' else torch.float32,
                       device='meta' if kind == 'device' else 'cpu')
    if kind == 'q_shape':
        q = q[:, :512]
    monkeypatch.setattr(m, 'moe_kernel_quantize_input', lambda *a, **k: (q,scale))
    monkeypatch.setattr(p, '_do_dispatch', lambda **k: pytest.fail('invalid wire reached collective'))
    with pytest.raises(RuntimeError, match='INT8'):
        p.prepare_async(x, weights, ids, 272, None, False, config)


def test_one_dimensional_native_scales_normalized_without_quantizing_twice(active, monkeypatch):
    m, p, base, config = active
    x, weights, ids = inputs(m.MIN_TOKENS)
    q, scale = torch.empty_like(x,dtype=torch.int8), torch.ones(x.shape[0],dtype=torch.float32)
    receiver = object()
    monkeypatch.setattr(m, 'moe_kernel_quantize_input', lambda *a, **k: (q,scale))
    def dispatch(**kwargs):
        actual = kwargs['token_scales']
        assert actual.shape == (x.shape[0],1) and actual.data_ptr() == scale.data_ptr()
        return receiver
    monkeypatch.setattr(p, '_do_dispatch', dispatch)
    assert p.prepare_async(x, weights, ids, 272, None, False, config) is receiver
