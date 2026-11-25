# SPDX-License-Identifier: Apache-2.0
"""CPU contracts; vendor GPU leaves are doubled, no native accuracy claim."""
from types import SimpleNamespace
import importlib

import pytest
import torch


def provider():
    return importlib.import_module('hcu_vllm_das.models.hy_v4_flash.aiter_w4a8_moe')


def test_all_byte_patterns_preserve_both_signed_nibbles():
    module = provider()
    source = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 256)
    result = module.ct_to_aiter_packed(source)
    decoded_ct = torch.stack([(source & 15).short() - 8, (source >> 4).short() - 8], -1)
    high, low = (result >> 4).short(), (result & 15).short()
    decoded_native = torch.stack([torch.where(high >= 8, high - 16, high), torch.where(low >= 8, low - 16, low)], -1)
    assert torch.equal(decoded_ct, decoded_native)
    assert result.data_ptr() != source.data_ptr()
    assert source.flatten().tolist() == list(range(256))


@pytest.mark.parametrize('value', [torch.zeros(2, dtype=torch.uint8), torch.zeros(1, 2, 8, dtype=torch.int32), torch.zeros(1, 8, 8, dtype=torch.uint8)[:, :, ::2]])
def test_invalid_packed_contract_refused(value):
    with pytest.raises(ValueError):
        provider().ct_to_aiter_packed(value)


def fixture(monkeypatch, *, layout='raw'):
    module = provider()
    calls = []
    method = module.HYV4FlashAiterW4A8Method.__new__(module.HYV4FlashAiterW4A8Method)
    method.moe = SimpleNamespace(tp_size=4, moe_backend='aiter', max_num_tokens=64, experts_per_token=2)
    method.hidden = method.intermediate = 8
    method._weights_processed = False
    method._graph_admitted = True
    layer = torch.nn.Module()
    fields = {'w13_weight_packed': torch.zeros(2, 16, 1, dtype=torch.int32),
              'w2_weight_packed': torch.zeros(2, 8, 1, dtype=torch.int32),
              'w13_weight_scale': torch.ones(2, 16, 1, dtype=torch.bfloat16),
              'w2_weight_scale': torch.ones(2, 8, 1, dtype=torch.bfloat16),
              'w13_weight_shape': torch.tensor([[32, 8], [32, 8]]),
              'w2_weight_shape': torch.tensor([[8, 32], [8, 32]])}
    for name, value in fields.items():
        setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))
    layer.swiglu_limit = 10.
    layer.apply_router_weight_on_input = False
    qt = SimpleNamespace(per_Token='token')
    def config(*args, **kwargs):
        calls.append(('config', args, kwargs))
        return True, SimpleNamespace(solution_type='triton', need_shuffle=layout != 'raw')
    def prepare(w1, w2, **kwargs):
        calls.append(('prepare', (w1, w2), kwargs))
        return w1, w2, SimpleNamespace(layout=layout, w1_scale=kwargs['w1_scale'], w2_scale=kwargs['w2_scale'])
    def execute(*args, **kwargs):
        # The native sorter stores one slot per (expert, token). This leaf
        # checks its real contract independently of the caller's algorithm.
        ordered = args[4].sort(1).values
        assert bool((ordered[:, 1:] != ordered[:, :-1]).all()), 'native sorter requires unique expert IDs in each row'
        calls.append(('execute', args, kwargs))
        return torch.zeros_like(args[0])
    vendor = SimpleNamespace(config=config, prepare=prepare, execute=execute,
                             quant_type=qt.per_Token, w4a8='w4a8')
    monkeypatch.setattr(module, '_vendor_api', lambda: vendor)
    monkeypatch.setattr(module, '_device_supported', lambda value: True)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    return module, method, layer, calls, vendor


@pytest.mark.parametrize('layout', ['raw', 'asm_shuffle'])
def test_public_decision_and_storage_ownership_are_frozen(monkeypatch, layout):
    module, method, layer, calls, vendor = fixture(monkeypatch, layout=layout)
    owners = (layer.w13_weight_packed, layer.w2_weight_packed)
    method.process_weights_after_loading(layer)
    assert [c[0] for c in calls] == ['config', 'prepare']
    assert calls[0][1][:6] == (1, 2, 16, 8, 8, 2)
    assert calls[1][2]['quant_dtype_a'] is torch.int8
    assert calls[1][2]['backend'] == 'auto'
    assert layer.w13_weight_packed is owners[0] and layer.w2_weight_packed is owners[1]
    assert layer.w13_weight_packed.dtype == layer.w2_weight_packed.dtype == torch.int32
    assert layer.w13_weight_packed.shape == (2, 16, 1)
    assert layer.w2_weight_packed.shape == (2, 8, 1)
    assert method._native_weights[0].data_ptr() == layer.w13_weight_packed.data_ptr()
    assert method._native_weights[1].data_ptr() == layer.w2_weight_packed.data_ptr()
    assert method._prepared.layout == layout
    with pytest.raises(RuntimeError, match='repack|processed'):
        method.process_weights_after_loading(layer)


def test_apply_maps_integer_activation_clipping_and_propagates_vendor_error(monkeypatch):
    module, method, layer, calls, vendor = fixture(monkeypatch)
    method.process_weights_after_loading(layer)
    x = torch.zeros(1, 8, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1]], dtype=torch.int32)
    weights = torch.ones(1, 2, dtype=torch.float32) / 2
    result = method.apply(layer, x, weights, ids, None, None)
    assert result.shape == x.shape
    assert calls[-1][2]['gemm1_limit'] == 10.
    assert calls[-1][2]['quant_dtype_a'] is torch.int8
    assert calls[-1][2]['prepared'] is method._prepared
    assert 'swiglu_limit' not in calls[-1][2]
    def fail(*args, **kwargs):
        raise RuntimeError('native launch failed')
    vendor.execute = fail
    with pytest.raises(RuntimeError, match='native launch failed'):
        method.apply(layer, x, weights, ids, None, None)


def test_changed_parameter_storage_cannot_reuse_prepared_graph(monkeypatch):
    module, method, layer, calls, vendor = fixture(monkeypatch)
    method.process_weights_after_loading(layer)
    layer.w13_weight_packed.data = layer.w13_weight_packed.clone()
    with pytest.raises(RuntimeError, match='storage|lifecycle'):
        method.apply(layer, torch.zeros(1, 8, dtype=torch.bfloat16), torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.int32), None, None)


def test_explicit_backend_required_before_postload(monkeypatch):
    module, method, layer, calls, vendor = fixture(monkeypatch)
    method.moe.moe_backend = 'auto'
    with pytest.raises(ValueError, match='aiter'):
        method.process_weights_after_loading(layer)
    assert not calls


def test_vendor_prepare_failure_leaves_method_unusable(monkeypatch):
    module, method, layer, calls, vendor = fixture(monkeypatch)
    def fail(*args, **kwargs):
        raise RuntimeError('layout unavailable')
    vendor.prepare = fail
    with pytest.raises(RuntimeError, match='layout unavailable'):
        method.process_weights_after_loading(layer)
    assert not method._weights_processed


def test_padded_nan_rows_and_minus_one_routes_are_sanitized_on_device(monkeypatch):
    module, method, layer, calls, vendor = fixture(monkeypatch)
    method.process_weights_after_loading(layer)
    x = torch.ones(2, 8, dtype=torch.bfloat16)
    x[0].fill_(float('nan'))
    ids = torch.tensor([[-1, -1], [0, -1]], dtype=torch.int32)
    weights = torch.tensor([[float('nan'), float('nan')], [1., float('nan')]])
    result = method.apply(layer, x, weights, ids, None, None)
    native_args = calls[-1][1]
    assert torch.equal(native_args[0][0], torch.zeros(8, dtype=torch.bfloat16))
    assert torch.equal(native_args[0][1], x[1])
    assert torch.equal(native_args[3], torch.tensor([[0., 0.], [1., 0.]]))
    assert torch.equal(native_args[4].sort(1).values, torch.tensor([[0, 1], [0, 1]], dtype=torch.int32))
    assert torch.isfinite(result).all()


def test_out_of_range_route_refused_before_vendor_call(monkeypatch):
    module, method, layer, calls, vendor = fixture(monkeypatch)
    method.process_weights_after_loading(layer)
    with pytest.raises(RuntimeError, match='expert ID'):
        method.apply(layer, torch.zeros(1, 8, dtype=torch.bfloat16), torch.ones(1, 2), torch.tensor([[0, 2]], dtype=torch.int32), None, None)
    assert not any(c[0] == 'execute' for c in calls)


def test_missing_vendor_fails_closed(monkeypatch):
    module = provider()
    import sys
    monkeypatch.setitem(sys.modules, 'aiter.fused_moe', None)
    with pytest.raises(ModuleNotFoundError):
        module._vendor_api()


def test_callable_adapter_missing_sentinel_is_not_a_vendor_api(monkeypatch):
    module = provider()
    from types import ModuleType
    from hcu_adapter._base import Missing
    import sys
    root = ModuleType('aiter')
    root.__path__ = []
    root.QuantType = SimpleNamespace(per_Token='token')
    functions = ModuleType('aiter.fused_moe')
    functions.fused_moe = Missing('aiter.fused_moe', 'aiter')
    layout = ModuleType('aiter.fused_moe_layout')
    layout.fused_moe_prepare_weights = lambda *a, **kw: None
    configs = ModuleType('aiter.moe')
    configs.MoeQuantType = SimpleNamespace(W4A8='w4a8')
    configs.get_aiter_moe_config = lambda *a, **kw: None
    for name, value in [('aiter', root), ('aiter.fused_moe', functions), ('aiter.fused_moe_layout', layout), ('aiter.moe', configs)]:
        monkeypatch.setitem(sys.modules, name, value)
    assert callable(functions.fused_moe)
    with pytest.raises(RuntimeError, match='unavailable'):
        module._vendor_api()


def test_partial_padding_does_not_collide_with_valid_expert_zero(monkeypatch):
    module, method, layer, calls, vendor = fixture(monkeypatch)
    method.process_weights_after_loading(layer)
    result = method.apply(layer, torch.ones(1, 8, dtype=torch.bfloat16),
                          torch.tensor([[1., float('nan')]]),
                          torch.tensor([[0, -1]], dtype=torch.int32), None, None)
    assert torch.equal(calls[-1][1][4], torch.tensor([[0, 1]], dtype=torch.int32))
    assert torch.equal(calls[-1][1][3], torch.tensor([[1., 0.]]))
    assert torch.isfinite(result).all()


def test_valid_ids_zero_weights_and_nan_activations_return_zero(monkeypatch):
    module, method, layer, calls, vendor = fixture(monkeypatch)
    method.process_weights_after_loading(layer)
    original = vendor.execute
    def execute(*args, **kwargs):
        original(*args, **kwargs)
        assert torch.equal(args[0], torch.zeros(1, 8, dtype=torch.bfloat16))
        # Native output for a completely zero contribution may be undefined;
        # the caller must mask it before it can enter the model.
        return torch.full_like(args[0], float('nan'))
    vendor.execute = execute
    result = method.apply(layer, torch.full((1, 8), float('nan'), dtype=torch.bfloat16),
                          torch.zeros(1, 2), torch.tensor([[0, 1]], dtype=torch.int32), None, None)
    assert torch.equal(result, torch.zeros_like(result))


def test_nonzero_contribution_predicate_does_not_hide_valid_nan_weights(monkeypatch):
    module, method, layer, calls, vendor = fixture(monkeypatch)
    method.process_weights_after_loading(layer)
    original = vendor.execute
    def execute(*args, **kwargs):
        original(*args, **kwargs)
        assert torch.equal(args[0], torch.ones(1, 8, dtype=torch.bfloat16))
        assert torch.isnan(args[3][0, 0])
        return torch.full_like(args[0], float('nan'))
    vendor.execute = execute
    result = method.apply(layer, torch.ones(1, 8, dtype=torch.bfloat16),
                          torch.tensor([[float('nan'), 0.]]),
                          torch.tensor([[0, 1]], dtype=torch.int32), None, None)
    assert torch.isnan(result).all()


@pytest.mark.parametrize('experts,topk', [(2, 2), (4, 4), (512, 16)])
def test_dummy_ids_preserve_valid_routes_and_fill_distinct_free_ids(experts, topk):
    module = provider()
    torch.manual_seed(177 + experts)
    source = torch.rand(37, experts).argsort(1)[:, :topk].int()
    source[torch.rand_like(source.float()) < .5] = -1
    source[-1] = -1
    result = module._unique_padded_route_ids(source)
    assert torch.equal(source[source >= 0], result[source >= 0])
    assert bool(((result >= 0) & (result < experts)).all())
    ordered = result.sort(1).values
    assert bool((ordered[:, 1:] != ordered[:, :-1]).all())
