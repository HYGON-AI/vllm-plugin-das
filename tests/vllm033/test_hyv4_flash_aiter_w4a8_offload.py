# SPDX-License-Identifier: Apache-2.0
"""CPU admission for native packed experts; Core offload configs are real."""
from types import SimpleNamespace

import pytest

from test_hyv4_flash_aiter_w4a8_selector import context, source


def offload(*, budget=0, backend='auto', params=(), group_size=0):
    from vllm.config.offload import OffloadConfig, PrefetchOffloadConfig, UVAOffloadConfig
    return OffloadConfig(offload_backend=backend,
                         uva=UVAOffloadConfig(cpu_offload_gb=budget, cpu_offload_params=set(params)),
                         prefetch=PrefetchOffloadConfig(offload_group_size=group_size))


@pytest.mark.parametrize('for_moe', [False, True], ids=['dense-dfly-admission', 'routed-admission'])
@pytest.mark.parametrize('disable_uva', ['0', '1'])
@pytest.mark.parametrize('params', [(), ('experts',)], ids=['all-params', 'experts-only'])
def test_aiter_w4a8_cpu_weight_offload_admits_only_non_uva(context, monkeypatch, for_moe, disable_uva, params):
    from hcu_vllm_das.models.hy_v4_flash.quantization import prepare_flash_quant_config
    monkeypatch.setenv('VLLM_WEIGHT_OFFLOADING_DISABLE_UVA', disable_uva)
    context.offload_config = offload(budget=8, params=params)
    if disable_uva == '0':
        with pytest.raises(NotImplementedError, match='AITER W4A8.*CPU weight offload'):
            prepare_flash_quant_config(SimpleNamespace(), source(), for_moe=for_moe)
    else:
        from hcu_vllm_das.models.hy_v4_flash.aiter_w4a8_moe import HYV4FlashAiterW4A8Config
        original = source()
        selected = prepare_flash_quant_config(SimpleNamespace(), original, for_moe=for_moe)
        assert isinstance(selected, HYV4FlashAiterW4A8Config) if for_moe else selected is original


def test_aiter_w4a8_prefetch_offload_rejected_before_method_selection(context):
    from hcu_vllm_das.models.hy_v4_flash.quantization import prepare_flash_quant_config
    context.offload_config = offload(backend='prefetch', group_size=2)
    with pytest.raises(NotImplementedError, match='AITER W4A8.*CPU weight offload'):
        prepare_flash_quant_config(SimpleNamespace(), source(), for_moe=True)


@pytest.mark.parametrize('for_moe', [False, True])
@pytest.mark.parametrize('backend', ['auto', 'uva'])
def test_aiter_w4a8_zero_offload_budget_keeps_original_admission(context, for_moe, backend):
    from hcu_vllm_das.models.hy_v4_flash.quantization import prepare_flash_quant_config
    from hcu_vllm_das.models.hy_v4_flash.aiter_w4a8_moe import HYV4FlashAiterW4A8Config
    context.offload_config = offload(backend=backend)
    original = source()
    selected = prepare_flash_quant_config(SimpleNamespace(), original, for_moe=for_moe)
    assert isinstance(selected, HYV4FlashAiterW4A8Config) if for_moe else selected is original


def test_exact_reference_cpu_offload_selection_is_unchanged(context):
    from hcu_vllm_das.models.hy_v4_flash.quantization import prepare_flash_quant_config
    from hcu_vllm_das.models.hy_v4_flash.w4a8 import HYV4FlashW4A8Config
    context.kernel_config.moe_backend = 'auto'
    context.offload_config = offload(budget=8)
    selected = prepare_flash_quant_config(SimpleNamespace(flash_w4a8_backend='adapter_reference'), source(), for_moe=True)
    assert type(selected) is HYV4FlashW4A8Config


def test_fp8_cpu_offload_selection_is_unchanged(context):
    from test_hyv4_flash_aiter_fp8_moe import _recipe
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
    from hcu_vllm_das.models.hy_v4_flash.quantization import prepare_flash_quant_config
    from hcu_vllm_das.models.hy_v4_flash.aiter_fp8_moe import HYV4FlashAiterFP8Config
    context.offload_config = offload(budget=8)
    selected = prepare_flash_quant_config(SimpleNamespace(), CompressedTensorsConfig.from_config(_recipe()), for_moe=True)
    assert isinstance(selected, HYV4FlashAiterFP8Config)


def test_core_functional_call_parameter_swap_breaks_frozen_native_aliases(monkeypatch):
    # Core's non-UVA offloader creates device_state then functional_call swaps
    # these Parameters during forward. Native views are not module Parameters.
    import torch
    from test_hyv4_flash_aiter_w4a8_moe import fixture
    module, method, layer, calls, vendor = fixture(monkeypatch)
    method.process_weights_after_loading(layer)
    original = module._identity(layer)
    native_pointers = tuple(w.data_ptr() for w in method._native_weights)
    layer.forward = lambda x: method.apply(layer, x, torch.ones(1, 2), torch.tensor([[0, 1]], dtype=torch.int32), None, None)
    device_state = {name: p.detach().clone() for name, p in layer.state_dict().items()}
    with pytest.raises(RuntimeError, match='storage/lifecycle changed'):
        torch.func.functional_call(layer, device_state, (torch.zeros(1, 8, dtype=torch.bfloat16),), tie_weights=False)
    assert module._identity(layer) == original
    assert tuple(w.data_ptr() for w in method._native_weights) == native_pointers
    assert not any(c[0] == 'execute' for c in calls)


def test_create_weights_freezes_non_uva_mode_while_core_context_exists(context, monkeypatch):
    from test_hyv4_flash_aiter_w4a8_moe import fixture
    module, method, layer, calls, vendor = fixture(monkeypatch)
    context.offload_config = offload(budget=8)
    monkeypatch.setenv('VLLM_WEIGHT_OFFLOADING_DISABLE_UVA', '1')
    monkeypatch.setattr(module.HYV4FlashChannelW4A8Method, 'create_weights', lambda *a, **kw: None)
    method.create_weights(layer)
    assert method._raw_offload is True


def test_non_uva_raw_uses_live_functional_call_parameters_and_scales(monkeypatch):
    import torch
    from test_hyv4_flash_aiter_w4a8_moe import fixture
    module, method, layer, calls, vendor = fixture(monkeypatch)
    method._raw_offload = True
    method.process_weights_after_loading(layer)
    assert method._prepared is None and method._native_weights is None
    seen = []
    def execute(x, w1, w2, weights, ids, **kwargs):
        assert kwargs.get('prepared') is None
        assert w1.data_ptr() == layer.w13_weight_packed.data_ptr()
        assert w2.data_ptr() == layer.w2_weight_packed.data_ptr()
        assert kwargs['w1_scale'] is layer.w13_weight_scale
        assert kwargs['w2_scale'] is layer.w2_weight_scale
        seen.append((w1.data_ptr(), kwargs['w1_scale'].data_ptr()))
        return x * kwargs['w1_scale'][0, 0, 0] + w1.flatten()[0].float()
    vendor.execute = execute
    x = torch.ones(1, 8, dtype=torch.bfloat16)
    layer.forward = lambda values: method.apply(layer, values, torch.ones(1, 2), torch.tensor([[0, 1]], dtype=torch.int32), None, None)
    original = module._identity(layer)
    for generation in (1, 2):
        live = {name: p.detach().clone() for name, p in layer.state_dict().items()}
        live['w13_weight_packed'].add_(generation * 0x01010101)
        live['w13_weight_scale'].fill_(generation + 1)
        result = torch.func.functional_call(layer, live, (x,), tie_weights=False)
        assert torch.equal(result, x * (generation + 1) + (136 + generation))
        assert module._identity(layer) == original
    assert len(seen) == 2
    assert [c[0] for c in calls] == ['config', 'prepare']


def test_non_uva_raw_drops_all_postload_native_and_prepared_tensor_refs(monkeypatch):
    import gc
    import weakref
    from test_hyv4_flash_aiter_w4a8_moe import fixture
    module, method, layer, calls, vendor = fixture(monkeypatch)
    method._raw_offload = True
    refs = []
    class Prepared:
        layout = 'raw'
    def prepare(w1, w2, **kwargs):
        p = Prepared()
        p.w1_scale = kwargs['w1_scale'].detach().clone()
        p.w2_scale = kwargs['w2_scale'].detach().clone()
        refs.extend(weakref.ref(v) for v in (w1, w2, p, p.w1_scale, p.w2_scale))
        return w1, w2, p
    vendor.prepare = prepare
    method.process_weights_after_loading(layer)
    # Simulate Core restoring the processed parameters to their CPU owner.
    # This CPU test verifies Python ownership, not actual accelerator release.
    for p in layer.parameters():
        p.data = p.detach().clone()
    calls.clear()
    gc.collect()
    assert method._prepared is None and method._native_weights is None
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize('layout', ['asm_shuffle', 'marlin'])
def test_non_uva_offload_rejects_nonraw_public_prepare_decision(monkeypatch, layout):
    from test_hyv4_flash_aiter_w4a8_moe import fixture
    module, method, layer, calls, vendor = fixture(monkeypatch, layout=layout)
    method._raw_offload = True
    with pytest.raises(NotImplementedError, match='offload.*raw'):
        method.process_weights_after_loading(layer)
    assert not method._weights_processed


@pytest.mark.parametrize('change', ['shape', 'stride', 'dtype'])
def test_non_uva_live_storage_schema_is_still_guarded(monkeypatch, change):
    import torch
    from test_hyv4_flash_aiter_w4a8_moe import fixture
    module, method, layer, calls, vendor = fixture(monkeypatch)
    method._raw_offload = True
    method.process_weights_after_loading(layer)
    value = layer.w13_weight_packed
    if change == 'shape':
        layer.w13_weight_packed.data = value.reshape(1, 32, 1)
    elif change == 'stride':
        layer.w13_weight_packed.data = torch.empty(2, 16, 2, dtype=value.dtype)[:, :, ::2]
    else:
        layer.w13_weight_packed.data = value.to(torch.int64)
    with pytest.raises(RuntimeError, match='raw offload.*schema'):
        method.apply(layer, torch.zeros(1, 8, dtype=torch.bfloat16), torch.ones(1, 2), torch.tensor([[0, 1]], dtype=torch.int32), None, None)
    assert not any(c[0] == 'execute' for c in calls)


def test_non_uva_raw_cannot_repack_processed_weights(monkeypatch):
    from test_hyv4_flash_aiter_w4a8_moe import fixture
    module, method, layer, calls, vendor = fixture(monkeypatch)
    method._raw_offload = True
    method.process_weights_after_loading(layer)
    with pytest.raises(RuntimeError, match='repack|processed'):
        method.process_weights_after_loading(layer)
