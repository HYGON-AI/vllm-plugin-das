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
def test_aiter_w4a8_cpu_weight_offload_rejected_before_method_selection(context, monkeypatch, for_moe, disable_uva, params):
    from hcu_vllm_das.models.hy_v4_flash.quantization import prepare_flash_quant_config
    monkeypatch.setenv('VLLM_WEIGHT_OFFLOADING_DISABLE_UVA', disable_uva)
    context.offload_config = offload(budget=8, params=params)
    with pytest.raises(NotImplementedError, match='AITER W4A8.*CPU weight offload'):
        prepare_flash_quant_config(SimpleNamespace(), source(), for_moe=for_moe)


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
