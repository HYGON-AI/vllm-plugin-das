# SPDX-License-Identifier: Apache-2.0
"""Real Core CT selector contracts; no GPU arithmetic is simulated or claimed."""
from copy import deepcopy
from types import SimpleNamespace

import pytest


def recipe():
    def group(bits, targets):
        return {'targets': targets,
                'format': 'pack-quantized' if bits == 4 else 'int-quantized',
                'weights': {'num_bits': bits, 'type': 'int', 'strategy': 'channel',
                            'symmetric': True, 'dynamic': False},
                'input_activations': {'num_bits': 8, 'type': 'int', 'strategy': 'token',
                                      'symmetric': True, 'dynamic': True}}
    return {'quant_method': 'compressed-tensors', 'format': 'pack-quantized',
            'quantization_status': 'compressed', 'ignore': ['lm_head'],
            'config_groups': {'experts': group(4, ['re:.*experts.*']),
                              'linears': group(8, ['re:.*q_b_proj.*'])}}


@pytest.fixture
def context(monkeypatch):
    import vllm.config
    from hcu_vllm_das.models.hy_v4_flash import runtime
    from test_hyv4_flash_w4a8_graph_admission import current
    cfg = current()
    cfg.model_config.enforce_eager = True
    cfg.kernel_config = SimpleNamespace(moe_backend='aiter')
    monkeypatch.setattr(vllm.config, 'get_current_vllm_config', lambda: cfg)
    monkeypatch.setattr(runtime, 'is_hcu_platform', lambda: True)
    return cfg


def source():
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
    return CompressedTensorsConfig.from_config(recipe())


@pytest.mark.parametrize('for_moe', [False, True], ids=['dense-dfly', 'routed'])
@pytest.mark.parametrize('flag', ['disabled', 'adapter_reference'])
def test_explicit_aiter_admits_original_metadata_without_mutation(context, for_moe, flag):
    from hcu_vllm_das.models.hy_v4_flash.quantization import prepare_flash_quant_config
    from hcu_vllm_das.models.hy_v4_flash.aiter_w4a8_moe import HYV4FlashAiterW4A8Config
    original = source()
    serialized, targets = deepcopy(original.config), deepcopy(original.target_scheme_map)
    model = SimpleNamespace(flash_w4a8_backend=flag)
    selected = prepare_flash_quant_config(model, original, for_moe=for_moe)
    if for_moe:
        assert isinstance(selected, HYV4FlashAiterW4A8Config)
        assert selected is not original
    else:
        assert selected is original
    assert original.config == serialized and original.target_scheme_map == targets
    assert model.flash_w4a8_backend == flag


def test_auto_disabled_still_refuses_implicit_w4a8(context):
    from hcu_vllm_das.models.hy_v4_flash.quantization import prepare_flash_quant_config
    context.kernel_config.moe_backend = 'auto'
    with pytest.raises(ValueError, match='W4A8'):
        prepare_flash_quant_config(SimpleNamespace(flash_w4a8_backend='disabled'), source(), for_moe=True)


def test_auto_reference_keeps_exact_ct_method(context):
    from hcu_vllm_das.models.hy_v4_flash.quantization import prepare_flash_quant_config
    from hcu_vllm_das.models.hy_v4_flash.w4a8 import HYV4FlashW4A8Config
    context.kernel_config.moe_backend = 'auto'
    selected = prepare_flash_quant_config(SimpleNamespace(flash_w4a8_backend='adapter_reference'), source(), for_moe=True)
    assert type(selected) is HYV4FlashW4A8Config


def test_explicit_aiter_cannot_admit_foreign_platform(context, monkeypatch):
    from hcu_vllm_das.models.hy_v4_flash import runtime
    from hcu_vllm_das.models.hy_v4_flash.quantization import prepare_flash_quant_config
    monkeypatch.setattr(runtime, 'is_hcu_platform', lambda: False)
    with pytest.raises(ValueError, match='HCU'):
        prepare_flash_quant_config(SimpleNamespace(), source(), for_moe=True)


@pytest.mark.parametrize('field,value', [('enable_expert_parallel', True), ('data_parallel_size', 2), ('use_sequence_parallel_moe', True)])
def test_explicit_aiter_preserves_parallel_admission(context, field, value):
    from hcu_vllm_das.models.hy_v4_flash.quantization import prepare_flash_quant_config
    setattr(context.parallel_config, field, value)
    with pytest.raises(NotImplementedError, match='ordinary TP'):
        prepare_flash_quant_config(SimpleNamespace(), source(), for_moe=True)


def test_explicit_aiter_cannot_drop_int8_activation_metadata(context):
    from hcu_vllm_das.models.hy_v4_flash.quantization import prepare_flash_quant_config
    original = source()
    original.config['config_groups']['experts']['input_activations'] = None
    with pytest.raises(ValueError, match='W4A8'):
        prepare_flash_quant_config(SimpleNamespace(), original, for_moe=True)


def test_explicit_aiter_preserves_qualified_graph_gate(context, monkeypatch):
    import vllm.compilation.breakable_cudagraph
    from hcu_vllm_das.models.hy_v4_flash.quantization import prepare_flash_quant_config
    monkeypatch.setattr(vllm.compilation.breakable_cudagraph, 'is_breakable_cudagraph_enabled', lambda: False)
    context.model_config.enforce_eager = False
    context.parallel_config.tensor_parallel_size = 2
    with pytest.raises(NotImplementedError, match='FULL_AND_PIECEWISE'):
        prepare_flash_quant_config(SimpleNamespace(), source(), for_moe=True)


def test_explicit_aiter_accepts_qualified_graph_with_original_metadata(context, monkeypatch):
    import vllm.compilation.breakable_cudagraph
    from hcu_vllm_das.models.hy_v4_flash.quantization import prepare_flash_quant_config
    from hcu_vllm_das.models.hy_v4_flash.aiter_w4a8_moe import HYV4FlashAiterW4A8Config
    monkeypatch.setattr(vllm.compilation.breakable_cudagraph, 'is_breakable_cudagraph_enabled', lambda: False)
    context.model_config.enforce_eager = False
    selected = prepare_flash_quant_config(SimpleNamespace(), source(), for_moe=True)
    assert isinstance(selected, HYV4FlashAiterW4A8Config)


def test_explicit_aiter_cannot_admit_groupwise_weight_recipe(context):
    from hcu_vllm_das.models.hy_v4_flash.quantization import prepare_flash_quant_config
    original = source()
    original.config['config_groups']['experts']['weights']['strategy'] = 'group'
    with pytest.raises(ValueError, match='W4A8'):
        prepare_flash_quant_config(SimpleNamespace(), original, for_moe=True)


def test_explicit_aiter_fp8_selection_is_unchanged(context):
    from test_hyv4_flash_aiter_fp8_moe import _recipe
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
    from hcu_vllm_das.models.hy_v4_flash.quantization import prepare_flash_quant_config
    from hcu_vllm_das.models.hy_v4_flash.aiter_fp8_moe import HYV4FlashAiterFP8Config
    selected = prepare_flash_quant_config(SimpleNamespace(), CompressedTensorsConfig.from_config(_recipe()), for_moe=True)
    assert isinstance(selected, HYV4FlashAiterFP8Config)
