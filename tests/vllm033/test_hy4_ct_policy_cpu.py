# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HY4 checkpoint admission; no device/provider availability is implied."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from hcu_vllm_das.quantization.hy4_ct_policy import accepts_checkpoint


def checkpoint():
    return {'quant_method': 'compressed-tensors', 'format': 'pack-quantized',
            'config_groups': {'experts': {
                'weights': {'num_bits': 4, 'type': 'int', 'strategy': 'channel',
                            'symmetric': True, 'dynamic': False},
                'input_activations': {'num_bits': 8, 'type': 'int', 'strategy': 'token',
                                      'symmetric': True, 'dynamic': True}}}}


@pytest.mark.parametrize('choice', [None, 'compressed-tensors', 'hcu_hy4_ct_w4a8'])
@pytest.mark.parametrize('model_type', ['hy_v4', 'hy_v4_mtp'])
def test_compatible_hy4_backbone_or_mtp_config_is_selected(choice, model_type):
    # SpeculativeConfig.hf_config_override renames the draft to hy_v4_mtp
    # and propagates the target's selected quantizer to ModelConfig.
    hf = SimpleNamespace(model_type=model_type)
    config = checkpoint()
    before = deepcopy(config)
    assert accepts_checkpoint(config, choice, hf)
    assert config == before


@pytest.mark.parametrize('model_type', ['deepseek_v41', 'hy_v3', 'hy_v4_vl',
                                       'hy_v4_vl_mtp', 'hy_v3_mtp', None])
def test_other_model_families_are_not_claimed(model_type):
    assert not accepts_checkpoint(checkpoint(), None, SimpleNamespace(model_type=model_type))


@pytest.mark.parametrize('choice', [None, 'compressed-tensors', 'hcu_hy4_ct_w4a8'])
def test_standard_vl_outer_config_selects_its_hy4_text_quantizer(choice):
    hf = SimpleNamespace(model_type='hy_v4_vl',
                         text_config=SimpleNamespace(model_type='hy_v4'))
    quant = checkpoint()
    before = deepcopy(quant)
    assert accepts_checkpoint(quant, choice, hf)
    assert quant == before
    assert hf.model_type == 'hy_v4_vl'
    assert hf.text_config.model_type == 'hy_v4'


@pytest.mark.parametrize('text', [None, {}, SimpleNamespace(),
    SimpleNamespace(model_type='hy_v4_flash'), SimpleNamespace(model_type='hy_v3')])
def test_vl_with_missing_or_foreign_text_family_is_not_claimed(text):
    hf = SimpleNamespace(model_type='hy_v4_vl', text_config=text)
    assert not accepts_checkpoint(checkpoint(), None, hf)


@pytest.mark.parametrize('choice', [None, 'compressed-tensors', 'hcu_hy4_ct_w4a8'])
def test_standard_dfly_derived_config_keeps_target_quantizer(choice):
    hf = SimpleNamespace(model_type='hy_v4_dfly', dfly_target_model_type='hy_v4')
    assert accepts_checkpoint(checkpoint(), choice, hf)


@pytest.mark.parametrize('target', [None, 'hy_v4_flash', 'hy_v3', 'qwen3'])
def test_dfly_without_standard_target_marker_is_not_claimed(target):
    hf = SimpleNamespace(model_type='hy_v4_dfly', dfly_target_model_type=target)
    assert not accepts_checkpoint(checkpoint(), None, hf)


@pytest.mark.parametrize('side,key,value', [
    ('weights', 'num_bits', 8), ('weights', 'type', 'float'),
    ('weights', 'strategy', 'group'), ('weights', 'symmetric', False),
    ('weights', 'dynamic', True), ('weights', 'group_size', 128),
    ('weights', 'block_structure', [128, 128]),
    ('input_activations', 'num_bits', 4), ('input_activations', 'dynamic', False),
    ('input_activations', 'type', 'float'), ('input_activations', 'symmetric', False),
])
def test_unsupported_quantization_is_not_selected(side, key, value):
    config = checkpoint()
    config['config_groups']['experts'][side][key] = value
    assert not accepts_checkpoint(config, None, SimpleNamespace(model_type='hy_v4'))


@pytest.mark.parametrize('config', [None, {}, {'config_groups': []}, {'quant_method': 'fp8'}])
def test_malformed_or_different_checkpoint_is_not_claimed(config):
    assert not accepts_checkpoint(config, None, SimpleNamespace(model_type='hy_v4'))


def test_explicit_other_quantizer_and_nonpacked_weights_are_preserved():
    hf = SimpleNamespace(model_type='hy_v4')
    config = checkpoint()
    assert not accepts_checkpoint(config, 'fp8', hf)
    config['format'] = 'dense'
    assert not accepts_checkpoint(config, None, hf)


def test_real_hy4_config_delegates_without_claiming_other_models(monkeypatch):
    """Only unavailable framework/base imports are doubled, not HY4 policy."""
    import importlib
    import sys
    from types import ModuleType
    calls = []
    current = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type='hy_v4')))
    fake_config = ModuleType('vllm.config')
    fake_config.get_current_vllm_config = lambda: current
    fake_base = ModuleType('hcu_vllm_das.quantization.ct_w4a8')
    class Base:
        def get_quant_method(self, layer, prefix):
            calls.append((layer, prefix))
            return 'delegated'
    fake_base.HcuDSV41CTW4A8Config = Base
    platform = ModuleType('hcu_vllm_das.quantization.dsv41_channel_fp8')
    platform._is_hcu = lambda: True
    health = ModuleType('hcu_vllm_das.hy4_quant_activation')
    health.check_health = lambda: calls.append('health')
    # Keep optional dense/provider imports isolated from the fake Core config.
    dense = ModuleType('hcu_vllm_das.hy4_dense_fusion')
    dense.enabled = lambda cfg: False
    linear = ModuleType('hcu_vllm_das.quantization.dsv41_int8_linear')
    linear.HcuDSV41Int8LinearMethod = type('UnusedDenseMethod', (), {})
    for name, value in {'vllm.config': fake_config,
        fake_base.__name__: fake_base, platform.__name__: platform,
        health.__name__: health, dense.__name__: dense,
        linear.__name__: linear}.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(importlib.import_module('hcu_vllm_das'),
                        'hy4_dense_fusion', dense, raising=False)
    name = 'hcu_vllm_das.quantization.hy4_ct_w4a8'
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    cls = module.HcuHY4CTW4A8Config
    try:
        assert cls.override_quantization_method(checkpoint(), None, current.model_config.hf_config) == cls.get_name()
        assert calls == ['health']
        assert cls().get_quant_method('layer', 'model.layers.1') == 'delegated'
        assert calls[-1] == ('layer', 'model.layers.1')
        current.model_config.hf_config.model_type = 'hy_v4_mtp'
        assert cls.override_quantization_method(checkpoint(), cls.get_name(),
            current.model_config.hf_config) == cls.get_name()
        assert cls().get_quant_method('draft', 'model.layers.80') == 'delegated'
        current.model_config.hf_config = SimpleNamespace(model_type='hy_v4_vl',
            text_config=SimpleNamespace(model_type='hy_v4'))
        assert cls.override_quantization_method(checkpoint(), None,
            current.model_config.hf_config) == cls.get_name()
        assert cls().get_quant_method('vision_or_text', 'visual.blocks.0') == 'delegated'
        current.model_config.hf_config.model_type = 'deepseek_v41'
        assert cls.override_quantization_method(checkpoint(), None, current.model_config.hf_config) is None
        with pytest.raises(ValueError, match='HY4'):
            cls().get_quant_method('layer', 'other')
    finally:
        sys.modules.pop(name, None)
