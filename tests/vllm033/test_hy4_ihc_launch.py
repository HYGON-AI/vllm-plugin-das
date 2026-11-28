# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Owned optional launch policy; native constructors/bytecode stay intact."""
from importlib import import_module, util
from types import SimpleNamespace as NS

import pytest
import torch


def module():
    name = 'hcu_vllm_das.hy4_ihc_launch'
    assert util.find_spec(name) is not None, 'Owned native iHC launch policy is missing'
    return import_module(name)


def config(**changes):
    result = NS(model_config=NS(dtype=torch.bfloat16,
        hf_config=NS(enable_ihc=True, hidden_size=6144, hc_mult=4)),
        compilation_config=NS(mode=0), use_v2_model_runner=True,
        speculative_config=None)
    for name, value in changes.items():
        setattr(result, name, value)
    return result


@pytest.mark.parametrize('master,child,expected', [
    (None, None, False), ('0', '1', False), ('1', '0', False),
    ('1', '1', True), ('true', 'on', True), ('off', 'yes', False)])
def test_optional_master_and_child_admission(monkeypatch, master, child, expected):
    owned = module()
    for key, value in [('VLLM_HCU_USE_CUSTOM_OPS', master),
                       ('HCU_VLLM_HY4_IHC_LAUNCH', child)]:
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    monkeypatch.setattr(owned, '_supported_config', lambda cfg: True)
    assert owned.enabled(config()) is expected


def test_invalid_flag_is_rejected(monkeypatch):
    owned = module()
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '1')
    monkeypatch.setenv('HCU_VLLM_HY4_IHC_LAUNCH', 'invalid')
    with pytest.raises(ValueError, match='boolean'):
        owned.enabled(config())


@pytest.mark.parametrize('change', ['hardware', 'width', 'channels', 'ihc', 'dtype', 'compile', 'runner'])
def test_unsupported_configs_retain_native(monkeypatch, change):
    owned = module()
    monkeypatch.setattr(owned, '_device_arch', lambda: 'gfx936' if change == 'hardware' else 'gfx938')
    cfg = config()
    if change == 'width': cfg.model_config.hf_config.hidden_size = 8192
    if change == 'channels': cfg.model_config.hf_config.hc_mult = 8
    if change == 'ihc': cfg.model_config.hf_config.enable_ihc = False
    if change == 'dtype': cfg.model_config.dtype = torch.float32
    if change == 'compile': cfg.compilation_config.mode = 3
    if change == 'runner': cfg.use_v2_model_runner = False
    assert owned._supported_config(cfg) is False


def test_launch_bindings_keep_native_code_and_precision():
    owned = module()
    for original, candidate in [
        (owned.kernels.triton_ihc_pre, owned.tuned_pre),
        (owned.kernels.triton_ihc_post_pre_rms_norm, owned.tuned_chain)]:
        assert candidate.__code__ is original.__code__
        assert candidate.__defaults__ == original.__defaults__
        assert candidate.__globals__['_PRE_STAGE1_WARPS'] == 4
        assert candidate.__globals__['_PRE_STAGE2_WARPS'] == 4
        assert original.__globals__['_PRE_STAGE1_WARPS'] == 8
        assert original.__globals__['_PRE_STAGE2_WARPS'] == 4
    assert owned.TunedHCPre.forward.__code__ is owned.hc.HYV4HCPreLayer.forward.__code__
    assert owned.TunedHCPre.forward.__globals__['triton_ihc_pre'] is owned.tuned_pre
    assert owned.hc.HYV4HCPreLayer.forward.__globals__['triton_ihc_pre'] is owned.kernels.triton_ihc_pre
    assert owned.TunedDecoder.forward is owned.native.HYV4DecoderLayer.forward
    assert owned.TunedDecoder._forward_ihc.__code__ is owned.native.HYV4DecoderLayer._forward_ihc.__code__


@pytest.mark.parametrize('selected', [False, True])
@pytest.mark.parametrize('kind', ['ct', 'static'])
def test_backbone_factory_keeps_constructor_arguments(monkeypatch, selected, kind):
    owned = module()
    from hcu_vllm_das.models import hy4_ct as ct, hy4_static as static
    calls = []
    monkeypatch.setattr(owned, 'enabled', lambda cfg: selected)
    owner = ct if kind == 'ct' else static
    base = 'HcuHY4CTBackbone' if kind == 'ct' else 'HcuHY4StaticBackbone'
    tuned = 'HcuHY4CTLaunchBackbone' if kind == 'ct' else 'HcuHY4StaticLaunchBackbone'
    monkeypatch.setattr(owner, base,
                        lambda *a, **kw: calls.append(('native', a, kw)) or 'native')
    monkeypatch.setattr(owner, tuned,
                        lambda *a, **kw: calls.append(('tuned', a, kw)) or 'tuned')
    cfg = config()
    kwargs = {'vllm_config': cfg, 'prefix': 'model'}
    factory = owner.make_ct_backbone if kind == 'ct' else owner.make_static_backbone
    assert factory(**kwargs) == ('tuned' if selected else 'native')
    assert calls == [('tuned' if selected else 'native', (), kwargs)]


def test_backbone_and_static_keep_native_loops_and_mtp():
    owned = module()
    from hcu_vllm_das.models import hy4_ct as ct, hy4_static as static
    assert ct.HcuHY4CTBackbone.forward is owned.native.HYV4Model.forward
    assert ct.HcuHY4CTBackbone.__init__.__code__ is owned.native.HYV4Model.__init__.__code__
    assert static.HcuHY4StaticBackbone.forward is owned.native.HYV4Model.forward
    assert static.HcuHY4StaticMTP.forward is static.mtp.HYV4MTP.forward
    assert static.HcuHY4StaticLaunchDecoder._forward_ihc is owned.TunedDecoder._forward_ihc
    assert owned.native.HYV4DecoderLayer._forward_ihc.__globals__['triton_ihc_post_pre_rms_norm'] is owned.kernels.triton_ihc_post_pre_rms_norm


def _raw_constructor(function):
    while function.__module__ == 'vllm.compilation.decorators':
        cells = dict(zip(function.__code__.co_freevars, function.__closure__))
        function = cells['old_init'].cell_contents
    return function


@pytest.mark.parametrize('kind', ['ct', 'static', 'ct-launch', 'static-launch'])
def test_actual_native_backbone_constructor_keeps_decoder_type_check(monkeypatch, kind):
    """Execute the full native ctor body; only heavy layer leaves are doubled."""
    owned = module()
    from hcu_vllm_das.hy4_constructor import bind_constructor
    from hcu_vllm_das.models import hy4_ct as ct, hy4_static as static

    def tiny_decoder(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.block_type = 'feedforward'

    for cls in [owned.native.HYV4DecoderLayer, owned.TunedDecoder,
                static.HcuHY4StaticDecoder, static.HcuHY4StaticLaunchDecoder]:
        monkeypatch.setattr(cls, '__init__', tiny_decoder)
    monkeypatch.setattr(owned, 'selected', lambda cfg: True)
    cls = {'ct': ct.HcuHY4CTBackbone, 'ct-launch': ct.HcuHY4CTLaunchBackbone,
           'static': static.HcuHY4StaticBackbone,
           'static-launch': static.HcuHY4StaticLaunchBackbone}[kind]
    raw = _raw_constructor(cls.__init__)
    body = bind_constructor(raw, {
        'current_platform': NS(device_type='cpu'),
        'get_pp_group': lambda: NS(is_first_rank=True, is_last_rank=True),
        'VocabParallelEmbedding': torch.nn.Embedding,
        'RMSNorm': torch.nn.RMSNorm,
        'make_layers': lambda count, make, prefix: (0, count, torch.nn.ModuleList([
            make(f'{prefix}.{i}') for i in range(count)])),
    })
    hf = NS(hidden_size=8, vocab_size=8, num_hidden_layers=1, enable_ihc=False,
            tie_word_embeddings=False, rms_norm_eps=1e-6, routed_scaling_factor=1.,
            n_routed_experts=4, moe_intermediate_size=8, n_shared_experts=1,
            norm_topk_prob=True)
    cfg = NS(model_config=NS(hf_config=hf), cache_config=None, quant_config=None,
             speculative_config=None,
             parallel_config=NS(eplb_config=NS(num_redundant_experts=0)))
    instance = cls.__new__(cls)
    body(instance, vllm_config=cfg)
    assert isinstance(instance.layers[0], owned.native.HYV4DecoderLayer)
    assert instance.num_moe_layers == 0
