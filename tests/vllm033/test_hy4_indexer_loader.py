# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Real Core fusion loader with CT INT8 WK checkpoint values."""
import itertools
from types import SimpleNamespace as NS

import pytest
import torch

from hcu_vllm_das import hy4_indexer_loader as owned
from vllm.models.hy_v4.amd import model as native
from vllm.models.hy_v4.nvidia import mtp


@pytest.mark.parametrize('scale_first', [False, True])
@pytest.mark.parametrize('prefix', ['model.layers.0.self_attn.indexer',
                                   'model.layers.78.mtp_block.self_attn.indexer'])
def test_int8_wk_uses_channel_scales_and_real_native_fusion(scale_first, prefix):
    weight = torch.tensor([[-128, -3, 5], [127, 1, -11]], dtype=torch.int8)
    scale = torch.tensor([[0.01], [0.125]], dtype=torch.float32)
    output = torch.full((3, 3), float('nan'), dtype=torch.bfloat16)
    output[2].fill_(7)
    output.weight_loader = lambda param, value, shard: param[:2].copy_(value)
    params = {prefix + '.wk_weights_proj.weight': output}
    buffer, loaded = {}, set()
    entries = [(prefix + '.wk.weight', weight), (prefix + '.wk.weight_scale', scale)]
    if scale_first:
        entries.reverse()
    for name, value in entries:
        assert owned.load_indexer_wk(name, value, buffer, params, loaded, [])
    assert not buffer
    assert loaded == set(params)
    torch.testing.assert_close(output[:2], (weight.float() * scale).bfloat16())
    assert torch.all(output[2] == 7)


def test_pp_missing_int8_weight_is_not_buffered():
    buffer = {}
    assert owned.load_indexer_wk('model.layers.1.self_attn.indexer.wk.weight',
        torch.ones((2, 3), dtype=torch.int8), buffer, {}, set(), ['model.layers.1.'])
    assert not buffer


def test_asymmetric_wk_is_rejected_instead_of_ignoring_zero_point():
    with pytest.raises(ValueError, match='symmetric'):
        owned.load_indexer_wk('model.layers.0.self_attn.indexer.wk.weight_zero_point',
            torch.zeros((2, 1)), {}, {}, set(), [])


def test_non_int8_and_unrelated_names_delegate_to_native(monkeypatch):
    calls = []
    monkeypatch.setattr(owned, '_native', lambda *args: calls.append(args) or False)
    for name, dtype in itertools.product(['layer.indexer.wk.weight', 'layer.other.weight'],
                                         [torch.bfloat16, torch.float8_e4m3fn]):
        assert not owned.load_indexer_wk(name, torch.empty((2, 3), dtype=dtype), {}, {}, set(), [])
    assert len(calls) == 4


def test_native_loader_body_is_retained_without_global_mutation():
    from hcu_vllm_das.models.hy4_ct import HcuHY4CTBackbone, HcuHY4CTMTP
    assert HcuHY4CTBackbone.load_weights.__code__ is native.HYV4Model.load_weights.__code__
    assert HcuHY4CTMTP.load_weights.__code__ is mtp.HYV4MTP.load_weights.__code__
    assert HcuHY4CTBackbone.load_weights.__globals__['_try_load_fp8_indexer_wk'] is owned.load_indexer_wk
    assert native.HYV4Model.load_weights.__globals__['_try_load_fp8_indexer_wk'] is not owned.load_indexer_wk


def test_ct_model_publication_is_explicit_and_defers_to_static(monkeypatch):
    from hcu_vllm_das import hy4_ct_model_activation as owner
    monkeypatch.setattr(owner, '_publication', None)
    monkeypatch.setattr(owner, '_registry', lambda: pytest.fail('Core registry touched'))
    for caps in [set(), {'hy4-ct-w4a8', 'hy4-static-eplb'}]:
        owner.activate(NS(requested_capabilities=caps, requested_extensions=set()))


def test_ct_model_publication_checks_both_owners_and_retains_health(monkeypatch):
    from hcu_vllm_das import hy4_ct_model_activation as owner
    from hcu_adapter.extensions import ExtensionCommitError
    monkeypatch.setattr(owner, '_publication', None)
    entries = {'HYV4ForCausalLM': NS(module_name='vllm.models.hy_v4', class_name='HYV4ForCausalLM'),
               'HYV4MTPModel': NS(module_name='foreign', class_name='Draft')}
    calls = []
    def register(arch, path):
        calls.append(arch)
        module, cls = path.split(':')
        entries[arch] = NS(module_name=module, class_name=cls)
    registry = NS(models=entries, register_model=register)
    monkeypatch.setattr(owner, '_registry', lambda: registry)
    ctx = NS(requested_capabilities={'hy4-ct-w4a8'}, requested_extensions=set())
    with pytest.raises(RuntimeError, match='owner'):
        owner.activate(ctx)
    assert not calls
    entries['HYV4MTPModel'] = NS(module_name='vllm.models.hy_v4', class_name='HYV4MTP')
    owner.activate(ctx)
    owner.activate(ctx)
    assert calls == ['HYV4ForCausalLM', 'HYV4MTPModel']
    entries['HYV4MTPModel'] = NS(module_name='foreign', class_name='Draft')
    with pytest.raises(ExtensionCommitError, match='owner'):
        owner.check_health()
