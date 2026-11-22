# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Optional model policy over Adapter's native CT activation contract."""
from importlib import import_module
from types import SimpleNamespace as NS

import pytest
import torch


def owned():
    return import_module('hcu_vllm_das.hy4_dense_fusion')


@pytest.fixture(autouse=True)
def native_config():
    from vllm.config import CompilationConfig, set_current_vllm_config
    cfg = NS(compilation_config=CompilationConfig(mode=0, custom_ops=['none']))
    with set_current_vllm_config(cfg, check_compile=False):
        yield


def config():
    return NS(model_config=NS(dtype=torch.bfloat16,
        hf_config=NS(model_type='hy_v4')), lora_config=None,
        use_v2_model_runner=True, compilation_config=NS(mode=0))


@pytest.mark.parametrize('master,child,expected', [(None,None,False),
    ('0','1',False), ('1','0',False), ('1','1',True)])
def test_default_off_and_master_rollback(monkeypatch, master, child, expected):
    m = owned()
    for key, value in [('VLLM_HCU_USE_CUSTOM_OPS',master), ('HCU_VLLM_HY4_DENSE_FUSION',child)]:
        if value is None: monkeypatch.delenv(key,raising=False)
        else: monkeypatch.setenv(key,value)
    monkeypatch.setattr(m, '_supported_config', lambda cfg: True)
    assert m.enabled(config()) is expected


@pytest.mark.parametrize('change', ['hardware','dtype','lora','model','compile','runner'])
def test_policy_unsupported_scope(monkeypatch,change):
    m=owned();cfg=config()
    monkeypatch.setattr(m, '_device_arch',lambda: 'gfx936' if change=='hardware' else 'gfx938')
    if change=='dtype': cfg.model_config.dtype=torch.float16
    if change=='lora': cfg.lora_config=object()
    if change=='model': cfg.model_config.hf_config.model_type='other'
    if change=='compile': cfg.compilation_config.mode=3
    if change=='runner': cfg.use_v2_model_runner=False
    assert not m._supported_config(cfg)


def test_actual_dtype_attachment_and_idempotence(monkeypatch):
    m=owned()
    from hcu_adapter.features.ct_w4a8.lightop_int8_linear import HcuLightopInt8LinearMethod, FusedSiluInt8Producer
    from vllm.models.hy_v4.nvidia.moe import HYV4FeedForward
    from vllm.model_executor.layers.activation import SiluAndMul
    def ffn(dtype, method):
        obj=HYV4FeedForward.__new__(HYV4FeedForward);torch.nn.Module.__init__(obj)
        obj.act_fn=SiluAndMul();obj.down_proj=torch.nn.Module()
        obj.down_proj.weight=torch.nn.Parameter(torch.ones(4,8,dtype=dtype),requires_grad=False)
        obj.down_proj.quant_method=method
        obj.down_proj.input_is_parallel=True
        return obj
    method=HcuLightopInt8LinearMethod.__new__(HcuLightopInt8LinearMethod)
    tree=torch.nn.ModuleList([ffn(torch.int8,method),ffn(torch.bfloat16,method),ffn(torch.int8,object())])
    before=set(tree.state_dict())
    monkeypatch.setattr(m,'enabled',lambda cfg: True)
    assert m.attach(tree,config()) is tree
    assert isinstance(tree[0].act_fn,FusedSiluInt8Producer)
    assert type(tree[1].act_fn) is SiluAndMul and type(tree[2].act_fn) is SiluAndMul
    assert set(tree.state_dict()) == before
    first=tree[0].act_fn;m.attach(tree,config())
    assert tree[0].act_fn is first
    tree[2].down_proj.input_is_parallel=False
    tree[2].down_proj.quant_method=method;m.attach(tree,config())
    assert type(tree[2].act_fn) is SiluAndMul


@pytest.mark.parametrize('kind',['ct','static','mtp'])
def test_owned_factories_attach_after_native_constructor(monkeypatch,kind):
    m=owned()
    from hcu_vllm_das.models import hy4_ct as ct,hy4_static as static
    calls=[];result=object();cfg=config()
    monkeypatch.setattr(m,'attach',lambda model,cfg: calls.append(('attach',model,cfg)) or model)
    monkeypatch.setattr(ct.ihc_launch,'selected',lambda cfg: False)
    if kind=='mtp':
        monkeypatch.setattr(ct.mtp,'HYV4MultiTokenPredictor',lambda **kw: calls.append(('native',kw)) or result)
        factory=ct.make_ct_predictor
    else:
        owner=ct if kind=='ct' else static
        name='HcuHY4CTBackbone' if kind=='ct' else 'HcuHY4StaticBackbone'
        monkeypatch.setattr(owner,name,lambda **kw:calls.append(('native',kw)) or result)
        factory=owner.make_ct_backbone if kind=='ct' else owner.make_static_backbone
    assert factory(vllm_config=cfg,prefix='model') is result
    assert calls==[('native',dict(vllm_config=cfg,prefix='model')),('attach',result,cfg)]


def test_native_execution_loops_and_private_mtp_ctor():
    owned()
    from hcu_vllm_das.models import hy4_ct as ct,hy4_static as static
    assert ct.HcuHY4CTForCausalLM.forward is ct.native.HYV4ForCausalLM.forward
    assert ct.HcuHY4CTMTP.forward is ct.mtp.HYV4MTP.forward
    assert static.HcuHY4StaticMTP.forward is ct.mtp.HYV4MTP.forward
    assert ct.HcuHY4CTMTP.__init__.__code__ is ct.mtp.HYV4MTP.__init__.__code__
    assert ct.mtp.HYV4MTP.__init__.__globals__['HYV4MultiTokenPredictor'] is ct.mtp.HYV4MultiTokenPredictor
