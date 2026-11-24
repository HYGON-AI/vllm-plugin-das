# SPDX-License-Identifier: Apache-2.0
"""Host metadata admission only; real vendor numerics need the GPU component lane."""
from types import SimpleNamespace as NS
import pytest
import torch
from hcu_vllm_das import hy4_ihc_launch as policy
REAL_TOPOLOGY = getattr(policy, '_vendor_topology', None)


@pytest.fixture(autouse=True)
def p_topology(monkeypatch):
    monkeypatch.setattr(policy, '_vendor_topology', lambda cfg=None: True, raising=False)


def residual(rows, *, dtype=torch.bfloat16, device='cuda', contiguous=True):
    return NS(shape=(rows, 4, 6144), ndim=3, dtype=dtype,
              device=NS(type=device), is_contiguous=lambda: contiguous)


@pytest.mark.parametrize('rows,expected', [(1,False),(32,False),(377,False),
                                        (378,True),(756,True),(1134,True),(1890,True),(6168,True)])
def test_actual_size_policy_and_forwarded_parameters(monkeypatch, rows, expected):
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '1')
    monkeypatch.setenv('HCU_VLLM_HY4_IHC_VENDOR', '1')
    calls=[]
    monkeypatch.setattr(policy, 'tuned_pre', lambda *a: calls.append(('core',a)) or 'core')
    monkeypatch.setattr(policy, '_vendor', lambda: NS(pre=lambda *a: calls.append(('vendor',a)) or 'vendor'), raising=False)
    args=(residual(rows),object(),object(),object(),2.0,1e-6,1e-5)
    assert policy._pre(*args)==('vendor' if expected else 'core')
    assert calls==[('vendor' if expected else 'core',args)]


@pytest.mark.parametrize('change', ['master','child','cpu','dtype','stride'])
def test_rollback_and_unsupported_inputs_do_not_import_vendor(monkeypatch, change):
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS', '0' if change=='master' else '1')
    monkeypatch.setenv('HCU_VLLM_HY4_IHC_VENDOR', '0' if change=='child' else '1')
    x=residual(1134,device='cpu' if change=='cpu' else 'cuda',
               dtype=torch.float32 if change=='dtype' else torch.bfloat16,
               contiguous=change!='stride')
    monkeypatch.setattr(policy,'_vendor',lambda:pytest.fail('vendor imported'),raising=False)
    monkeypatch.setattr(policy,'tuned_pre',lambda *a:'native')
    assert policy._pre(x,None,None,None,2.,1e-6,1e-5)=='native'


def test_chain_retains_native_argument_order(monkeypatch):
    monkeypatch.setenv('VLLM_HCU_USE_CUSTOM_OPS','1')
    monkeypatch.setenv('HCU_VLLM_HY4_IHC_VENDOR','1')
    calls=[]
    monkeypatch.setattr(policy,'_vendor',lambda:NS(post_pre_rms_norm=lambda *a:calls.append(a) or 'vendor'),raising=False)
    x=NS(shape=(1134,6144),ndim=2,dtype=torch.bfloat16,device=NS(type='cuda'),is_contiguous=lambda:True)
    args=(x,residual(1134),object(),object(),object(),object(),object(),2.,1e-6,1e-5)
    assert policy._chain(*args)=='vendor'
    assert calls==[args]


@pytest.mark.parametrize('change', [None,'multimodal','not_eager','tp','pcp','dp','model'])
def test_vendor_is_limited_to_the_validated_text_p_topology(change):
    cfg=NS(model_config=NS(hf_config=NS(model_type='hy_v4'),
                           is_multimodal_model=False,enforce_eager=True),
           parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1,
                              prefill_context_parallel_size=16,decode_context_parallel_size=1,
                              data_parallel_size=1))
    if change=='multimodal':cfg.model_config.is_multimodal_model=True
    if change=='not_eager':cfg.model_config.enforce_eager=False
    if change=='tp':cfg.parallel_config.tensor_parallel_size=2
    if change=='pcp':cfg.parallel_config.prefill_context_parallel_size=1
    if change=='dp':cfg.parallel_config.data_parallel_size=16
    if change=='model':cfg.model_config.hf_config.model_type='other'
    assert REAL_TOPOLOGY(cfg) is (change is None)
