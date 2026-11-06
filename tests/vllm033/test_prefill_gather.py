"""CPU contracts for public gather composition; device/Graph checks are separate."""
from types import ModuleType, SimpleNamespace
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

PLUGIN_SOURCE_ROOT = Path(__file__).resolve().parents[2] / 'adapter_extension/src'
sys.path.insert(0, os.fspath(PLUGIN_SOURCE_ROOT))
# Import this CPU helper independently of attention/__init__.py, which publishes
# the actual Flash backend and requires the framework at module import time.
_gather_spec = importlib.util.spec_from_file_location(
    '_hcu_v033_prefill_gather_contract',
    PLUGIN_SOURCE_ROOT / 'hcu_vllm_das/attention/prefill_gather.py',
)
gather = importlib.util.module_from_spec(_gather_spec)
sys.modules[_gather_spec.name] = gather
_gather_spec.loader.exec_module(gather)
from hcu_vllm_das.environment import (
    owns_prefill_compile_factor, prefill_gather_enabled,
    register_prefill_compile_factor,
)


def test_source_only_collection_does_not_import_flash_backend():
    code = f'''
import sys, importlib.abc, importlib.util
import torch
assert 'vllm' not in sys.modules
class MissingCore(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'vllm' or fullname.startswith('vllm.'):
            raise ModuleNotFoundError('vllm is not installed', name=fullname)
sys.meta_path.insert(0, MissingCore())
spec = importlib.util.spec_from_file_location(
    '_hcu_prefill_source_only_collection_probe', {str(Path(__file__).resolve())!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'hcu_vllm_das.attention.flash_attn' not in sys.modules
assert not torch.cuda.is_initialized()
'''
    run = subprocess.run([sys.executable, '-c', code], env=dict(os.environ),
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr


def test_policy_defaults_off_and_rejects_invalid_values(monkeypatch):
    monkeypatch.delenv("VLLM_HCU_FLASH_PREFILL_GATHER",raising=False)
    assert prefill_gather_enabled() is False
    monkeypatch.setenv("VLLM_HCU_FLASH_PREFILL_GATHER","yes")
    assert prefill_gather_enabled() is True
    monkeypatch.setenv("VLLM_HCU_FLASH_PREFILL_GATHER","invalid")
    with pytest.raises(ValueError,match="boolean"):
        prefill_gather_enabled()


def test_factor_ownership_and_disabled_values(monkeypatch):
    envs=SimpleNamespace(environment_variables={},compile_factors=lambda:{})
    register_prefill_compile_factor(envs)
    register_prefill_compile_factor(envs)
    assert owns_prefill_compile_factor(envs)
    factor=envs.environment_variables['VLLM_HCU_FLASH_PREFILL_GATHER_ENABLED']
    for raw in ('0','false','off'):
        monkeypatch.setenv('VLLM_HCU_FLASH_PREFILL_GATHER',raw)
        assert factor() is False
    envs.environment_variables['VLLM_HCU_FLASH_PREFILL_GATHER_ENABLED']=lambda:False
    assert not owns_prefill_compile_factor(envs)
    with pytest.raises(RuntimeError,match='another owner'):
        register_prefill_compile_factor(envs)


def test_actual_aot_factors_distinguish_prefill_policy():
    target=Path(os.environ['VLLM_TARGET_ROOT']).resolve()
    code=f'''
import os,sys
from pathlib import Path
sys.path.insert(0,{str(target)!r})
sys.path.insert(0,{str(PLUGIN_SOURCE_ROOT)!r})
import vllm
assert Path(vllm.__file__).resolve().is_relative_to({str(target)!r}), vllm.__file__
import vllm.envs as envs
from types import SimpleNamespace
from vllm.compilation.caching import aot_compile_hash_factors
from hcu_vllm_das.environment import register_prefill_compile_factor
register_prefill_compile_factor(envs)
config=SimpleNamespace(compute_hash=lambda:'same-model')
observed={{}}
for raw in ('0','false','1','yes'):
    os.environ['VLLM_HCU_FLASH_PREFILL_GATHER']=raw
    observed[raw]=aot_compile_hash_factors(config)
assert observed['0']==observed['false']
assert observed['1']==observed['yes']
assert observed['0']!=observed['1']
assert observed['0'][1]==observed['1'][1]
'''
    run=subprocess.run([sys.executable,'-c',code],env=dict(os.environ),capture_output=True,text=True,timeout=120)
    assert run.returncode==0,run.stdout+run.stderr


def inputs():
    # Two variable contexts including a partial final page; GQA heads4/2.
    k=torch.arange(4*2*3*4,dtype=torch.bfloat16).reshape(4,2,3,4)
    return dict(q=torch.zeros(5,4,4,dtype=torch.bfloat16),k=k,v=k+100,
        out=torch.full((5,4,4),-7,dtype=torch.bfloat16),
        cu_seqlens_q=torch.tensor([0,2,5],dtype=torch.int32),
        seqused_k=torch.tensor([5,4],dtype=torch.int32),
        block_table=torch.tensor([[2,0],[3,1]],dtype=torch.int32),
        max_seqlen_q=3,max_seqlen_k=6,causal=True,
        softmax_scale=.5,q_descale=None,k_descale=None,v_descale=None)


@pytest.mark.parametrize('change',[{'max_seqlen_q':1},{'max_seqlen_k':131073},
    {'block_table':None},{'k_descale':torch.ones(())},
    {'q':torch.ones(5,4,4,dtype=torch.float32)}])
def test_fallback_boundaries(change):
    kw=inputs();assert gather.can_run(kw,'LBHNC')
    kw.update(change);assert not gather.can_run(kw,'LBHNC')


@pytest.mark.parametrize('layout',['LBHNC','LBNHC'])
def test_reuses_strided_head_gather_preserves_arguments_and_caller_output(monkeypatch,layout):
    kw=inputs()
    if layout=='LBNHC':
        kw['k']=kw['k'].transpose(1,2);kw['v']=kw['v'].transpose(1,2)
    original=dict(kw);calls=[]
    ops=ModuleType('vllm._custom_ops')
    def cp(src,dst,table,cuk,batch):
        calls.append((src.stride(),dst.stride()))
        for b in range(batch):
            for position in range(int(cuk[b+1]-cuk[b])):
                dst[int(cuk[b])+position]=src[int(table[b,position//src.shape[1]]),position%src.shape[1]]
    ops.cp_gather_cache=cp
    monkeypatch.setitem(sys.modules,'vllm._custom_ops',ops)
    captured=[]
    def vendor(**arguments):
        captured.append(arguments)
        assert 'block_table' not in arguments and 'seqused_k' not in arguments
        assert arguments['cu_seqlens_k'].tolist()==[0,5,9]
        assert arguments['max_seqlen_q']==3 and arguments['max_seqlen_k']==6
        assert arguments['causal'] and arguments['softmax_scale']==.5
        assert arguments['fa_version']==2
        for b,n in enumerate((5,4)):
            for pos in range(n):
                source=original['k']
                expected=source[int(kw['block_table'][b,pos//3]),:,pos%3] if layout=='LBHNC' else source[int(kw['block_table'][b,pos//3]),pos%3,:]
                assert torch.equal(arguments['k'][sum((5,4)[:b])+pos],expected)
        return torch.full_like(arguments['out'],17)
    result=gather.run(vendor,kw,layout)
    assert result is kw['out'] and torch.all(result==17)
    assert len(calls)==4 and len(captured)==1
    assert kw.keys()==original.keys() and all(kw[n] is original[n] for n in kw)
