"""Selection contracts for the independently measured TP1 native plans."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[2] / 'adapter_extension/src'))
from hcu_adapter.bridges import vllm as bridge
from hcu_vllm_das.moe import fused_moe_aiter as provider

PROFILES = json.loads((Path(__file__).parent / 'data/int8_qwen3_tp1_gfx936_qualified_profiles.json').read_text())['profiles']


class Tensor:
    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype
        self.device = torch.device('cuda:0')


def inputs(tokens=1, dtype=torch.bfloat16, *, experts=128, hidden=2048, intermediate=768, topk=8):
    parallel = SimpleNamespace(tp_size=1,ep_size=1,ep_rank=0,dp_size=1,pcp_size=1,
        use_ep=False,is_sequence_parallel=False,enable_eplb=False,use_passthrough_all2all=False)
    return dict(hidden_states=Tensor((tokens,hidden),dtype),
        w1=Tensor((experts,2*intermediate,hidden),torch.int8),
        w2=Tensor((experts,hidden,intermediate),torch.int8),
        topk_weights=Tensor((tokens,topk),torch.float32),
        topk_ids=Tensor((tokens,topk),torch.int32),
        moe_config=SimpleNamespace(num_experts=experts,moe_parallel_config=parallel),
        quant_config=SimpleNamespace(use_int8_w8a8=True,per_act_token_quant=True,
            w1_scale=object(),w2_scale=object(),w1_zp=None,w2_zp=None,a1_scale=None,a2_scale=None),
        activation=SimpleNamespace(value='silu'),expert_map=None,output_dtype=dtype,asm_only=True)


def config(tokens=1, **overrides):
    values=dict(solution_type='asm',config=dict(PROFILES[str(tokens)]),
                need_shuffle=False,need_shuffle_scale=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def setup(monkeypatch, *, donor=None, donor_found=True, arch='gfx936', detected=True, local=None):
    lookups, native, fallback, events = [], [], [], []
    donor = config() if donor is None else donor
    def lookup(*args, **kwargs):
        lookups.append((args,kwargs))
        if len(lookups)==1:
            return (True,local) if local is not None else (False,SimpleNamespace(solution_type=None,config=None))
        assert args[1:6] == (256,1024,2048,2048,8)
        return donor_found,donor
    monkeypatch.setattr(bridge,'detect_hcu',lambda torch_module=None:SimpleNamespace(detected=detected,architecture=arch))
    monkeypatch.setattr(provider,'_load_aiter_api',lambda:SimpleNamespace(
        quant_type_w8a8='int8_w8a8',get_config=lookup,
        aiter_moe=lambda **kw:native.append(kw) or 'native'))
    monkeypatch.setattr(provider,'_load_triton_fallback',lambda:lambda **kw:fallback.append(kw) or 'fallback')
    monkeypatch.setattr(provider,'_record_once',lambda level,event,*a:events.append(event))
    return lookups,native,fallback,events,donor


@pytest.mark.parametrize('tokens',sorted(map(int,PROFILES)))
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
def test_qualified_tp1_reuses_actual_donor_after_local_miss(monkeypatch,tokens,dtype):
    args=inputs(tokens,dtype)
    output=object()
    lookups,native,fallback,events,donor=setup(monkeypatch,donor=config(tokens))
    assert provider.apply_aiter_int8_moe(**args,output=output)=='native'
    assert [x[0][1:6] for x in lookups]==[(128,1536,2048,2048,8),(256,1024,2048,2048,8)]
    assert all(x[1]['spec_sol_type']=='asm' for x in lookups)
    assert native[0]['moe_config'] is donor and native[0]['output'] is output
    assert native[0]['w1'] is args['w1'] and native[0]['expert_map'] is None
    assert not fallback and any('tp1-qualified' in event for event in events)


@pytest.mark.parametrize('tokens',[5,7,24,2048])
def test_unqualified_token_count_keeps_whole_official_fallback(monkeypatch,tokens):
    args=inputs(tokens)
    lookups,native,fallback,_,_=setup(monkeypatch)
    assert provider.apply_aiter_int8_moe(**args)=='fallback'
    assert len(lookups)==1 and not native
    assert fallback[0]['hidden_states'] is args['hidden_states'] and fallback[0]['w1'] is args['w1']


def test_empty_tensor_retains_existing_shape_rejection(monkeypatch):
    lookups,native,fallback,_,_=setup(monkeypatch)
    with pytest.raises(ValueError,match='positive rank'):
        provider.apply_aiter_int8_moe(**inputs(0))
    assert not lookups and not native and not fallback


@pytest.mark.parametrize('field,value',[('tp_size',2),('ep_size',2),('ep_rank',1),
    ('dp_size',2),('pcp_size',2),('use_ep',True),('is_sequence_parallel',True),
    ('enable_eplb',True),('use_passthrough_all2all',True)])
def test_other_parallel_modes_do_not_borrow_tp1_profile(monkeypatch,field,value):
    args=inputs()
    setattr(args['moe_config'].moe_parallel_config,field,value)
    lookups,native,_,_,_=setup(monkeypatch)
    assert provider.apply_aiter_int8_moe(**args)=='fallback'
    assert len(lookups)==1 and not native


@pytest.mark.parametrize('kind',['no-parallel','mask','generic-provider','float32','output-dtype',
    'experts','hidden','intermediate','topk','gelu'])
def test_other_metadata_does_not_borrow_qualified_geometry(monkeypatch,kind):
    args=inputs(**{'experts':256} if kind=='experts' else
                {'hidden':4096} if kind=='hidden' else
                {'intermediate':512} if kind=='intermediate' else
                {'topk':4} if kind=='topk' else {})
    if kind=='no-parallel':del args['moe_config'].moe_parallel_config
    if kind=='mask':args['expert_map']=object()
    if kind=='generic-provider':args['asm_only']=False
    if kind=='float32':args['hidden_states'].dtype=args['output_dtype']=torch.float32
    if kind=='output-dtype':args['output_dtype']=torch.float16
    if kind=='gelu':args['activation']=SimpleNamespace(value='gelu')
    lookups,native,_,_,_=setup(monkeypatch)
    assert provider.apply_aiter_int8_moe(**args)=='fallback'
    assert len(lookups)==1 and not native


@pytest.mark.parametrize('field',['per_act_token_quant','w1_zp','w2_zp','a1_scale','a2_scale'])
def test_static_or_asymmetric_quantization_does_not_borrow(monkeypatch,field):
    args=inputs()
    setattr(args['quant_config'],field,False if field=='per_act_token_quant' else object())
    lookups,native,_,_,_=setup(monkeypatch)
    assert provider.apply_aiter_int8_moe(**args)=='fallback'
    assert len(lookups)==1 and not native


@pytest.mark.parametrize('arch,detected',[('gfx938',True),('gfx936',False),('gfx92a',True)])
def test_new_profile_is_qualified_only_on_detected_gfx936(monkeypatch,arch,detected):
    lookups,native,_,_,_=setup(monkeypatch,arch=arch,detected=detected)
    assert provider.apply_aiter_int8_moe(**inputs())=='fallback'
    assert len(lookups)==1 and not native


@pytest.mark.parametrize('tokens',[1,512,1024])
@pytest.mark.parametrize('field',['SOL_ID1','SOL_ID2','BLOCK_SIZE_M'])
def test_any_changed_donor_stage_or_block_keeps_official_fallback(monkeypatch,tokens,field):
    donor=config(tokens)
    donor.config[field]+=1
    lookups,native,fallback,_,_=setup(monkeypatch,donor=donor)
    assert provider.apply_aiter_int8_moe(**inputs(tokens))=='fallback'
    assert len(lookups)==2 and not native and len(fallback)==1


@pytest.mark.parametrize('kind',['miss','ck','shuffle','scale-shuffle','padded','use-shuffle','non-dict'])
def test_unqualified_donor_layout_or_provider_keeps_fallback(monkeypatch,kind):
    donor=config()
    if kind=='ck':donor.solution_type='ck'
    if kind=='shuffle':donor.need_shuffle=True
    if kind=='scale-shuffle':donor.need_shuffle_scale=True
    if kind=='padded':donor.config['PADDED_K']=4096
    if kind=='use-shuffle':donor.config['USE_SHUFFLE']=1
    if kind=='non-dict':donor.config=None
    lookups,native,_,_,_=setup(monkeypatch,donor=donor,donor_found=kind!='miss')
    assert provider.apply_aiter_int8_moe(**inputs())=='fallback'
    assert len(lookups)==2 and not native


def test_real_local_vendor_profile_takes_precedence(monkeypatch):
    local=config()
    lookups,native,fallback,events,_=setup(monkeypatch,local=local,arch='gfx938')
    assert provider.apply_aiter_int8_moe(**inputs())=='native'
    assert len(lookups)==1 and native[0]['moe_config'] is local and not fallback
    assert not any('tp1-qualified' in event for event in events)


@pytest.mark.parametrize('actual_gfx,expected', [
    ('gfx936:sramecc+:xnack-', 'native'), ('gfx938', 'fallback'),
    ('missing-property', 'fallback'), ('unavailable', 'fallback'),
])
def test_tp1_plan_checks_nonzero_actual_tensor_device(monkeypatch, actual_gfx, expected):
    args = inputs()
    device = torch.device('cuda:1')
    args['hidden_states'].device = device
    # Adapter inspects cuda:0. A TP1 worker may actually be on cuda:1.
    lookups,native,_,_,_=setup(monkeypatch, arch='gfx936')
    queried = []
    def properties(ordinal):
        queried.append(ordinal)
        if actual_gfx == 'unavailable':
            raise RuntimeError('Device properties unavailable')
        if actual_gfx == 'missing-property':
            return SimpleNamespace()
        return SimpleNamespace(gcnArchName=actual_gfx)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', properties)
    assert provider.apply_aiter_int8_moe(**args) == expected
    assert queried == [device]
    assert len(lookups) == (2 if expected == 'native' else 1)
    assert bool(native) == (expected == 'native')


@pytest.mark.parametrize('device',[torch.device('cpu'),torch.device('cuda')])
def test_tp1_plan_declines_unresolved_or_non_cuda_tensor_device(monkeypatch,device):
    args = inputs()
    args['hidden_states'].device = device
    lookups,native,_,_,_=setup(monkeypatch)
    assert provider.apply_aiter_int8_moe(**args) == 'fallback'
    assert len(lookups) == 1 and not native
