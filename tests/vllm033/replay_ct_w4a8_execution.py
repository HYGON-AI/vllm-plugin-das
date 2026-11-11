#!/usr/bin/env python3
"""Pre-implementation CPU integration: actual modular kernel, doubled providers/EP."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import importlib
import importlib.machinery
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import torch
import torch.nn.functional as F


def quantize(x):
    scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-7) / 127
    return torch.round(x.float()/scale).clamp(-127,127).to(torch.int8), scale


class Event:
    event = None
    def current_stream_wait(self): pass


class Buffer:
    def __init__(self, capacity=4, dispatchers=2):
        self.capacity, self.dispatchers = capacity, dispatchers
        self.calls = []
    @staticmethod
    def get_dispatch_config(n): return None
    @staticmethod
    def get_combine_config(n): return None
    def capture(self): return Event()
    def get_dispatch_layout(self, topk_idx, num_experts, **kwargs):
        return torch.tensor([topk_idx.shape[0],0]), None, torch.tensor([topk_idx.shape[0]]*2), None, Event()
    def dispatch(self, x, topk_idx, topk_weights, **kwargs):
        assert isinstance(x,torch.Tensor) and x.dtype==torch.bfloat16
        self.calls.append('ht_dispatch')
        return x, topk_idx, topk_weights, [x.shape[0]]*2, ('ht',), Event()
    def combine(self, x, handle, **kwargs):
        assert handle == ('ht',)
        self.calls.append('ht_combine')
        return x, None, Event()
    def low_latency_dispatch(self, x, topk_idx, topk_weight, num_max_dispatch_tokens_per_rank,
                             num_experts, quant_type=1, quant_group_size=0, fp8_round_scale=False,
                             async_finish=False, return_recv_hook=False):
        assert quant_type==1 and quant_group_size==0 and not fp8_round_scale
        assert return_recv_hook and not async_finish
        assert x.shape[0] <= self.capacity == num_max_dispatch_tokens_per_rank
        self.calls.append(('ll_dispatch',topk_weight.clone()))
        total = self.capacity*self.dispatchers
        data=torch.zeros((2,total,x.shape[1]),dtype=torch.int8)
        scales=torch.ones((2,total,1),dtype=torch.float32)
        counts=torch.zeros(2,dtype=torch.int32)
        routes=[]
        for e in range(2):
            rows=[row for row in range(x.shape[0]) if e in topk_idx[row].tolist()]
            counts[e]=len(rows)
            if rows:
                q,s=quantize(x[rows]); data[e,:len(rows)]=q; scales[e,:len(rows)]=s
            routes.append(rows)
        handle=(routes,topk_idx.clone(),topk_weight.clone())
        def hook(): self.calls.append('ll_recv')
        return (data,scales),counts,handle,Event(),hook
    def low_latency_combine(self,x,topk_idx,topk_weights,handle,**kwargs):
        routes,ids,weights=handle
        assert torch.equal(ids,topk_idx) and torch.equal(weights,topk_weights)
        assert self.calls[-1]=='ll_recv' or any(c=='ll_recv' for c in self.calls if isinstance(c,str))
        out=kwargs['out']; result=torch.zeros(out.shape,dtype=torch.float32)
        for e,rows in enumerate(routes):
            for slot,row in enumerate(rows):
                for j in range(topk_idx.shape[1]):
                    if int(topk_idx[row,j])==e:
                        result[row] += x[e,slot].float()*topk_weights[row,j]
        out.copy_(result.to(out.dtype)); self.calls.append('ll_combine')
        return out,Event(),lambda:None


fake_ep=ModuleType('deep_ep'); fake_ep.__spec__=importlib.machinery.ModuleSpec('deep_ep',loader=None)
fake_ep.Buffer=Buffer; fake_ep.Config=object; fake_ep.EventOverlap=Event
sys.modules['deep_ep']=fake_ep
from vllm.model_executor.layers.fused_moe import modular_kernel as mk, RoutedExperts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import DeepEPLLPrepareAndFinalize
from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import DeepEPHTPrepareAndFinalize
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
from vllm.model_executor.layers.fused_moe.oracle import w4a8_int8 as oracle
from vllm.config import ModelConfig
import vllm.config.model as model_module
import vllm.model_executor.layers.quantization as quant_registry
from hcu_adapter.extensions import ExtensionContext, ExtensionCommitError
from replay_ct_w4a8_weights import fixture as weight_fixture, ct_pack, H, I, E
from hcu_vllm_das.quantization import dsv41_channel_fp8 as fp8_config
from hcu_vllm_das import dsv41_activation

MODULE='hcu_vllm_das.quantization.ct_w4a8'
try:
    config_module=importlib.import_module(MODULE)
    runtime=importlib.import_module('hcu_vllm_das.moe.ct_w4a8_runtime')
    prepare_module=importlib.import_module('hcu_vllm_das.moe.ct_w4a8_prepare')
    activate_module=importlib.import_module('hcu_vllm_das.ct_w4a8_activation')
except ModuleNotFoundError as error:
    if error.name not in (MODULE,'hcu_vllm_das.moe.ct_w4a8_runtime','hcu_vllm_das.moe.ct_w4a8_prepare','hcu_vllm_das.ct_w4a8_activation'):
        raise
    config_module=runtime=prepare_module=activate_module=None
NAME='hcu_dsv41_ct_w4a8'
SCHEMA={'quant_method':'compressed-tensors','format':'pack-quantized','config_groups':{'w4':{
 'targets':['RoutedExperts'],'weights':{'num_bits':4,'type':'int','strategy':'channel','symmetric':True,'dynamic':False,'group_size':None},
 'input_activations':{'num_bits':8,'type':'int','strategy':'token','symmetric':True,'dynamic':True,'group_size':None}}}}
RESULTS=[]


def required():
    assert config_module is not None,'owned W4A8 execution and registration are missing'


def case(name,fn):
    try: fn(); RESULTS.append({'case':name,'status':'passed'})
    except Exception as error: RESULTS.append({'case':name,'status':'failed','error':f'{type(error).__name__}: {error}'})


def rejects(fn):
    try: fn()
    except (RuntimeError,ValueError,AssertionError,TypeError): return
    raise AssertionError('invalid W4A8 execution contract accepted')


def raw_weight(w):
    if w.ndim==6:
        w=w.view(w.shape[0],w.shape[2]*32,w.shape[1]*32)
    b=w.to(torch.int16)
    hi=b>>4; lo=b&15; lo=torch.where(lo>=8,lo-16,lo)
    return torch.stack((hi,lo),-1).reshape(w.shape[0],w.shape[1],-1).float()


class Ops:
    def pack(self,t): return t
    def view_n32(self,t): return t.view(t.shape[0],t.shape[2]//32,t.shape[1]//32,4,32,8)
    def gemm_contiguous(self,a,b,out,indices):
        x,s=a; w,ws=b; logical=raw_weight(w)*16; out.zero_()
        for e in range(w.shape[0]):
            rows=indices==e
            out[rows]=((x[rows].float()@logical[e].t())*s[rows]*ws[e].t()).to(out.dtype)
    def gemm_masked(self,a,b,out,counts,expected_m):
        x,s=a; w,ws=b; logical=raw_weight(w)*16; out.zero_()
        assert expected_m==x.shape[1]
        for e in range(x.shape[0]):
            n=int(counts[e]); out[e,:n]=((x[e,:n].float()@logical[e].t())*s[e,:n].reshape(n,1)*ws[e].t()).to(out.dtype)
    def silu_quant(self,x,output=None,expert_ids=None):
        q,s=quantize(F.silu(x[...,:x.shape[-1]//2].float())*x[...,x.shape[-1]//2:].float())
        if output is not None: output.copy_(q);q=output
        return q,s
    def silu_quant_ep(self,x,tokens_per_expert): return self.silu_quant(x)
    def clamp_quant(self,x,limit,output=None):
        return self.silu_quant(x,output)
    def clamp_quant_ep(self,x,limit,mask_m,expect_m): return self.silu_quant(x)


def permutation(*,aq,aq_scale,topk_ids,local_num_experts,expert_map,expert_tokens_meta,aq_out,block_size):
    assert block_size==H and aq.dtype==torch.int8 and aq_scale.shape==(aq.shape[0],1)
    m=aq_out.shape[0]; output=aq_out;output.zero_(); scales=torch.ones((m,1));indices=torch.full((m,),-1,dtype=torch.int32)
    inverse=torch.full(topk_ids.shape,-1,dtype=torch.int64)
    for e in range(local_num_experts):
        offset=e*256
        for row in range(aq.shape[0]):
            for j in range(topk_ids.shape[1]):
                if int(topk_ids[row,j])==e:
                    output[offset]=aq[row]; scales[offset]=aq_scale[row];indices[offset]=e;inverse[row,j]=offset;offset+=1
    return output,scales,indices,inverse,None


def unpermute(*,a,topk_ids,topk_weights,inv_perm,expert_map,output):
    result=torch.zeros(output.shape,dtype=torch.float32)
    for row in range(output.shape[0]):
        for j in range(topk_ids.shape[1]):
            idx=int(inv_perm[row,j])
            if idx>=0: result[row] += a[idx].float()*topk_weights[row,j]
    output.copy_(result.to(output.dtype))


def context(mode='deepep_low_latency',dp=2):
    parallel=SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1,decode_context_parallel_size=1,
        prefill_context_parallel_size=1,data_parallel_size=dp,enable_eplb=False,enable_elastic_ep=False)
    config=SimpleNamespace(parallel_config=parallel,lora_config=None,model_config=SimpleNamespace(enforce_eager=True,dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1,2,4]))
    p=SimpleNamespace(use_ep=True,all2all_backend=mode,tp_size=1,dp_size=dp,pcp_size=1,ep_size=dp,ep_rank=0,
        is_sequence_parallel=False,use_all2all_kernels=True)
    moe=SimpleNamespace(moe_parallel_config=p,moe_backend='deep_gemm',activation=MoEActivation.SILU,is_act_and_mul=True,
        has_bias=False,in_dtype=torch.bfloat16,hidden_dim=H,intermediate_size_per_partition=I,
        hidden_dim_unpadded=H,intermediate_size_per_partition_unpadded=I,
        swiglu_alpha=1.0,swiglu_beta=0.0,swiglu_limit=None,activation_situ_beta=None,activation_situ_linear_beta=None,max_num_tokens=4,
        num_experts=E*dp,num_local_experts=E,dp_size=dp,tp_rank=0,tp_shard_with_padding=False,
        use_deepep_ht_kernels=mode=='deepep_high_throughput',use_deepep_ll_kernels=mode=='deepep_low_latency')
    return config,moe


def native_factory(moe,quant_config,routing_tables=None,allow_new_interface=False,use_monolithic=False,all2all_manager=None,input_dtype=None):
    assert allow_new_interface and not use_monolithic and input_dtype is None
    buffer=Buffer(moe.max_num_tokens,moe.moe_parallel_config.ep_size)
    if moe.use_deepep_ll_kernels:
        return DeepEPLLPrepareAndFinalize(buffer,moe.max_num_tokens,moe.moe_parallel_config.ep_size)
    return DeepEPHTPrepareAndFinalize(buffer,moe.moe_parallel_config.ep_size,moe.dp_size,0)


def pipeline(mode):
    required(); config,moe=context(mode)
    ops=Ops()
    with ExitStack() as stack:
        stack.enter_context(patch.object(config_module,'get_current_vllm_config',return_value=config))
        stack.enter_context(patch.object(runtime,'native_make_prepare_finalize',side_effect=native_factory))
        stack.enter_context(patch.object(runtime,'get_ops',return_value=ops))
        stack.enter_context(patch.object(prepare_module.HcuCTLLPrepareAndFinalize,'SUPPORTED_HIDDEN_SIZES',[H]))
        stack.enter_context(patch.object(runtime,'deepgemm_moe_permute',side_effect=permutation))
        stack.enter_context(patch.object(runtime,'deepgemm_unpermute_and_reduce',side_effect=unpermute))
        def token_quant(x,scale,quant_dtype,per_act_token_quant,block_shape,**kwargs):
            assert quant_dtype==torch.int8 and per_act_token_quant and block_shape is None
            return quantize(x)
        stack.enter_context(patch.object(prepare_module,'moe_kernel_quantize_input',side_effect=token_quant))
        qc=config_module.HcuDSV41CTW4A8Config.from_config(SCHEMA)
        layer,_,logical=weight_fixture()
        layer.quant_config=qc;layer.moe_config=moe
        layer.ckpt_gate_proj_name='w1';layer.ckpt_up_proj_name='w3';layer.ckpt_down_proj_name='w2'
        layer.activation=MoEActivation.SILU;layer.global_num_experts=moe.num_experts;layer.apply_router_weight_on_input=False
        layer.register_buffer('_expert_map',torch.tensor([0,1]+[-1]*(moe.num_experts-2),dtype=torch.int32),persistent=False)
        method=qc.get_quant_method(layer,'model.layers.0.ffn.experts');layer.quant_method=method
        method.create_weights(layer,E,H,I,torch.bfloat16,weight_loader=layer.weight_loader)
        for e in range(E):
            for shard in ('w1','w3','w2'):
                q,s=logical[e,shard]; stem='w2' if shard=='w2' else 'w13'
                # Use binary-exact CT scales for the independent CPU reference.
                s=torch.full_like(s,1/32);logical[e,shard]=(q,s)
                for suffix,data in (('weight_packed',ct_pack(q)),('weight_scale',s),('weight_shape',torch.tensor(q.shape,dtype=torch.int64))):
                    name=f'{stem}_{suffix}'
                    getattr(layer,name).weight_loader(getattr(layer,name),data,name,shard_id=shard,expert_id=e,return_success=True)
        method.process_weights_after_loading(layer)
        assert isinstance(method.moe_kernel,mk.FusedMoEKernel) and method.supports_internal_mk
        x=(torch.arange(3*H).reshape(3,H).remainder(13).float()/32).to(torch.bfloat16)
        ids=torch.tensor([[0,1]]*3,dtype=torch.int64);weights=torch.tensor([[0.25,0.75]]*3)
        actual=method.apply(layer,x,weights,ids,None,None)
        qx,sx=quantize(x); reference=torch.zeros((3,H),dtype=torch.float32)
        for e in range(E):
            q1=torch.cat((logical[e,'w1'][0],logical[e,'w3'][0]),0)
            gateup=((qx.float()@q1.float().t())*sx/32).to(torch.bfloat16)
            mid=F.silu(gateup[:,:I].float())*gateup[:,I:].float()
            qmid,smid=quantize(mid)
            y=((qmid.float()@logical[e,'w2'][0].float().t())*smid/32).to(torch.bfloat16)
            reference += y.float()*weights[:,e:e+1]
        assert torch.equal(actual,reference.to(torch.bfloat16)), (actual-reference.to(actual.dtype)).abs().max().item()
        before=method.moe_kernel
        method.process_weights_after_loading(layer)
        assert method.moe_kernel is before
        empty=method.apply(layer,x[:0],weights[:0],ids[:0],None,None)
        assert empty.shape==(0,H)
        buf=method.moe_kernel.prepare_finalize.buffer
        assert any((isinstance(c,tuple) and c[0]=='ll_dispatch') or c=='ht_dispatch' for c in buf.calls)
        assert 'll_combine' in buf.calls if mode=='deepep_low_latency' else 'ht_combine' in buf.calls
        rejects(lambda: method.apply(layer,torch.zeros(5,H,dtype=torch.bfloat16),torch.ones(5,2),torch.zeros(5,2,dtype=torch.int64),None,None))
        assert method.moe_quant_config.quant_dtype==torch.int8 and method.moe_quant_config.weight_quant_dtype=='int4'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    torch.set_num_threads(1)
    original=(CompressedTensorsConfig.get_quant_method,oracle.select_w4a8_int8_moe_backend)
    if config_module is not None:
        config_module._is_hcu=lambda:True
        fp8_config._is_hcu=lambda:True
        activate_module.activate_extension(ExtensionContext('0.1.0','0.33.0'))
    def selector():
        required()
        hfcfg=SimpleNamespace(model_type='deepseek_v41',architectures=['DeepseekV41ForCausalLM'])
        assert config_module.HcuDSV41CTW4A8Config.override_quantization_method(SCHEMA,None,hfcfg)==NAME
        assert config_module.HcuDSV41CTW4A8Config.override_quantization_method(SCHEMA,None,SimpleNamespace(model_type='llama')) is None
        assert config_module.HcuDSV41CTW4A8Config.override_quantization_method(SCHEMA,'quark',hfcfg) is None
        fakeplatform=SimpleNamespace(verify_quantization=lambda n:None,supported_quantization=None)
        cfg=SimpleNamespace(quantization=None,model_arch_config=SimpleNamespace(quantization_config=SCHEMA),hf_config=hfcfg,allow_deprecated_quantization=False)
        with patch.object(model_module,'current_platform',fakeplatform),patch.object(model_module,'me_quant',quant_registry):
            ModelConfig._verify_quantization(cfg)
        assert cfg.quantization==NAME
    case('actual_selector_and_model_scope',selector)
    case('native_modular_LL_pipeline_and_exact_reference',lambda:pipeline('deepep_low_latency'))
    case('native_modular_HT_pipeline_and_exact_reference',lambda:pipeline('deepep_high_throughput'))
    def topology():
        required()
        for dp in (1,4,8,16):
            cfg,moe=context(dp=dp);config_module.validate_topology(cfg,moe)
        for field,value in (('tensor_parallel_size',2),('pipeline_parallel_size',2),('decode_context_parallel_size',2),('enable_eplb',True),('enable_elastic_ep',True)):
            cfg,moe=context();setattr(cfg.parallel_config,field,value)
            rejects(lambda:config_module.validate_topology(cfg,moe))
        cfg,moe=context();moe.max_num_tokens=3;rejects(lambda:config_module.validate_topology(cfg,moe))
        cfg,moe=context();cfg.compilation_config.cudagraph_capture_sizes=[8];rejects(lambda:config_module.validate_topology(cfg,moe))
    case('parallel_capacity_invariants_without_DP_whitelist',topology)
    def negative_schema():
        required(); cfg,moe=context();q=config_module.HcuDSV41CTW4A8Config.from_config(SCHEMA)
        layer,_,_=weight_fixture();layer.moe_config=moe
        layer.ckpt_gate_proj_name='w1';layer.ckpt_up_proj_name='w3';layer.ckpt_down_proj_name='w2'
        with patch.object(config_module,'get_current_vllm_config',return_value=cfg):
            moe.has_bias=True;rejects(lambda:q.get_quant_method(layer,'model.layers.0.ffn.experts'))
    case('unsupported_expert_contract_rejected',negative_schema)
    def activation():
        required();before=quant_registry.get_quantization_config(NAME)
        activate_module.activate_extension(ExtensionContext('0.1.0','0.33.0'));activate_module.check_health()
        assert quant_registry.get_quantization_config(NAME) is before
    case('public_registration_health_and_idempotence',activation)
    case('official_CT_and_oracle_symbols_unchanged',lambda:assert_original(original))
    assert not torch.cuda.is_initialized()
    root=Path(__file__).resolve().parents[2];sha=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
    sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'adapter_extension/src/hcu_vllm_das').rglob('ct_w4a8*.py')}
    report={'level':'CPU actual modular-kernel integration; EP, providers, quantizer and permutation doubled',
            'plugin':sha(root),'core':sha(root.parent/'vllm'),'adapter':sha(root.parent/'adapter-das'),'cuda_initialized':False,
            'source_sha256':sources,'cases':RESULTS,'passed':sum(r['status']=='passed' for r in RESULTS),'total':len(RESULTS)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['passed']==report['total'] else 1


def assert_original(original):
    required();assert original==(CompressedTensorsConfig.get_quant_method,oracle.select_w4a8_int8_moe_backend)

if __name__=='__main__':raise SystemExit(main())
