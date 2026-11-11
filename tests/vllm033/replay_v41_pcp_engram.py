#!/usr/bin/env python3
"""Pre-implementation CPU source replay; HIP/lookup/transport are doubles."""
from __future__ import annotations
from contextlib import ExitStack
import gc
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from vllm.models.deepseek_v41.common import engram as native
from vllm.models.deepseek_v41.amd import model
from hcu_vllm_das import v41_pcp
from hcu_vllm_das.models import v41_pcp as model_bridge
def cfg(width=8):
    return NS(use_v2_model_runner=True,model_config=NS(enforce_eager=True,hf_config=NS(model_type='deepseek_v41',engram_layer_ids=[],vision_n_layers=0)),
        parallel_config=NS(prefill_context_parallel_size=width,decode_context_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,tensor_parallel_size=1,enable_dbo=False,enable_eplb=False),
        speculative_config=None,lora_config=None,kv_transfer_config=None,kernel_config=NS(monokernel=False),cache_config=NS(swa_bounded_replay=False),
        attention_config=NS(backend=None,indexer_sparse_logits=False,resolve_indexer_kv_dtype=lambda _: 'fp8'))
try:
    prod=importlib.import_module('hcu_vllm_das.v41_engram')
    host=importlib.import_module('hcu_vllm_das.engram_host')
except ModuleNotFoundError as e:
    if not e.name.startswith('hcu_vllm_das.'):raise
    prod=host=None
RESULTS=[]
def require():assert prod is not None,'production Engram module absent'
def case(name,fn):
    try:fn();RESULTS.append(dict(case=name,status='passed'))
    except Exception as e:
        import traceback;traceback.print_exc()
        RESULTS.append(dict(case=name,status='failed',error=f'{type(e).__name__}: {e}'))
def reject(fn):
    try:fn()
    except (ValueError,RuntimeError,AssertionError):return
    raise AssertionError('unsupported contract accepted')
def config(width=8):
    c=cfg(width);c.engram_config=NS(cpu_offload=True,dp_shared_memory=False,embedding_across_dp=False,use_thp=False)
    c.scheduler_config=NS(max_num_batched_tokens=8)
    return c
class Hip:
    def __init__(self):self.registered=[];self.released=[];self.bad_pointer=False;self.fail=False
    def register(self,ptr,nbytes):
        if self.fail:raise RuntimeError('injected registration failure')
        self.registered.append((ptr,nbytes))
    def device_pointer(self,ptr):
        if self.bad_pointer:raise RuntimeError('injected device pointer failure')
        return ptr
    def unregister(self,ptr):self.released.append(ptr)
class Group:
    world_size=8
    rank_in_group=0
    def all_gather(self,x,dim):return self.collect(x,dim)

def configure(stack,c,rank=0,tp_size=1):
    group=Group();group.rank_in_group=rank
    hip=Hip()
    stack.enter_context(patch.object(prod,'get_current_vllm_config',return_value=c))
    stack.enter_context(patch.object(native,'get_current_vllm_config',return_value=c))
    stack.enter_context(patch.dict(prod.HcuV41PcpEngram._create_embedding.__globals__,get_current_vllm_config=lambda:c))
    stack.enter_context(patch.object(prod,'get_pcp_group',return_value=group))
    stack.enter_context(patch.object(native,'get_engram_dp_size',return_value=1))
    stack.enter_context(patch.object(native,'is_uva_available',return_value=True))
    stack.enter_context(patch.object(native,'get_tensor_model_parallel_world_size',return_value=tp_size))
    stack.enter_context(patch.object(prod,'get_tensor_model_parallel_rank',return_value=0))
    stack.enter_context(patch.object(host,'runtime',return_value=hip))
    stack.enter_context(patch.object(torch.Tensor,'is_pinned',return_value=True))
    stack.enter_context(patch.object(torch.cuda,'get_device_properties',return_value=NS(multi_processor_count=8)))
    stack.enter_context(patch.object(torch.accelerator,'current_device_index',return_value=0))
    return group,hip

class Projection(torch.nn.Module):
    def __init__(self,*args,**kwargs):super().__init__();self.width=args[1];self.inputs=[]
    def forward(self,x):self.inputs.append(x.clone());return x.new_zeros((x.shape[0],self.width))
def layer(layout):
    return prod.HcuV41PcpEngram(NS(hidden_size=4,hc_mult=2,rms_norm_eps=1e-6),None,layout,0,False,'model.layers.1.engram')
def layout_for(sizes):
    return NS(max_ngram_size=2,n_heads=len(sizes),head_dim=32,num_embeddings=(sum(sizes),),primes=((tuple(sizes),),))

def admission_and_constructor():
    require();c=config();c.model_config.hf_config.engram_layer_ids=[1]
    assert v41_pcp.validate_config(c)
    with ExitStack() as s:
        group,hip=configure(s,c)
        s.enter_context(patch.object(native,'ColumnParallelLinear',Projection))
        obj=layer(layout_for([2]*24))
        assert isinstance(obj,native.Engram) and type(obj.embed_tokens) is prod.HcuV41PcpEmbedding
        assert obj.embed_tokens.part_n_hash_cols==3
        assert obj.embed_tokens.weight.shape==(6,32)
        assert obj.embed_tokens.weight_scale_inv.shape==(6,1)
        assert len(hip.registered)==1
        assert model_bridge._DECODER_INIT.__globals__['Engram'] is prod.HcuV41PcpEngram
        assert model.DeepseekV4DecoderLayer.__init__.__globals__['Engram'] is native.Engram
        clone=prod.HcuV41PcpEngram._create_embedding
        assert clone.__code__ is native.Engram._create_embedding.__code__
        assert clone.__globals__['ParallelEngramEmbedding'] is prod.HcuV41PcpEmbedding
        assert native.Engram._create_embedding.__globals__['ParallelEngramEmbedding'] is native.ParallelEngramEmbedding
        assert native.NgramHashState is model.NgramHashState
        for field,value in [('dp_shared_memory',True),('embedding_across_dp',True),('use_thp',True),('cpu_offload',False)]:
            old=getattr(c.engram_config,field);setattr(c.engram_config,field,value)
            reject(lambda:v41_pcp.validate_config(c));setattr(c.engram_config,field,old)
        reject(lambda:obj.embed(torch.zeros(1,24,dtype=torch.int32)))
        assert not v41_pcp.validate_config(config(1))
        # Native Engram.forward is retained and actually entered; gate kernel double.
        assert obj.forward.__func__ is native.Engram.forward
        def lookup(table,ids,out,background=False):out.zero_()
        group.collect=lambda x,dim: torch.cat([x]*8,dim=dim)
        s.enter_context(patch.object(prod.HcuV41PcpEmbedding,'lookup',lookup))
        ids=torch.zeros(1,24,dtype=torch.int32);obj.prepare_embeddings(ids)
        class Gate:
            def __getitem__(self,grid):
                return lambda *args,**kw:args[5].copy_(args[0])
        s.enter_context(patch.object(native,'_fused_engram_post_wkv_kernel',Gate()))
        residual=torch.zeros(1,2,4);assert torch.equal(obj(residual,ids),residual)
        assert len(obj.wkv.inputs)==1 and obj.wkv.inputs[0].shape==(1,24*32)
        reject(lambda:obj.embed(ids))

def seeded_gather_200():
    require()
    for seed in range(200):
        g=torch.Generator().manual_seed(seed)
        heads=48 if seed%2==0 else 47
        sizes=torch.randint(1,6,(heads,),generator=g).tolist()
        offsets=torch.tensor([0,*torch.cumsum(torch.tensor(sizes),0).tolist()])
        counts=torch.randint(0,6,(8,),generator=g).tolist()
        if seed==0:counts=[0]*8
        if seed==1:counts=[1]+[0]*7
        slot=max(counts)
        ids=[]
        for count in counts:
            backing=torch.full((count,heads*2),-1,dtype=torch.int32)
            values=backing[:,::2]
            for h in range(heads):
                if count:values[:,h]=torch.randint(0,sizes[h],(count,),generator=g)+offsets[h]
            if count and seed%3==0:values[-1].fill_(-1)
            ids.append(values)
        all_ids=torch.cat([torch.cat((v,torch.full((slot-len(v),heads),-1,dtype=torch.int32))) for v in ids])
        # Independent reference: table row's value encodes the global row id.
        def full_lookup(rows):return rows.clamp_min(0).to(torch.bfloat16)[:,:,None].expand(-1,-1,32).clone().masked_fill((rows<0)[:,:,None],0)
        full=full_lookup(all_ids)
        c=config();c.scheduler_config.max_num_batched_tokens=8
        with ExitStack() as stack:
            group,_=configure(stack,c)
            stack.enter_context(patch.object(native,'ColumnParallelLinear',Projection))
            def lookup(table,rows,out,background=False):
                assert torch.equal(rows,all_ids)
                out.zero_();start=table.head_start;stop=min(heads,start+table.part_n_hash_cols)
                out[:,:stop-start]=full[:,start:stop]
            stack.enter_context(patch.object(prod.HcuV41PcpEmbedding,'lookup',lookup))
            for rank in range(8):
                group.rank_in_group=rank;obj=layer(layout_for(sizes));table=obj.embed_tokens
                assert table.head_start==rank*((heads+7)//8)
                start=int(offsets[table.head_start]);stop=int(offsets[min(heads,table.head_start+table.part_n_hash_cols)])
                weight=torch.arange(sum(sizes)*32,dtype=torch.int32).remainder(7).reshape(sum(sizes),32).to(torch.float8_e4m3fn)
                scales=torch.arange(sum(sizes),dtype=torch.uint8)[:,None]
                for param,loaded in ((table.weight,weight),(table.weight_scale_inv,scales)):
                    param.weight_loader(param,loaded)
                    assert torch.equal(param.view(torch.uint8),loaded[start:stop].view(torch.uint8))
                def collect(x,dim):
                    if dim==0 and x.ndim==1:
                        assert x.item()==counts[rank];return torch.tensor(counts,dtype=x.dtype)
                    if dim==0:
                        assert x.is_contiguous() and torch.equal(x,all_ids[rank*slot:(rank+1)*slot]);return all_ids
                    assert dim==1 and x.shape==(8*slot,table.part_n_hash_cols,32)
                    assert torch.equal(x[:,:min(heads-table.head_start,table.part_n_hash_cols)],full[:,table.head_start:min(heads,table.head_start+table.part_n_hash_cols)])
                    span=table.part_n_hash_cols*8;begin=(table.head_start//span)*span
                    padded=torch.zeros(8*slot,span,32,dtype=x.dtype);available=min(span,heads-begin);padded[:,:available]=full[:,begin:begin+available];return padded
                group.collect=collect
                obj.prepare_embeddings(ids[rank]);actual=obj.embed(ids[rank]);expected=full_lookup(ids[rank])
                assert actual.dtype==expected.dtype and actual.shape==expected.shape and torch.equal(actual,expected),(seed,rank)
                reject(lambda:obj.embed(ids[rank]))
                obj.prepare_embeddings(ids[rank]);reject(lambda:obj.embed(torch.empty(counts[rank]+1,heads,dtype=torch.int32)))
            # TP2 x PCP8 order, independent global-head gather double.
            stack.enter_context(patch.object(native,'get_tensor_model_parallel_world_size',return_value=2))
            stack.enter_context(patch.object(prod,'tensor_model_parallel_all_gather',side_effect=lambda rows,dim: full_lookup(ids[group.rank_in_group])))
            for tp in range(2):
                with patch.object(prod,'get_tensor_model_parallel_rank',return_value=tp):
                    for rank in range(8):
                        group.rank_in_group=rank
                        obj=layer(layout_for(sizes));table=obj.embed_tokens
                        assert table.head_start==(tp*8+rank)*((heads+15)//16)
                        group.collect=collect
                        obj.prepare_embeddings(ids[rank]);actual=obj.embed(ids[rank]);expected=full_lookup(ids[rank])
                        assert torch.equal(actual,expected),(seed,tp,rank)
            reject(lambda:layer(layout_for([2]*24)))

def host_lifetime():
    require();hip=Hip()
    with patch.object(host,'runtime',return_value=hip),patch.object(torch.Tensor,'is_pinned',return_value=True):
        storage=host.allocate_registered(4096);ptr=storage.data_ptr();assert ptr%4096==0
        parameter=torch.nn.Parameter(storage[:32],requires_grad=False);del storage;gc.collect()
        assert not hip.released
        del parameter;gc.collect();assert hip.released==[ptr]
        hip.bad_pointer=True;reject(lambda:host.allocate_registered(4096));assert len(hip.released)==2
        hip.bad_pointer=False
        with patch.object(torch.Tensor,'is_pinned',return_value=False):reject(lambda:host.allocate_registered(4096))
        assert len(hip.released)==3
        hip.fail=True;reject(lambda:host.allocate_registered(4096));assert len(hip.released)==3
    with ExitStack() as stack:
        c=config();configure(stack,c)
        table=prod.HcuV41PcpEmbedding(48,32,(2,)*24,cpu_offload=True)
        # Replacing storage invalidates registration; no fallback to private copies.
        table.weight=torch.nn.Parameter(torch.zeros_like(table.weight),requires_grad=False)
        reject(table._storage)

for name,fn in [('real_Engram_constructor_forward_and_admission',admission_and_constructor),('seeded_rank_gather_loader_200x8',seeded_gather_200),('host_lifetime_and_rollback',host_lifetime)]:case(name,fn)
root=Path(__file__).resolve().parents[2]
report=dict(cases=RESULTS,passed=sum(x['status']=='passed' for x in RESULTS),total=len(RESULTS),cuda_initialized=torch.cuda.is_initialized(),native_source=native.__file__,source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'adapter_extension/src/hcu_vllm_das').glob('*engram*.py')},boundary='CPU source replay; HIP APIs, GPU lookup/gate kernels, projection and rank transport doubled. Native Engram constructor/forward, loader, owned storage/staging code execute; no full model or native GPU hash verification.')
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));sys.exit(0 if report['passed']==report['total'] and not report['cuda_initialized'] else 1)
