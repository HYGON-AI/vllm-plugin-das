#!/usr/bin/env python3
"""Pre-implementation DP Engram source replay. HIP, projection, lookup/select,
attention/FFN and heavy outer allocation boundaries are explicit CPU doubles.
"""
from __future__ import annotations
from contextlib import ExitStack
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
from hcu_adapter.extensions import ExtensionContext
from vllm.models.deepseek_v41.common import engram as native
from vllm.models.deepseek_v41.amd import model as amd
from vllm.model_executor.models import ModelRegistry
from hcu_vllm_das.models import v41_pcp as bridge
from hcu_vllm_das import v41_pcp_activation as activation
from hcu_vllm_das import engram_host as host
try:prod=importlib.import_module('hcu_vllm_das.v41_target_engram')
except ModuleNotFoundError as e:
    if e.name!='hcu_vllm_das.v41_target_engram':raise
    prod=None
RESULTS=[]
def require():assert prod is not None,'DP target Engram module absent'
def case(name,fn):
    try:fn();RESULTS.append(dict(case=name,status='passed'))
    except Exception as e:
        import traceback;traceback.print_exc()
        RESULTS.append(dict(case=name,status='failed',error=f'{type(e).__name__}: {e}'))
def reject(fn):
    try:fn()
    except (RuntimeError,ValueError,AssertionError):return
    raise AssertionError('invalid host contract accepted')
def cfg(dp=8):
    hf=NS(model_type='deepseek_v41',engram_layer_ids=[1],hidden_size=4,rms_norm_eps=1e-6,hc_eps=1e-6,hc_mult=2,hc_sinkhorn_iters=3)
    return NS(model_config=NS(enforce_eager=True,hf_config=hf),use_v2_model_runner=True,
        parallel_config=NS(data_parallel_size=dp,prefill_context_parallel_size=1,decode_context_parallel_size=1,pipeline_parallel_size=1,enable_eplb=False,enable_elastic_ep=False,enable_dbo=False),
        engram_config=NS(cpu_offload=True,dp_shared_memory=False,use_thp=False,embedding_across_dp=True),lora_config=None,
        scheduler_config=NS(max_num_batched_tokens=8),quant_config=None)
class Projection(torch.nn.Module):
    def __init__(self,*args,**kwargs):super().__init__();self.width=args[1];self.inputs=[]
    def forward(self,x):self.inputs.append(x.clone());return x.new_zeros((len(x),self.width))
class Hip:
    def __init__(self):self.ptrs=[]
    def register(self,p,n):self.ptrs.append(p)
    def unregister(self,p):pass
    def device_pointer(self,p):return p
class Group:
    def __init__(self,rank,width,offset=0):self.rank_in_group=rank;self.world_size=width;self.ranks=list(range(offset,offset+width))
    def all_gather(self,x,dim):return self.collect(x,dim)
def configure(stack,c,group,tp=1,tp_rank=0,global_rank=None):
    stack.enter_context(patch.object(prod,'get_current_vllm_config',return_value=c))
    stack.enter_context(patch.object(native,'get_current_vllm_config',return_value=c))
    stack.enter_context(patch.dict(prod.HcuV41HostEngram._create_embedding.__globals__,get_current_vllm_config=lambda:c))
    stack.enter_context(patch.object(native,'get_engram_dp_size',return_value=group.world_size))
    stack.enter_context(patch.object(native,'get_engram_dp_group',return_value=group if group.world_size>1 else None))
    stack.enter_context(patch.object(native,'get_dp_group',return_value=NS(rank_in_group=group.rank_in_group if global_rank is None else global_rank)))
    stack.enter_context(patch.object(native,'get_tensor_model_parallel_world_size',return_value=tp))
    stack.enter_context(patch.object(native,'get_tensor_model_parallel_rank',return_value=tp_rank))
    stack.enter_context(patch.object(native,'is_uva_available',return_value=True))
    stack.enter_context(patch.object(host,'runtime',return_value=Hip()))
    stack.enter_context(patch.object(torch.Tensor,'is_pinned',return_value=True))
    stack.enter_context(patch.object(torch.cuda,'get_device_properties',return_value=NS(multi_processor_count=8)))
    stack.enter_context(patch.object(torch.accelerator,'current_device_index',return_value=0))
    stack.enter_context(patch.object(native,'ColumnParallelLinear',Projection))
def layout(sizes):return NS(layer_ids=(1,),max_ngram_size=2,n_heads=len(sizes),head_dim=32,num_embeddings=(sum(sizes),),primes=((tuple(sizes),),))
def make_layer(sizes):return prod.HcuV41HostEngram(NS(hidden_size=4,hc_mult=2,rms_norm_eps=1e-6),None,layout(sizes),0,False,'model.layers.1.engram')

def registration_and_decoder():
    require();context=ExtensionContext('0.1.0','0.33.0')
    with ExitStack() as s:
        s.enter_context(patch.dict(os.environ,{'VLLM_HCU_DSV41_ENGRAM_HOST':'1','VLLM_HCU_DSV41_PCP_ATTENTION':'0'}))
        s.enter_context(patch.object(activation,'_state',None));s.enter_context(patch.object(activation,'_failure',None))
        s.enter_context(patch.object(activation,'_is_hcu',return_value=True))
        s.enter_context(patch.object(ModelRegistry,'models',dict(ModelRegistry.models)))
        activation.activate_extension(context);entry=ModelRegistry.models['DeepseekV41ForCausalLM']
        assert entry.module_name=='hcu_vllm_das.models.v41_pcp'
        activation.activate_extension(context);assert ModelRegistry.models['DeepseekV41ForCausalLM'] is entry
        activation.check_health();assert activation.host_enabled() and not activation.pcp_enabled()
        c=cfg();group=Group(0,8);configure(s,c,group)
        assert prod.use_host_target(c)
        ctor=bridge._HOST_DECODER_INIT
        assert ctor.__code__ is amd.DeepseekV4DecoderLayer.__init__.__code__
        assert ctor.__globals__['Engram'] is prod.HcuV41HostEngram
        assert ctor.__globals__['_select_dsv4_attn_cls'] is amd._select_dsv4_attn_cls
        assert amd.DeepseekV4DecoderLayer.__init__.__globals__['Engram'] is native.Engram
        # Execute real native decoder constructor, retaining real owned Engram.
        class Noop(torch.nn.Module):
            def __init__(self,*a,**k):super().__init__();self.mono=None
            @staticmethod
            def create(*a,**k):return None
        replacements={'_select_dsv4_attn_cls':lambda c:Noop,'DeepseekV4MoE':Noop,'RMSNorm':Noop,'MHCPreDelayedOp':Noop,'MHCPostOp':Noop,'MonoDecodeLayer':Noop}
        with patch.dict(ctor.__globals__,replacements):
            decoder=bridge.HcuV41HostDecoderLayer(c,'model.layers.1',engram_layout=layout([2]*48))
        assert type(decoder.engram) is prod.HcuV41HostEngram and type(decoder.engram.embed_tokens) is prod.HcuV41HostEmbedding
        assert isinstance(decoder.attn,Noop)
        assert decoder.engram.forward.__func__ is native.Engram.forward
        assert amd.DeepseekV4Model.forward is bridge.HcuV41PcpBackbone.forward
        c.engram_config.dp_shared_memory=True;reject(lambda:prod.validate_host_config(c));c.engram_config.dp_shared_memory=False
        c.model_config.enforce_eager=False;reject(lambda:prod.validate_host_config(c));c.model_config.enforce_eager=True
        c.parallel_config.prefill_context_parallel_size=2;reject(lambda:prod.validate_host_config(c));c.parallel_config.prefill_context_parallel_size=1
        # Public target entry uses a checked native outer-code bridge; heavy VL
        # allocation boundary is doubled while its actual decoder was built above.
        obj=object.__new__(bridge.HcuV41PcpForCausalLM)
        def outer(self,*,vllm_config,prefix=''):torch.nn.Module.__init__(self);self.language_model=NS(model=NS(layers=[decoder]))
        with patch.object(bridge,'_HOST_OUTER_INIT',outer):obj.__init__(vllm_config=c)
        assert type(obj.language_model.model.layers[0].engram.embed_tokens) is prod.HcuV41HostEmbedding
        with patch.dict(os.environ,{'VLLM_HCU_DSV41_ENGRAM_HOST':'0'}):reject(activation.check_health)


def native_dp_gather_200():
    require()
    for seed in range(200):
        gen=torch.Generator().manual_seed(seed);width=(2,4,8)[seed%3];tp=1 if seed%2==0 else 2;heads=48 if tp==1 else 64
        sizes=torch.randint(1,6,(heads,),generator=gen).tolist();offsets=torch.tensor([0,*torch.cumsum(torch.tensor(sizes),0).tolist()])
        counts=torch.randint(0,5,(width,),generator=gen).tolist()
        if seed==0:counts=[0]*width
        slot=max(counts);ids=[]
        for count in counts:
            value=torch.empty(count,heads,dtype=torch.int32)
            for h in range(heads):
                if count:value[:,h]=offsets[h]+torch.randint(0,sizes[h],(count,),generator=gen)
            if count and seed%3==0:value[-1].fill_(-1)
            ids.append(value)
        padded=[torch.cat((v,torch.full((slot-len(v),heads),-1,dtype=torch.int32))) for v in ids]
        hashes=torch.cat(padded)
        full=hashes.clamp_min(0).to(torch.bfloat16)[:,:,None].expand(-1,-1,32).clone().masked_fill((hashes<0)[:,:,None],0)
        for trank in range(tp):
            for rank in range(width):
                global_rank=rank+width;c=cfg(width*2);group=Group(rank,width,offset=width)
                with ExitStack() as s:
                    configure(s,c,group,tp,trank,global_rank)
                    obj=make_layer(sizes);table=obj.embed_tokens
                    part=(heads+tp*width-1)//(tp*width);assert table.head_start==(trank*width+rank)*part
                    start=int(offsets[table.head_start]);stop=int(offsets[min(heads,table.head_start+part)])
                    raw=torch.arange(sum(sizes)*32).remainder(7).reshape(sum(sizes),32).to(torch.float8_e4m3fn);scales=torch.arange(sum(sizes),dtype=torch.uint8)[:,None]
                    for param,value in ((table.weight,raw),(table.weight_scale_inv,scales)):
                        param.weight_loader(param,value);assert torch.equal(param.view(torch.uint8),value[start:stop].view(torch.uint8))
                    allcounts=torch.tensor([*([slot+1]*width),*counts])
                    s.enter_context(patch.object(native,'get_forward_context',return_value=NS(dp_metadata=NS(num_tokens_across_dp_cpu=allcounts))))
                    calls=[]
                    def collect(x,dim):
                        calls.append(dim)
                        if x.ndim==3 and x.shape[1:]==(1,heads):return hashes[:,None,:]
                        assert x.ndim==3 and tuple(x.shape)==(slot*width,part,32)
                        # EDP gather stores [source rank][all token slots][heads].
                        return torch.cat([full[:,(trank*width+r)*part:(trank*width+r+1)*part] for r in range(width)],dim=0)
                    group.collect=collect
                    def lookup(table,rows,out,background=False):
                        assert torch.equal(rows,hashes);out.copy_(full[:,table.head_start:table.head_start+part])
                    s.enter_context(patch.object(prod.HcuV41HostEmbedding,'lookup',lookup))
                    def select(gathered,out,source_tokens,token_start,local_width):
                        for r in range(width):out[:,r*part:(r+1)*part]=gathered[r*source_tokens+token_start:r*source_tokens+token_start+len(out)]
                    s.enter_context(patch.object(native,'_engram_select_rows',side_effect=select))
                    expected=ids[rank].clamp_min(0).to(torch.bfloat16)[:,:,None].expand(-1,-1,32).clone().masked_fill((ids[rank]<0)[:,:,None],0)
                    s.enter_context(patch.object(native,'tensor_model_parallel_all_gather',side_effect=lambda rows,dim:expected))
                    gathered=native.gather_engram_hashes(ids[rank][:,None,:])
                    obj.prepare_embeddings(gathered[:,0]);actual=obj.embed(ids[rank])
                    assert actual.shape==expected.shape and actual.dtype==expected.dtype and torch.equal(actual,expected),(seed,trank,rank)
                    assert calls==[0,0]
                    s.enter_context(patch.object(native,'get_forward_context',return_value=NS(dp_metadata=None)))
                    reject(lambda:native.gather_engram_hashes(ids[rank][:,None,:]))
                    assert obj.prepare_embeddings.__func__ is prod.HcuV41HostEngram.prepare_embeddings
                    # Execute native residual gate forward, after native DP staging.
                    class Gate:
                        def __getitem__(self,grid):return lambda *a,**kw:a[5].copy_(a[0])
                    s.enter_context(patch.object(native,'_fused_engram_post_wkv_kernel',Gate()))
                    s.enter_context(patch.object(native,'get_forward_context',return_value=NS(dp_metadata=NS(num_tokens_across_dp_cpu=allcounts))))
                    residual=torch.zeros(len(ids[rank]),2,4);out=obj(residual,ids[rank]);assert torch.equal(out,residual)

for name,fn in [('public_host_flag_and_real_decoder_Engram_construction',registration_and_decoder),('native_DP_hash_stage_head_restore_200',native_dp_gather_200)]:case(name,fn)
root=Path(__file__).resolve().parents[2]
report=dict(passed=sum(x['status']=='passed' for x in RESULTS),total=len(RESULTS),cases=RESULTS,cuda_initialized=torch.cuda.is_initialized(),native_source=native.__file__,source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'adapter_extension/src/hcu_vllm_das').glob('*target_engram*.py')},boundary='CPU native decoder/Engram construction, loaders and DP hash/gather/head orchestration; HIP, kernel lookup/select/gate and heavy VL outer allocation are doubles. No full target model/hash numerics/GPU/Graph verification.')
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));sys.exit(0 if report['passed']==report['total'] and not report['cuda_initialized'] else 1)
