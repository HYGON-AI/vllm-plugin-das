#!/usr/bin/env python3
"""Fixed before implementation: CPU source integration of HCU V41 PCP attention.
Transport, cache writer and attention kernels are explicit CPU doubles. Native
common scheduling, CR2 linking and production FP8 scoring execute on CPU.
"""
from __future__ import annotations
from contextlib import ExitStack
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from vllm.models.deepseek_v41.amd import model as native_model, rocm
from vllm.models.deepseek_v41.amd import vl_model
from vllm.models.deepseek_v41 import attention as common, compressor
from vllm.model_executor.models import ModelRegistry
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata, DeepSeekV32IndexerDecodeMetadata
from hcu_adapter.extensions import ExtensionContext, ExtensionCommitError
from replay_v41_fp8_indexer import pack, reference_decode

try:
    prod=importlib.import_module('hcu_vllm_das.v41_pcp')
    models=importlib.import_module('hcu_vllm_das.models.v41_pcp')
    activation=importlib.import_module('hcu_vllm_das.v41_pcp_activation')
except ModuleNotFoundError as error:
    if not error.name.startswith('hcu_vllm_das.'):raise
    prod=models=activation=None
RESULTS=[]

def require():assert prod is not None,'PCP production modules are absent'
def case(name, fn):
    try:fn();RESULTS.append(dict(case=name,status='passed'))
    except Exception as error:
        import traceback
        traceback.print_exc()
        RESULTS.append(dict(case=name,status='failed',error=f'{type(error).__name__}: {error}'))
def reject(fn, kind=(ValueError, RuntimeError)):
    try:fn()
    except kind:return
    raise AssertionError('invalid PCP contract admitted')
def cfg(width=8):
    return NS(model_config=NS(enforce_eager=True,hf_config=NS(model_type='deepseek_v41',engram_layer_ids=[],vision_n_layers=0)),
        parallel_config=NS(prefill_context_parallel_size=width,decode_context_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,tensor_parallel_size=1,enable_dbo=False,enable_eplb=False),
        use_v2_model_runner=True,speculative_config=None,lora_config=None,kv_transfer_config=None,
        kernel_config=NS(monokernel=False),cache_config=NS(swa_bounded_replay=False),
        attention_config=NS(backend=None,indexer_sparse_logits=False,resolve_indexer_kv_dtype=lambda _: 'fp8'))

def admission():
    require()
    for n in (2,4,8,16):assert prod.validate_config(cfg(n))
    assert not prod.validate_config(cfg(1))
    x=cfg();x.cache_config.swa_bounded_replay=True;assert prod.validate_config(x)
    for field,value in [('enforce_eager',False)]:
        x=cfg();setattr(x.model_config,field,value);reject(lambda:prod.validate_config(x))
    for field in ('decode_context_parallel_size','pipeline_parallel_size','data_parallel_size'):
        x=cfg();setattr(x.parallel_config,field,2);reject(lambda:prod.validate_config(x))
    for field in ('speculative_config','lora_config','kv_transfer_config'):
        x=cfg();setattr(x,field,object());reject(lambda:prod.validate_config(x))
    x=cfg();x.model_config.hf_config.engram_layer_ids=[1];reject(lambda:prod.validate_config(x))
    x=cfg();x.kernel_config.monokernel=True;reject(lambda:prod.validate_config(x))
    x=cfg();x.attention_config.resolve_indexer_kv_dtype=lambda _: 'mxfp4';reject(lambda:prod.validate_config(x))
    assert prod.HcuV41PcpBackend.supports_pcp() and prod.HcuV41PcpSwaBackend.supports_pcp()
    assert prod.HcuV41PcpBackend.get_builder_cls() is rocm.DeepseekV41ROCMAiterMLASparseMetadataBuilder
    assert prod.HcuV41PcpSwaBackend.get_builder_cls() is rocm.DeepseekV41ROCMAiterSparseSWAMetadataBuilder

def constructors():
    require()
    pairs=[(models._DECODER_INIT,native_model.DeepseekV4DecoderLayer.__init__,'_select_dsv4_attn_cls',models.select_attention),
           (models._BACKBONE_INIT,native_model.DeepseekV4Model.__init__,'DeepseekV4DecoderLayer',models.HcuV41PcpDecoderLayer),
           (models._OUTER_INIT,vl_model.DeepseekV41ForCausalLM.__init__,'DeepseekV41LLMForCausalLM',models.HcuV41PcpLLM)]
    for owned,native,key,replacement in pairs:
        assert owned.__code__ is native.__code__ and owned.__closure__ is native.__closure__
        assert inspect.signature(owned)==inspect.signature(native)
        assert owned.__globals__ is not native.__globals__
        assert owned.__globals__[key] is replacement
        assert native.__globals__[key] is not replacement
    assert issubclass(models.HcuV41PcpDecoderLayer,native_model.DeepseekV4DecoderLayer)
    assert models.HcuV41PcpLLM.model_cls is models.HcuV41PcpBackbone
    assert models.select_attention(cfg(1)) is rocm.DeepseekV41ROCMAiterMLAAttention
    assert models.select_attention(cfg()) is prod.HcuV41PcpAttention
    reject(lambda:models._constructor(native_model.DeepseekV4Model.__init__,'bad',{}))


def decoder_replay_gate():
    require()
    gate=models.HcuV41PcpBackbone._decoder_replay_supported
    def backbone(**kw):
        return NS(start_layer=0,end_layer=40,use_sequence_parallel=False,
                  config=NS(num_hidden_layers=40,engram_layer_ids=kw.pop('engram',[1,14])),**kw)
    def replay_cfg(width=8,ubatch=False):
        x=cfg(width);x.cache_config.swa_bounded_replay=True
        x.parallel_config.use_ubatching=ubatch;return x
    assert gate(backbone(),replay_cfg(),20) is True
    assert gate(backbone(),replay_cfg(ubatch=True),20) is False
    assert gate(backbone(engram=[1,20]),replay_cfg(),20) is False
    sp=backbone();sp.use_sequence_parallel=True;assert gate(sp,replay_cfg(),20) is False
    stage=backbone();stage.end_layer=30;assert gate(stage,replay_cfg(),20) is False
    # PCP1 keeps the exact native gate, which still refuses native PCP elsewhere.
    calls=[]
    with patch.object(models,'_REPLAY_NATIVE',lambda *a:calls.append(a) or 'native'):
        assert gate(backbone(),replay_cfg(1),20)=='native' and len(calls)==1
    assert models._REPLAY_NATIVE is native_model.DeepseekV4Model._decoder_replay_supported

def publication():
    require()
    with ExitStack() as s:
        s.enter_context(patch.dict(os.environ,{'VLLM_HCU_DSV41_PCP_ATTENTION':'1'}))
        s.enter_context(patch.object(activation,'_state',None));s.enter_context(patch.object(activation,'_failure',None))
        s.enter_context(patch.object(activation,'_is_hcu',return_value=True))
        s.enter_context(patch.object(ModelRegistry,'models',dict(ModelRegistry.models)))
        old=ModelRegistry.models['DeepseekV41ForCausalLM']
        activation.activate_extension(ExtensionContext('0.1.0','0.33.0'))
        entry=ModelRegistry.models['DeepseekV41ForCausalLM']
        assert entry is not old and entry.module_name=='hcu_vllm_das.models.v41_pcp'
        activation.activate_extension(ExtensionContext('0.1.0','0.33.0'))
        assert ModelRegistry.models['DeepseekV41ForCausalLM'] is entry
        activation.check_health()
        ModelRegistry.models['DeepseekV41ForCausalLM']=old
        reject(activation.check_health,ExtensionCommitError)
    with ExitStack() as s:
        s.enter_context(patch.dict(os.environ,{'VLLM_HCU_DSV41_PCP_ATTENTION':'0'}))
        s.enter_context(patch.object(activation,'_state',None));s.enter_context(patch.object(activation,'_failure',None))
        s.enter_context(patch.object(activation,'_is_hcu',return_value=True))
        original=ModelRegistry.models['DeepseekV41ForCausalLM']
        activation.activate_extension(ExtensionContext('0.1.0','0.33.0'))
        assert ModelRegistry.models['DeepseekV41ForCausalLM'] is original


def native_ring_200():
    require()
    for seed in range(200):
        n=seed%5+1;total=16*n;capacity=8
        # Two symmetric fragments per rank, one request, no fake third fragment.
        rows=[];keys=[]
        for rank in range(8):
            pos=torch.tensor([*range(rank*n,(rank+1)*n),*range((15-rank)*n,(16-rank)*n)])
            rows.append(pos);keys.extend([(3<<32)|int(pos[n-1]),(3<<32)|int(pos[-1])])
        gathered=torch.tensor(keys,dtype=torch.int64)
        for rank,pos in enumerate(rows):
            builder=object.__new__(compressor.CompressorMetadataBuilder)
            builder.capacity=capacity;builder.device=torch.device('cpu');builder.vllm_config=NS(scheduler_config=NS(max_num_seqs=1))
            meta=compressor.CompressorMetadata(torch.full((2*n,),3*capacity)+pos%capacity,torch.tensor([0,n,2*n]),torch.cat([torch.zeros(n),torch.ones(n)]).int())
            with patch.object(compressor,'get_pcp_group',return_value=NS(all_gather=lambda x,dim:gathered)):
                builder._link_pcp_chunks(meta,2,pos)
            for req,start in enumerate((0,n)):
                p=int(pos[start]);actual=int(meta.prev_row_indices[req])
                expected=next((i for i,key in enumerate(keys) if key==((3<<32)|(p-1))),-1) if p%2 else -1
                assert actual==expected,(seed,rank,req,actual,expected)
            slots=meta.ring_write_slots
            valid=slots[slots>=0]
            expected=[] if (total-1)%2 else [3*capacity+(total-1)%capacity]
            assert valid.tolist()==expected


def native_preparation_200():
    require()
    for seed in range(200):
        size=seed%4+1
        positions=[torch.tensor([*range(r*size,(r+1)*size),*range((15-r)*size,(16-r)*size)]) for r in range(8)]
        allpos=torch.cat(positions);global_kv=allpos[:,None].float();global_latent=global_kv+100
        for rank,pos in enumerate(positions):
            events=[];attn=object.__new__(prod.HcuV41PcpAttention);torch.nn.Module.__init__(attn)
            attn.use_pcp=True;attn.aux_stream_list=None;attn.n_local_heads=1;attn.head_dim=1;attn.ln_events=[None,None]
            attn.rotary_emb=attn.indexer_rotary_emb=NS(cos_sin_cache=torch.empty(1));attn.swa_cache_layer=NS(prefix='swa',block_size=32,kv_cache=torch.empty(1,32,584,dtype=torch.uint8))
            attn.padded_heads=1;attn.kv_mxfp8=False;attn.eps=1e-6
            attn._wq_b_proj=lambda qr,scale:qr
            global_slots=allpos.long()+32
            def swa_store(values,cache,slots,p,*args):
                events.append('swa_store');assert torch.equal(values,global_kv) and torch.equal(slots,global_slots) and torch.equal(p,allpos)
            def q_prepare(q,kv,cache,slots,p,*args):
                events.append('local_q_prepare');assert slots.numel()==0 and torch.equal(p,pos);return q
            class Compressor:
                def __call__(self,scores,p):events.append('compress');return p[:,None].float()+100
                def insert_cache(self,latent,p,rope):
                    events.append('main_store');assert torch.equal(p,allpos) and torch.equal(latent,global_latent)
            source_indexer=object.__new__(common.DeepseekV4Indexer);torch.nn.Module.__init__(source_indexer)
            source_indexer.owns_k=True;source_indexer.compress_ratio=1;source_indexer.use_fp4_kv=False;source_indexer.n_head=1
            source_indexer.wk=lambda x:(x,None);source_indexer.k_norm=NS(weight=torch.ones(1),variance_epsilon=1e-6)
            source_indexer.k_cache=NS(prefix='k',kv_cache=torch.empty(1))
            def k_store(k,p,rope,norm,eps,cache,slots,*args):
                events.append('index_store');assert torch.equal(k,global_latent) and torch.equal(p,allpos) and torch.equal(slots,global_slots)
            class Indexer:
                def __call__(self,qr,latent,weights,p,rope,scale,*,cache_latent,cache_positions):
                    assert torch.equal(latent,pos[:,None].float()+100)
                    source_indexer._produce_k(cache_latent,cache_positions,rope)
                    return None,None,None
            attn.compressor=Compressor();attn.indexer=Indexer()
            def gather(x,dim):
                if torch.equal(x,pos[:,None].float()):events.append('swa_gather');return global_kv
                events.append('latent_gather');assert torch.equal(x,pos[:,None].float()+100);return global_latent
            def finish(hidden,iq,iqs,iw,q,kv,p,out):events.append('local_attention');out.copy_(q);assert torch.equal(p,pos)
            attn._sparse_indexer_and_attn=finish
            q=pos[:,None].float();out=torch.empty(2*size,1,1)
            metadata={'swa':NS(block_size=32,cache_positions=allpos,cache_slot_mapping=global_slots,slot_mapping=global_slots[rank*2*size:(rank+1)*2*size]),'k':NS(slot_mapping=global_slots)}
            with ExitStack() as stack:
                stack.enter_context(patch.object(common,'get_forward_context',return_value=NS(attn_metadata=metadata)))
                stack.enter_context(patch.object(common,'get_pcp_group',return_value=NS(all_gather=gather)))
                stack.enter_context(patch.object(common,'indexer_k_norm_rope_store',side_effect=k_store))
                stack.enter_context(patch.object(torch.ops._C,'fused_deepseek_v4_kv_rope_insert',side_effect=swa_store,create=True))
                stack.enter_context(patch.object(torch.ops._C,'fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert',side_effect=q_prepare,create=True))
                common.DeepseekV4Attention._prepare_and_attn(attn,q,q,q,None,q,q,pos,out)
            assert torch.equal(out[:,0,0],pos.float())
            assert events==['swa_gather','swa_store','local_q_prepare','compress','latent_gather','index_store','main_store','local_attention'],events
    assert prod.HcuV41PcpAttention.forward is common.DeepseekV4Attention.forward


def local_indexer_200():
    require()
    for seed in range(200):
        cache,logical,scales=pack(seed);snapshot=cache.clone()
        for rank in range(8):
            # Different absolute local query bounds, full shared K cache.
            limit=1+(seed+rank)%30
            q=torch.ones(1,2,128,dtype=prod.reference.FP8_DTYPE)
            weights=torch.tensor([[.5,1.]])
            tables=torch.tensor([[3,1]],dtype=torch.int32)
            decode=DeepSeekV32IndexerDecodeMetadata(tables,torch.tensor([limit]),torch.tensor([1]),False,torch.empty(0))
            meta=DeepseekV32IndexerMetadata(torch.tensor([limit]),limit,torch.zeros(16,dtype=torch.int64),1,1,0,0,decode=decode)
            layer=object.__new__(prod.HcuV41PcpSparseIndexer);torch.nn.Module.__init__(layer)
            layer.k_cache=NS(prefix='k',kv_cache=cache);layer.topk_tokens=32;layer.head_dim=128;layer.max_model_len=32;layer.max_total_seq_len=64;layer.compress_ratio=1
            layer.topk_indices_buffer=torch.full((4,32),999,dtype=torch.int32);layer.candidate_blocks=None;layer.candidate_block_size=0;layer.candidate_write=False
            with patch.object(prod.reference,'get_forward_context',return_value=NS(attn_metadata={'k':meta})):
                actual=layer.forward_hip(torch.zeros(4,1),q,None,weights)
            torch.testing.assert_close(actual[0,:limit],torch.arange(limit,dtype=torch.int32),atol=0,rtol=0)
            assert (actual[0,limit:]==-1).all() and (actual[1:]==-1).all()
            expected,_=reference_decode(q.unsqueeze(0),logical,scales,weights,torch.tensor([limit]),tables,32)
            got=prod.reference.decode_logits(q.unsqueeze(0),cache,weights,torch.tensor([limit]),tables,32)
            torch.testing.assert_close(got,expected,atol=0,rtol=0)
        assert torch.equal(cache,snapshot)


def local_output():
    require()
    for actual in (0,1,3):
        obj=object.__new__(prod.HcuV41PcpAttention);torch.nn.Module.__init__(obj)
        obj.swa_cache_layer=NS(prefix='swa');obj.use_pcp=True
        meta=NS(num_decode_tokens=actual,num_prefill_tokens=0)
        q=torch.ones(5,2,4);kv=torch.ones(5,4);positions=torch.arange(5);out=torch.full_like(q,777)
        calls=[]
        def native(self,q,kv,p,output):calls.append(q.shape[0]);assert q.shape[0]==actual;output.copy_(q*2)
        with patch.object(prod,'get_forward_context',return_value=NS(attn_metadata={'swa':meta})),patch.object(rocm.DeepseekV41ROCMAiterMLAAttention,'forward_mqa',native):
            obj.forward_mqa(q,kv,positions,out)
        assert calls==([] if actual==0 else [actual])
        assert (out[:actual]==2).all() and (out[actual:]==0).all()
    assert prod.HcuV41PcpAttention._o_proj is rocm.DeepseekV41ROCMAiterMLAAttention._o_proj


for name,fn in [('admission_and_owned_backends',admission),('native_constructor_namespace',constructors),('decoder_replay_gate',decoder_replay_gate),('public_model_publication',publication),('native_CR2_link_200x8',native_ring_200),('native_preparation_200x8',native_preparation_200),('production_FP8_local_scoring_200x8',local_indexer_200),('actual_local_output_and_padding',local_output)]:case(name,fn)
repo=Path(__file__).resolve().parents[2]
functions=[native_model.DeepseekV4DecoderLayer.__init__,native_model.DeepseekV4Model.__init__,vl_model.DeepseekV41ForCausalLM.__init__]
report=dict(cases=RESULTS,passed=sum(r['status']=='passed' for r in RESULTS),total=len(RESULTS),cuda_initialized=torch.cuda.is_initialized(),native_model_source=native_model.__file__,native_rocm_source=rocm.__file__,native_constructor_hashes={f'{f.__module__}.{f.__qualname__}':hashlib.sha256(inspect.getsource(f).encode()).hexdigest() for f in functions},vllm_metadata=importlib.metadata.version('vllm'),source_hashes={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (repo/'adapter_extension/src/hcu_vllm_das').glob('v41_pcp*.py')},boundaries='CPU source replay; transports, projection/cache/GPU kernels doubled; no GPU/full model/Graph accuracy')
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
sys.exit(0 if report['passed']==report['total'] and not report['cuda_initialized'] else 1)
