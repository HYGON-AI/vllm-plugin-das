#!/usr/bin/env python3
"""Pre-implementation CPU source integration for the eager V41 FP8 indexer.
Actual OOT construction/metadata; cache/logits math real CPU, native gather and
candidate GPU kernels doubled. No GPU/Graph/model execution or installation.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
import torch
from hcu_adapter.extensions import ExtensionContext, ExtensionCommitError
from vllm.model_executor import custom_op
from vllm.model_executor.layers import sparse_attn_indexer as native
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata, DeepseekV32IndexerPrefillMetadata, DeepSeekV32IndexerDecodeMetadata
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as rocm

try:
    activation=importlib.import_module('hcu_vllm_das.v41_indexer_activation')
    candidate=importlib.import_module('hcu_vllm_das.v41_indexer')
except ModuleNotFoundError as error:
    if error.name not in ('hcu_vllm_das.v41_indexer_activation','hcu_vllm_das.v41_indexer'):raise
    activation=candidate=None
RESULTS=[]
CONTEXT=ExtensionContext('0.1.0','0.33.0')
B,D,HEADS=16,128,2


def require():assert candidate is not None,'V41 eager indexer production module missing'

def case(name,fn):
    try:fn();RESULTS.append({'case':name,'status':'passed'})
    except Exception as e:RESULTS.append({'case':name,'status':'failed','error':f'{type(e).__name__}: {e}'})

def rejects(fn,kind=(RuntimeError,ValueError,AssertionError)):
    try:fn()
    except kind:return
    raise AssertionError('unsupported indexer contract was accepted')


def pack(seed):
    g=torch.Generator().manual_seed(seed)
    logical=torch.randint(-4,5,(5,B,D),generator=g).to(rocm.FP8_DTYPE)
    scales=torch.randint(1,8,(5,B),generator=g).float()/4
    raw=torch.zeros((5,B*(D+4)),dtype=torch.uint8)
    # Independent explicit writer offset formula, not the candidate reshape.
    cols=torch.arange(D)
    for token in range(B):
        offsets=(token//16)*D*16+(cols//16)*256+(token%16)*16+cols%16
        raw[:,offsets]=logical[:,token].view(torch.uint8)
    raw[:,B*D:]=scales.contiguous().view(torch.uint8).reshape(5,B*4)
    return raw.view(5,B,D+4),logical,scales


def reference_decode(q,logical,scale,weights,lens,tables,maxlen):
    reqs,nextn,heads,dim=q.shape
    out=torch.full((reqs*nextn,maxlen),-float('inf'))
    limits=lens.reshape(reqs,-1)
    if lens.ndim==1:
        limits=(limits-nextn+torch.arange(nextn)[None,:]+1).clamp_min(0)
    for req in range(reqs):
        for j in range(nextn):
            end=int(limits[req,j]);row=req*nextn+j
            for pos in range(end):
                page=int(tables[req,pos//B]);slot=pos%B
                dots=(q[req,j].float()*logical[page,slot].float()).sum(-1)
                out[row,pos]=(dots.relu()*weights[row]).sum()*scale[page,slot]
    return out,limits


def dense_topk(logits,starts,ends,k):
    out=torch.full((len(ends),k),-1,dtype=torch.int32)
    for row in range(len(ends)):
        valid=[i for i in range(int(starts[row]),int(ends[row])) if logits[row,i]!=-float('inf')]
        if int(ends[row]-starts[row])<=k:
            selected=valid
        else:
            selected=sorted(valid,key=lambda i:float(logits[row,i]),reverse=True)[:k]
        if selected:out[row,:len(selected)]=torch.tensor([v-int(starts[row]) for v in selected],dtype=torch.int32)
    return out


def mask(logits,starts,ends,blocks,block_size):
    for row in range(len(ends)):
        allowed=blocks[row].tolist()
        for col in range(logits.shape[1]):
            if col<int(starts[row]) or col>=int(ends[row]) or (col-int(starts[row]))//block_size not in allowed:
                logits[row,col]=-float('inf')


def select(logits,starts,ends,count,block_size,out):
    out.fill_(-1)
    for row in range(len(ends)):
        n=(int(ends[row]-starts[row])+block_size-1)//block_size
        scores=[]
        for b in range(n):
            v=float('inf') if b==n-1 else float(logits[row,int(starts[row])+b*block_size:min(int(ends[row]),int(starts[row])+(b+1)*block_size)].max())
            scores.append((v,b))
        chosen=sorted(scores,reverse=True)[:count]
        if chosen:out[row,:len(chosen)]=torch.tensor([b for _,b in chosen],dtype=torch.int32)


def setup(stack,enabled=True,model_type='deepseek_v41'):
    require()
    cfg=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type),enforce_eager=True),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1,decode_context_parallel_size=1,pcp_shard_decode_requests=False,cp_kv_cache_interleave_size=1),
        kernel_config=SimpleNamespace(sparse_indexer_topk_backend='auto',enable_jit_warmup=False,monokernel=False))
    stack.enter_context(patch.dict(os.environ,{'VLLM_HCU_DSV41_FP8_INDEXER_REFERENCE':'1' if enabled else '0'}))
    stack.enter_context(patch.object(activation,'_state',None));stack.enter_context(patch.object(activation,'_failure',None))
    stack.enter_context(patch.object(custom_op,'op_registry_oot',dict(custom_op.op_registry_oot)))
    stack.enter_context(patch.object(activation,'_is_hcu',return_value=True))
    stack.enter_context(patch.object(candidate,'current_config',return_value=cfg))
    stack.enter_context(patch.object(native,'get_current_vllm_config',return_value=cfg))
    stack.enter_context(patch.object(custom_op,'get_cached_compilation_config',return_value=SimpleNamespace(custom_ops=['all'],enabled_custom_ops=set(),disabled_custom_ops=set())))
    stack.enter_context(patch.object(custom_op,'current_platform',SimpleNamespace(is_rocm=lambda:True)))
    stack.enter_context(patch.object(candidate,'select_candidate_blocks',side_effect=select))
    stack.enter_context(patch.object(candidate,'apply_candidate_mask',side_effect=mask))
    activation.activate_extension(CONTEXT)
    return cfg


def obj(cache,out,blocks=None,write=False,**overrides):
    opts=dict(skip_k_cache_insert=True,use_fp4_cache=False,compress_ratio=1,candidate_blocks=blocks,candidate_block_size=4 if blocks is not None else 0,candidate_write=write)
    opts.update(overrides)
    return native.SparseAttnIndexer(SimpleNamespace(prefix='cache',kv_cache=cache),128,'ue8m0',out.shape[1],D,32,64,out,**opts)


def decode_200():
    require()
    with ExitStack() as stack:
        setup(stack)
        for seed in range(200):
            cache,logical,scales=pack(seed);g=torch.Generator().manual_seed(seed+999)
            n=seed%3+1;q=torch.randint(-3,4,(2,n,HEADS,D),generator=g).to(rocm.FP8_DTYPE)
            weights=torch.randint(1,8,(2*n,HEADS),generator=g).float()/8
            lens=torch.tensor([seed%24+1,seed%17+3],dtype=torch.int32)
            if seed%2:lens=(lens[:,None]-n+torch.arange(n)+1).clamp_min(0)
            tables=torch.tensor([[3,1],[4,2]],dtype=torch.int32)
            actual=candidate.decode_logits(q,cache,weights,lens,tables,32)
            expected,_=reference_decode(q,logical,scales,weights,lens,tables,32)
            assert torch.equal(torch.isneginf(actual),torch.isneginf(expected)),seed
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        r,_,_=pack(6)
        bad=r[...,:-1];rejects(lambda:candidate.decode_logits(q,bad,weights,lens,tables,32))
    # One correct packed token page uses NORMAL storage, but must retain tail scales.
    raw=torch.zeros((1,1,D+4),dtype=torch.uint8)
    raw[0,0,:D]=torch.ones(D,dtype=rocm.FP8_DTYPE).view(torch.uint8)
    raw[0,0,D:]=torch.tensor([.5]).view(torch.uint8)
    got=candidate.decode_logits(torch.ones(1,1,HEADS,D,dtype=rocm.FP8_DTYPE),raw,torch.ones(1,HEADS),torch.tensor([1]),torch.tensor([[0]]),1)
    assert got.item()==D


def runtime_decode():
    require()
    with ExitStack() as stack:
        setup(stack)
        cache,logical,scales=pack(32);snapshot=cache.clone()
        q=torch.ones((3,HEADS,D),dtype=rocm.FP8_DTYPE);weights=torch.ones(3,HEADS)
        out=torch.full((6,8),999,dtype=torch.int32);blocks=torch.full((6,2),-1,dtype=torch.int32)
        layer=obj(cache,out,blocks,True)
        assert type(layer).__module__=='hcu_vllm_das.v41_indexer'
        decode=DeepSeekV32IndexerDecodeMetadata(torch.tensor([[3,1],[4,2]]),torch.tensor([6,9]),torch.tensor([1,2]),True,torch.empty(0))
        meta=DeepseekV32IndexerMetadata(torch.tensor([6,9]),9,torch.zeros(3,dtype=torch.int64),2,3,0,0,decode=decode)
        stack.enter_context(patch.object(candidate,'get_forward_context',return_value=SimpleNamespace(attn_metadata={'cache':meta})))
        result=layer(torch.zeros(3,4),q,None,weights)
        assert result is out and torch.equal(cache,snapshot)
        assert not bool((out[:3]==999).any())
        assert torch.equal(out[0],torch.tensor([0,1,2,3,4,5,-1,-1],dtype=torch.int32))
        assert not bool((out[1]>=8).any()) and not bool((out[2]>=9).any())
        assert int(blocks[2,0])==1 or int(blocks[2,1])==1
        layer.candidate_write=False;blocks[:4]=torch.tensor([[0,-1],[0,-1],[0,-1],[0,-1]])
        layer(torch.zeros(3,4),q,None,weights)
        assert not bool((out[:3]>=4).any())
        rejects(lambda:layer(torch.zeros(3,4),q,torch.ones(3,D),weights))


def prefill():
    require()
    with ExitStack() as stack:
        setup(stack)
        cache,logical,scales=pack(10);out=torch.full((4,8),999,dtype=torch.int32)
        layer=obj(cache,out)
        q=torch.ones((2,HEADS,D),dtype=rocm.FP8_DTYPE);weights=torch.ones(2,HEADS)
        chunk=SimpleNamespace(token_start=0,token_end=2,total_seq_lens=8,
            block_table=torch.tensor([[3]]),cu_seq_lens=torch.tensor([0,8]),token_to_seq=torch.zeros(8,dtype=torch.int32),
            cu_seqlen_ks=torch.tensor([0,0]),cu_seqlen_ke=torch.tensor([3,8]))
        meta=DeepseekV32IndexerMetadata(torch.tensor([8]),8,torch.zeros(2,dtype=torch.int64),0,0,1,2,prefill=DeepseekV32IndexerPrefillMetadata([chunk]))
        stack.enter_context(patch.object(candidate,'get_forward_context',return_value=SimpleNamespace(attn_metadata={'cache':meta})))
        def gather(raw,k,s,table,cu,token_to_seq=None,cache_layout=None):
            k.copy_(logical[3,:8]);s.copy_(scales[3,:8].contiguous().view(torch.uint8).reshape(8,4))
        stack.enter_context(patch.object(candidate,'cp_gather_indexer_k_quant_cache_triton',side_effect=gather))
        layer(torch.zeros(2,4),q,None,weights)
        assert torch.equal(out[0],torch.tensor([0,1,2,-1,-1,-1,-1,-1],dtype=torch.int32))
        assert torch.equal(out[1],torch.arange(8,dtype=torch.int32))
        assert torch.equal(out[2:],torch.full((2,8),999,dtype=torch.int32))


def topk_and_native():
    require()
    with ExitStack() as stack:
        cfg=setup(stack)
        values=torch.tensor([[-float('inf'),1.,3.,-float('inf')]])
        got=candidate.local_topk(values,torch.tensor([1]),torch.tensor([4]),5)
        assert torch.equal(got,torch.tensor([[0,1,-1,-1,-1]],dtype=torch.int32))
        cache,_,_=pack(1);out=torch.zeros(2,8,dtype=torch.int32)
        for field,val in [('prefill_context_parallel_size',2),('decode_context_parallel_size',2)]:
            with patch.object(cfg.parallel_config,field,val):rejects(lambda:obj(cache,out))
        with patch.object(cfg.model_config,'enforce_eager',False):rejects(lambda:obj(cache,out))
        rejects(lambda:obj(cache,out,use_fp4_cache=True))
        rejects(lambda:obj(cache,out,skip_k_cache_insert=False))
    with ExitStack() as stack:
        setup(stack,False)
        layer=obj(cache,out)
        with patch.object(native.SparseAttnIndexer,'forward_hip',return_value='native'):
            assert layer.forward_hip(torch.zeros(1,4),torch.ones(1,HEADS,D),None,torch.ones(1,HEADS))=='native'
    with ExitStack() as stack:
        setup(stack,True,'llama')
        layer=obj(cache,out)
        with patch.object(native.SparseAttnIndexer,'forward_hip',return_value='native'):
            assert layer.forward_hip(torch.zeros(1,4),torch.ones(1,HEADS,D),None,torch.ones(1,HEADS))=='native'


def health():
    require()
    with ExitStack() as stack:
        setup(stack);before=dict(custom_op.op_registry_oot)
        activation.activate_extension(CONTEXT);activation.check_health();assert before==custom_op.op_registry_oot
        with patch.dict(os.environ,{'VLLM_HCU_DSV41_FP8_INDEXER_REFERENCE':'0'}):rejects(activation.check_health,ExtensionCommitError)
    with ExitStack() as stack:
        setup(stack)
        custom_op.op_registry_oot['SparseAttnIndexer']=object
        rejects(activation.check_health,ExtensionCommitError)
        rejects(lambda:activation.activate_extension(CONTEXT),ExtensionCommitError)


def graph_registration_timing():
    require()
    with ExitStack() as stack:
        cfg=setup(stack)
        from vllm.config import CompilationMode, CUDAGraphMode
        from hcu_vllm_das.v41_indexer_graph import OP_NAME
        # Activation happens before the resolved VllmConfig exists.  The
        # construction path must perform the split registration instead.
        assert not hasattr(cfg, 'compilation_config')
        cfg.compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            use_inductor_graph_partition=False,
            splitting_ops=[],
        )
        cfg.model_config.enforce_eager=False
        cache,_,_=pack(1);out=torch.zeros(2,8,dtype=torch.int32)
        obj(cache,out)
        assert cfg.compilation_config.splitting_ops==[OP_NAME]
        obj(cache,out)
        assert cfg.compilation_config.splitting_ops==[OP_NAME]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    torch.set_num_threads(1)
    original=native.SparseAttnIndexer.forward_hip
    case('200_seed_SHUFFLE_scale_tail_nextn_and_causal_logits',decode_200)
    case('actual_OOT_decode_ragged_candidates_and_cache_immutability',runtime_decode)
    case('native_prefill_chunk_contract_and_short_row_order',prefill)
    case('masked_topk_construction_admission_and_other_paths_native',topk_and_native)
    case('registration_idempotence_policy_and_owner_health',health)
    case('splitting_registration_at_post_activation_construction',graph_registration_timing)
    assert original is native.SparseAttnIndexer.forward_hip and not torch.cuda.is_initialized()
    root=Path(__file__).resolve().parents[2];sha=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
    sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'adapter_extension/src/hcu_vllm_das').glob('v41_indexer*.py')}
    report={'level':'CPU public OOT and packed-cache math; native GPU gather/candidate kernels doubled',
        'plugin':sha(root),'core':sha(root.parent/'vllm'),'adapter':sha(root.parent/'adapter-das'),
        'installed_vllm_metadata':metadata.version('vllm'),'cuda_initialized':False,'source_sha256':sources,
        'native_indexer_source':inspect.getsourcefile(native.SparseAttnIndexer),'native_layout_source':rocm.__file__,
        'cases':RESULTS,'passed':sum(r['status']=='passed' for r in RESULTS),'total':len(RESULTS)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['passed']==report['total'] else 1

if __name__=='__main__':raise SystemExit(main())
