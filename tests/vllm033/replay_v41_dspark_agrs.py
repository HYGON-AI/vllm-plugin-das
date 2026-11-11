#!/usr/bin/env python3
"""Fixed before implementation. Actual native AG-RS/PF and CT modular kernel;
borrowed groups/quant/provider/permutation/constructor-device boundaries doubled.
"""
from __future__ import annotations
from contextlib import ExitStack
import copy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
import replay_ct_w4a8_execution as base
from vllm.config import get_current_vllm_config, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.distributed.device_communicators import all2all as native_a2a
from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep as naive
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from hcu_vllm_das.models.dsv41_dspark import HcuDSparkDeepseekV41ForCausalLM,NativeDSpark
from hcu_vllm_das.quantization.ct_w4a8_method import HcuCTW4A8MoEMethod
from hcu_adapter.extensions import ExtensionContext
try:
    prod=importlib.import_module('hcu_vllm_das.dspark_agrs')
except ModuleNotFoundError as e:
    if e.name!='hcu_vllm_das.dspark_agrs':raise
    prod=None
RESULTS=[]
def require():assert prod is not None,'owned DSpark AG-RS module absent'
def case(name,fn):
    try:fn();RESULTS.append(dict(case=name,status='passed'))
    except Exception as e:
        import traceback;traceback.print_exc()
        RESULTS.append(dict(case=name,status='failed',error=f'{type(e).__name__}: {e}'))
def reject(fn):
    try:fn()
    except (RuntimeError,ValueError,TypeError,AssertionError):return
    raise AssertionError('invalid DSpark contract accepted')
def config(dp=8,n=5,requests=4,budget=32,anchor=True):
    cfg,moe=base.context(dp=dp)
    cfg.parallel_config.enable_expert_parallel=True;cfg.parallel_config.enable_dbo=False
    cfg.parallel_config.all2all_backend='deepep_low_latency'
    cfg.model_config.hf_config=NS(model_type='deepseek_v41')
    cfg.use_v2_model_runner=True;cfg.kernel_config=NS(moe_backend='deep_gemm',monokernel=False)
    cfg.scheduler_config=NS(max_num_batched_tokens=budget,max_num_seqs=requests)
    cfg.speculative_config=NS(method='dspark',num_speculative_tokens=n,
        draft_model_config=NS(enforce_eager=True,hf_config=NS(model_type='deepseek_v41',sample_from_anchor=anchor)))
    cfg.quant_config=object.__new__(base.config_module.HcuDSV41CTW4A8Config)
    cfg.compilation_config.cudagraph_capture_sizes=[]
    moe.max_num_tokens=budget
    return cfg,moe
class CPUGroup:
    def __init__(self,rank,width):self._rank=rank;self._width=width
    def rank(self):return self._rank
    def size(self):return self._width
class Group:
    def __init__(self,rank,width):
        self.rank_in_group=rank;self.world_size=width;self.ranks=list(range(width));self.cpu_group=CPUGroup(rank,width)
        self.device_communicator=NS(all2all_manager=object(),get_symmetric_memory_buffer=lambda *a:None)
        self.payload=None;self.reduced=None;self.calls=[]
    def all_gatherv(self,tensors,dim,sizes):
        self.calls.append('gather');assert sizes==[x.shape[0] for x in self.payload[0]]
        for i,x in enumerate(tensors):assert torch.equal(x,self.payload[i][self.rank_in_group])
        return [torch.cat(parts,dim=dim) for parts in self.payload]
    def reduce_scatterv_into_output(self,x,out,dim,sizes):
        self.calls.append('scatter');assert sum(sizes)==x.shape[0]
        first=sum(sizes[:self.rank_in_group]);expected=self.reduced if self.reduced is not None else x
        out.copy_(expected[first:first+sizes[self.rank_in_group]]);return out

def groups(stack,rank=0,width=8):
    require();dp=Group(rank,width);ep=Group(rank,width);tp=Group(0,1);pcp=Group(0,1)
    stack.enter_context(patch.object(parallel_state,'get_dp_group',return_value=dp))
    stack.enter_context(patch.object(parallel_state,'get_tp_group',return_value=tp))
    stack.enter_context(patch.object(parallel_state,'in_the_same_node_as',return_value=[True]*width))
    stack.enter_context(patch.object(prod,'get_dp_group',return_value=dp))
    stack.enter_context(patch.object(prod,'get_ep_group',return_value=ep))
    stack.enter_context(patch.object(prod,'get_tp_group',return_value=tp))
    stack.enter_context(patch.object(prod,'get_pcp_group',return_value=pcp))
    return dp,ep,tp

def capacities_and_scope():
    require()
    with ExitStack() as s:
        s.enter_context(patch.object(base.config_module,'_is_hcu',return_value=True))
        for seed in range(200):
            g=torch.Generator().manual_seed(seed)
            n=int(torch.randint(1,8,(1,),generator=g));r=int(torch.randint(1,20,(1,),generator=g));budget=r*(n+1)+seed%4
            cfg,moe=config(dp=(2,4,8,16)[seed%4],n=n,requests=r,budget=budget,anchor=bool(seed%2))
            local=prod.draft_config(cfg)
            assert local is not cfg and local.parallel_config is not cfg.parallel_config
            assert local.parallel_config.all2all_backend=='allgather_reducescatter'
            assert cfg.parallel_config.all2all_backend=='deepep_low_latency'
            assert local.scheduler_config is cfg.scheduler_config and local.compilation_config is cfg.compilation_config
            assert prod.is_owned_draft(local) and not prod.is_owned_draft(cfg)
            base.config_module.validate_topology(local,NS(**{**vars(moe),'moe_parallel_config':NS(**{**vars(moe.moe_parallel_config),'all2all_backend':'allgather_reducescatter'})}))
            cfg.scheduler_config.max_num_batched_tokens=r*(n+1)-1;reject(lambda:prod.draft_config(cfg))
        cfg,_=config();bad=copy.copy(cfg);bad.model_config=NS(**{**vars(cfg.model_config),'enforce_eager':False});reject(lambda:prod.draft_config(bad))
        for key in ('tensor_parallel_size','pipeline_parallel_size','decode_context_parallel_size','prefill_context_parallel_size'):
            bad=copy.copy(cfg);bad.parallel_config=copy.copy(cfg.parallel_config);setattr(bad.parallel_config,key,2);reject(lambda:prod.draft_config(bad))
        seen=[]
        def ctor(self,*,vllm_config,prefix=''):
            torch.nn.Module.__init__(self);seen.append(vllm_config)
            assert get_current_vllm_config() is vllm_config
            self.quant_config=vllm_config.quant_config
        with patch.object(NativeDSpark,'__init__',ctor),set_current_vllm_config(cfg):
            obj=HcuDSparkDeepseekV41ForCausalLM(vllm_config=cfg)
            assert seen[-1] is not cfg and prod.is_owned_draft(seen[-1])
            assert get_current_vllm_config() is cfg
            failure=RuntimeError('native ctor failed')
            with patch.object(NativeDSpark,'__init__',side_effect=failure):
                reject(lambda:HcuDSparkDeepseekV41ForCausalLM(vllm_config=cfg))
            assert get_current_vllm_config() is cfg
        cfg.parallel_config.data_parallel_size=1
        with patch.object(NativeDSpark,'__init__',ctor),set_current_vllm_config(cfg):
            HcuDSparkDeepseekV41ForCausalLM(vllm_config=cfg);assert seen[-1] is cfg

def transport_200():
    require()
    descriptor=FusedMoEQuantConfig.make(torch.int8,per_act_token_quant=True,weight_dtype='int4')
    for seed in range(200):
        width=(2,4,8,16)[seed%4];gen=torch.Generator().manual_seed(seed)
        counts=torch.randint(0,5,(width,),generator=gen).tolist()
        if seed==0:counts=[0]*width
        xs=[torch.randint(-5,6,(m,8),generator=gen).to(torch.bfloat16)/8 for m in counts]
        weights=[torch.rand((m,2),generator=gen) for m in counts];ids=[torch.randint(0,width*2,(m,2),generator=gen) for m in counts]
        gathered=torch.cat(xs);expectedq,expecteds=(base.quantize(gathered) if gathered.shape[0] else (gathered.to(torch.int8),torch.empty(0,1)))
        total=sum(counts)
        # Sum independently computed expert-rank contributions before RS.
        reduced=sum((torch.arange(total*8).reshape(total,8).float()+rank) for rank in range(width)).to(torch.bfloat16)
        for rank in range(width):
            with ExitStack() as stack:
                dp,ep,tp=groups(stack,rank,width);target_manager=ep.device_communicator.all2all_manager
                cfg,moe=config(width);local=prod.draft_config(cfg)
                manager=None
                with set_current_vllm_config(local):manager=prod.HcuDSparkAgRsManager(moe.max_num_tokens)
                sizes=NS(get_chunk_sizes_across_dp_rank=lambda:counts)
                stack.enter_context(patch.object(native_a2a,'get_forward_context',return_value=NS(dp_metadata=sizes)))
                stack.enter_context(patch.object(naive,'dbo_enabled',return_value=False))
                stack.enter_context(patch.object(prod,'moe_kernel_quantize_input',side_effect=lambda x,*a,**k:base.quantize(x)))
                dp.payload=[xs,weights,ids];dp.reduced=reduced
                prepare=prod.HcuCTAgRsPrepareAndFinalize(manager)
                actual=prepare.prepare(xs[rank],weights[rank],ids[rank],width*2,None,False,descriptor)
                assert torch.equal(actual[0],expectedq) and torch.equal(actual[1],expecteds)
                assert torch.equal(actual[3],torch.cat(ids)) and torch.equal(actual[4],torch.cat(weights))
                out=torch.empty(counts[rank],8,dtype=torch.bfloat16)
                prepare.finalize(out,reduced/width,actual[4],actual[3],False,base.runtime.TopKWeightAndReduceNoOP())
                begin=sum(counts[:rank]);assert torch.equal(out,reduced[begin:begin+counts[rank]])
                assert dp.calls==['gather','scatter'];assert ep.device_communicator.all2all_manager is target_manager
                assert naive.get_ep_group is not prepare._native_prepare.__func__.__globals__['get_ep_group']
                ep.ranks=list(reversed(ep.ranks));reject(manager.check_health)

def modular_pipeline():
    require();cfg,moe=config(dp=2,n=1,requests=2,budget=4)
    local=prod.draft_config(cfg);moe.moe_parallel_config.all2all_backend='allgather_reducescatter'
    moe.use_deepep_ht_kernels=moe.use_deepep_ll_kernels=False
    ops=base.Ops()
    with ExitStack() as stack:
        dp,ep,tp=groups(stack,width=2)
        stack.enter_context(patch.object(base.config_module,'get_current_vllm_config',return_value=local))
        stack.enter_context(patch.object(base.runtime,'get_ops',return_value=ops))
        stack.enter_context(patch.object(base.runtime,'deepgemm_moe_permute',side_effect=base.permutation))
        stack.enter_context(patch.object(base.runtime,'deepgemm_unpermute_and_reduce',side_effect=base.unpermute))
        stack.enter_context(patch.object(prod,'moe_kernel_quantize_input',side_effect=lambda x,*a,**k:base.quantize(x)))
        stack.enter_context(patch.object(naive,'dbo_enabled',return_value=False))
        with set_current_vllm_config(local):
            qc=base.config_module.HcuDSV41CTW4A8Config.from_config(base.SCHEMA)
            layer,_,logical=base.weight_fixture();layer.quant_config=qc;layer.moe_config=moe
            layer.ckpt_gate_proj_name='w1';layer.ckpt_up_proj_name='w3';layer.ckpt_down_proj_name='w2'
            layer.activation=base.MoEActivation.SILU;layer.global_num_experts=moe.num_experts;layer.apply_router_weight_on_input=False
            layer.register_buffer('_expert_map',torch.tensor([0,1,-1,-1],dtype=torch.int32),persistent=False)
            method=qc.get_quant_method(layer,'model.layers.40.ffn.experts');layer.quant_method=method
            method.create_weights(layer,base.E,base.H,base.I,torch.bfloat16,weight_loader=layer.weight_loader)
            draft=object.__new__(HcuDSparkDeepseekV41ForCausalLM);torch.nn.Module.__init__(draft)
            draft.config=NS(dspark_n_routed_experts=moe.num_experts,n_routed_experts=moe.num_experts,expert_dtype='fp8')
            draft.quant_config=qc;draft.pad_shared_expert=False;draft.linear_scale_name='weight_scale'
            draft.model=torch.nn.Module();draft.model.confidence_head=None
            block=torch.nn.Module();block.ffn=torch.nn.Module();block.ffn.experts=torch.nn.Module();block.ffn.experts.routed_experts=layer
            draft.model.layers=torch.nn.ModuleList([block]);checkpoint=[]
            for e in range(base.E):
                for shard in ('w1','w3','w2'):
                    q,s=logical[e,shard];s=torch.full_like(s,1/32);logical[e,shard]=(q,s)
                    for suffix,data in (('weight_packed',base.ct_pack(q)),('weight_scale',s),('weight_shape',torch.tensor(q.shape,dtype=torch.int64))):
                        checkpoint.append((f'mtp.0.ffn.experts.{e}.{shard}.{suffix}',data))
            loaded=draft.load_weights(iter(checkpoint))
            assert len(loaded)==6 and all(name.startswith('model.layers.0.ffn.experts.routed_experts.') for name in loaded)
            method.process_weights_after_loading(layer)
        assert type(method.moe_kernel.prepare_finalize) is prod.HcuCTAgRsPrepareAndFinalize
        assert isinstance(method.moe_kernel.fused_experts,base.runtime.HcuCTContiguousExperts)
        x=torch.arange(3*base.H).reshape(3,base.H).remainder(13).to(torch.bfloat16)/32
        ids=torch.tensor([[0,1]]*3,dtype=torch.int64);weights=torch.tensor([[.25,.75]]*3)
        dp.payload=[[x,x[:0]],[weights,weights[:0]],[ids,ids[:0]]]
        sizes=NS(get_chunk_sizes_across_dp_rank=lambda:[3,0])
        stack.enter_context(patch.object(native_a2a,'get_forward_context',return_value=NS(dp_metadata=sizes)))
        actual=method.apply(layer,x,weights,ids,None,None)
        qx,sx=base.quantize(x);reference=torch.zeros(3,base.H)
        for e in range(base.E):
            q1=torch.cat((logical[e,'w1'][0],logical[e,'w3'][0]),0)
            gateup=((qx.float()@q1.float().t())*sx/32).to(torch.bfloat16)
            mid=torch.nn.functional.silu(gateup[:,:base.I].float())*gateup[:,base.I:].float();qm,sm=base.quantize(mid)
            y=((qm.float()@logical[e,'w2'][0].float().t())*sm/32).to(torch.bfloat16);reference+=y.float()*weights[:,e:e+1]
        assert torch.equal(actual,reference.to(torch.bfloat16))
        assert dp.calls==['gather','scatter']
        before=method.moe_kernel;method.process_weights_after_loading(layer);assert method.moe_kernel is before

def main():
    torch.set_num_threads(1)
    base.config_module._is_hcu=lambda:True;base.fp8_config._is_hcu=lambda:True
    base.activate_module.activate_extension(ExtensionContext('0.1.0','0.33.0'))
    case('owned_draft_constructor_scope_capacity_200',capacities_and_scope)
    case('native_variable_AG_RS_prepare_finalize_200',transport_200)
    case('packed_CT_modular_AG_RS_pipeline',modular_pipeline)
    root=Path(__file__).resolve().parents[2]
    report=dict(passed=sum(x['status']=='passed' for x in RESULTS),total=len(RESULTS),cases=RESULTS,cuda_initialized=torch.cuda.is_initialized(),native_agrs_source=native_a2a.__file__,source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'adapter_extension/src/hcu_vllm_das').rglob('*agrs*.py')},boundary='CPU native manager/PF, CT loader and modular orchestration; transport/provider/quant/permutation and native constructor device boundary doubled. No whole model, GPU or Graph validation.')
    Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return 0 if report['passed']==report['total'] and not report['cuda_initialized'] else 1
if __name__=='__main__':sys.exit(main())
