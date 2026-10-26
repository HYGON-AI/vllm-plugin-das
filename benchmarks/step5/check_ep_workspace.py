"""Real Step5 EP kernels and WorkspaceManager: bounded scratch and replay checks.

Run on one idle GPU in the matching DTK image. This checks operator equivalence,
not end-to-end model quality or sixteen-GPU serving throughput.
"""
import ast
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.plugins import load_general_plugins
from vllm_hcu.patch.worker import apply_worker_patches

torch.set_num_threads(1)
torch.cuda.set_device(0)
load_general_plugins(); apply_worker_patches()
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep
from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend, backend_to_kernel_cls
from vllm.v1.worker.workspace import current_workspace_manager, init_workspace_manager
from vllm_hcu.model_executor.layers.fused_moe import ep_fused_moe as ep
from vllm_hcu.model_executor.layers.fused_moe import modular_kernel as mk
from vllm_hcu.model_executor.layers.fused_moe.experts.deep_gemm_moe import DeepGemmExperts

assert torch.cuda.mem_get_info()[0] > 40 * 1024**3
torch.manual_seed(61220)
E,K,I,TOPK=22,4096,1536,8
w1=(torch.randn(E,2*I,K,device='cuda',dtype=torch.bfloat16)*8).to(ep.FP8)
w2=(torch.randn(E,K,I,device='cuda',dtype=torch.bfloat16)*8).to(ep.FP8)
s1=torch.rand(E,2*I,1,device='cuda')*.0002+.0005
s2=torch.rand(E,K,1,device='cuda')*.0002+.0005
mapping=torch.full((352,),-1,device='cuda',dtype=torch.int32)
mapping[:E]=torch.arange(E,device='cuda',dtype=torch.int32)
qconfig=SimpleNamespace(block_shape=None,use_fp8_w8a8=True,use_int8_w8a8=False,
    per_out_ch_quant=True,per_act_token_quant=True,w1_scale=s1,w2_scale=s2,
    a2_scale=None,gemm1_clamp_limit=7.0)
inner=object.__new__(DeepGemmExperts)
inner.quant_config=qconfig;inner._hcu_logical_n=2*I;inner._hcu_logical_k=K
inner.gemm1_clamp_limit=7.;inner._lora_context=None
dispatcher_cls=backend_to_kernel_cls(Fp8MoeBackend.DEEPGEMM)[0]
expert=object.__new__(dispatcher_cls)
expert.experts=inner;expert.fallback_experts=SimpleNamespace()
expert.quant_config=qconfig;expert._lora_context=None
runner=object.__new__(mk.FusedMoEKernelModularImpl)
runner.fused_experts=expert;runner.N=2*I;runner.K=K
runner.prepare_finalize=naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPModular(num_dispatchers=16)
ep.STEP5_EXPERT_GRID=False;ep.shared_add_provider=None;ep.DIRECT_MIN=0
reports=[]
cases=[(256,64,'all-local'),(131072,32768,'global'),(131073,32768,'all-local'),
       (262144,32768,'global'),(262145,32768,'all-local')]
if os.environ.get('STEP5_WORKSPACE_LARGE_ONLY')=='1':cases=cases[-2:]
for rows,chunk,distribution in cases:
    ep.CHUNK=chunk
    x=(torch.randn(rows,K,device='cuda')*8).to(ep.FP8)
    scale=torch.rand(rows,1,device='cuda')*.005+.01
    experts=E if distribution=='all-local' else 352
    ids=torch.rand(rows,experts,device='cuda').topk(TOPK,dim=-1).indices.to(torch.int32)
    weights=torch.rand(rows,TOPK,device='cuda');weights/=weights.sum(-1,keepdim=True)
    assert expert._select_experts_impl(x,w1,w2) is inner
    def reference():
        output=torch.empty(rows,K,device='cuda',dtype=torch.bfloat16)
        return ep.ep_fused_experts(x,w1,w2,weights,ids,mapping,s1,s2,
            a1_scale=scale,global_num_experts=352,output_dtype=torch.bfloat16,out=output)
    def native():
        return runner._fused_experts(torch.bfloat16,x,scale,w1,w2,weights,ids,
            MoEActivation.SWIGLUSTEP,352,E,mapping,False,None)
    init_workspace_manager(torch.device('cuda',0))
    os.environ['HCU_STEP5_BOUNDED_EP_WORKSPACE']='1'
    expected=reference();actual=native();torch.cuda.synchronize()
    if not torch.equal(actual,expected):
        again=reference();torch.cuda.synchronize()
        diagnostic=dict(rows=rows,chunk=chunk,distribution=distribution,
            different_elements=int((actual!=expected).sum()),
            reference_repeat_bitexact=torch.equal(expected,again),
            actual_nan=int(actual.isnan().sum()),reference_nan=int(expected.isnan().sum()),
            relative_l2=float((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12)),
            per_chunk_mismatches=[int((actual[s:s+chunk]!=expected[s:s+chunk]).sum()) for s in range(0,rows,chunk)])
        Path(os.environ['STEP5_PROOF_OUTPUT']).write_text(json.dumps(dict(passed=False,diagnostic=diagnostic),indent=2))
        print(json.dumps(diagnostic),flush=True)
        raise AssertionError((rows,'eager mismatch'))
    expected_shape=expert.workspace_shapes(chunk,2*I,K,TOPK,352,E,None,MoEActivation.SWIGLUSTEP)
    full_shapes=expert.workspace_shapes(rows,2*I,K,TOPK,352,E,None,MoEActivation.SWIGLUSTEP)
    actual_bytes=current_workspace_manager()._current_workspaces[0].numel()
    old_bytes=(max(math.prod(full_shapes[0]),math.prod(full_shapes[2]))+math.prod(full_shapes[1]))*2
    bounded_bytes=(math.prod(expected_shape[0])+math.prod(expected_shape[1])+rows*K)*2
    assert actual_bytes==bounded_bytes,(actual_bytes,bounded_bytes)
    assert actual_bytes<old_bytes
    current_workspace_manager().lock()
    a,b=torch.cuda.CUDAGraph(),torch.cuda.CUDAGraph()
    with torch.cuda.graph(a):ref_output=reference()
    with torch.cuda.graph(b):native_output=native()
    for value in [0,1,3]:
        x.fill_(value)
        ids.copy_(torch.rand(rows,experts,device='cuda').topk(TOPK,dim=-1).indices.to(torch.int32))
        weights.copy_(torch.rand_like(weights));weights/=weights.sum(-1,keepdim=True)
        ref_output.zero_();native_output.zero_()
        a.replay();b.replay();torch.cuda.synchronize()
        assert torch.equal(native_output,ref_output),(rows,value,'replay mismatch')
    report=dict(rows=rows,chunk=chunk,distribution=distribution,eager_bitexact=True,
        changed_input_graph_replays=3,graph_bitexact=True,old_workspace_bytes=old_bytes,
        bounded_workspace_bytes=actual_bytes)
    reports.append(report);print(json.dumps(report),flush=True)
    del a,b,ref_output,native_output,actual,expected,x,scale,ids,weights
    current_workspace_manager().unlock()

# Decode's one-chunk M256 path retains its original scratch/output layout.
ep.CHUNK=32768
init_workspace_manager(torch.device('cuda',0))
x=torch.zeros(256,K,device='cuda',dtype=ep.FP8);scale=torch.ones(256,1,device='cuda')*.01
ids=torch.randint(0,352,(256,TOPK),device='cuda',dtype=torch.int32)
weights=torch.full((256,TOPK),1/TOPK,device='cuda')
os.environ['HCU_STEP5_BOUNDED_EP_WORKSPACE']='0';baseline=native().clone()
bytes_before=current_workspace_manager()._current_workspaces[0].numel()
os.environ['HCU_STEP5_BOUNDED_EP_WORKSPACE']='1';candidate=native().clone()
torch.cuda.synchronize();assert torch.equal(candidate,baseline)
assert bytes_before==current_workspace_manager()._current_workspaces[0].numel()
result=dict(passed=True,cases=reports,expert_dispatcher=dispatcher_cls.__module__+'.'+dispatcher_cls.__name__,decode_one_chunk_bitexact=True,
    decode_workspace_unchanged=True,full_model_quality_validated=False,
    full_model_performance_validated=False)
Path(os.environ['STEP5_PROOF_OUTPUT']).write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
