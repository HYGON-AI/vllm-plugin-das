"""One idle HCU: validate native expert-grid selection and changed-input graphs.

Run with the matching CP312 extension and native source tree. This process
toggles its own module flag only; the model service is not modified. Synthetic
FP8 weights preserve the measured shapes/strides; no accuracy claim is made
for a complete language model from these operator checks.
"""
import copy,json,os
from pathlib import Path
import torch
import vllm
from vllm.platforms import current_platform
from vllm.plugins import load_general_plugins
from vllm_hcu.patch.worker import apply_worker_patches
load_general_plugins();apply_worker_patches()
from vllm_hcu.model_executor.layers.fused_moe import ep_fused_moe as ep
from vllm_hcu.model_executor.layers.fused_moe.step5_expert_grid import can_use_expert_grid

torch.set_num_threads(1);torch.cuda.set_device(0);torch.manual_seed(642064)
E,N,K=22,3072,4096
w1=(torch.randn(E,N,K,device='cuda',dtype=torch.bfloat16)*32).to(torch.float8_e4m3fn)
w2=(torch.randn(E,K,N//2,device='cuda',dtype=torch.bfloat16)*32).to(torch.float8_e4m3fn)
s1=torch.rand(E,N,1,device='cuda')*.0002+.0005
s2=torch.rand(E,K,1,device='cuda')*.0002+.0005
mapping=torch.full((352,),-1,device='cuda',dtype=torch.int32);mapping[:E]=torch.arange(E,device='cuda',dtype=torch.int32)


def error(a,b):
    d=a.float()-b.float();value=float(d.norm()/b.float().norm().clamp_min(1e-12))
    assert value<.01,(value,float(d.abs().max()))
    return dict(relative_l2=value,max_abs=float(d.abs().max()),exact=torch.equal(a,b))


rows=[]
for m in [128,256,272]:
    # A hot local expert plus empty and single-route local experts exercises
    # the bounded worker grid, as well as fallback shapes on both sides.
    ids=torch.stack([torch.randperm(352,device='cuda')[:8] for _ in range(m)]).to(torch.int32)
    ids[:,0]=0
    # Keep top-k unique even on rows which already contained expert zero.
    for slot in range(1,8):ids[:,slot]=torch.where(ids[:,slot]==0,torch.full_like(ids[:,slot],351),ids[:,slot])
    # Rebuild any rows with a duplicated replacement, outside measurement.
    for i in range(m):
        if ids[i].unique().numel()!=8:
            tail=torch.randperm(351,device='cuda')[:7]+1;ids[i,1:]=tail
    weights=torch.rand(m,8,device='cuda');weights/=weights.sum(1,keepdim=True)
    x=torch.randn(m,K,device='cuda',dtype=torch.bfloat16)*.1
    aq,asc=ep.per_token_quant_triton(x,quant_dtype=ep.FP8)
    assert can_use_expert_grid(aq,asc,w1,w2,s1,s2,ids,torch.bfloat16)==(m==256)
    if m==256:
        assert not can_use_expert_grid(aq,asc[:1],w1,w2,s1,s2,ids,torch.bfloat16)
        assert not can_use_expert_grid(aq,asc.cpu(),w1,w2,s1,s2,ids,torch.bfloat16)
        assert not can_use_expert_grid(aq,asc,w1,w2,s1.half(),s2,ids,torch.bfloat16)
        assert not can_use_expert_grid(aq,asc,w1,w2,None,s2,ids,torch.bfloat16)
    for input_kind in ['prequantized','bf16']:
        if m==256 and input_kind=='bf16':ids=ids.to(torch.int64)
        def function():
            return ep.ep_fused_experts(aq if input_kind=='prequantized' else x,w1,w2,weights,ids,mapping,s1,s2,
                a1_scale=asc if input_kind=='prequantized' else None,global_num_experts=352,output_dtype=torch.bfloat16)
        ep.STEP5_EXPERT_GRID=False;reference=function().clone()
        ep.STEP5_EXPERT_GRID=True;actual=function()
        row=dict(rows=m,input_kind=input_kind,eager=error(actual,reference),replays=[])
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=function()
        for amplitude in [0.,.2,1.]:
            x.copy_(torch.randn_like(x)*amplitude)
            q,s=ep.per_token_quant_triton(x,quant_dtype=ep.FP8);aq.copy_(q);asc.copy_(s)
            out.zero_();graph.replay();torch.cuda.synchronize()
            ep.STEP5_EXPERT_GRID=False;ref=function().clone();ep.STEP5_EXPERT_GRID=True
            row['replays'].append(error(out,ref))
        rows.append(row);del graph
result=dict(native_source=ep.__file__,gpu_numerics_passed=True,changed_input_zeroed_output_graph_replay=True,results=rows)
destination=os.environ.get('STEP5_EXPERT_GRID_CHECK_OUTPUT')
if destination:Path(destination).write_text(json.dumps(result,indent=2))
print(json.dumps(result))
