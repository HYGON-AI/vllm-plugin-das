"""DP16 collective screen at the real Prefill tensor sizes; not a model TPS score."""
import datetime
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
import vllm
from vllm.plugins import load_general_plugins
from vllm_hcu.patch.worker import apply_worker_patches

rank,local=int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
torch.set_num_threads(1);torch.cuda.set_device(local)
assert torch.cuda.mem_get_info()[0]>32*1024**3,'Requires an idle GPU, with model services stopped'
load_general_plugins();apply_worker_patches()
dist.init_process_group('gloo',timeout=datetime.timedelta(seconds=150))
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

comm=CudaCommunicator(dist.group.WORLD,device=torch.device('cuda',local),unique_name='dp:prefill-communication-check')
assert comm.pynccl_comm is not None and not comm.pynccl_comm.disabled
root=Path(os.environ['STEP5_PROOF_DIR'])
M,K,WORLD=8192,4096,16
sizes=[M]*WORLD
torch.manual_seed(20261021+rank)
rs=torch.empty((WORLD*M,K),dtype=torch.bfloat16,device='cuda')


def dispatch(inputs):
    return comm.all_gatherv(inputs,dim=0,sizes=sizes)


def combine():
    return comm.reduce_scatterv(rs,dim=0,sizes=sizes)


def check_dispatch(inputs,outputs,factor):
    assert len(inputs)==len(outputs)
    for original,output in zip(inputs,outputs):
        assert tuple(output.shape)==(WORLD*M,*original.shape[1:]) and output.dtype==original.dtype
        for peer in range(WORLD):
            part=output[peer*M:(peer+1)*M]
            expected=torch.tensor(factor*(peer+1),device='cuda',dtype=original.dtype)
            if original.element_size()==1:
                assert (part.view(torch.uint8)==expected.view(torch.uint8)).all().item()
            else:assert (part==expected).all().item()


# Validate the sum against FP32 and the standard BF16 rounding bound.
rs.normal_()
reference=comm.all_reduce(rs.float())
absolute_sum=comm.all_reduce(rs.float().abs())
actual=combine().float();target=reference[rank*M:(rank+1)*M]
delta=actual-target
relative_l2=float(delta.norm()/target.norm().clamp_min(1e-12))
unit=torch.finfo(torch.bfloat16).eps/2
gamma=(WORLD-1)*unit/(1-(WORLD-1)*unit)
bound=gamma*absolute_sum[rank*M:(rank+1)*M]+1e-5
assert relative_l2<.01 and (delta.abs()<=bound).all().item(),(rank,relative_l2)
numerical=dict(relative_l2=relative_l2,max_abs=float(delta.abs().max()),bf16_rounding_bound_passed=True)
del reference,absolute_sum,actual,target,delta,bound

results=[]
for family in ['fp8-and-routing','bf16-and-logits']:
    if family=='fp8-and-routing':
        shapes=[(M,K),(M,8),(M,8),(M,1)]
        dtypes=[torch.float8_e4m3fn,torch.float32,torch.int32,torch.float32]
    else:
        shapes=[(M,K),(M,352)];dtypes=[torch.bfloat16,torch.float32]
    inputs=[torch.empty(shape,device='cuda',dtype=dtype) for shape,dtype in zip(shapes,dtypes)]
    for factor in [1,2]:
        for tensor in inputs:tensor.fill_(factor*(rank+1))
        for peer in range(WORLD):rs[peer*M:(peer+1)*M].fill_(factor*(peer+1))
        gathered=dispatch(inputs);out=combine();torch.cuda.synchronize()
        check_dispatch(inputs,gathered,factor)
        assert (out==WORLD*factor*(rank+1)).all().item(),('Incorrect reduce-scatter destination or stale input',rank)
    # Random data for timing avoids a constant/zero-data bandwidth artifact.
    for tensor in inputs:
        if tensor.dtype==torch.int32:tensor.random_(0,352)
        elif tensor.element_size()==1:tensor.copy_(torch.randn(tensor.shape,device='cuda',dtype=torch.bfloat16).to(tensor.dtype))
        else:tensor.normal_()
    rs.normal_()
    for _ in range(10):gathered=dispatch(inputs);out=combine()
    torch.cuda.synchronize();samples=[]
    for _ in range(20):
        dist.barrier()
        a,b,c=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
        wall=time.perf_counter();a.record();gathered=dispatch(inputs);b.record();out=combine();c.record()
        c.synchronize();torch.cuda.synchronize()
        wall=(time.perf_counter()-wall)*1000
        samples.append(dict(dispatch_ms=a.elapsed_time(b),combine_ms=b.elapsed_time(c),total_ms=a.elapsed_time(c),wall_ms=wall))
    assert statistics.median(x['total_ms'] for x in samples)>.5*statistics.median(x['wall_ms'] for x in samples),'GPU events do not bracket the collective work'
    results.append(dict(family=family,local_rows=M,hidden=K,input_shapes=shapes,input_dtypes=list(map(str,dtypes)),samples=samples,changed_input_checks=2))
    del inputs,gathered,out

result=dict(rank=rank,world=WORLD,local_device=local,results=results,numerical=numerical,
            nccl_environment={k:os.environ.get(k) for k in ['NCCL_MIN_NCHANNELS','NCCL_MAX_NCHANNELS','NCCL_BUFFSIZE']},
            source=comm.__class__.__module__,full_model_performance_validated=False,completed=True,finished=time.time())
(root/f'rank{rank}.json').write_text(json.dumps(result,indent=2))
all_results=[None]*WORLD;dist.all_gather_object(all_results,result)
if rank==0:
    summaries=[]
    for j,family in enumerate(['fp8-and-routing','bf16-and-logits']):
        maxima=[max(r['results'][j]['samples'][i]['total_ms'] for r in all_results) for i in range(20)]
        summaries.append(dict(family=family,median_rank_max_ms=statistics.median(maxima),samples_rank_max_ms=maxima))
    report=dict(passed=True,world=WORLD,summaries=summaries,
                max_relative_l2=max(r['numerical']['relative_l2'] for r in all_results),
                environment=result['nccl_environment'],full_model_performance_validated=False)
    (root/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
dist.barrier();comm.destroy();dist.destroy_process_group()
