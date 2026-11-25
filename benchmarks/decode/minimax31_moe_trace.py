"""Eight-rank VERIFY-M128 MoE component timing; not a full decode benchmark.

Run via torchrun with HOSTnone. Export all-rank traces and graph timings;
correctness coverage lives in tests/vllm033/minimax31_ll_distributed.py.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from hcu_vllm_das.moe.minimax31_ll import LowLatencyMoE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--experts',type=int,choices=(128,136),default=128)
    p.add_argument('--replays',type=int,default=100)
    p.add_argument('--profiler',choices=('torch','hipprof','none'),default='torch')
    args=p.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
    dist.init_process_group('nccl')
    from deepgemm import pack_w4a8_moe_hipc_weight
    n,h,i=128,6144,3072;e=args.experts//8
    torch.manual_seed(117+rank)
    w13=pack_w4a8_moe_hipc_weight(torch.full((e,2*i,h//2),0x11,device='cuda',dtype=torch.int8))
    w2=pack_w4a8_moe_hipc_weight(torch.full((e,h,i//2),0x11,device='cuda',dtype=torch.int8))
    s13=torch.full((e,2*i,1),1/(16*h),device='cuda',dtype=torch.float32)
    s2=torch.full((e,h,1),1/(16*i),device='cuda',dtype=torch.float32)
    weights=w13,w2,s13,s2
    x=(torch.randn((n,h),device='cuda',dtype=torch.bfloat16)*.1).contiguous()
    ids=((torch.arange(n,device='cuda')[:,None]*7+rank*3+
          torch.tensor([0,32,64,96],device='cuda')[None,:])%args.experts).long().contiguous()
    rw=torch.tensor([.1,.2,.3,.4],device='cuda').expand(n,4).contiguous()
    executor=LowLatencyMoE(dist.group.WORLD,device=torch.device('cuda',rank),
        physical_experts=args.experts,max_tokens=128)
    for _ in range(5):output=executor(x,ids,rw,weights)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):output=executor(x,ids,rw,weights)
    for _ in range(10):graph.replay()
    torch.cuda.synchronize();dist.barrier()
    durations=[]
    for trial in range(3):
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        dist.barrier();begin.record()
        for _ in range(args.replays):graph.replay()
        end.record();end.synchronize()
        durations.append(begin.elapsed_time(end)/args.replays)
    torch.cuda.synchronize();dist.barrier()
    if args.profiler=='torch':
      with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                         torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as prof:
        for step in range(8):
            with torch.profiler.record_function('VERIFY_MOE_COMPONENT_M128'):
                graph.replay()
            torch.cuda.synchronize()
      path=root/('rank%d.trace.json'%rank);prof.export_chrome_trace(str(path))
      with path.open('rb') as source,gzip.open(str(path)+'.gz','wb') as target:
        while True:
            data=source.read(1024*1024)
            if not data:break
            target.write(data)
      path.unlink()
    elif args.profiler=='hipprof':
      import ctypes
      from hcu_vllm_das.parallel.minimax31_hisparse import loaded_hip_runtime
      runtime=loaded_hip_runtime()
      runtime.hipProfilerStart.restype=ctypes.c_int;runtime.hipProfilerStop.restype=ctypes.c_int
      roctx=ctypes.CDLL('/opt/dtk/roctracer/lib/libroctx64.so.1')
      roctx.roctxRangePushA.argtypes=[ctypes.c_char_p]
      dist.barrier();assert runtime.hipProfilerStart()==0
      for step in range(8):
        roctx.roctxRangePushA(('VERIFY_MOE_COMPONENT_M128_STEP%d'%step).encode())
        graph.replay();torch.cuda.synchronize();roctx.roctxRangePop()
      assert runtime.hipProfilerStop()==0
    result={'scope':'single MoE component; synthetic balanced routing; not full model',
        'rank':rank,'query_rows':n,'requests':16,'verify_width':8,
        'hidden':h,'intermediate':i,'physical_experts':args.experts,
        'max_dispatch_tokens':128,'graph_ms_per_call':durations,
        'transport':'DeepEP LL INT8','gemm':'masked HIPC',
        'gpu':torch.cuda.get_device_name(rank),'torch':torch.__version__,'profiler':args.profiler}
    (root/('rank%d.timing.json'%rank)).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    torch.cuda.synchronize();dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
