"""Compare the actual DBO pipe's grouped roots with uniform collectives."""
import datetime,json,os,statistics,time
from pathlib import Path
import torch,torch.distributed as dist
from vllm.plugins import load_general_plugins
from vllm_hcu.patch.worker import apply_worker_patches

rank,local=int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local);torch.set_num_threads(1)
assert torch.cuda.mem_get_info()[0]>32*1024**3
load_general_plugins();apply_worker_patches()
dist.init_process_group('gloo',timeout=datetime.timedelta(seconds=180))
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import buffer_type,cudaStream_t,ncclDataTypeEnum,ncclRedOpTypeEnum
comm=PyNcclCommunicator(dist.group.WORLD,torch.device('cuda',local))
W,K=16,4096;rows=[];checks=[]
torch.manual_seed(102200+rank)
def dtype(t):return ncclDataTypeEnum.from_torch(torch.uint8 if t.element_size()==1 else t.dtype)
def maximum(x):
 t=torch.tensor([x],dtype=torch.float64);dist.all_reduce(t,op=dist.ReduceOp.MAX);return float(t.item())
try:
 for M in [4096,8192]:
  ins=[torch.empty((M,n),device='cuda',dtype=d) for n,d in [(K,torch.float8_e4m3fn),(8,torch.float32),(8,torch.int32),(1,torch.float32)]]
  outs=[torch.empty((W*M,t.shape[1]),device='cuda',dtype=t.dtype) for t in ins]
  rin=torch.empty((W*M,K),device='cuda',dtype=torch.bfloat16);rout=torch.empty((M,K),device='cuda',dtype=rin.dtype)
  def dispatch(uniform):
   stream=torch.cuda.current_stream();st=cudaStream_t(stream.cuda_stream)
   comm.nccl.ncclGroupStart()
   for src,dst in zip(ins,outs):
    if uniform:
     comm.nccl.ncclAllGather(buffer_type(src.data_ptr()),buffer_type(dst.data_ptr()),src.numel(),dtype(src),comm.comm,st)
    else:
     for peer in range(W):
      sl=dst[peer*M:(peer+1)*M]
      comm.nccl.ncclBroadcast(buffer_type(src.data_ptr()),buffer_type(sl.data_ptr()),sl.numel(),dtype(src),peer,comm.comm,st)
   comm.nccl.ncclGroupEnd()
  def combine(uniform):
   st=cudaStream_t(torch.cuda.current_stream().cuda_stream);op=ncclRedOpTypeEnum.from_torch(dist.ReduceOp.SUM)
   if uniform:
    comm.nccl.ncclReduceScatter(buffer_type(rin.data_ptr()),buffer_type(rout.data_ptr()),rout.numel(),dtype(rin),op,comm.comm,st)
   else:
    comm.nccl.ncclGroupStart()
    for peer in range(W):
     sl=rin[peer*M:(peer+1)*M]
     comm.nccl.ncclReduce(buffer_type(sl.data_ptr()),buffer_type(rout.data_ptr()),sl.numel(),dtype(sl),op,peer,comm.comm,st)
    comm.nccl.ncclGroupEnd()
  # Power-of-two factors keep all partial BF16 integer sums representable.
  # General random sums are tested against the BF16 rounding bound below.
  for factor in [1,2,4]:
   for t in ins:t.fill_(factor*(rank+1))
   for peer in range(W):rin[peer*M:(peer+1)*M].fill_(factor*(peer+1))
   for uniform in [False,True]:
    dispatch(uniform);combine(uniform);torch.cuda.synchronize()
    for original,out in zip(ins,outs):
     for peer in range(W):assert torch.equal(out[peer*M:(peer+1)*M].view(torch.uint8),torch.full_like(original,factor*(peer+1)).view(torch.uint8))
    assert (rout==W*factor*(rank+1)).all().item(),(rank,M,factor,uniform,float(rout.min()),float(rout.max()))
    checks.append(dict(M=M,uniform=uniform,factor=factor,exact=True))
  rin.normal_();ref=comm.all_reduce(rin.float());absolute=comm.all_reduce(rin.float().abs())
  target=ref[rank*M:(rank+1)*M];unit=torch.finfo(rin.dtype).eps/2;gamma=(W-1)*unit/(1-(W-1)*unit)
  for uniform in [False,True]:
   combine(uniform);torch.cuda.synchronize();delta=rout.float()-target
   error=float(delta.norm()/target.norm().clamp_min(1e-12));assert error<.02
   assert (delta.abs()<=gamma*absolute[rank*M:(rank+1)*M]+1e-5).all().item()
   checks.append(dict(M=M,uniform=uniform,random_relative_l2=error,bf16_rounding_bound=True))
  del ref,absolute,target,delta
  for t in ins:
   if t.dtype==torch.int32:t.random_(0,352)
   else:t.copy_(torch.randn(t.shape,device='cuda',dtype=torch.bfloat16).to(t.dtype))
  for name in ['dispatch','combine','pair']:
   def run(uniform):
    if name!='combine':dispatch(uniform)
    if name!='dispatch':combine(uniform)
   for _ in range(3):run(False);run(True)
   torch.cuda.synchronize();samples=[]
   for i in range(10):
    values={}
    for uniform in ([False,True] if i%2 else [True,False]):
     dist.barrier();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
     a.record();run(uniform);b.record();b.synchronize();values['uniform' if uniform else 'grouped']=maximum(a.elapsed_time(b))
    samples.append(values)
   row=dict(M=M,name=name,grouped_ms=statistics.median(x['grouped'] for x in samples),uniform_ms=statistics.median(x['uniform'] for x in samples),samples=samples)
   rows.append(row)
   if rank==0:print(json.dumps({k:v for k,v in row.items() if k!='samples'}),flush=True)
  del ins,outs,rin,rout
 result=dict(rank=rank,complete=True,checks=checks,timings=rows)
 (Path(os.environ['STEP5_PROOF_DIR'])/f'result-rank{rank}.json').write_text(json.dumps(result,indent=2))
 dist.barrier()
finally:
 comm.destroy();dist.destroy_process_group()
