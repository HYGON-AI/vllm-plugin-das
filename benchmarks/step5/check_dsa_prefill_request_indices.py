"""Validate exact staged GPU row-owner mapping and absence of host stream wait."""
import argparse,ast,importlib.util,json,pathlib,random,statistics,sys,tempfile,time
import torch
from vllm.triton_utils import tl,triton
torch.set_num_threads(1);torch.cuda.set_device(0)
parser=argparse.ArgumentParser(description='Run on an idle HCU GPU: byte-exact row mapping and asynchronous launch check.')
parser.add_argument('--source',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[2]/'vllm_hcu/model_executor/layers/step4_dsa.py')
parser.add_argument('--output',type=pathlib.Path)
args=parser.parse_args()
root=pathlib.Path(tempfile.mkdtemp(prefix='step5-dsa-mapper-'))
p=args.source;text=p.read_text();tree=ast.parse(text)
planner=next(f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name=='build_plan')
calls=[f for f in ast.walk(planner) if isinstance(f,ast.Call) and isinstance(f.func,ast.Name) and f.func.id=='_prefill_request_indices']
assert len(calls)==1 and len(calls[0].args)==5 and isinstance(calls[0].args[0],ast.Attribute) and calls[0].args[0].attr=='query_start_loc', 'Planner must use the tested GPU helper with device request boundaries'
selected=[]
for f in tree.body:
 if isinstance(f,ast.FunctionDef) and f.name in ['_prefill_request_indices_kernel','_prefill_request_indices']:
  code=ast.get_source_segment(text,f)
  if f.decorator_list:code='@triton.jit\n'+code
  selected.append(code)
assert len(selected)==2
module=root/'metadata_helpers.py';module.write_text('from __future__ import annotations\nimport torch\nfrom vllm.triton_utils import tl,triton\n'+'\n\n'.join(selected))
spec=importlib.util.spec_from_file_location('step5_dsa_mapper_proof',module);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
rng=random.Random(20261022);cases=[(2,[4,4],[1024]*6+[1020]),(0,[],[32768]),(1,[4],[4]+[1024]*7),(3,[1,4,4],[0,1,0,64,0,1024,0])]
for n in [1,2,3,7,8,15,16,31,32,63,64]:
 for _ in range(6):
  nd=rng.randrange(0,5);cases.append((nd,[rng.choice([1,4]) for _ in range(nd)],[rng.randrange(0,1100) for _ in range(n)]))
proof=[]
for nd,prefix,counts in cases:
 full=prefix+counts;starts=torch.tensor([0]+list(__import__('itertools').accumulate(full)),dtype=torch.int32,device='cuda');N=sum(counts);decode=sum(prefix)
 expected=torch.repeat_interleave(torch.arange(nd,nd+len(counts),dtype=torch.int32),torch.tensor(counts),output_size=N)
 out=m._prefill_request_indices(starts,N,nd,len(counts),decode);assert torch.equal(out.cpu(),expected),(nd,counts)
 poisoned=torch.full((N,),-1,dtype=torch.int32,device='cuda')
 if N:m._prefill_request_indices_kernel[(triton.cdiv(N,256),)](starts,poisoned,N,nd,len(counts),decode,STEPS=len(counts).bit_length(),BLOCK=256,num_warps=4)
 assert torch.equal(poisoned.cpu(),expected),(nd,counts,'unwritten')
 for groups in [1,2,4]:assert torch.equal(out.repeat(groups).cpu(),expected.repeat(groups))
 proof.append(dict(first_request=nd,prefill_requests=len(counts),tokens=N,exact=True,poisoned=True))
nd=2;prefix=[4,4];counts=[1024]*6+[1020];N=sum(counts);decode=sum(prefix);starts=torch.tensor([0]+list(__import__('itertools').accumulate(prefix+counts)),dtype=torch.int32,device='cuda');cpu_counts=torch.tensor(counts,dtype=torch.int32)
expected=torch.repeat_interleave(torch.arange(nd,nd+len(counts),dtype=torch.int32),cpu_counts,output_size=N)
def old():return torch.repeat_interleave(torch.arange(nd,nd+len(counts),dtype=torch.int32,device='cuda'),cpu_counts.to('cuda'))
def new():return m._prefill_request_indices(starts,N,nd,len(counts),decode)
stream=torch.cuda.Stream();latency=[]
with torch.cuda.stream(stream):
 for _ in range(3):old();new()
torch.cuda.synchronize()
for repeat in range(3):
 for name,fn in [('old',old),('new',new)]:
  torch.cuda.synchronize()
  with torch.cuda.stream(stream):
   torch.cuda._sleep(100000000);tail=torch.cuda.Event();tail.record();start=time.perf_counter();out=fn();cpu_ms=(time.perf_counter()-start)*1000;delay_finished=tail.query()
  torch.cuda.synchronize();assert torch.equal(out.cpu(),expected)
  latency.append(dict(repeat=repeat,method=name,host_call_ms=cpu_ms,queued_delay_finished_when_call_returned=delay_finished))
assert all(z['queued_delay_finished_when_call_returned'] for z in latency if z['method']=='old'),latency
assert all(not z['queued_delay_finished_when_call_returned'] for z in latency if z['method']=='new'),latency
result=dict(complete=True,source_sha256=__import__('hashlib').sha256(p.read_bytes()).hexdigest(),cases=proof,checks=len(proof)*5,all_indices_byte_exact=True,latency=latency,model_tps_not_measured=True)
destination=args.output or root/'result.json';destination.write_text(json.dumps(result,indent=2));print(json.dumps(dict(complete=True,cases=len(proof),checks=result['checks'],latency=latency,source_sha256=result['source_sha256'])),flush=True)
