"""CPU-only numerical, workload-identity and timing audit of the EP8 module."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics
import torch

p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
root=Path(args.root);refroot=root/'sglang_reference_module'
torch.set_num_threads(8)
ours=[json.loads((root/('bench_ours_rank%d.json'%r)).read_text()) for r in range(8)]
sg=[json.loads((refroot/('bench_sglang_rank%d.json'%r)).read_text()) for r in range(8)]
summary={'schema':'standalone-minimax31-moe-v1','status':'running',
 'scope':'EP8 dispatch + received experts + combine with fixed physical Top4; synthetic target-shape fixtures; gate/router and real shared MLP excluded from latency numbers',
 'timing_statistic':'median over 15 repetitions of maximum wall time across 8 ranks; 4 warmups; profiler disabled for timed repetitions',
 'software':{'ours':ours[0]['torch'],'sglang':sg[0]['torch']},'fixture_sha256':[],
 'cases':{},'numerical':[],'profiles':{},'tests':{}}
for rank in range(8):
 assert ours[rank]['rank']==sg[rank]['rank']==rank
 assert ours[rank]['fixture_sha256']==sg[rank]['fixture_sha256']
 summary['fixture_sha256'].append(ours[rank]['fixture_sha256'])
cases=[k.split('/',1)[1] for k in sg[0]['cases']]
for case in cases:
 timings={}
 for variant,rs in [('legacy24',ours),('aligned24',ours),('aligned64',ours),('sglang64',sg)]:
  key=variant+'/'+case
  values=[max(r['cases'][key]['wall_ms'][i] for r in rs) for i in range(15)]
  events=[max(r['cases'][key]['event_ms'][i] for r in rs) for i in range(15)]
  timings[variant]={'median_wall_ms':statistics.median(values),'wall_max_rank_per_repeat_ms':values,
                   'median_event_ms':statistics.median(events),'min_wall_ms':min(values),'max_wall_ms':max(values)}
 new=timings['aligned64']['median_wall_ms'];old=timings['legacy24']['median_wall_ms'];s=timings['sglang64']['median_wall_ms']
 summary['cases'][case]={'timings':timings,'latency_reduction_vs_legacy_pct':(1-new/old)*100,
                        'latency_delta_vs_sglang_pct':(new/s-1)*100}
 print('TIMING',case,round(old,4),round(new,4),round(s,4),flush=True)
 for rank in range(8):
  r=ours[rank];t=sg[rank]['cases']['sglang64/'+case]
  for variant in ('aligned24','aligned64'):
   o=r['cases'][variant+'/'+case]
   assert o['input_quant_sha256']==t['input_quant_sha256'],('quantized input mismatch',rank,case,variant)
   assert o['dispatch']==t['dispatch'],('receive workload mismatch',rank,case,variant)
  got=torch.load(root/('aligned64_%s_rank%d.pt'%(case,rank)),weights_only=True,map_location='cpu')
  expected=torch.load(refroot/('sglang64_%s_rank%d.pt'%(case,rank)),weights_only=True,map_location='cpu')
  assert got.shape==expected.shape and got.isfinite().all() and expected.isfinite().all()
  exact=torch.equal(got,expected)
  if exact:rel=0.;cos=1.;maxabs=0.
  else:
   a,b=got.double().flatten(),expected.double().flatten();delta=a-b
   rel=float(delta.norm()/b.norm().clamp_min(1e-12));cos=float(a.dot(b)/(a.norm()*b.norm()).clamp_min(1e-12));maxabs=float(delta.abs().max())
  assert rel<=.0031 and cos>=.9999967,(case,rank,rel,cos,maxabs)
  control=torch.load(root/('aligned24_%s_rank%d.pt'%(case,rank)),weights_only=True,map_location='cpu')
  assert torch.equal(got,control),('24/64 outputs differ',case,rank)
  summary['numerical'].append({'case':case,'rank':rank,'relative_l2_vs_sglang':rel,'cosine_vs_sglang':cos,
                              'max_abs_vs_sglang':maxabs,'bit_exact_vs_sglang':exact,
                              'bit_exact_24_vs64':True,'INT8_and_scales_byte_exact':True,
                              'receive_counts_and_rows_identical':True,'dispatch':t['dispatch']})
  print('NUMERIC',case,rank,rel,exact,flush=True)

def merge(intervals):
 out=[]
 for a,b in sorted(intervals):
  if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
  else:out.append((a,b))
 return out

for variant,base in [('legacy24',root),('aligned24',root),('aligned64',root),('sglang64',refroot)]:
 summary['profiles'][variant]=[]
 for rank in range(8):
  path=base/('%s_rank%d.trace.json'%(variant,rank));raw=path.read_bytes()
  ev=json.loads(raw)['traceEvents'];ks=sorted([e for e in ev if e.get('cat')=='kernel'],key=lambda e:e['ts'])
  gems=[e for e in ks if 'wasp_kernel' in e['name'] or 'MARLIN_HIP_NT_CONTIGUOUS' in e['name']]
  gi={id(e):'gemm1' if i%2==0 else 'gemm2' for i,e in enumerate(gems)}
  groups=collections.defaultdict(list)
  for e in ks:
   n=e['name'].lower();g='other'
   if id(e) in gi:g=gi[id(e)]
   else:
    for label,needle in [('dispatch','::dispatch<'),('combine','::combine<'),('notify_dispatch','::notify_dispatch'),('notify_combine','cached_notify_combine'),('layout','deep_ep::layout'),('pack','_pad_sorted_routes'),('scatter','ep_scatter_kernel'),('gather','ep_gather_kernel'),('restore','_restore_routes'),('swiglu_quant','swiglu_oai_int8_quant'),('input_quant','per_token_quant_int8'),('input_quant','_quant_reciprocal'),('sort','rocprim::'),('guard','_route_guard'),('assert','assert_async'),('fill','fillfunctor')]:
     if needle in n:g=label;break
   groups[g].append(e)
  start=min(e['ts'] for e in ks);end=max(e['ts']+e['dur'] for e in ks)
  union=sum(b-a for a,b in merge([(e['ts'],e['ts']+e['dur']) for e in ks]))/1000
  span=(end-start)/1000
  record={'rank':rank,'sha256':hashlib.sha256(raw).hexdigest(),'gpu_span_ms':span,
          'kernel_count':len(ks),'kernel_sum_ms':sum(e['dur'] for e in ks)/1000,
          'no_kernel_ms':span-union,'groups':{k:{'count':len(es),'ms':sum(e['dur'] for e in es)/1000} for k,es in groups.items()},
          'communication_grids':{k:dict(collections.Counter(str(e['args']['grid']) for e in groups[k])) for k in ('dispatch','combine')},
          'gemm_kernel_names':sorted(set(e['name'] for e in gems))}
  assert len(gems)%2==0 and record['communication_grids']['dispatch']=={('[24, 1, 1]' if variant.endswith('24') else '[64, 1, 1]'):1}
  if variant!='legacy24':assert len(gems)==2 and not record['groups'].get('sort')
  summary['profiles'][variant].append(record)

for name in ('oracle_optimized.json','oracle_sglang.json','guard_validation.json'):
 v=json.loads((root/name).read_text());assert v['status']=='pass';summary['tests'][name]=v
assert all(r['cases']['aligned64/2048_balanced'].get('shared_stream_scale_join_bit_exact') for r in ours)
summary['tests']['shared_stream_scale_join_all8_bit_exact']=True
summary['max_relative_l2_vs_sglang']=max(r['relative_l2_vs_sglang'] for r in summary['numerical'])
summary['min_cosine_vs_sglang']=min(r['cosine_vs_sglang'] for r in summary['numerical'])
summary['all_outputs_bit_exact_vs_sglang']=all(r['bit_exact_vs_sglang'] for r in summary['numerical'])
summary['status']='pass'
(root/'module_alignment_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('AUDIT_PASS',summary['max_relative_l2_vs_sglang'],summary['min_cosine_vs_sglang'],summary['all_outputs_bit_exact_vs_sglang'],flush=True)
