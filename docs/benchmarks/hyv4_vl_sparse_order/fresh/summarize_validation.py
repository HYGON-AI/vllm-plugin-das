import json,torch
from pathlib import Path
r=Path('/root/hy4-precision-fix/fresh')
def save(n,x):(r/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def load(p):return torch.load(p,weights_only=False,map_location='cpu')
for size in [32,128]:
 p=r/f'caption{size}-results.json'
 if not p.exists():continue
 results=json.loads(p.read_text());groups={}
 for v in results:groups.setdefault((v['index'],v['second_index']),[]).append(v)
 report={'requests':len(results),'groups':[],'logit_comparisons':[],'indices_validation':[]}
 for (idx,second),values in groups.items():
  report['groups'].append({'index':idx,'second_index':second,'answers_all_identical':len(set(v['answer'] for v in values))==1,'requests':len(values),'prompt_tokens':sorted(set(v['usage']['prompt_tokens'] for v in values)),'modes':sorted(set(v['mode'] for v in values))})
 reference=r/f'caption{size}-0-eager-478-618.logits.pt'
 if reference.exists() and (r/f'caption{size}-0-eager-478-618-trace.json').exists():
  ref=load(reference)
  for p in sorted(r.glob(f'caption{size}-*-478-618.logits.pt')):
   if not (r/(p.name.replace('.logits.pt','-trace.json'))).exists():continue
   values=load(p)
   changed=[{'record':i,'max_abs':(u['logits']-v['logits']).abs().max().item()} for i,(u,v) in enumerate(zip(ref,values)) if not torch.equal(u['logits'],v['logits'])]
   report['logit_comparisons'].append({'file':p.name,'records':len(values),'reference_records':len(ref),'changed':changed,'actual_modes':{name:sum(v['context']['runtime_mode']==name for v in values) for name in ['NONE','FULL','PIECEWISE']}})
  for p in sorted(r.glob(f'caption{size}-*-478-618.indices.rank*.pt')):
   values=load(p)
   ordered=[]
   for t in values:
    keys=torch.where(t<0,torch.iinfo(torch.int32).max,t)
    ordered.append(bool((keys[:,1:]>=keys[:,:-1]).all()))
   report['indices_validation'].append({'file':p.name,'records':len(values),'all_canonical':all(ordered)})
 save(f'caption{size}-comparison.json',report)
 print('CAPTIONS',size,len(results),'all groups same',all(v['answers_all_identical'] for v in report['groups']),'logit differences',sum(len(v['changed']) for v in report['logit_comparisons']))
p=r/'long-qa-results.json'
if p.exists():
 results=json.loads(p.read_text());report={'requests':len(results),'labels':{},'pairs':[]}
 for label in ['native_eager','fixed_eager','fixed_full_piecewise']:
  data=[v for v in results if v['label']==label]
  if not data:continue
  report['labels'][label]={'runs':len(data),'correct_by_repetition':{str(rep):sum(v['correct'] for v in data if v['repetition']==rep) for rep in [0,1]},'changed_repeats':[],'context_lengths':sorted(set(v['usage']['prompt_tokens'] for v in data))}
  for idx in dict.fromkeys(v['index'] for v in data):
   values=[v for v in data if v['index']==idx]
   if len(values)==2 and values[0]['answer']!=values[1]['answer']:report['labels'][label]['changed_repeats'].append(idx)
 eager={v['index']:v for v in results if v['label']=='fixed_eager' and v['repetition']==0}
 for v in results:
  if v['label']=='fixed_full_piecewise' and v['index'] in eager:
   old=eager[v['index']]
   report['pairs'].append({'index':v['index'],'repetition':v['repetition'],'raw_equal':old['answer']==v['answer'],'eager_correct':old['correct'],'graph_correct':v['correct']})
 save('long-qa-comparison.json',report);print('LONG QA',report['labels'])
if (r/'perf-eager/summary.json').exists() and (r/'perf-full-piecewise/summary.json').exists():
 a=json.loads((r/'perf-full-piecewise/summary.json').read_text());b=json.loads((r/'perf-eager/summary.json').read_text());report={}
 for name in ['qa_c1','qa_c2','caption32_c1','two_images32_c1']:
  x=json.loads((r/'perf-full-piecewise'/f'{name}.json').read_text());y=json.loads((r/'perf-eager'/f'{name}.json').read_text())
  report[name]={'requests':len(x),'eager_p50_ms':b[name]['latency_p50_seconds']*1000,'graph_p50_ms':a[name]['latency_p50_seconds']*1000,'eager_rps':b[name]['requests_per_second'],'graph_rps':a[name]['requests_per_second'],'throughput_gain_percent':(a[name]['requests_per_second']/b[name]['requests_per_second']-1)*100,'raw_changes':[{'index':u['index'],'second_index':u['second_index'],'eager':v['answer'],'graph':u['answer']} for u,v in zip(x,y) if u['answer']!=v['answer']]}
 save('performance-comparison.json',report);print('PERFORMANCE',report)
