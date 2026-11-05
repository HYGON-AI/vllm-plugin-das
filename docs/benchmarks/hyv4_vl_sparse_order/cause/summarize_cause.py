import json,hashlib
from pathlib import Path
import torch
r=Path('/root/hy4-precision-fix')
def load(p):return torch.load(p,weights_only=False,map_location='cpu')
def save(n,x):(r/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
answers=json.loads((r/'selector-ablation.json').read_text())
report={'baseline_head':'7b5b973698f7447b1cf0cea6e66a44f929915328','selectors':{},'indexer_boundary':[]}
for name in ['original','canonical','stable']:
 x=[v for v in answers if v['selector']==name]
 a=load(r/f'{name}-0-478-618.pt');b=load(r/f'{name}-1-478-618.pt')
 diffs=[]
 for i,(u,v) in enumerate(zip(a,b)):
  if not torch.equal(u['logits'],v['logits']):diffs.append({'record':i,'logits_max_abs':(u['logits']-v['logits']).abs().max().item()})
 report['selectors'][name]={'requests':len(x),'pair_repeats_exact':{str(idx):len(set(v['answer'] for v in x if v['index']==idx))==1 for idx in [112,422,287,478]},'logit_records':len(a),'changed_logit_records':diffs}
 for rank in range(8):
  aa=load(r/f'{name}-0-478-618-topk.rank{rank}.pt');bb=load(r/f'{name}-1-478-618-topk.rank{rank}.pt')
  entry={'selector':name,'rank':rank,'calls':len(aa),'first_scores_changed':None,'first_set_changed':None,'first_order_changed':None,'first_call_score_max_abs':(aa[0]['logits'][...,:2049]-bb[0]['logits'][...,:2049]).abs().max().item(),'first_call_reordered_positions':int((aa[0]['indices']!=bb[0]['indices']).sum())}
  for i,(u,v) in enumerate(zip(aa,bb)):
   if entry['first_scores_changed'] is None and not torch.equal(u['logits'][...,:2049],v['logits'][...,:2049]):entry['first_scores_changed']=i
   if entry['first_set_changed'] is None and not torch.equal(u['indices'].sort().values,v['indices'].sort().values):entry['first_set_changed']=i
   if entry['first_order_changed'] is None and not torch.equal(u['indices'],v['indices']):entry['first_order_changed']=i
  report['indexer_boundary'].append(entry)
a=load(r/'source-0.pt');b=load(r/'source-1.pt');c=load(r/'canonical-0-478-618.pt')
report['source_eager']={'repeated_records':len(a),'all_logits_exact':all(torch.equal(u['logits'],v['logits']) for u,v in zip(a,b)),'matches_canonical_reference_logits':all(torch.equal(u['logits'],v['logits']) for u,v in zip(a,c))}
report['attention_fixed_input']=json.loads((r/'attention-reference-metrics.json').read_text())
save('cause-summary.json',report)
manifest=[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(r.glob('*.pt'))]
save('external-tensor-manifest.json',manifest)
print('cause summary',report['selectors']['canonical']['pair_repeats_exact'],report['source_eager'],'tensors',len(manifest))
