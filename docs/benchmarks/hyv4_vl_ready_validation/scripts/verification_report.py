import hashlib
import json
import shutil
from pathlib import Path
import torch
r=Path('/root/hy4-pr-ready-validation')
d=Path('/tmp/vllm-plugin-das-pr/docs/benchmarks/hyv4_vl_ready_validation');d.mkdir(exist_ok=True)
def load(p):return torch.load(p,weights_only=False,map_location='cpu')
def save(name,obj):
 (r/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
 shutil.copy2(r/name,d/name)
results=json.loads((r/'two-image-repeats.json').read_text())
summary={'tested_head':'125d5e5a63c20d2e61563e572140398bd8313419','requests':len(results),'unexpected_http_errors':0,'pairs':[], 'repeat_logits':[], 'selector_ablations':[]}
for a,b in [(112,507),(422,204),(287,719),(478,618)]:
 entry={'indices':[a,b],'modes':{}}
 for mode in ['full_piecewise','padded_eager','decode_only','eager']:
  rows=[x for x in results if x['index']==a and x['mode']==mode]
  entry['modes'][mode]={'runs':len(rows),'prompt_tokens':[x['usage']['prompt_tokens'] for x in rows],'answers_identical':rows[0]['answer']==rows[1]['answer']}
 summary['pairs'].append(entry)
def compare(a,b):
 rows=[]
 for i,(x,y) in enumerate(zip(a,b)):
  delta=x['logits'].float()-y['logits'].float()
  rows.append({'record':i,'logits_exact':torch.equal(x['logits'],y['logits']),'logits_max_abs':delta.abs().max().item(),'hidden_max_abs':(x['hidden_states'].float()-y['hidden_states'].float()).abs().max().item(),'argmax_equal':torch.equal(x['logits'].argmax(-1),y['logits'].argmax(-1)),'context':x['context']})
 return {'records':len(rows),'first_changed_record':next((v['record'] for v in rows if not v['logits_exact']),None),'steps':rows}
for mode in ['full_piecewise','padded_eager','decode_only','eager']:
 entry=compare(load(r/f'0-{mode}-478-618.pt'),load(r/f'1-{mode}-478-618.pt'))
 entry['mode']=mode;summary['repeat_logits'].append(entry)
for name in ['original','legacy','torch']:
 entry={'selector':name,'repetition_0_vs_1':compare(load(r/f'topk-{name}-0.pt'),load(r/f'topk-{name}-1.pt')),'first_indexer_comparison':[]}
 aa=load(r/f'topk-{name}-0-selection.pt');bb=load(r/f'topk-{name}-1-selection.pt')
 for x,y in zip(aa,bb):
  length=x['length'];v=x['logits'][...,:length];w=y['logits'][...,:length];ix=x['indices'];iy=y['indices']
  entry['first_indexer_comparison'].append({'context_length':length,'valid_scores_exact':torch.equal(v,w),'valid_scores_max_abs':(v-w).abs().max().item(),'index_order_exact':torch.equal(ix,iy),'index_set_exact':torch.equal(ix.sort().values,iy.sort().values),'different_index_positions':int((ix!=iy).sum()),'invalid_indices':int(((ix<0)|(ix>=length)).sum()),'unique_indices':ix.unique().numel(),'excluded_index':sorted(set(range(length))-set(ix.flatten().tolist()))})
 summary['selector_ablations'].append(entry)
summary['excluded_ablation']={'label':'canonical','requests':3,'reason':'The selector wrapper was already installed before its source file gained the canonical branch. The closure was not refreshed. These requests used the original selector, not a sorted selector.'}
save('verification-summary.json',summary)
manifest=[]
for p in sorted(r.glob('*.pt')):
 manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
save('external-tensor-manifest.json',manifest)
for name in ['two-image-repeats.json','answer-comparison.json','logit-comparison.json','topk-ablation.json','topk-canonical-ablation.json','unit-tests.log','model-tests.log']:
 shutil.copy2(r/name,d/name)
for p in sorted(r.glob('*-dispatch.json')):shutil.copy2(p,d/p.name)
scripts=d/'scripts';scripts.mkdir(exist_ok=True)
for name in ['run.py','trace_install.py','trace_reset.py','trace_save.py','analyze.py','topk_probe.py','topk_probe_install.py','topk_probe_save.py','canonical_probe.py','verification_report.py','serve.sh']:
 shutil.copy2(r/name,scripts/name)
print([(x['mode'],x['first_changed_record'],x['steps'][x['first_changed_record']]['logits_max_abs']) for x in summary['repeat_logits']])
print('saved',len(list(d.rglob('*'))),'files','external tensors',len(manifest))
