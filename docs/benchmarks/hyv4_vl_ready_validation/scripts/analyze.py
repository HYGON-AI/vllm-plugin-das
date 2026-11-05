import json
from pathlib import Path
import torch
r=Path('/root/hy4-pr-ready-validation')
p=r/'two-image-repeats.json'
if not p.exists():raise SystemExit('no requests yet')
x=json.loads(p.read_text())
summary={}
for case in [112,422,287,478]:
    selected=[v for v in x if v['index']==case]
    modes={}
    for mode in ['full_piecewise','padded_eager','decode_only','eager']:
        data=[v for v in selected if v['mode']==mode]
        if data:modes[mode]={'runs':len(data),'repeat_exact':len(set(v['answer'] for v in data))==1,'answers':[v['answer'] for v in data]}
    entry={'modes':modes,'pairs':[]}
    for rep in [0,1]:
        answers={v['mode']:v['answer'] for v in selected if v['repetition']==rep}
        if 'eager' in answers:
            entry['pairs'].append({'repetition':rep,'differs_from_eager':[mode for mode,pred in answers.items() if pred!=answers['eager']]})
    summary[str(case)]=entry
(r/'answer-comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print('ANSWER SUMMARY',[(k,v['pairs'],{mode:d['repeat_exact'] for mode,d in v['modes'].items()}) for k,v in summary.items()])
trace=[]
for rep in [0,1]:
    refs=r/f'{rep}-eager-478-618.pt'
    if not refs.exists():continue
    a=torch.load(refs,weights_only=False,map_location='cpu')
    a=[v for v in a if v['logits'].numel()]
    for mode in ['full_piecewise','padded_eager','decode_only','eager']:
        path=r/f'{rep}-{mode}-478-618.pt'
        if not path.exists():continue
        b=torch.load(path,weights_only=False,map_location='cpu');b=[v for v in b if v['logits'].numel()]
        rows=[]
        for i,(u,v) in enumerate(zip(a,b)):
            da=u['logits'].float()-v['logits'].float()
            dh=u['hidden_states'].float()-v['hidden_states'].float()
            rows.append({'step':i,'logits_exact':torch.equal(u['logits'],v['logits']),'logits_max_abs':da.abs().max().item(),'hidden_max_abs':dh.abs().max().item(),'eager_argmax':u['logits'].argmax(-1).tolist(),'candidate_argmax':v['logits'].argmax(-1).tolist(),'context':v['context']})
        entry={'repetition':rep,'mode':mode,'steps':rows}
        trace.append(entry)
        print('TRACE',rep,mode,'steps',len(rows),'first_changed_logits',next((v for v in rows if not v['logits_exact']),None),'first_changed_argmax',next((v['step'] for v in rows if v['eager_argmax']!=v['candidate_argmax']),None))
(r/'logit-comparison.json').write_text(json.dumps(trace,indent=2)+'\n')
