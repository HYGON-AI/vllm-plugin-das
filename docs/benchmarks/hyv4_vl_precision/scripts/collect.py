import hashlib
import json
from pathlib import Path
import statistics
import re
import torch
r=Path('/root/hy4-perf-stage4')
def load(name):return json.loads((r/name).read_text())
def save(name,value):(r/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def rows(name):return [json.loads(x) for x in (r/name).read_text().splitlines()]
def tensors(value):
    if isinstance(value,dict):
        for v in value.values():yield from tensors(v)
    elif isinstance(value,list):
        for v in value:yield from tensors(v)
    elif isinstance(value,torch.Tensor):yield value
ablation=[]
ref={x['index']:x for x in load('ablation-0-decode_only.json')}
for i,name in enumerate(['decode_only','padded_eager','full_piecewise']*2):
    label=f'ablation-{i}-{name}'
    data=load(label+'.json');dispatch=load(label+'-dispatch.json')['results']
    def canonical(d):
        d=dict(d)
        d['shapes']={re.sub(r"'valid_modes': \{([^}]+)\}", lambda m: "'valid_modes': {" + ", ".join(sorted(m.group(1).split(", "))) + "}", k):v for k,v in d['shapes'].items()}
        return d
    assert len(dispatch)==8 and all(canonical(x)==canonical(dispatch[0]) for x in dispatch)
    changes=[{'index':x['index'],'before':ref[x['index']]['prediction'],'after':x['prediction']} for x in data if x['prediction']!=ref[x['index']]['prediction']]
    ablation.append({'label':label,'mode':name,'items':len(data),'correct':sum(x['correct'] for x in data),'raw_changed':len(changes),'changes':changes,'per_rank_dispatch':dispatch[0]})
    if name=='padded_eager':assert dispatch[0]['bypass']==dispatch[0]['counts']['PIECEWISE']
assert load('ablation-1-padded_eager.json')[0]['index']==load('ablation-2-full_piecewise.json')[0]['index']
for i in ref:
    assert all({x['index']:x for x in load(f'ablation-{k}-{name}.json')}[i]['prediction']==ref[i]['prediction'] for k,name in [(3,'decode_only')])
    preds=[{x['index']:x for x in load(f'ablation-{k}-{name}.json')}[i]['prediction'] for k,name in [(1,'padded_eager'),(2,'full_piecewise'),(4,'padded_eager'),(5,'full_piecewise')]]
    assert len(set(preds))==1
save('ablation-summary.json',ablation)
snapshots=[]
for idx in [119,94]:
    info={'index':idx,'repeats':[],'layer_errors':[]}
    for mode,its in [('decode_only',[0,2]),('padded_eager',[1,3])]:
        data=[torch.load(r/f'snapshot-{idx}-{it}-{mode}.pt',weights_only=False,map_location='cpu') for it in its]
        a,b=[list(tensors(x)) for x in data]
        info['repeats'].append({'mode':mode,'tensors':len(a),'all_exact':len(a)==len(b) and all(torch.equal(x,y) for x,y in zip(a,b))})
        assert info['repeats'][-1]['all_exact']
    a=torch.load(r/f'snapshot-{idx}-0-decode_only.pt',weights_only=False,map_location='cpu');b=torch.load(r/f'snapshot-{idx}-1-padded_eager.pt',weights_only=False,map_location='cpu')
    info['sampled_1024_token_chunk_all_layers_exact']=all(all(torch.equal(x,y) for x,y in zip(tensors(a[f'model.layers.{j}:0']['output']),tensors(b[f'model.layers.{j}:0']['output']))) for j in range(78))
    assert info['sampled_1024_token_chunk_all_layers_exact']
    for name in ['model.layers.0:1','model.layers.1:1','model.layers.10:1','model.layers.40:1','model.layers.77:1','model.norm:1']:
        x=list(tensors(a[name]['output']))[0].float();y=list(tensors(b[name]['output']))[0].float();d=x-y
        info['layer_errors'].append({'module':name,'sampled_shape':list(x.shape),'max_abs':d.abs().max().item(),'relative_l2':(d.norm()/x.norm()).item(),'nonzero':int(torch.count_nonzero(d))})
    snapshots.append(info)
save('snapshot-summary.json',snapshots)
raw=[]
for p in sorted(r.glob('*.pt')):
    raw.append({'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
save('raw-tensor-manifest.json',raw)
if (r/'accuracy-decode_only/summary.json').exists():
    a=rows('accuracy-guard/results.jsonl');b=rows('accuracy-decode_only/results.jsonl')
    assert len(a)==len(b)==765 and [x['index'] for x in a]==[x['index'] for x in b]==list(range(765))
    assert load('accuracy-guard/manifest.json')==load('accuracy-decode_only/manifest.json')
    paired=[]
    for x,y in zip(a,b):
        assert (x['question'],x['answer'],x['image_sha256'])==(y['question'],y['answer'],y['image_sha256'])
        paired.append({'index':x['index'],'image_sha256':x['image_sha256'],'question':x['question'],'reference':x['answer'],'guard_prediction':x['prediction'],'decode_only_prediction':y['prediction'],'guard_correct':x['correct'],'decode_only_correct':y['correct']})
    (r/'paired-predictions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in paired))
    comparison={'items':765,'guard_correct':sum(x['correct'] for x in a),'decode_only_correct':sum(x['correct'] for x in b),'guard_strict_correct':sum(x['strict_correct'] for x in a),'decode_only_strict_correct':sum(x['strict_correct'] for x in b),'raw_changed':sum(x['prediction']!=y['prediction'] for x,y in zip(a,b)),'errors':{label:{k:load(label+'/summary.json')[k] for k in ['http_errors','empty_answers','truncated']} for label in ['accuracy-guard','accuracy-decode_only']}}
    comparison['strict_acceptance_pass']=comparison['raw_changed']==0 and not any(value for errors in comparison['errors'].values() for value in errors.values())
    save('accuracy-comparison.json',comparison)
    actual={}
    for label in ['accuracy-guard','accuracy-decode_only']:
        data=load(label+'-actual-dispatch.json')['results']
        assert len(data)==8 and all({k:v for k,v in d.items() if k!='rank'}=={k:v for k,v in data[0].items() if k!='rank'} for d in data)
        actual[label]=data[0]
    save('actual-dispatch-validation.json',{'ranks':8,'passed':True,'modes':actual})
if (r/'e2e-decode_only/summary.json').exists():
    timings={}
    for suite in ['qa_c1','qa_c2','caption32_c1','two_images32_c1']:
        a=load('e2e-guard/summary.json')[suite];b=load('e2e-decode_only/summary.json')[suite]
        xa=load('e2e-guard/'+suite+'.json');xb=load('e2e-decode_only/'+suite+'.json')
        assert [(x['index'],x['second_index']) for x in xa]==[(x['index'],x['second_index']) for x in xb]
        changes=[{'index':x['index'],'second_index':x['second_index'],'guard':x['answer'],'decode_only':y['answer']} for x,y in zip(xa,xb) if x['answer']!=y['answer']]
        timings[suite]={'requests':a['requests'],'guard_p50_ms':a['latency_p50_seconds']*1000,'decode_only_p50_ms':b['latency_p50_seconds']*1000,'guard_rps':a['requests_per_second'],'decode_only_rps':b['requests_per_second'],'throughput_gain_percent':(a['requests_per_second']/b['requests_per_second']-1)*100,'raw_changed':len(changes),'changes':changes}
    save('performance-comparison.json',timings)
print('COLLECTED',flush=True)
