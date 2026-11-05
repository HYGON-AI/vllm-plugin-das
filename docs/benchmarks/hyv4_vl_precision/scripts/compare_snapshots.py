import json
from pathlib import Path
import torch
r=Path('/root/hy4-perf-stage4')
def flatten(x,p=''):
    if isinstance(x,dict):
        if 'value' in x and isinstance(x['value'],torch.Tensor):yield p,x['value']
        else:
            for k,v in x.items():yield from flatten(v,p+'/'+str(k))
    elif isinstance(x,list):
        for i,v in enumerate(x):yield from flatten(v,p+'/'+str(i))
for idx in [119,94]:
    a=r/f'snapshot-{idx}-0-decode_only.pt';b=r/f'snapshot-{idx}-1-padded_eager.pt'
    if not a.exists() or not b.exists():continue
    x=torch.load(a,weights_only=False,map_location='cpu');y=torch.load(b,weights_only=False,map_location='cpu')
    diffs=[]
    for name,v in x.items():
        if name not in y:continue
        va=dict(flatten(v));vb=dict(flatten(y[name]))
        for path,t in va.items():
            u=vb[path]
            if t.shape==u.shape and not torch.equal(t,u):
                d=t.float()-u.float()
                diffs.append({'module':name,'path':path,'real_tokens':v['real_tokens'],'shape':list(t.shape),'nonzero':int(torch.count_nonzero(d)), 'elements':d.numel(),'max_abs':d.abs().max().item(),'relative_l2':(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(t.float())).item()})
    (r/f'snapshot-{idx}-diffs.json').write_text(json.dumps(diffs,indent=2)+'\n')
    print('INDEX',idx,'diffs',len(diffs))
    for v in diffs[:18]:print(v)
