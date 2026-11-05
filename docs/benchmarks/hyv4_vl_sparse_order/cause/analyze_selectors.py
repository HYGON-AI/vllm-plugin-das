import json
from pathlib import Path
import torch
r=Path('/root/hy4-precision-fix')
def load(p):return torch.load(p,weights_only=False,map_location='cpu')
if (r/'selector-ablation.json').exists():
 results=json.loads((r/'selector-ablation.json').read_text())
 for selector in ['original','canonical','stable']:
  x=[p for p in results if p['selector']==selector]
  for idx in [112,422,287,478]:
   y=[p for p in x if p['index']==idx]
   if len(y)==2:print('TEXT',selector,idx,y[0]['answer']==y[1]['answer'])
for selector in ['original','canonical','stable']:
 a=r/f'{selector}-0-478-618.pt';b=r/f'{selector}-1-478-618.pt'
 if not a.exists() or not b.exists():continue
 aa=load(a);bb=load(b)
 first=next((i for i,(u,v) in enumerate(zip(aa,bb)) if not torch.equal(u['logits'],v['logits'])),None)
 print('LOGITS',selector,len(aa),first)
 for rank in range(8):
  p=r/f'{selector}-0-478-618-topk.rank{rank}.pt';q=r/f'{selector}-1-478-618-topk.rank{rank}.pt'
  aa=load(p);bb=load(q)
  print('TOPK',selector,rank,'calls',len(aa),'first score change',next((i for i,(u,v) in enumerate(zip(aa,bb)) if not torch.equal(u['logits'][...,:2049],v['logits'][...,:2049])),None),'first set change',next((i for i,(u,v) in enumerate(zip(aa,bb)) if not torch.equal(u['indices'].sort().values,v['indices'].sort().values)),None),'first order change',next((i for i,(u,v) in enumerate(zip(aa,bb)) if not torch.equal(u['indices'],v['indices'])),None))
