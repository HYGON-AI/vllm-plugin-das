# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU oracle tests using the untouched checkpoint's own Python functions."""
import ast
import importlib.util
import json
from pathlib import Path
import random
import torch

root = Path(importlib.util.find_spec('vllm_hcu').origin).parent/'v1/locateanything'
tree = ast.parse((root/'core.py').read_text())
nodes = [n for n in tree.body if isinstance(n, ast.Assign)
         or (isinstance(n, ast.FunctionDef) and n.name == 'select_blocks')]
ns = {'torch':torch}
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'rules_only', 'exec'), ns)
spec = importlib.util.spec_from_file_location('reference', '/workspace/LocateAnything/generate_utils.py')
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
ids=ref.get_token_ids_from_config(object())
select=ns['select_blocks']
random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(4)
V=152681
cases=[]
# Every pattern branch, plus interior EOS/null and duplicated ref suffixes.
B,E,R,RE,C,N,EOS,NULL=(ns[k] for k in ('BOX','BOX_END','REF','REF_END','COORD_LO','NONE','EOS','NULL'))
patterns=[[EOS,1,2,3,4,5],[NULL,1,2,3,4,5],[B,N,8,9,10,11],
          [B,C,C+2,C+3,C+4,E],[B,C,C+1,E,55,66],[B,7,8,9,NULL,EOS],
          [R,22,RE,RE,NULL,99],[22,EOS,7,8,9,10],[22,NULL,1,2,3,4]]
pool=[B,E,R,RE,C,C+50,N,EOS,NULL,7,8,9]
patterns += [[random.choice(pool) for _ in range(6)] for _ in range(300)]
for start in range(0,len(patterns),16):
    sample=torch.tensor(patterns[start:start+16])
    probs=torch.full((len(sample),6,V),1/V)
    tokens,counts=select(probs,sample)
    for i,p in enumerate(sample):
        want=ref.handle_pattern(p,ids,generation_mode='fast')['tokens']
        assert tokens[i,:counts[i]].tolist()==want, (p,want,tokens[i],counts[i])
        cases.append('pattern')
for kind in ('box','empty','ref','fallback'):
    probs=torch.rand((12,6,V))*.000001
    for row in range(12):
        if kind=='box':
            probs[row,5,E]=.3
            for col in range(1,5): probs[row,col,C+row*5+col]=.8
        elif kind=='empty':
            for col,token,value in [(0,B,.8),(1,N,.3),(2,E,.3),(3,NULL,.2),(4,NULL,.2)]:probs[row,col,token]=value
        elif kind=='ref':
            probs[row,0,R]=.7
            for col in range(1,6):probs[row,col,10+row+col]=.8
    sample=torch.randint(0,V,(12,6))
    out,count=select(probs,sample)
    for row in range(12):
        box=ref.decode_bbox_avg(probs[row],probs[row],ids,keep_k=4,generation_mode='fast')
        chosen=box if box is not None else ref.decode_ref(probs[row],probs[row],ids)
        if chosen is None:chosen=sample[row]
        want=ref.handle_pattern(chosen,ids,generation_mode='fast')['tokens']
        assert out[row,:count[row]].tolist()==want, (kind,row,want,out[row],count[row])
        cases.append(kind)
# Visible keys for full prompt and variable-length real suffixes.
for c in (0,1,15,16,31,32,511,8192):
    for r in range(1,7):
        length=c+r+6
        baseline=torch.full((r+6,length),float('-inf'))
        for i in range(r+6):baseline[i,:c+i+1]=0
        baseline=ref_mask=baseline.clone()
        refspec=importlib.util.spec_from_file_location('mask_ref','/workspace/LocateAnything/mask_sdpa_utils.py')
        maskref=importlib.util.module_from_spec(refspec);refspec.loader.exec_module(maskref)
        expected=maskref.update_causal_mask_for_one_gen_window_2d(None,baseline,6,True,False)
        query=c+torch.arange(r+6)[:,None];keys=torch.arange(length)[None,:];real=c+r
        actual=torch.where(query<real,keys<=query,(keys<real-1)|(keys>=real))
        assert torch.equal(actual,expected==0)
print(json.dumps({'status':'passed','pattern_and_decode_cases':len(cases),'mask_cases':48}))
