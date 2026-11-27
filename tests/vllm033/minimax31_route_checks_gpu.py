"""Exact predicate equivalence, adversarial inputs, and changed-input graph."""
import argparse
import json
from pathlib import Path
import sys
sys.path.append('/usr/local/lib/python3.12/site-packages')
import torch
from triton_tmp_dir.minimax31_route_checks import route_checks

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
torch.cuda.set_device(0);cases=[]
for experts in (128,136):
    for rows in (0,1,128,256):
        ids=torch.zeros((rows,4),device='cuda',dtype=torch.int64)
        weights=torch.ones((rows,4),device='cuda',dtype=torch.float32)
        checks=torch.empty(2,device='cuda',dtype=torch.bool)
        variants=[('valid',0,1.),('padding',-1,0.),('negative_zero_padding',-1,-0.),
                  ('upper_expert',experts-1,-1.),('too_negative',-2,0.),('too_large',experts,1.),
                  ('padding_nonzero',-1,1.),('nan',0,float('nan')),('positive_inf',0,float('inf')),
                  ('negative_inf',0,-float('inf')),('subnormal_padding',-1,1.401298464324817e-45)]
        for name,index,value in variants:
            ids.fill_(0);weights.fill_(1.)
            if rows:ids[-1,-1]=index;weights[-1,-1]=value
            route_checks(ids,weights,checks,experts)
            expected=torch.stack([((ids>=-1)&(ids<experts)).all(),
                (torch.isfinite(weights)&((ids>=0)|(weights==0))).all()])
            if not torch.equal(checks,expected):raise RuntimeError('predicate mismatch: '+str((experts,rows,name,checks.cpu().tolist(),expected.cpu().tolist())))
            cases.append({'experts':experts,'rows':rows,'case':name,'checks':checks.cpu().tolist()})
ids=torch.zeros((128,4),device='cuda',dtype=torch.int64)
weights=torch.ones_like(ids,dtype=torch.float32);checks=torch.empty(2,device='cuda',dtype=torch.bool)
flags=(checks[0],checks[1]);route_checks(ids,weights,checks,128)
torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    route_checks(ids,weights,checks,128)
    torch._assert_async(flags[0],'invalid physical expert ID')
    torch._assert_async(flags[1],'invalid route/padding weights')
for index,value in [(127,1.),(-1,0.),(0,-1.)]:
    ids.fill_(index);weights.fill_(value);graph.replay();torch.cuda.synchronize()
    assert checks.cpu().tolist()==[True,True]
result={'passed':True,'exact_torch_predicate_cases':cases,'valid_changed_input_assert_graph_cases':3,
        'checks_retained':['-1 <= expert_id < physical_experts','finite weights','padding expert has exactly zero weight'],
        'scope':'guard equivalence only; no model performance result'}
args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'exact_predicate_cases':len(cases),'valid_assert_graph_cases':3}),flush=True)
