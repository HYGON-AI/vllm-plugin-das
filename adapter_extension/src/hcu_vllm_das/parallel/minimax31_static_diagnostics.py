# SPDX-License-Identifier: Apache-2.0
"""Bounded append-only route diagnostics; disabled in performance windows."""
import json,os
from pathlib import Path
import torch
from .minimax31_static_runtime import layout
_PENDING={}

def selected(layer,logical,physical):
    control=os.environ.get('HCU_MINIMAX_STATIC_ROUTE_CONTROL')
    if not control:return
    cfg=json.loads(Path(control).read_text())
    if cfg.get('schema')!='minimax31-static-route-diagnostic-v1':raise ValueError('invalid static route diagnostic control')
    if cfg.get('active') is not True:return
    batch=layout(logical.shape[0])
    if batch['is_dummy']:return
    rows=batch['rows'];positions=batch['positions'][:rows].cpu().tolist()
    # A target diagnostic is exactly the specified append window. Prefix and
    # decode must never be silently included or normalized into these counts.
    if rows!=8192 or any(not 131072<=p<147456 for p in positions):raise ValueError('static diagnostic is not the target local8192 append')
    keys=[];expected=[]
    for lo,hi,key,start in batch['segments']:keys.extend([key]*(hi-lo));expected.extend(range(start,start+hi-lo))
    if positions!=expected or sorted(set(keys))!=sorted(cfg['request_keys']):raise ValueError('static PCP position/request identity mismatch')
    lid=layer._static_layer_id;key=(cfg['generation'],lid)
    if key in _PENDING:raise RuntimeError('duplicate static diagnostic forward/layer')
    lc=torch.bincount(logical[:rows].to(torch.int64).reshape(-1),minlength=128).cpu().tolist();pc=torch.bincount(physical[:rows].reshape(-1),minlength=136).cpu().tolist()
    if sum(lc)!=32768 or sum(pc)!=32768:raise RuntimeError('static logical/selected physical routes are not conserved')
    ix=[i for i,p in enumerate(positions) if (p-131072)%1024==0];li=logical[ix].cpu().tolist();pi=physical[ix].cpu().tolist();samples=[[keys[i],positions[i],l,p] for i,ls,ps in zip(ix,li,pi) for l,p in zip(ls,ps)]
    _PENDING[key]={'schema':'minimax31-static-route-rank-v1','generation':cfg['generation'],'group_id':cfg['group_id'],'rank':layer.expert_map_manager.ep_rank,'layer':lid,'map_sha256':layer._static_placement.map_sha256,'request_keys':sorted(set(keys)),'logical_counts':lc,'selected_physical_counts':pc,'route_samples':samples,'coverage_segments':[list(x) for x in batch['segments']],'output_dir':cfg['output_dir']}

def received(layer,counts):
    control=os.environ.get('HCU_MINIMAX_STATIC_ROUTE_CONTROL')
    if not control:return
    cfg=json.loads(Path(control).read_text())
    if cfg.get('active') is not True:return
    key=(cfg['generation'],layer._static_layer_id);record=_PENDING.pop(key,None)
    if record is None:raise RuntimeError('missing selected-route diagnostic for receive')
    if len(counts)!=17 or any(type(n) is not int or n<0 for n in counts):raise ValueError('invalid17-slot actual receive counts')
    record['received_local_counts']=list(counts);folder=Path(record.pop('output_dir'));folder.mkdir(parents=True,exist_ok=True);path=folder/('rank%d_layer%02d.json'%(record['rank'],record['layer']))
    with path.open('x') as f:json.dump(record,f,sort_keys=True,separators=(',',':'));f.write('\n')
