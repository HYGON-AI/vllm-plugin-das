# SPDX-License-Identifier: Apache-2.0
"""Content-derived request identity and PCP global positions for static routing."""
import hashlib,json,threading
from pathlib import Path
import numpy as np
import torch
from .minimax31_static_contract import canonical,from_current,request_base
_MANIFEST_PATH=None
_REQUESTS={};_DUMMY={};_MANIFEST_KEYS=None;_LAYOUT=None;_HOOKS=None;_PLACEMENT=None;_FRAME_LOCK=threading.Lock();_LAST_LAYER=None

def content_key(token_ids):
    global _MANIFEST_KEYS
    placement=_PLACEMENT
    if placement is None:raise RuntimeError('static runtime requires explicit placement')
    if _MANIFEST_KEYS is None:
        path=_MANIFEST_PATH
        if not path:raise ValueError('static routing requires the frozen dataset manifest')
        raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=placement.dataset_manifest_sha256:raise ValueError('static dataset manifest SHA differs from placement')
        manifest=json.loads(raw);lookup={}
        for group in manifest['groups']:
            for q in group['requests']:
                digest=q['token_ids_sha256'];key=hashlib.sha256(canonical({'schema':'minimax31-request-key-v1','group_id':group['group_id'],'slot':q['slot'],'token_ids_sha256':digest})).hexdigest()
                if q['request_key']!=key or digest in lookup:raise ValueError('ambiguous or invalid manifest content/request-key binding')
                lookup[digest]=key
        _MANIFEST_KEYS=lookup
    a=np.asarray(token_ids,dtype='<u4');digest=hashlib.sha256(a.tobytes()).hexdigest()
    if digest in _MANIFEST_KEYS:return _MANIFEST_KEYS[digest]
    return hashlib.sha256(canonical({'schema':'minimax31-request-key-v1','group_id':'runtime','slot':0,'token_ids_sha256':digest})).hexdigest()

def install_identity_hooks():
    global _HOOKS,_PLACEMENT,_LAST_LAYER,_MANIFEST_PATH
    from vllm.v1.worker.gpu.states import RequestState
    from vllm.v1.worker.gpu.input_batch import InputBatch
    if _HOOKS is not None:
        from .minimax31_config import artifacts
        if from_current()!=_PLACEMENT or artifacts().require("dataset_manifest")!=_MANIFEST_PATH:
            raise RuntimeError("static deployment identity changed; restart the engine")
        if RequestState.add_request is not _HOOKS[0] or RequestState.remove_request is not _HOOKS[1] or InputBatch.make_dummy.__func__ is not _HOOKS[2]:raise RuntimeError('static request identity hook ownership changed')
        return
    _PLACEMENT=from_current()
    if _PLACEMENT is None:raise RuntimeError("identity hooks require a static model")
    from vllm.config import get_current_vllm_config
    config=get_current_vllm_config()
    from .minimax31_config import artifacts
    _MANIFEST_PATH=artifacts(config).require("dataset_manifest")
    hf=config.model_config.hf_config
    _LAST_LAYER=int(getattr(hf,"text_config",hf).num_hidden_layers)-1
    add=RequestState.add_request;remove=RequestState.remove_request;dummy=InputBatch.make_dummy.__func__
    def add_request(self,req_id,prompt_len,all_token_ids,num_computed_tokens,max_tokens):
        # The initial prompt remains the identity across decode/resumption.
        key=content_key(all_token_ids[:prompt_len]);entry=(key,request_base(key))
        old=_REQUESTS.get(req_id)
        if old is not None and old!=entry:raise RuntimeError('request identity changed while active')
        result=add(self,req_id,prompt_len,all_token_ids,num_computed_tokens,max_tokens)
        _REQUESTS[req_id]=entry
        return result
    def remove_request(self,req_id):
        result=remove(self,req_id);_REQUESTS.pop(req_id,None);return result
    def make_dummy(cls,*args,**kwargs):
        batch=dummy(cls,*args,**kwargs);_DUMMY.clear()
        for i,req in enumerate(batch.req_ids):
            key=hashlib.sha256(canonical({'schema':'minimax31-static-dummy-v1','request_ordinal':i})).hexdigest();_DUMMY[req]=(key,request_base(key))
        return batch
    RequestState.add_request=add_request;RequestState.remove_request=remove_request;InputBatch.make_dummy=classmethod(make_dummy)
    _HOOKS=(add_request,remove_request,make_dummy)

def _set_forward_layout(batch):
    global _LAYOUT
    rows=int(batch.num_tokens);padded=int(batch.num_tokens_after_padding);lengths=[int(x) for x in batch.num_scheduled_tokens.tolist()];reqs=list(batch.req_ids)
    if len(lengths)!=len(reqs) or sum(lengths)!=rows or padded<rows:raise ValueError('static PCP local row layout mismatch')
    bases=np.zeros(padded,dtype=np.uint64);keys=[];offset=0;segments=[]
    for req,n,start in zip(reqs,lengths,batch.num_computed_tokens_np.tolist()):
        entry=_REQUESTS.get(req,_DUMMY.get(req))
        if entry is None:raise RuntimeError('static selector has no content-bound identity for '+req)
        key,base=entry;bases[offset:offset+n]=base;segments.append((offset,offset+n,key,int(start)));keys.append(key);offset+=n
    _LAYOUT={'rows':rows,'padded':padded,'base':torch.from_numpy(bases).to(device=batch.positions.device),'positions':batch.positions,'segments':segments,'is_dummy':all(req in _DUMMY for req in reqs),'thread':threading.get_ident(),'next_layer':3,'completed':False}

def set_forward_layout(batch):
    if not _FRAME_LOCK.acquire(blocking=False):raise RuntimeError("static routing supports only one in-flight model forward")
    try:_set_forward_layout(batch)
    except BaseException:
        _FRAME_LOCK.release()
        raise

def complete_layer(layer):
    if (_LAYOUT is None or _LAYOUT["completed"] or _LAYOUT["thread"]!=threading.get_ident()
            or layer!=_LAYOUT["next_layer"]):raise RuntimeError("static routed layer/frame order changed")
    _LAYOUT["next_layer"]+=1
    if layer==_LAST_LAYER:
        _LAYOUT["completed"]=True
        _FRAME_LOCK.release()

def layout(rows):
    if _LAYOUT is None or rows!=_LAYOUT['padded']:raise RuntimeError('static selector missing matching PCP token layout')
    if _LAYOUT['completed'] or _LAYOUT['thread']!=threading.get_ident():raise RuntimeError('static frame is completed or belongs to another in-flight caller')
    return _LAYOUT
