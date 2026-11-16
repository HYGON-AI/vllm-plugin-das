# SPDX-License-Identifier: Apache-2.0
"""Immutable physical capacity contract. HF/router logical cardinality is 128."""
from dataclasses import dataclass
import hashlib,json,re
from contextlib import contextmanager
from contextvars import ContextVar
_MODEL_SCOPE=ContextVar("minimax31_static_model_scope",default=None)
LAYERS=tuple(range(3,60));LOGICAL=128;PHYSICAL=136;LOCAL=17
SEED=0x4d33315043503831
SELECTOR={'version':'splitmix64-request256-position-layer-expert-v1','seed_u64':SEED,'byte_order':'little','overflow':'unsigned-mod-2^64','replica_order':'ascending-physical-id'}
ABI='minimax31-static136-hipc-loader-v1'
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
def sha(value):return hashlib.sha256(canonical(value)).hexdigest()
def sha_string(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None
@dataclass(frozen=True)
class StaticPlacement:
    rows:tuple
    map_sha256:str
    capture_sha256:str
    dataset_manifest_sha256:str
    precision_decision_sha256:str
    def row(self,layer):
        if type(layer) is not int or layer not in LAYERS:raise ValueError('static placement requires model layer3..59')
        return self.rows[layer-3]
    def local(self,layer,rank):
        if type(rank) is not int or not 0<=rank<8:raise ValueError('invalid static EP rank')
        return self.row(layer)[rank*LOCAL:(rank+1)*LOCAL]
def validate(value):
    fields={'schema','logical_experts','physical_experts','ep_size','slots_per_rank','model_layer_ids','physical_to_logical','map_sha256','selector','loader_abi','capture_sha256','dataset_manifest_sha256','precision_decision_sha256'}
    if not isinstance(value,dict) or set(value)!=fields or value.get('schema')!='minimax31-static-runtime-contract-v1':raise ValueError('invalid static placement contract schema')
    for k,v in [('logical_experts',128),('physical_experts',136),('ep_size',8),('slots_per_rank',17)]:
        if type(value.get(k)) is not int or value[k]!=v:raise ValueError('invalid static capacity: '+k)
    if value['model_layer_ids']!=list(LAYERS) or value['selector']!=SELECTOR or value['loader_abi']!=ABI:raise ValueError('static layer/selector/loader ABI changed')
    rows=value['physical_to_logical']
    if not isinstance(rows,list) or len(rows)!=57:raise ValueError('static map must contain all57 layers')
    for row in rows:
        if not isinstance(row,list) or len(row)!=136 or any(type(e) is not int or not 0<=e<128 for e in row) or set(row)!=set(range(128)):raise ValueError('static map must cover128 logical experts in136 physical slots')
    digest=sha({'model_layer_ids':list(LAYERS),'physical_to_logical':rows})
    if value['map_sha256']!=digest or not all(sha_string(value[k]) for k in ['map_sha256','capture_sha256','dataset_manifest_sha256','precision_decision_sha256']):raise ValueError('invalid static map/provenance digest')
    return StaticPlacement(tuple(tuple(r) for r in rows),digest,value['capture_sha256'],value['dataset_manifest_sha256'],value['precision_decision_sha256'])
def from_model_config(model_config):
    q=getattr(model_config.hf_config,'quantization_config',None)
    value=q.get('hcu_minimax_static_placement') if isinstance(q,dict) else None
    return None if value is None else validate(value)
@contextmanager
def model_construction_scope(model_config):
    placement=from_model_config(model_config)
    previous=_MODEL_SCOPE.get()
    if previous is not None and previous!=placement:raise RuntimeError("conflicting static model construction scope")
    token=_MODEL_SCOPE.set(placement)
    try:yield placement
    finally:_MODEL_SCOPE.reset(token)

def from_current():
    from vllm.config import get_current_vllm_config
    model_config=get_current_vllm_config().model_config
    direct=from_model_config(model_config)
    if direct is not None:return direct
    scoped=_MODEL_SCOPE.get()
    if scoped is not None:
        if getattr(model_config.hf_config,"architectures",None)!=["MiniMaxM3SparseForCausalLM"]:
            raise RuntimeError("static scope may be inherited only by the nested MiniMax text model")
        return scoped
    return None
def layer_id(prefix):
    m=re.search(r'(?:^|\.)layers\.(\d+)\.block_sparse_moe\.experts$',prefix)
    if m is None or int(m.group(1)) not in LAYERS:raise ValueError('invalid static routed-layer identity: '+prefix)
    return int(m.group(1))
def mix(x):
    mask=(1<<64)-1;x=(x+0x9e3779b97f4a7c15)&mask;x=((x^(x>>30))*0xbf58476d1ce4e5b9)&mask;x=((x^(x>>27))*0x94d049bb133111eb)&mask;return (x^(x>>31))&mask

def request_base(key):
    if not sha_string(key):raise ValueError('invalid content-derived request key')
    h=SEED;raw=bytes.fromhex(key)
    for i in range(0,32,8):h=mix(h^int.from_bytes(raw[i:i+8],'little'))
    return h
