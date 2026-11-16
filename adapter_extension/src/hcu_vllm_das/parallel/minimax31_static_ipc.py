# SPDX-License-Identifier: Apache-2.0
"""Static publication envelope bound to the same native-key/IPC response."""
import hashlib,json,os,re
from pathlib import Path
import torch
from .minimax31_static_contract import from_model_config,sha,ABI
from .minimax31_static_experts import StaticRoutedExperts,parameter_samples
COMMAND='get_minimax31_static136_state_v1'

def metadata(t,kind):return {'kind':kind,'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),'storage_offset':t.storage_offset()}
def model_inventory(model):
    tensors={};aliases={};ids={};refs={}
    for kind,items in [('param',model.named_parameters(remove_duplicate=False)),('buffer',model.named_buffers(remove_duplicate=False))]:
        for name,t in items:
            if name in tensors or name in aliases:continue
            if id(t) in ids:aliases[name]=ids[id(t)];continue
            ids[id(t)]=name;tensors[name]=metadata(t,kind);refs[name]=t
    return tensors,aliases,refs

def version(t):
    try:return t._version
    except RuntimeError:return None

def freeze_publication(daemon):
    placement=from_model_config(daemon.model_config)
    if placement is None:return
    generation=os.environ.get('HCU_MINIMAX_STATIC_GENERATION','')
    if not re.fullmatch('[0-9a-f]{32}',generation):raise ValueError('static producer requires a fresh32-hex generation')
    tensors,aliases,refs=model_inventory(daemon.model);proofs={}
    for module in daemon.model.modules():
        if not isinstance(module,StaticRoutedExperts):continue
        lid=module._static_layer_id;proof=getattr(module,'_static_loading_proof',None)
        if not proof or proof['permanent_pack_count']!=1 or module._static_pack_count!=1 or proof['map_sha256']!=placement.map_sha256:raise RuntimeError('static publication lacks complete pack-once loading proof')
        expected=torch.tensor(placement.row(lid),dtype=torch.int32)
        if not torch.equal(module._static_physical_to_logical.cpu(),expected):raise RuntimeError('static physical map differs from loaded-slot contract')
        if parameter_samples(module)!=proof['packed_samples']:raise RuntimeError('static packed weights changed since loading receipt')
        if str(lid) in proofs:raise RuntimeError("duplicate static layer proof")
        proofs[str(lid)]=proof
    text=getattr(daemon.model_config.hf_config,'text_config',daemon.model_config.hf_config);layers=int(text.num_hidden_layers)
    if set(proofs)!=set(map(str,range(3,layers))):raise RuntimeError('static publication has missing or duplicate MoE layer proofs')
    manifest={'schema':'minimax31-static-publication-v1','generation':generation,'rank':daemon.global_rank,'gpu_uuid':daemon._partition.gpu_uuid,'map_sha256':placement.map_sha256,'weight_contract_sha256':daemon._partition.weight_contract_sha256,'loader_abi':ABI,'logical_experts':128,'physical_experts':136,'slots_per_rank':17,'tensors':tensors,'aliases':aliases,'layer_loading_proofs':proofs}
    daemon._static_manifest=manifest;daemon._static_manifest_sha256=sha(manifest)
    daemon._static_storage_snapshot={n:(id(t),t.data_ptr(),version(t)) for n,t in refs.items()}
    path=Path(os.environ['HCU_MINIMAX_STATIC_RECEIPTS'])/('rank%d_publication.json'%daemon.global_rank)
    with path.open('x') as f:json.dump({'manifest_sha256':daemon._static_manifest_sha256,'manifest':manifest},f,sort_keys=True);f.write('\n')

def publication_identity(daemon):
    if not hasattr(daemon,'_static_manifest'):return None
    m=daemon._static_manifest
    return {k:m[k] for k in ('generation','rank','gpu_uuid','map_sha256')}|{'manifest_sha256':daemon._static_manifest_sha256}

def assert_unchanged(daemon):
    tensors,aliases,refs=model_inventory(daemon.model);m=daemon._static_manifest
    if tensors!=m['tensors'] or aliases!=m['aliases'] or {n:(id(t),t.data_ptr(),version(t)) for n,t in refs.items()}!=daemon._static_storage_snapshot:raise RuntimeError('static producer storage or metadata changed after READY')
    for module in daemon.model.modules():
        if isinstance(module,StaticRoutedExperts) and parameter_samples(module)!=m['layer_loading_proofs'][str(module._static_layer_id)]['packed_samples']:raise RuntimeError('static producer weight sample changed after READY')

def _hash_bytes(value):return None if value is None else hashlib.sha256(bytes(value)).hexdigest()
def entry_metadata(entry):
    if entry.ipc_args is None:
        t=entry.cpu_tensor
        if t is None:raise ValueError('static entry has neither IPC nor CPU storage')
        out=metadata(t,entry.kind);out['cpu_bytes_sha256']=_hash_bytes(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes());return out
    a=entry.ipc_args
    if len(a)!=15:raise ValueError('unsupported native IPC reduction ABI')
    return {'kind':entry.kind,'shape':list(a[1]),'stride':list(a[2]),'storage_offset':a[3],'dtype':str(a[5]),'device':a[6],'storage_handle_sha256':_hash_bytes(a[7]),'storage_size_bytes':a[8],'storage_offset_bytes':a[9],'requires_grad':a[10],'ref_counter_handle_sha256':_hash_bytes(a[11]),'ref_counter_offset':a[12],'event_handle_sha256':_hash_bytes(a[13]),'event_sync_required':a[14]}

def response_digest(response):
    return sha({'entries':{n:entry_metadata(e) for n,e in response['entries'].items()},'aliases':response.get('aliases',{}),'attrs':response.get('attrs',{})})
class CaptureWire:
    def __init__(self):self.buffer=bytearray()
    def sendall(self,b):self.buffer.extend(b)
    def recv(self,n):
        data=bytes(self.buffer[:n]);del self.buffer[:n];return data

def serve_static(daemon,conn,request):
    from vllm.model_executor.model_loader.weight_cache.daemon import WeightCacheDaemon
    from vllm.model_executor.model_loader.weight_cache.protocol import send_msg,recv_msg
    expected=publication_identity(daemon)
    if expected is None or request.get('publication')!=expected:
        send_msg(conn,{'status':'mismatch','fields':['static_generation_or_publication']});return
    # All supplemental fields are checked before native export. The unchanged
    # native method independently checks every WeightCacheKey field.
    assert_unchanged(daemon)
    wire=CaptureWire();WeightCacheDaemon._handle_get_state(daemon,wire,request);response=recv_msg(wire)
    if wire.buffer:raise RuntimeError('native IPC emitted unexpected extra framing')
    if response.get('status')=='ok':response['static_publication']={'identity':expected,'manifest':daemon._static_manifest,'response_sha256':response_digest(response)}
    send_msg(conn,response)

def expected_publication(partition):
    path=os.environ.get('HCU_MINIMAX_STATIC_READY_MANIFEST')
    if not path:raise ValueError('static consumer requires a matching READY observation')
    ready=json.loads(Path(path).read_text());rows=ready.get('ranks',[])
    if len(rows)!=8 or sorted(row['rank'] for row in rows)!=list(range(8)):raise ValueError('static READY lacks8 ranks')
    row=next(x for x in rows if x['rank']==partition.ep_rank);identity=row.get('static_publication')
    if not isinstance(identity,dict) or set(identity)!={'generation','rank','gpu_uuid','map_sha256','manifest_sha256'}:raise ValueError('static READY omitted publication identity')
    if (identity['rank']!=partition.ep_rank or identity['gpu_uuid']!=partition.gpu_uuid or identity['map_sha256']!=partition.map_sha256 or not re.fullmatch('[0-9a-f]{32}',identity['generation'])):raise ValueError('static READY map/rank/device mismatch')
    if len({x['static_publication']['generation'] for x in rows})!=1 or len({x['static_publication']['map_sha256'] for x in rows})!=1:raise ValueError('mixed static READY generations or maps')
    return identity

def verify_response(response,expected,partition):
    envelope=response.get('static_publication')
    if not isinstance(envelope,dict) or set(envelope)!={'identity','manifest','response_sha256'}:raise ValueError('static IPC response lacks same-session publication envelope')
    manifest=envelope['manifest']
    if envelope['identity']!=expected or sha(manifest)!=expected['manifest_sha256'] or envelope['response_sha256']!=response_digest(response):raise ValueError('static IPC generation/manifest/handle binding mismatch')
    fields={'schema':'minimax31-static-publication-v1','generation':expected['generation'],'rank':partition.ep_rank,'gpu_uuid':partition.gpu_uuid,'map_sha256':partition.map_sha256,'weight_contract_sha256':partition.weight_contract_sha256,'loader_abi':ABI,'logical_experts':128,'physical_experts':136,'slots_per_rank':17}
    if any(manifest.get(k)!=v for k,v in fields.items()) or response.get('gpu_uuid')!=partition.gpu_uuid:raise ValueError('static IPC publication contract mismatch')
    if set(response['entries'])!=set(manifest['tensors']) or response.get('aliases',{})!=manifest['aliases']:raise ValueError('static IPC tensor inventory differs from publication')
    for name,entry in response['entries'].items():
        info=entry_metadata(entry)
        if any(info.get(k)!=v for k,v in manifest['tensors'][name].items()):raise ValueError('static IPC tensor metadata differs: '+name)
    return manifest

def verify_mapped_model(model,manifest):
    tensors,aliases,refs=model_inventory(model)
    for name,info in manifest['tensors'].items():
        if tensors.get(name)!=info:raise RuntimeError('mapped static tensor metadata differs: '+name)
    if any(aliases.get(k)!=v for k,v in manifest['aliases'].items()):raise RuntimeError('mapped static alias changed')
    seen=set()
    for module in model.modules():
        if not isinstance(module,StaticRoutedExperts):continue
        lid=str(module._static_layer_id);proof=manifest['layer_loading_proofs'].get(lid)
        if lid in seen or proof is None or module._static_pack_count!=0 or module._static_load_receipts:raise RuntimeError('static consumer repacked/reloaded or has duplicate layer')
        seen.add(lid)
        if tuple(proof['logical_owners'])!=module._static_local_owners or proof['map_sha256']!=module._static_placement.map_sha256:raise RuntimeError('mapped static expert owner mismatch')
        if parameter_samples(module)!=proof['packed_samples']:raise RuntimeError('mapped static weight/scale samples differ from same-generation producer')
        if module._static_physical_to_logical.cpu().tolist()!=list(module._static_placement.row(module._static_layer_id)):raise RuntimeError('mapped physical-to-logical table differs')
    if seen!=set(manifest['layer_loading_proofs']):raise RuntimeError('mapped static layer coverage mismatch')
    target=os.environ.get('HCU_MINIMAX_STATIC_CONSUMER_RECEIPTS')
    if not target:raise ValueError('static consumer requires a receipt directory')
    folder=Path(target);folder.mkdir(parents=True,exist_ok=True);path=folder/('rank%d_%d.json'%(manifest['rank'],os.getpid()))
    with path.open('x') as f:json.dump({'schema':'minimax31-static-consumer-map-proof-v1','generation':manifest['generation'],'map_sha256':manifest['map_sha256'],'manifest_sha256':sha(manifest),'rank':manifest['rank'],'gpu_uuid':manifest['gpu_uuid'],'layers':sorted(map(int,seen)),'slots_per_rank':17,'consumer_permanent_pack_count':0,'mapped_weight_scale_samples_match':True,'strict_native_and_partition_validation':True},f,sort_keys=True);f.write('\n')
