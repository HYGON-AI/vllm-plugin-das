"""Explicit idle HYV4-VL static-placement inspection; never installs hooks.

Weight fingerprints cover 32 fixed flattened positions per expert. Both
channel-scale tensors are hashed in full after native loading/processing.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from build_vl_maps import MODEL_KEY, require, validate_map
from vl_diagnostic_probe import WorkerExtension as DiagnosticWorkerExtension
from vl_diagnostic_probe import DiagnosticMiddleware, all_core_rpc, tensor_metadata

TENSORS=('w13_weight','w2_weight','w13_weight_scale','w2_weight_scale')
FLAGS=('HY4_EXPERIMENTAL_STATIC_R16','VLLM_HCU_STATIC_HT_W4A8_COUNT_SCRATCH',
       'VLLM_HCU_STATIC_HT_W4A8_COUNT_DIAGNOSTIC','VLLM_USE_V1','VLLM_USE_V2_MODEL_RUNNER',
       'CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def fixed_sample_indices(width):
    require(type(width) is int and width>0,'empty expert tensor')
    count=min(32,width)
    return [0] if count==1 else [(i*(width-1))//(count-1) for i in range(count)]


def _sample_expert(tensor,local_id,indices):
    import torch
    expert=tensor.detach()[local_id]
    # Advanced indexing reads only32 elements even for transposed/noncontiguous
    # expert storage; flatten/reshape could copy an entire huge weight tensor.
    coordinates=[]
    for axis,size in enumerate(expert.shape):
        trailing=math.prod(expert.shape[axis+1:])
        coordinates.append(torch.tensor([(flat//trailing)%size for flat in indices],
                          dtype=torch.int64,device=expert.device))
    return expert[tuple(coordinates)].cpu().contiguous()


def fingerprint_expert(owner,local_id):
    import torch
    digest=hashlib.sha256();components={}
    for name in TENSORS:
        tensor=getattr(owner,name,None)
        require(isinstance(tensor,torch.Tensor) and tensor.ndim>=2,'missing/invalid '+name)
        require(type(local_id) is int and 0<=local_id<tensor.shape[0],'local expert ID outside tensor')
        shape=list(tensor.shape[1:]);width=math.prod(shape)
        require(width>0,'empty '+name)
        full=name.endswith('_scale')
        indices=None if full else fixed_sample_indices(width)
        selected=tensor.detach()[local_id].cpu().contiguous() if full else _sample_expert(tensor,local_id,indices)
        header={'tensor':name,'dtype':str(tensor.dtype),'expert_shape':shape,
                'mode':'full' if full else 'fixed_floor_linspace32',
                'flat_indices':indices,'selected_elements':width if full else len(indices)}
        encoded=json.dumps(header,sort_keys=True,separators=(',',':')).encode()
        raw=selected.reshape(-1).view(torch.uint8).numpy().tobytes()
        part=hashlib.sha256(len(encoded).to_bytes(8,'big')+encoded+raw).hexdigest()
        digest.update(name.encode()+b'\x00'+bytes.fromhex(part))
        components[name]={**header,'sha256':part,'hashed_bytes':len(raw)}
    return {'sha256':digest.hexdigest(),'components':components}


def _integer_tensor(value,name):
    import torch
    require(isinstance(value,torch.Tensor) and value.dtype in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8),
            name+' must be an integer tensor')
    return value.detach().cpu()


def _validate_topology(runner,rank,world_size):
    require(type(rank) is int and 0<=rank<16 and type(world_size) is int and world_size==16,'exactly16 EP ranks required')
    config=getattr(runner,'vllm_config',None)
    require(config is not None and getattr(config,'speculative_config',None) is None,'noMTP config required')
    require(getattr(runner,'speculator',None) is None and getattr(runner,'drafter',None) is None,'draft model present')
    require(getattr(runner,'num_speculative_steps',0)==0,'speculative steps present')
    model_config=getattr(config,'model_config',None)
    require(tuple(getattr(model_config,'architectures',()))==(MODEL_KEY,),'exact singleton VL architecture required')
    require(getattr(model_config,'enforce_eager',None) is True,'staticP inspector requires eager execution')
    parallel=getattr(config,'parallel_config',None)
    expected={'tensor_parallel_size':1,'prefill_context_parallel_size':16,'data_parallel_size':1,
              'decode_context_parallel_size':1,'pipeline_parallel_size':1}
    for name,value in expected.items():
        actual=getattr(parallel,name,None)
        require(type(actual) is int and actual==value,'wrong staticP topology '+name)
    require(getattr(parallel,'enable_eplb',None) is True and getattr(parallel,'enable_expert_parallel',None) is True,
            'staticP EPLB+EP must be enabled')
    require(getattr(parallel,'all2all_backend',None)=='deepep_high_throughput','fixed HT backend required')
    return {**expected,'ep_world_size':world_size,'all2all_backend':parallel.all2all_backend,
            'enable_eplb':True,'enable_expert_parallel':True,'enforce_eager':True,
            'moe_backend':getattr(getattr(config,'kernel_config',None),'moe_backend',None),
            'speculative_config':None,'num_speculative_steps':0}


def inspect_runner(runner,*,rank,world_size,samples=False,calibration_active=False,diagnostic_active=False):
    import torch
    require(type(samples) is bool,'samples must be boolean')
    require(not calibration_active and not diagnostic_active,'capture hooks must be stopped before idle inspection')
    require(getattr(runner,'execute_model_state',None) is None,'runner has an in-flight execute_model_state')
    topology=_validate_topology(runner,rank,world_size)
    model=runner.model
    require(type(model).__name__==MODEL_KEY,'exact VL model required')
    layers=tuple(getattr(model,'moe_layers',()))
    require(len(layers)==77 and len({id(x) for x in layers})==77,'exactly77 distinct MoE rows required')
    plan=getattr(model,'_vllm_hcu_static_eplb_plan',None)
    require(plan is not None,'bound static plan required; D inspector unsupported')
    require(getattr(plan,'model_key',None)==MODEL_KEY,'wrong static plan model key')
    physical=getattr(plan,'num_physical_experts',None)
    require(type(physical) is int and physical in (256,272),'physical count must be256 or272')
    require(type(plan.num_logical_experts) is int and plan.num_logical_experts==256,'wrong plan logical count')
    require(type(plan.num_redundant_experts) is int and plan.num_redundant_experts==physical-256,'wrong plan redundancy')
    for name,value in [('num_logical_experts',256),('num_physical_experts',physical),('num_redundant_experts',physical-256)]:
        require(type(getattr(model,name,None)) is int and getattr(model,name)==value,'model/plan count mismatch '+name)
    rows=[list(plan.layer_map(i)) for i in range(77)]
    validate_map(rows,physical)
    require(file_sha256(plan.source_path)==plan.source_sha256,'live plan source SHA256 mismatch')
    local=physical//16;layer_records=[];fingerprints=[];owners=set();routers=set()
    for index,(layer,row) in enumerate(zip(layers,rows)):
        owner=getattr(layer,'routed_experts',None);router=getattr(layer,'router',None)
        require(owner is not None and id(owner) not in owners,'missing/duplicate routed-expert owner');owners.add(id(owner))
        require(router is not None and id(router) not in routers,'missing/duplicate router');routers.add(id(router))
        require('simulat' not in type(router).__name__.lower(),'simulated router cannot inspect real execution')
        require(getattr(router,'capture_fn',None) is None,'router capture callback still active')
        require(type(getattr(owner,'local_num_experts',None)) is int and owner.local_num_experts==local,'wrong local expert count')
        owner_row=list(getattr(owner,'_vllm_hcu_static_eplb_row',()))
        require(owner_row==row and all(type(e) is int for e in owner_row),'owner static row differs from plan')
        moe_config=getattr(owner,'moe_config',None)
        require(getattr(moe_config,'num_logical_experts',None)==256 and getattr(moe_config,'num_experts',None)==physical,
                'owner expert metadata differs from plan')
        state=getattr(router,'eplb_state',None)
        counts=_integer_tensor(getattr(state,'logical_replica_count',None),'logical_replica_count')
        expected=torch.bincount(torch.tensor(row,dtype=torch.int64),minlength=256)
        require(counts.shape==(256,) and torch.equal(counts.long(),expected),'replica counts differ from plan')
        inverse=_integer_tensor(getattr(state,'logical_to_physical_map',None),'logical_to_physical_map')
        require(inverse.ndim==2 and inverse.shape[0]==256,'wrong inverse-map shape')
        require(bool(((inverse>=-1)&(inverse<physical)).all()),'invalid inverse-map ID')
        for logical in range(256):
            actual=inverse[logical][inverse[logical]>=0].tolist()
            wanted=[p for p,e in enumerate(row) if e==logical]
            require(len(actual)==len(set(actual)) and set(actual)==set(wanted),'inverse-map replica set differs from plan')
        tensors={}
        for name in TENSORS:
            value=getattr(owner,name,None)
            require(isinstance(value,torch.Tensor) and value.ndim>=2 and value.shape[0]==local and value.numel()>0,
                    'wrong weight/scale leading dimension '+name)
            require(value.device.type!='meta','unmaterialized weight/scale '+name)
            tensors[name]=tensor_metadata(value)
        extras={}
        for name,value in vars(owner).items():
            if isinstance(value,torch.Tensor) and any(word in name.lower() for word in ('packed','cache')):
                extras[name]=tensor_metadata(value)
        layer_records.append({'layer':index,'router_class':type(router).__module__+'.'+type(router).__name__,
            'owner_class':type(owner).__module__+'.'+type(owner).__name__,'local_experts':local,
            'row_sha256':hashlib.sha256(json.dumps(row,separators=(',',':')).encode()).hexdigest(),
            'replica_counts':counts.tolist(),'inverse_replica_sets_verified':True,'tensors':tensors,'optional_storage':extras})
        if samples:
            for local_id in range(local):
                physical_id=rank*local+local_id
                fingerprints.append({'model':MODEL_KEY,'layer':index,'local_expert':local_id,
                    'physical':physical_id,'logical':row[physical_id],**fingerprint_expert(owner,local_id)})
    return {'rank':rank,'ep_world_size':world_size,'runner':type(runner).__module__+'.'+type(runner).__name__,
        'model_state_class':type(getattr(runner,'model_state',None)).__module__+'.'+type(getattr(runner,'model_state',None)).__name__,
        'topology':topology,'no_mtp':True,'calibration_active':False,'diagnostic_hook_installed':False,
        'models':[{'model':MODEL_KEY,'model_key':plan.model_key,'layers':77,'logical':256,'physical':physical,
            'redundant':physical-256,'map_path':plan.source_path,'map_sha256':plan.source_sha256,
            'local_experts':local,'bindings_checked':True}],
        'layers':layer_records,'fingerprints':fingerprints,'samples':samples,
        'flags':{name:os.environ.get(name) for name in FLAGS},
        'scope':'idle read-only inspection; sampled weights, complete per-expert channel scales; unsampled weight bytes are not proven; caller must prevent concurrent inference'}


def loaded_sources(runner):
    names={'vllm_hcu','vllm_hcu.hcu_ops','vllm_hcu.v1.hcu_model_runner_v2',
        'vl_inspection_probe','vl_diagnostic_probe','vl_calibration_probe','build_vl_maps',
        type(runner).__module__,type(runner.model).__module__,type(runner.model_state).__module__}
    names.update(name for name,module in tuple(sys.modules.items())
                 if name.startswith('vllm_hcu') and str(getattr(module,'__file__','')).endswith(('.so','.pyd','.dll')))
    result={}
    for name in sorted(names):
        module=sys.modules.get(name);source=getattr(module,'__file__',None)
        if source and Path(source).is_file():
            path=Path(source).resolve();result[name]={'path':str(path),'sha256':file_sha256(path)}
        else:result[name]={'path':source,'unavailable':True}
    # Module may have been loaded using an alternate extension import name.
    result['inspection_source']={'path':str(Path(__file__).resolve()),'sha256':file_sha256(__file__)}
    return result


class WorkerExtension(DiagnosticWorkerExtension):
    def hy4_vl_inspect_info(self,samples=False):
        from vllm.distributed import get_ep_group
        self._vl_model()
        group=get_ep_group()
        session=getattr(self,'_hy4_vl_diag',None)
        result=inspect_runner(self.model_runner,rank=group.rank_in_group,world_size=group.world_size,
            samples=samples,calibration_active=bool(getattr(self,'_hy4_vl_capture',None)),
            diagnostic_active=bool(session and session.get('hook_installed')))
        result['files']=loaded_sources(self.model_runner)
        return result


class InspectionMiddleware(DiagnosticMiddleware):
    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope.get('path')!='/hy4/vl-inspect/info':
            return await super().__call__(scope,receive,send)
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        if (scope.get('client') or ('',))[0] not in ('127.0.0.1','::1'):
            return await JSONResponse({'error':'loopback only'},status_code=403)(scope,receive,send)
        if scope['method']!='POST':
            return await JSONResponse({'error':'POST only'},status_code=400)(scope,receive,send)
        request=Request(scope,receive=receive)
        try:
            body=await request.json();samples=body.get('samples',False)
            require(type(samples) is bool,'samples must be boolean')
            workers=await all_core_rpc(request.app.state.engine_client,'hy4_vl_inspect_info',(samples,))
            require(len(workers)==16 and {w['rank'] for w in workers}==set(range(16)),
                    'inspection requires16 unique EP-rank responses')
            signatures={(w['models'][0]['model_key'],w['models'][0]['map_sha256'],w['models'][0]['physical']) for w in workers}
            require(len(signatures)==1,'cross-rank plan fingerprint mismatch')
            await JSONResponse({'workers':workers,'all16_ranks_verified':True})(scope,receive,send)
        except Exception as error:
            await JSONResponse({'error':str(error)},status_code=400)(scope,receive,send)

Middleware=InspectionMiddleware
