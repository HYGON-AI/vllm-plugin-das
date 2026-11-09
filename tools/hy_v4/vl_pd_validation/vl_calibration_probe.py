"""Temporary opt-in real HYV4-VL routing recorder; never enable in timing runs.

Use WorkerExtension plus CalibrationMiddleware, then explicit loopback POST
/hy4/vl-eplb/start and /stop. No generic EPLB recorder lifecycle is changed.
"""
import hashlib
import json
from pathlib import Path
from types import MethodType

from build_vl_maps import MODEL_KEY, require, validate_capture, validate_map, output_path


class WorkerExtension:
    def _vl_model(self):
        runner=self.model_runner
        for name in ('speculator','drafter'):
            require(getattr(runner,name,None) is None,'draft/speculator present; noMTP required')
        for owner in (runner,getattr(runner,'vllm_config',None),getattr(self,'vllm_config',None)):
            require(getattr(owner,'speculative_config',None) is None,'speculative config present; noMTP required')
        model=runner.model
        require(type(model).__name__==MODEL_KEY,'exact HYV4VLForConditionalGeneration model required')
        require(getattr(model,'num_logical_experts',None)==256,'expected 256 logical experts')
        layers=tuple(getattr(model,'moe_layers',()))
        require(len(layers)==77,'expected exactly 77 MoE rows')
        require(len({id(layer) for layer in layers})==77,'duplicate MoE layer objects')
        return model,layers

    def hy4_vl_eplb_info(self):
        model,layers=self._vl_model()
        plan=getattr(model,'_vllm_hcu_static_eplb_plan',None)
        return {'model_key':MODEL_KEY,'moe_layers':len(layers),'logical_experts':256,
                'physical_experts':len(plan.layer_map(0)) if plan is not None else None,
                'routers':[type(getattr(layer,'router',None)).__name__ for layer in layers],
                'capture_active':bool(getattr(self,'_hy4_vl_capture',None)),
                'scope':'executed real top-k including PCP padding; -1 slots excluded'}

    def _restore_vl_capture(self):
        records=getattr(self,'_hy4_vl_capture',None)
        self._hy4_vl_capture=None
        for record in records or ():
            router=record['router']
            # Restore exact saved instance values. Delete our override where the
            # original mapper came from the class descriptor.
            if record['had_mapper_override']:
                router.__dict__['_apply_eplb_mapping']=record['mapper_override']
            else:
                router.__dict__.pop('_apply_eplb_mapping',None)
            router.capture_fn=record['original_capture_fn']
        return records

    def hy4_vl_eplb_start(self):
        import torch
        require(not getattr(self,'_hy4_vl_capture',None),'capture already active')
        model,layers=self._vl_model()
        plan=getattr(model,'_vllm_hcu_static_eplb_plan',None)
        require(plan is not None,'bound static bootstrap/candidate map required')
        rows=[list(plan.layer_map(i)) for i in range(77)]
        validate_map(rows,len(rows[0]))
        records=[];routers=set()
        # Validate every router before installing the first callback.
        for index,(layer,row) in enumerate(zip(layers,rows)):
            router=getattr(layer,'router',None)
            require(router is not None and callable(getattr(router,'set_capture_fn',None)),'router capture API missing')
            require(id(router) not in routers,'duplicate router object');routers.add(id(router))
            require('simulat' not in type(router).__name__.lower(),'simulated routing cannot calibrate')
            require(getattr(router,'capture_fn',None) is None,'existing capture callback conflict')
            require(getattr(router,'top_k',None)==8,'router top-k must be 8')
            require(callable(getattr(router,'_apply_eplb_mapping',None)),'router mapping API missing')
            records.append({'model_key':MODEL_KEY,'layer_index':index,'num_logical_experts':256,'top_k':8,
                'calls':0,'mapping_calls':0,'rows':0,'ignored_invalid_slots':0,'physical_ignored_invalid_slots':0,
                'physical_to_logical':row,'histogram':torch.zeros(256,dtype=torch.int64,device=self.device),
                'physical_histogram':torch.zeros(len(row),dtype=torch.int64,device=self.device),
                'router':router,'original_capture_fn':router.capture_fn,'original_mapper':router._apply_eplb_mapping,
                'had_mapper_override':'_apply_eplb_mapping' in router.__dict__,
                'mapper_override':router.__dict__.get('_apply_eplb_mapping'),
                'map_tensor':torch.tensor(row,dtype=torch.int64,device=self.device),'pending':None})
        self._hy4_vl_capture=records
        def check_ids(ids,width):
            require(isinstance(ids,torch.Tensor) and ids.ndim==2 and ids.shape[1]==8,'expected [rows,8] top-k IDs')
            require(ids.dtype in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8),'IDs must be integers')
            require(not bool(((ids < -1)|(ids >= width)).any().item()),'unexpected invalid ID; only -1 sentinel allowed')
        try:
            for record in records:
                def capture(ids,record=record):
                    try:
                        check_ids(ids,256)
                        require(record['pending'] is None,'logical capture without preceding mapping')
                        record['pending']=ids.detach().clone()
                        valid=ids>=0
                        record['histogram'].add_(torch.bincount(ids[valid].long(),minlength=256))
                        record['calls']+=1;record['rows']+=ids.shape[0]
                        record['ignored_invalid_slots']+=int((~valid).sum().item())
                    except BaseException:
                        self._restore_vl_capture();raise
                def apply_mapping(router,ids,record=record):
                    try:
                        # Clone before original mapper: native implementations may
                        # mutate the logical-ID input in place.
                        logical=ids.detach().clone()
                        pending=record['pending']
                        require(pending is not None and torch.equal(pending,logical),'mapper must follow matching logical callback')
                        mapped=record['original_mapper'](ids)
                        check_ids(mapped,len(record['physical_to_logical']))
                        require(mapped.shape==logical.shape,'mapping changed top-k shape')
                        valid=logical>=0
                        require(torch.equal(mapped>=0,valid),'logical/physical invalid-slot mismatch')
                        require(torch.equal(record['map_tensor'][mapped[valid].long()],logical[valid].long()),
                                'mapped physical IDs do not match selected logical experts')
                        record['physical_histogram'].add_(torch.bincount(mapped[valid].long(),minlength=len(record['physical_to_logical'])))
                        record['physical_ignored_invalid_slots']+=int((~valid).sum().item())
                        record['mapping_calls']+=1;record['pending']=None
                        return mapped
                    except BaseException:
                        self._restore_vl_capture();raise
                record['router'].set_capture_fn(capture)
                record['router']._apply_eplb_mapping=MethodType(apply_mapping,record['router'])
        except BaseException:
            self._restore_vl_capture();raise
        return {'layers':77,'model_key':MODEL_KEY,'physical_experts':len(rows[0]),
                'scope':'executed real top-k including PCP padding; -1 slots excluded',
                'warning':'calibration hooks active; disable before timed performance'}

    def hy4_vl_eplb_stop(self,output_dir):
        # Restore first, even if output validation/serialization later fails.
        records=self._restore_vl_capture()
        require(bool(records),'capture not active')
        directory=output_path(output_dir)
        from vllm.distributed import get_ep_group
        rank=get_ep_group().rank_in_group
        public=('model_key','layer_index','num_logical_experts','top_k','calls','mapping_calls','rows',
                'ignored_invalid_slots','physical_ignored_invalid_slots','physical_to_logical')
        rows=[]
        for record in records:
            require(record['pending'] is None,'capture stopped with unmatched mapping')
            row={key:record[key] for key in public}
            row['histogram']=record['histogram'].cpu().tolist()
            row['physical_histogram']=record['physical_histogram'].cpu().tolist()
            row['physical_mapping_verified']=True
            rows.append(row)
        data={'rank':rank,'records':rows,'scope':'executed real top-k including PCP padding; -1 slots excluded',
              'invalid_slot_policy':'only -1 accepted and counted; other invalid IDs abort capture and restore hooks'}
        validate_capture(data)
        directory.mkdir(parents=True,exist_ok=True)
        path=directory/f'rank{rank}.json'
        raw=json.dumps(data,sort_keys=True,separators=(',',':')).encode()
        with path.open('xb') as stream:stream.write(raw)
        return {'rank':rank,'file':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'layers':77,
                'total_selected_ids':sum(sum(row['histogram']) for row in rows),
                'ignored_invalid_slots':sum(row['ignored_invalid_slots'] for row in rows),'capture_disabled':True}


class CalibrationMiddleware:
    def __init__(self,app):self.app=app
    async def __call__(self,scope,receive,send):
        prefix='/hy4/vl-eplb/'
        if scope['type']!='http' or not scope.get('path','').startswith(prefix):
            return await self.app(scope,receive,send)
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        if (scope.get('client') or ('',))[0] not in ('127.0.0.1','::1'):
            return await JSONResponse({'error':'loopback only'},status_code=403)(scope,receive,send)
        action=scope['path'][len(prefix):]
        if scope['method']!='POST' or action not in ('info','start','stop'):
            return await JSONResponse({'error':'POST info/start/stop only'},status_code=400)(scope,receive,send)
        request=Request(scope,receive=receive)
        try:
            body=await request.json()
            args=(body['output_dir'],) if action=='stop' else ()
            result=await request.app.state.engine_client.collective_rpc('hy4_vl_eplb_'+action,timeout=120,args=args)
            await JSONResponse({'workers':result})(scope,receive,send)
        except Exception as error:
            # A failed collective start may have succeeded on a subset of ranks.
            # Stop all workers; stop restores hooks before any output-path error.
            if action=='start':
                try:
                    await request.app.state.engine_client.collective_rpc('hy4_vl_eplb_stop',timeout=120,args=('',))
                except Exception:
                    pass
            await JSONResponse({'error':str(error)},status_code=400)(scope,receive,send)
