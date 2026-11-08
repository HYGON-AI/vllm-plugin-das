"""Opt-in bounded HYV4-VL native MRoPE/PCP diagnostic, never timed serving.

Extends the isolated calibration extension. Only an explicit diagnostic start
installs a prepare_inputs wrapper. It calls the saved method exactly once.
"""
import asyncio
import hashlib
import json
from pathlib import Path
from types import MethodType

import torch
from build_vl_maps import MODEL_KEY, output_path, require
from vl_calibration_probe import WorkerExtension as CalibrationWorkerExtension
from vl_calibration_probe import CalibrationMiddleware


def tensor_metadata(value):
    if not isinstance(value,torch.Tensor):return None
    return {'shape':list(value.shape),'stride':list(value.stride()),'dtype':str(value.dtype),
            'device':str(value.device),'data_ptr':value.data_ptr(),
            'storage_ptr':value.untyped_storage().data_ptr(),
            'storage_bytes':value.untyped_storage().nbytes(),'storage_offset':value.storage_offset()}


def cpu_copy(value):
    require(isinstance(value,torch.Tensor),'diagnostic expected native tensor')
    return value.detach().cpu().clone()


def position_evidence(value):
    copied=cpu_copy(value)
    return {**tensor_metadata(value),'values':copied.tolist(),
            'sha256':hashlib.sha256(copied.contiguous().numpy().tobytes()).hexdigest()}


def plain(value):
    if value is None or type(value) in (str,bool,int,float):return value
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    if hasattr(value,'tolist'):return value.tolist()
    return str(type(value).__name__)


async def all_core_rpc(engine_client,method,args):
    core=engine_client.engine_core
    engines=list(core.core_engines)
    require(bool(engines),'no managed native engine cores')
    groups=await asyncio.gather(*(core._call_utility_async(
        'collective_rpc',method,120,args,None,engine=engine) for engine in engines))
    return [worker for group in groups for worker in group]


def worker_ep_rank():
    try:
        from vllm.distributed import get_ep_group
    except ImportError:
        return None  # CPU contract tests have no native vLLM installation.
    return get_ep_group().rank_in_group


class WorkerExtension(CalibrationWorkerExtension):
    def hy4_vl_eplb_start(self):
        require(not self.hy4_vl_diag_info()['hook_installed'],'diagnostic and calibration must be separate')
        return super().hy4_vl_eplb_start()

    def hy4_vl_diag_info(self,include_records=False):
        # Status only: no tensor copies, CUDA queries, processing, or sync.
        session=getattr(self,'_hy4_vl_diag',None)
        result={'rank':worker_ep_rank(),'model_key':MODEL_KEY,'hook_installed':bool(session and session['hook_installed']),
                'captured_invocations':len(session['records']) if session else 0,
                'limit':session['limit'] if session else None,
                'exported':bool(session and session.get('exported')),
                'error':session.get('error') if session else None,
                'scope':'bounded prepare_inputs diagnostic; CPU copies only while capture active; never timing evidence'}
        if include_records:result['records']=session['records'] if session else []
        return result

    def _restore_vl_diag(self):
        session=getattr(self,'_hy4_vl_diag',None)
        if session and session['hook_installed']:
            state=session['state']
            if session['had_override']:state.__dict__['prepare_inputs']=session['override']
            else:state.__dict__.pop('prepare_inputs',None)
            session['hook_installed']=False
        return session

    def _diagnostic_budget(self):
        runner=self.model_runner
        budget=getattr(runner,'mm_budget',None)
        origin='existing runner.mm_budget'
        if budget is None:
            from vllm.multimodal.encoder_budget import MultiModalBudget
            budget=MultiModalBudget(runner.vllm_config,runner.mm_registry)
            origin='native MultiModalBudget constructed during explicit diagnostic; no encoder call'
        fields=('max_model_len','max_num_reqs','mm_limits','encoder_compute_budget','encoder_cache_size',
                'mm_max_toks_per_item','mm_max_items_per_prompt','mm_max_items_per_batch')
        return {'class':type(budget).__module__+'.'+type(budget).__name__,'origin':origin,
                **{field:plain(getattr(budget,field,None)) for field in fields}}

    def _diagnostic_buffers(self,state):
        rope=state.rope_state
        values={'_local_positions':getattr(state,'_local_positions',None),
                '_local_inputs_embeds':getattr(state,'_local_inputs_embeds',None),
                'rope_state.positions':getattr(rope,'positions',None),
                'rope_state.prefill_positions.gpu':getattr(getattr(rope,'prefill_positions',None),'gpu',None),
                'rope_state.prefill_delta.gpu':getattr(getattr(rope,'prefill_delta',None),'gpu',None),
                'encoder_runner.inputs_embeds':getattr(getattr(state,'encoder_runner',None),'inputs_embeds',None)}
        cache=getattr(self.model_runner,'encoder_cache',None)
        for key,value in getattr(cache,'encoder_outputs',{}).items():
            values['encoder_cache.encoder_outputs.'+str(key)]=value
        return {name:tensor_metadata(value) for name,value in values.items() if isinstance(value,torch.Tensor)}

    def _diagnostic_memory(self,state):
        device=torch.device(getattr(self.model_runner,'device',self.device))
        result={'device':str(device),'model_memory_usage':getattr(self.model_runner,'model_memory_usage',None),
                '_hcu_vl_persistent_memory_bytes':getattr(state,'_hcu_vl_persistent_memory_bytes',None),
                'torch_allocated_bytes':None,'torch_reserved_bytes':None,'torch_max_allocated_bytes':None}
        if device.type=='cuda':
            result.update(torch_allocated_bytes=torch.cuda.memory_allocated(device),
                torch_reserved_bytes=torch.cuda.memory_reserved(device),
                torch_max_allocated_bytes=torch.cuda.max_memory_allocated(device))
            try:
                free,total=torch.cuda.mem_get_info(device)
                result.update(device_free_bytes=free,device_total_bytes=total)
            except Exception as error:result['mem_get_info_error']=str(error)
        return result

    def _diagnostic_before(self,state,batch):
        dummy=bool(getattr(state,'_dummy_run',False))
        manager=getattr(state,'pcp_manager',None)
        global_batch=manager.get_global_batch(batch) if manager is not None and not dummy else batch
        result={'global_batch':global_batch,'dummy':dummy,
                'canonical_local':position_evidence(batch.positions),
                'canonical_global':position_evidence(global_batch.positions),
                'pcp_size':int(getattr(manager,'pcp_size',getattr(state,'_pcp_size',1))),
                'pcp_rank':getattr(manager,'pcp_rank',None)}
        if manager is not None and not dummy:
            indices,valid=manager.get_local_token_mapping(batch)
            result['indices']=cpu_copy(indices);result['valid']=cpu_copy(valid)
        else:
            result['indices']=torch.arange(batch.num_tokens_after_padding,dtype=torch.int64)
            padding=getattr(batch,'is_padding',None)
            result['valid']=~cpu_copy(padding) if padding is not None else torch.ones(batch.num_tokens_after_padding,dtype=torch.bool)
        return result

    def _diagnostic_record(self,state,batch,req_states,result,before,invocation):
        require(isinstance(result,dict) and isinstance(result.get('positions'),torch.Tensor),'native model positions missing')
        positions=result['positions']
        require(positions.ndim==2 and positions.shape[0]==3,'native model positions must have three axes')
        global_batch=before['global_batch'];rope=state.rope_state
        local_after=position_evidence(batch.positions);global_after=position_evidence(global_batch.positions)
        record={'invocation':invocation,'dummy_run':before['dummy'],'model_positions':position_evidence(positions),
            'canonical_local_before':before['canonical_local'],'canonical_local_after':local_after,
            'canonical_global_before':before['canonical_global'],'canonical_global_after':global_after,
            'canonical_local_unchanged':before['canonical_local']==local_after,
            'canonical_global_unchanged':before['canonical_global']==global_after,
            'persistent_buffers':self._diagnostic_buffers(state),'memory':self._diagnostic_memory(state),
            'requests':[],'native_rope':{'num_dims':getattr(rope,'num_dims',None),'has_delta':getattr(rope,'has_delta',None)}}
        indices=before['indices'];valid=before['valid']
        record['pcp']={'rank':before['pcp_rank'],'size':before['pcp_size'],
            'local_num_tokens':int(batch.num_tokens),'local_num_tokens_after_padding':int(batch.num_tokens_after_padding),
            'global_num_tokens':int(global_batch.num_tokens),'global_num_tokens_after_padding':int(global_batch.num_tokens_after_padding),
            'global_indices':indices.tolist(),'valid_tokens':valid.tolist(),
            'selected_positions_equal_native':None,'canonical_positions_equal_global_selection':None,
            'invalid_local_model_positions_zero':None}
        if before['dummy']:
            record['scope']='dummy invocation; no real MRoPE/PCP correctness assertion'
            return record
        count=int(global_batch.num_tokens_after_padding)
        require(isinstance(getattr(rope,'positions',None),torch.Tensor) and rope.positions.shape[1]>=count,
                'native rope buffer unavailable or too short')
        # Native getter is read-only slicing (inspected pinned rope.py). The
        # original prepare_inputs has already prepared the global buffer once.
        native=rope.get_positions(count)
        require(native.ndim==2 and native.shape==(3,count),'native global rope shape mismatch')
        global_positions=cpu_copy(native);local_positions=cpu_copy(positions)
        record['native_global_positions']=position_evidence(native)
        require(indices.dtype==torch.int64 and valid.dtype==torch.bool and indices.shape==valid.shape,
                'native PCP index/valid mapping has incorrect types/shapes')
        require(indices.numel()==local_positions.shape[1],'PCP index count differs from model positions')
        require(bool(((indices>=0)&(indices<count)).all()),'native PCP indices outside global rope buffer')
        expected=global_positions.index_select(1,indices)
        expected[:,~valid]=0
        canonical_global=torch.tensor(before['canonical_global']['values'])
        canonical_local=torch.tensor(before['canonical_local']['values'])
        canonical_expected=canonical_global.index_select(0,indices)
        # Canonical PCP padding is independently reported; comparison is only
        # over model-valid tokens because KV padding can have its own sentinel.
        record['pcp'].update(selected_positions_equal_native=torch.equal(expected,local_positions),
            invalid_local_model_positions_zero=bool((local_positions[:,~valid]==0).all()),
            canonical_positions_equal_global_selection=torch.equal(canonical_expected[valid],canonical_local[valid]))
        n=int(global_batch.num_reqs)
        mapping=cpu_copy(global_batch.idx_mapping)[:n].tolist()
        starts=cpu_copy(global_batch.query_start_loc)[:n+1].tolist()
        req_ids=list(getattr(global_batch,'req_ids',[]))
        for row,slot in enumerate(mapping):
            require(type(slot) is int and slot>=0,'request-state slot invalid')
            prefill=int(cpu_copy(req_states.prefill_len.gpu[slot:slot+1])[0])
            computed=int(cpu_copy(req_states.num_computed_tokens.gpu[slot:slot+1])[0])
            start,end=int(starts[row]),int(starts[row+1])
            require(0<=start<=end<=count,'global query offsets out of range')
            info={'batch_row':row,'request_id':str(req_ids[row]) if row<len(req_ids) else None,
                'request_state_slot':slot,'prefill_len':prefill,'computed_tokens':computed,
                'global_query_start':start,'global_query_end':end,
                'prefill_chunk_equal_staged_fullprompt':None}
            delta=getattr(rope,'prefill_delta',None)
            if delta is not None:
                if hasattr(delta,'np'):info['native_delta_host']=plain(delta.np[slot])
                if hasattr(delta,'gpu'):info['native_delta_device']=int(cpu_copy(delta.gpu[slot:slot+1])[0])
            # The staged full-prompt native positions are read only. Compare
            # the current prefill chunk, not a recomputed synthetic MRoPE rule.
            length=max(0,min(end-start,prefill-computed))
            if length and callable(getattr(rope,'read_prefill_positions',None)):
                staged=rope.read_prefill_positions(slot,prefill)[:,computed:computed+length]
                staged_cpu=cpu_copy(staged)
                info.update(staged_fullprompt_window_start=computed,staged_fullprompt_window_end=computed+length,
                    staged_fullprompt_positions=staged_cpu.tolist(),
                    prefill_chunk_equal_staged_fullprompt=torch.equal(staged_cpu,global_positions[:,start:start+length]))
            record['requests'].append(info)
        record['scope']='real native global preparation once; exact PCP gather and staged full-prompt prefill-window comparisons; canonical KV positions observed before/after'
        return record

    def hy4_vl_diag_start(self,limit=2):
        require(type(limit) is int and 1<=limit<=8,'diagnostic limit must be integer 1..8')
        require(not getattr(self,'_hy4_vl_capture',None),'diagnostic and calibration must be separate')
        previous=getattr(self,'_hy4_vl_diag',None)
        require(previous is None or previous.get('exported'),'stop/export previous diagnostic before starting another')
        self._vl_model()
        state=getattr(self.model_runner,'model_state',None)
        require(state is not None and callable(getattr(state,'prepare_inputs',None)),'MRV2 native model_state.prepare_inputs required')
        require(getattr(state,'rope_state',None) is not None,'native three-axis rope state required')
        session={'state':state,'original':state.prepare_inputs,'had_override':'prepare_inputs' in state.__dict__,
            'override':state.__dict__.get('prepare_inputs'),'hook_installed':False,'records':[],
            'limit':limit,'exported':False,'error':None}
        self._hy4_vl_diag=session
        try:
            session['hook_installed']=True
            session['budget']=self._diagnostic_budget()
            def prepare_inputs(state,input_batch,req_states,*args,**kwargs):
                try:
                    before=self._diagnostic_before(state,input_batch)
                    result=session['original'](input_batch,req_states,*args,**kwargs)
                    record=self._diagnostic_record(state,input_batch,req_states,result,before,len(session['records'])+1)
                    session['records'].append(record)
                    if len(session['records'])>=session['limit']:self._restore_vl_diag()
                    return result
                except BaseException as error:
                    session['error']=type(error).__name__+': '+str(error)
                    self._restore_vl_diag();raise
            state.prepare_inputs=MethodType(prepare_inputs,state)
        except BaseException as error:
            session['error']=type(error).__name__+': '+str(error)
            self._restore_vl_diag();raise
        return self.hy4_vl_diag_info()

    def hy4_vl_diag_stop(self,output_dir):
        session=self._restore_vl_diag()
        require(session is not None,'diagnostic not started')
        require(not session['exported'],'diagnostic already exported')
        directory=output_path(output_dir)
        from vllm.distributed import get_ep_group
        rank=get_ep_group().rank_in_group
        payload={'rank':rank,'model_key':MODEL_KEY,'limit':session['limit'],'records':session['records'],
            'budget':session.get('budget'),'error':session.get('error'),'hook_installed':False,
            'scope':'explicit bounded native-position diagnostic; CPU copies/synchronization allowed only during capture; no timing or GPU correctness claim from CPU tests',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        raw=json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
        directory.mkdir(parents=True,exist_ok=True);path=directory/f'diag-rank{rank}.json'
        with path.open('xb') as stream:stream.write(raw)
        session['exported']=True
        return {'rank':rank,'file':str(path),'sha256':hashlib.sha256(raw).hexdigest(),
                'captured_invocations':len(session['records']),'hook_installed':False,'error':session.get('error')}


class DiagnosticMiddleware(CalibrationMiddleware):
    async def __call__(self,scope,receive,send):
        prefix='/hy4/vl-diag/'
        if scope['type']!='http' or not scope.get('path','').startswith(prefix):
            return await super().__call__(scope,receive,send)
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
            args=(body['output_dir'],) if action=='stop' else ((body.get('limit',2),) if action=='start' else (bool(body.get('include_records',False)),))
            result=await all_core_rpc(request.app.state.engine_client,'hy4_vl_diag_'+action,args)
            await JSONResponse({'workers':result})(scope,receive,send)
        except Exception as error:
            if action=='start':
                try:await all_core_rpc(request.app.state.engine_client,'hy4_vl_diag_stop',('',))
                except Exception:pass
            await JSONResponse({'error':str(error)},status_code=400)(scope,receive,send)

# Conventional middleware name for the same launch flag shape as calibration.
Middleware=DiagnosticMiddleware
