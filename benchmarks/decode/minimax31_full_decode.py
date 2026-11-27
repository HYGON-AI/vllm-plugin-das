"""Offline real-weight DP8 decode harness. No HTTP server or fake GEMMs.

The load-only mode qualifies strict target/draft IPC and full-reference GPU
capacity before allocating the 1.2 TB mapped host history. It is not a decode
timing result. Full graph execution is added only after each backend is bound.
"""
import argparse
import json
import os
from pathlib import Path
import time
import sys


def device_index():
    return 0 if os.environ.get('M31_SINGLE_VISIBLE_DEVICE')=='1' else int(os.environ['LOCAL_RANK'])


def numa_spread():
    """Assign one NUMA node per GPU, minimizing actual PCI/NUMA distance."""
    import ctypes
    import itertools
    from hcu_vllm_das.parallel.minimax31_hisparse import loaded_hip_runtime
    rt=loaded_hip_runtime()
    rt.hipDeviceGetPCIBusId.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int]
    nodes=sorted(Path('/sys/devices/system/node').glob('node[0-9]*'),key=lambda p:int(p.name[4:]))
    if len(nodes)!=8:raise RuntimeError('node20 spread contract requires eight NUMA nodes')
    distances=[]
    # A single-visible worker uses the assignment fixed before site/HSA load.
    if os.environ.get('M31_SINGLE_VISIBLE_DEVICE')=='1':
        rank=int(os.environ['LOCAL_RANK']);node=int(os.environ['M31_NUMA_ASSIGNMENT'].split(',')[rank])
        print('MINIMAX_NUMA_SPREAD '+json.dumps({'rank':rank,'node':node,'early':True,
            'cpus':sorted(os.sched_getaffinity(0))}),flush=True)
        return
    for gpu in range(8):
        bus=ctypes.create_string_buffer(32)
        if rt.hipDeviceGetPCIBusId(bus,32,gpu):raise RuntimeError('cannot query GPU PCI ownership')
        local=int((Path('/sys/bus/pci/devices')/bus.value.decode()/'numa_node').read_text())
        if not 0<=local<8:raise RuntimeError('GPU has no explicit NUMA locality')
        distances.append([int(x) for x in (nodes[local]/'distance').read_text().split()])
    assignment=min(itertools.permutations(range(8)),key=lambda a:sum(distances[r][a[r]] for r in range(8)))
    rank=int(os.environ['LOCAL_RANK']);node=nodes[assignment[rank]]
    cpus=set()
    for field in (node/'cpulist').read_text().strip().split(','):
        ends=[int(x) for x in field.split('-')]
        cpus.update(range(ends[0],ends[-1]+1))
    cpus &= os.sched_getaffinity(0)
    if not cpus:raise RuntimeError('assigned NUMA node is outside allowed CPU affinity')
    os.sched_setaffinity(0,cpus)
    print('MINIMAX_NUMA_SPREAD '+json.dumps({'rank':rank,'assignment':assignment,'cpus':sorted(cpus)}),flush=True)



def _replay_decode_config():
    """Translate the historical replay switches into a per-model startup policy."""
    def on(name):
        return os.environ.get(name) == "1"

    if on("HCU_MINIMAX31_DECODE_SHARED_AUX_OVERLAP"):
        shared_schedule = "aux"
    elif on("HCU_MINIMAX31_DECODE_SHARED_DISPATCH_OVERLAP"):
        shared_schedule = "dual_window"
    elif on("HCU_MINIMAX31_DECODE_SHARED_COMBINE_OVERLAP"):
        shared_schedule = "combine"
    else:
        shared_schedule = "stock"
    w8a8 = on("HCU_MINIMAX31_DECODE_W8A8_FUSED")
    options = {
        "shared_expert_schedule": shared_schedule,
        "shared_w8a8_fused": w8a8 and shared_schedule != "stock",
        "hipc_verify_tuned": os.environ.get("HCU_MINIMAX31_COMBINE_GEMM_TUNED", "1") == "1",
        "lightop_moe_tuned": on("HCU_MINIMAX31_LIGHTOP_MOE_TUNED"),
    }
    return options, w8a8


def verify8_replay_options():
    """Keep historical fake-PD provider selection local to the replay launcher."""
    if os.environ.get("HCU_MINIMAX31_MSA_VERIFY8_TUNED") == "1":
        msa = "tuned"
    elif os.environ.get("HCU_MINIMAX31_MSA_FUSED_VERIFY8") == "1":
        msa = "fused"
    else:
        msa = "stock"
    return dict(
        qkv_gemma_int8=os.environ.get("HCU_MINIMAX31_QKV_GEMMA_INT8") == "1",
        qk_fp8=os.environ.get("HCU_MINIMAX31_LIGHTOP_QK_FP8") == "1",
        msa_selector=msa,
    )


def load_models(root,*,verify_transpose=False,load_draft=True):
    import torch
    rank=int(os.environ['LOCAL_RANK'])
    local_device=device_index();torch.cuda.set_device(local_device)
    from vllm.entrypoints.common.startup import cli_env_setup
    cli_env_setup()
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from vllm.entrypoints.cli.preload import PreloadSubcommand
    from vllm.engine.arg_utils import EngineArgs
    parser=argparse.ArgumentParser()
    PreloadSubcommand.add_cli_args(parser)
    spec=json.loads((root/'dspark_spec_v2.json').read_text())
    socket_dir=os.environ.get('M31_DECODE_SOCKET_DIR','/tmp/m31d20')
    extra={'socket_dir':socket_dir,'mode':'zero_copy','fallback':False,
           'decode_ready_manifest':str(root/('dual_ready_v2.json' if load_draft else 'target_ready_v1.json'))}
    options, verify_row_major = _replay_decode_config()
    argv=['--model',str(root/'config_target_v1'),'--trust-remote-code',
          '--quantization','slimquant_w4a8','--tensor-parallel-size','1',
          '--pipeline-parallel-size','1','--data-parallel-size','8',
          '--data-parallel-size-local','8','--prefill-context-parallel-size','1',
          '--decode-context-parallel-size','1','--enable-expert-parallel',
          '--expert-placement-strategy','linear','--all2all-backend','allgather_reducescatter',
          '--language-model-only','--distributed-executor-backend','mp',
          '--max-model-len','148608','--max-num-seqs','16',
          '--max-num-batched-tokens','256','--enforce-eager',
          '--kv-cache-dtype','fp8','--attention-config','{"indexer_kv_dtype":"bf16"}',
          '--disable-custom-all-reduce','--no-enable-flashinfer-autotune',
          '--speculative-config',json.dumps(spec),'--load-format','minimax31_decode_ipc',
          '--model-loader-extra-config',json.dumps(extra),
          '--additional-config',json.dumps({'minimax31_decode': options})]
    cfg=EngineArgs.from_cli_args(parser.parse_args(argv)).create_engine_config()
    from hcu_vllm_das.parallel.minimax31_decode_ipc import DecodeDaemon
    from vllm.model_executor.model_loader import get_model
    from vllm.v1.worker.workspace import init_workspace_manager
    class Consumer(DecodeDaemon):
        def get_model(self):
            init_workspace_manager(torch.device('cuda',local_device))
            return get_model(vllm_config=self.vllm_config,model_config=self.model_config)
    # Native daemon owns the supported DP/EP initialization sequence. This
    # consumer never calls serve_forever or binds an owner socket.
    address='tcp://127.0.0.1:'+os.environ['MASTER_PORT']
    consumer=Consumer(cfg,rank,local_device,address,socket_dir,dp_rank=rank)
    consumer.load_model()
    cfg=consumer.vllm_config;target=consumer.model
    from hcu_vllm_das.quantization.minimax31 import restore_minimax31_router_aliases
    from hcu_vllm_das.parallel.minimax31_decode import restore_execution_state
    restore_minimax31_router_aliases(target)
    restore_execution_state(target,torch.device('cuda',local_device))
    from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model
    from vllm.config import set_current_vllm_config
    draft=None
    if load_draft:
        with set_current_vllm_config(cfg):draft=load_dspark_model(target,cfg)
    from hcu_vllm_das.parallel.minimax31_int8_runtime import select_qualified_int8
    select_qualified_int8(target, verify_transpose=verify_transpose, verify_row_major=verify_row_major)
    if draft is not None:select_qualified_int8(draft)
    return consumer,cfg,target,draft


def verify(args,consumer,cfg,target,draft):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.distributed.parallel_state import get_world_group
    from hcu_vllm_das.parallel.minimax31_decode_attention import Verify8Runtime
    from minimax31_validation import finite,open_history_lease
    if args.host_history_ready and not args.staged_host_initialization:
        raise ValueError('history reuse requires explicit staged offline VERIFY initialization')
    if args.shared_zero_host_history and (not args.host_history_ready or not args.verify_eager_only):
        raise ValueError('shared zero history is only an explicit leased eager VERIFY bootstrap')
    if args.gpu_progress and not args.verify_eager_only:
        raise ValueError('GPU breadcrumbs are eager-only diagnostics')
    if args.gpu_progress and not args.preinitialize_moe:
        raise ValueError('GPU breadcrumbs require the stream-owned MoE executor to be preinitialized')
    root=args.task_root
    sys.path.insert(0,str(root/'native_decode/build_lib'))
    sys.path.insert(0,str(root/'hisparse_native/build_lib'))
    sys.path.insert(0,str(root.parent/'prototypes/sol/private_msa_build/build_lib'))
    from hcu_minimax_msa import _native as index_native
    import hcu_minimax_decode_native as decode_native
    import hcu_minimax_hisparse_native as page_native
    rank=int(os.environ['LOCAL_RANK'])
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
    numa_spread()
    # Execution workspace geometry does not change published weight ownership.
    # The runner retains this construction-time config object.
    cfg.model_config.hf_config.hcu_minimax_decode={'max_dispatch_tokens':128}
    for module in target.modules():
        execution=getattr(module,'_minimax_decode_config',None)
        if execution is not None:
            execution.model_config.hf_config.hcu_minimax_decode={'max_dispatch_tokens':128}
    if args.preinitialize_moe:
        from hcu_vllm_das.parallel.minimax31_decode import _get_executor
        runners=[m for m in target.modules() if hasattr(m,'_minimax_decode_config') and hasattr(m,'routed_experts')]
        if len(runners)!=57:raise RuntimeError('expected all 57 actual MoE runners')
        runner=runners[0]
        dummy=torch.empty((128,6144),device='cuda',dtype=torch.bfloat16)
        print('MINIMAX_MOE_INITIALIZE_BEGIN '+json.dumps({'rank':rank}),flush=True)
        _get_executor(runner.routed_experts,dummy,runner._minimax_decode_config)
        torch.cuda.synchronize();get_world_group().barrier()
        print('MINIMAX_MOE_INITIALIZE_END '+json.dumps({'rank':rank}),flush=True)
    execution_pool=None
    if args.execution_pool_library:
        from minimax31_execution_pool import ExecutionPool
        execution_pool=ExecutionPool(args.execution_pool_library,device_index())
        print('MINIMAX_EXECUTION_POOL '+json.dumps({'rank':rank,**execution_pool.check()}),flush=True)
    if args.prewarm_execution:
        if not args.preinitialize_moe:raise ValueError('execution prewarm requires initialized MoE')
        from minimax31_execution_prewarm import run as prewarm_execution
        from hcu_vllm_das.parallel import minimax31_decode
        prewarm_execution(target=target,runner=runner,executor=minimax31_decode._executor,
                          pool=execution_pool,rank=rank,world=get_world_group())
    print('MINIMAX_HOST_REGISTER_BEGIN '+json.dumps({'rank':rank}),flush=True)
    def vram_usage():
        return {p.parent.resolve().name:int(p.read_text())
                for p in Path('/sys/class/drm').glob('card*/device/mem_info_vram_used')}
    before_vram=vram_usage()
    backing=None
    try:
        if args.staged_host_initialization:
            from hcu_vllm_das.parallel.minimax31_hisparse import MappedHostStorage
            shape=(2,60,2378880,4,128)
            fd=open_history_lease(args.host_history_ready,rank,shape) if args.host_history_ready else None
            try:
                allocation_shape=(2378880,4,128) if args.shared_zero_host_history else shape
                backing=MappedHostStorage(allocation_shape,layer_bytes=2378880*512,zero=True,
                    registration_lock=root/'host_registration.lock' if args.serialize_host_registration else None,
                    address_hint=args.host_address_hint,zero_threads=8,existing_zero_backing_fd=fd,
                    zero_slice_source_bytes=146158387200 if args.shared_zero_host_history else None,
                    chunk_bytes=2*60*2378880*512 if args.whole_host_registration else 4*1024**3)
            finally:
                if fd is not None:os.close(fd)
            if args.shared_zero_host_history:backing.as_readonly_zero_layer_alias(60)
            print('MINIMAX_HOST_ONLY_READY '+json.dumps({'rank':rank,'vram_before':before_vram,
                'vram_after':vram_usage(),'host_pointer':hex(backing.storage.data_ptr()),
                'registration_seconds':backing.registration_seconds,
                'registered_bytes':backing.registered_bytes,'shared_zero_layer_alias':backing.zero_history_layer_alias}),flush=True)
            # Every rank must finish pinning before any rank allocates its large
            # GPU caches. Per-rank "host first" alone does not ensure this.
            get_world_group().barrier()
        runtime=Verify8Runtime(index_native=index_native,decode_native=decode_native,
                page_native=page_native,main_slots=args.main_slots,fake_zero_history=True,
                registration_lock=root/'host_registration.lock' if args.serialize_host_registration else None,
                address_hint=args.host_address_hint,host_backing=backing,
                registration_chunk_bytes=2*60*2378880*512 if args.whole_host_registration else 4*1024**3,
                **verify8_replay_options())
    except BaseException:
        print('MINIMAX_HOST_REGISTER_FAILURE_RESOURCES '+json.dumps({
            'rank':rank,'meminfo':Path('/proc/meminfo').read_text(),
            'vram_before':before_vram,'vram_after':vram_usage(),
            'gtt_used':{str(p):int(p.read_text()) for p in Path('/sys/class/drm').glob('card*/device/mem_info_gtt_used')}}),flush=True)
        raise
    print('MINIMAX_HOST_REGISTER_END '+json.dumps({'rank':rank,
        'prefault_seconds':runtime.backing.prefault_seconds,
        'zero_seconds':runtime.backing.zero_seconds,
        'registration_seconds':runtime.backing.registration_seconds,
        'gtt_used':{str(p):int(p.read_text()) for p in Path('/sys/class/drm').glob('card*/device/mem_info_gtt_used')}}),flush=True)
    runtime.bind(target)
    ids=torch.full((128,),1000+rank,device='cuda',dtype=torch.int64)
    dp_tokens=torch.full((8,),128,dtype=torch.int32)
    progress=None
    def forward():
        if progress:progress.start_forward()
        with set_current_vllm_config(cfg),set_forward_context(None,cfg,num_tokens=128,
                num_tokens_across_dp=dp_tokens):
            hidden=target(ids,runtime.positions)
            if isinstance(hidden,tuple):hidden=hidden[0]
            logits=target.compute_logits(hidden)
            if progress:progress.end_forward()
            return hidden,logits
    print('MINIMAX_OFFLINE_VERIFY_BACKEND_READY '+json.dumps({'rank':rank,
        'rope_dtype':str(runtime.rope.dtype),'main_slots':runtime.main_slots,
        'host_slots':runtime.host_slots,'host_registration_ranges':len(runtime.backing.ranges),
        'gpu_free_bytes':torch.cuda.mem_get_info()[0]}),flush=True)
    get_world_group().barrier()
    if args.gpu_progress:
        from minimax31_gpu_progress import GPUProgress
        from hcu_vllm_das.parallel import minimax31_decode
        progress=GPUProgress(args.output,rank);runtime.gpu_progress=progress
        minimax31_decode._executor.gpu_progress=progress
        minimax31_decode._executor.experts.gpu_progress=progress
    from contextlib import nullcontext
    if execution_pool:
        runtime.execution_pool_owner=execution_pool
        runtime.execution_pool_id=execution_pool.pool.id
        print('MINIMAX_EXECUTION_POOL '+json.dumps({'rank':rank,**execution_pool.check()}),flush=True)
    # Explicit diagnostic control for bounding the number of outstanding
    # full-model layers. Kept for the resident session and reported in results.
    if args.eager_layer_sync:
        synchronous_layers=[m for m in target.modules() if type(m).__name__=='MiniMaxM3DecoderLayer']
        if len(synchronous_layers)!=60:raise RuntimeError('expected all 60 decoder layers')
        def synchronize_layer(module,inputs,output):torch.cuda.current_stream().synchronize()
        runtime.eager_sync_layers=synchronous_layers
        runtime.eager_sync_handles=[m.register_forward_hook(synchronize_layer) for m in synchronous_layers]
    with torch.inference_mode(), (execution_pool.context() if execution_pool else nullcontext()):
        handles=[]
        if args.diagnostic_layer_sync:
            decoder_layers=[(name,m) for name,m in target.named_modules() if type(m).__name__=='MiniMaxM3DecoderLayer']
            if len(decoder_layers)!=60:raise RuntimeError('expected all 60 decoder layers for diagnostics')
            for name,module in decoder_layers:
                def before(module,inputs,_name=name):
                    print('MINIMAX_VERIFY_LAYER_BEGIN '+json.dumps({'rank':rank,'layer':_name}),flush=True)
                def after(module,inputs,output,_name=name):
                    torch.cuda.synchronize()
                    hidden=output[0] if isinstance(output,tuple) else output
                    if not finite(hidden):raise RuntimeError('nonfinite layer '+_name)
                    print('MINIMAX_VERIFY_LAYER_END '+json.dumps({'rank':rank,'layer':_name}),flush=True)
                handles.append(module.register_forward_pre_hook(before))
                handles.append(module.register_forward_hook(after))
        try:hidden,logits=forward()
        finally:
            for handle in handles:handle.remove()
        hidden,logits=forward()
        torch.cuda.synchronize();get_world_group().barrier()
        if not finite(hidden) or not finite(logits):
            raise RuntimeError('real model verify produced nonfinite hidden/logits')
        print('MINIMAX_VERIFY_EAGER_PASSED '+json.dumps({'rank':rank}),flush=True)
        get_world_group().barrier()
        from hcu_vllm_das.parallel.minimax31_decode_placement import placement_for_model
        placement=placement_for_model(cfg.model_config)
        result={'scope':'real target VERIFY only; excludes DRAFT/accept/commit; not complete cycle alignment',
                'rank':rank,'query_rows':128,'requests_per_rank':16,'prefix_tokens':147456,
                'target_layers':60,'physical_experts':128,
                'placement':'sglang_static128' if placement else 'linear',
                'placement_sha256':placement.map_sha256 if placement else None,
                'main_slots':runtime.main_slots,'reference_main_slots':992256,
                'actual_total_gpu_bytes':torch.cuda.get_device_properties(device_index()).total_memory,
                'reference_total_gpu_bytes':154602045440,'finite_hidden_and_logits':True,
                'input_token_pattern':'constant_per_rank',
                'fake_main_and_index_history':'zero',
                'shared_expert_stream_token_threshold':int(os.environ.get('VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD','256')),
                'preinitialized_moe':args.preinitialize_moe,
                'execution_prewarmed':args.prewarm_execution,
                'initial_shared_zero_host_history':args.shared_zero_host_history,
                'host_history_lease':str(args.host_history_ready) if args.host_history_ready else None,
                'gpu_progress_instrumented':args.gpu_progress,
                'eager_layer_sync':args.eager_layer_sync,
                'execution_pool':execution_pool.check() if execution_pool else None,
                'HSA_ENABLE_SDMA':os.environ.get('HSA_ENABLE_SDMA'),
                'torch_version':torch.__version__,'torch_hip_version':torch.version.hip,
                'index_length_metadata_devices':[str(runtime.q_lens.device),str(runtime.kv_lens.device)],
                'eager_only':args.verify_eager_only}
        if args.verify_eager_only:
            del hidden,logits
            from minimax31_eager_session import run
            run(args,cfg=cfg,runtime=runtime,ids=ids,forward=forward,base_result=result,
                profiler=getattr(args,'startup_profiler',None))
            return
        if not args.verify_eager_only:
            graph=torch.cuda.CUDAGraph()
            print('MINIMAX_VERIFY_CAPTURE_BEGIN '+json.dumps({'rank':rank}),flush=True)
            with (execution_pool.capture(graph,stream=stream) if execution_pool else torch.cuda.graph(graph,stream=stream)):
                hidden,logits=forward()
            print('MINIMAX_VERIFY_CAPTURE_END '+json.dumps({'rank':rank}),flush=True)
            get_world_group().barrier()
            print('MINIMAX_VERIFY_FIRST_REPLAY_BEGIN '+json.dumps({'rank':rank}),flush=True)
            for _ in range(5):graph.replay()
            torch.cuda.synchronize()
            print('MINIMAX_VERIFY_FIRST_REPLAY_END '+json.dumps({'rank':rank}),flush=True)
            times=[]
            for _ in range(3):
                get_world_group().barrier()
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(8):graph.replay()
                end.record();end.synchronize();times.append(begin.elapsed_time(end)/8)
            result['verify_graph_ms']=times
            misses=torch.stack([state.miss_count.sum() for state in runtime.states]).cpu().tolist()
            result['last_timed_replay_host_miss_pages_per_layer']=misses
            result['last_timed_replay_host_swap_bytes']=sum(misses)*2*128*128
            ids.add_(1);runtime.prefix.add_(13);runtime.prepare();graph.replay()
            changed_hidden=hidden.clone();changed_logits=logits.clone()
            eager_hidden,eager_logits=forward()
            torch.testing.assert_close(changed_hidden,eager_hidden,rtol=.02,atol=.02)
            torch.testing.assert_close(changed_logits,eager_logits,rtol=.02,atol=.04)
            result['changed_input_and_position_graph_matches_eager']=True
        (args.output/('rank%d.json'%rank)).write_text(json.dumps(result,indent=2)+'\n')
        print('MINIMAX_OFFLINE_VERIFY '+json.dumps(result),flush=True)
    get_world_group().barrier()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('load-only','capacity','verify','draft','va-probe'),default='load-only')
    p.add_argument('--main-slots',type=int,default=992256)
    p.add_argument('--verify-eager-only',action='store_true')
    p.add_argument('--serialize-host-registration',action='store_true')
    p.add_argument('--draft-preflight',action='store_true')
    p.add_argument('--host-address-hint',type=lambda value:int(value,0))
    p.add_argument('--whole-host-registration',action='store_true')
    p.add_argument('--staged-host-initialization',action='store_true')
    p.add_argument('--host-history-ready',type=Path)
    p.add_argument('--execution-pool-library',type=Path)
    p.add_argument('--gpu-progress',action='store_true')
    p.add_argument('--verify-transpose-int8',action='store_true')
    p.add_argument('--preinitialize-moe',action='store_true')
    p.add_argument('--diagnostic-layer-sync',action='store_true')
    p.add_argument('--resident-eager',action='store_true')
    p.add_argument('--eager-profile-startup',action='store_true')
    p.add_argument('--eager-iterations',type=int,default=5)
    p.add_argument('--eager-trials',type=int,default=3)
    p.add_argument('--shared-zero-host-history',action='store_true',help='diagnostic readonly zero alias bootstrap; attach full history before final trace comparison')
    p.add_argument('--prewarm-execution',action='store_true',help='warm validation and real-weight INT8/MoE execution before full KV allocation')
    p.add_argument('--eager-layer-sync',action='store_true',help='diagnostic per-layer stream sync retained in all eager forwards')
    p.add_argument('--resident-idle-seconds',type=int,default=3600)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.verify_transpose_int8 and args.mode!='verify':
        p.error('--verify-transpose-int8 is specific to VERIFY')
    if (args.resident_eager or args.eager_profile_startup) and not (args.mode=='verify' and args.verify_eager_only):
        p.error('resident/profile eager mode requires --mode verify --verify-eager-only')
    if not 1<=args.eager_iterations<=100 or not 1<=args.eager_trials<=10:
        p.error('bounded eager measurement count required')
    import torch
    import faulthandler,signal
    faulthandler.register(signal.SIGUSR1,all_threads=True)
    if args.eager_profile_startup:
        torch.cuda.set_device(device_index())
        args.startup_profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False)
        args.startup_profiler.start()
    started=time.monotonic()
    consumer,cfg,target,draft=load_models(args.task_root,verify_transpose=args.verify_transpose_int8,
        load_draft=args.mode!='verify' or args.draft_preflight)
    rank=int(os.environ['LOCAL_RANK'])
    from vllm.distributed.parallel_state import get_world_group
    get_world_group().barrier()
    if args.mode in ('va-probe','draft'):
        import ctypes,mmap
        size=2*60*2378880*4*128
        mapping=mmap.mmap(-1,size,flags=mmap.MAP_SHARED|mmap.MAP_ANONYMOUS,prot=mmap.PROT_READ|mmap.PROT_WRITE)
        base=ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        intersections=[];storages={}
        for role,model in [('target',target),('draft',draft)]:
            for name,tensor in list(model.named_parameters())+list(model.named_buffers()):
                if not tensor.is_cuda:continue
                storage=tensor.untyped_storage();lo=storage.data_ptr();hi=lo+storage.nbytes()
                storages[lo]=hi
                if lo<base+size and base<hi:
                    intersections.append({'role':role,'name':name,'gpu_begin':hex(lo),'gpu_end':hex(hi)})
        result={'rank':rank,'host_begin':hex(base),'host_end':hex(base+size),'host_bytes':size,
                'weight_gpu_min':hex(min(storages)),'weight_gpu_max':hex(max(storages.values())),
                'overlaps_with_live_gpu_weights':intersections}
        (args.output/('rank%d.va.json'%rank)).write_text(json.dumps(result,indent=2)+'\n')
        print('MINIMAX_HOST_VA_PROBE '+json.dumps(result),flush=True)
        mapping.close();get_world_group().barrier()
        if args.mode=='va-probe':return
    if args.mode=='draft':
        from minimax31_draft_probe import run
        run(args,cfg,draft)
        return
    if args.mode=='verify':
        if args.draft_preflight:
            from minimax31_draft_probe import run
            import gc
            subargs=argparse.Namespace(**vars(args));subargs.output=args.output/'draft_preflight'
            subargs.output.mkdir(exist_ok=True)
            run(subargs,cfg,draft)
            gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
        verify(args,consumer,cfg,target,draft)
        return
    pools=[]
    if args.mode=='capacity':
        # Same main capacity as the SGLang launch, plus the complete 147K
        # single-KV-head FP8 index history. Touch every allocation, per layer,
        # to exercise physical memory without a second full-size temporary.
        for layer in range(60):
            main=torch.empty((2,args.main_slots,4,128),device='cuda',dtype=torch.float8_e4m3fn)
            index=torch.empty((2378880,1,128),device='cuda',dtype=torch.float8_e4m3fn)
            main.view(torch.uint8).zero_();index.view(torch.uint8).zero_()
            pools.append((main,index))
        torch.cuda.synchronize();get_world_group().barrier()
    result={'scope':'strict real-weight IPC and capacity only; not full decode timing',
            'rank':rank,'mode':args.mode,'load_and_capacity_seconds':time.monotonic()-started,
            'target_class':type(target).__name__,'draft_class':type(draft).__name__,
            'main_slots':args.main_slots if pools else None,'index_slots':2378880 if pools else None,
            'reference_main_slots':992256,'reference_total_gpu_bytes':154602045440,
            'actual_total_gpu_bytes':torch.cuda.get_device_properties(device_index()).total_memory,
            'gpu_free_bytes':torch.cuda.mem_get_info()[0],
            'consumer_allocated_bytes':torch.cuda.memory_allocated(),
            'consumer_reserved_bytes':torch.cuda.memory_reserved()}
    (args.output/('rank%d.json'%rank)).write_text(json.dumps(result,indent=2)+'\n')
    print('MINIMAX_OFFLINE_LOAD '+json.dumps(result),flush=True)
    get_world_group().barrier()
    # Exit without releasing the owners. IPC lifetime watches belong to the
    # imported model; no unlink, daemon stop, or in-place weight mutation.


if __name__=='__main__':main()
