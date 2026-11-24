"""Attach the complete leased zero history after full GPU execution is warmed."""
import ctypes
import json
import os
import time
import torch
from minimax31_validation import open_history_lease


def compare(actual,expected):
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype:raise RuntimeError('history switch output geometry changed')
    maximum=0.
    for a,b in zip(actual.reshape(-1).split(128*6144),expected.reshape(-1).split(128*6144)):
        a=a.float();b=b.float()
        if not bool(torch.isclose(a,b,rtol=.02,atol=.04).all()):raise RuntimeError('history switch changed model outputs')
        maximum=max(maximum,float((a-b).abs().max()))
    return maximum


def attach(*,args,runtime,forward,rank,world,destination):
    from hcu_vllm_das.parallel.minimax31_hisparse import MappedHostStorage,loaded_hip_runtime
    old=runtime.backing
    if not getattr(old,'zero_history_layer_alias',False) or not old.readonly_zero_history:
        raise RuntimeError('full-history attachment requires the explicit readonly alias bootstrap')
    if torch.cuda.is_current_stream_capturing():raise RuntimeError('attach full history before graph capture')
    hip=loaded_hip_runtime();hip.hipGetDevice.argtypes=[ctypes.POINTER(ctypes.c_int)]
    hip.hipGetDevice.restype=ctypes.c_int;hip.hipSetDevice.argtypes=[ctypes.c_int];hip.hipSetDevice.restype=ctypes.c_int
    device=runtime.main[0].device.index;device_observations=[]
    def restore_device(phase):
        current=ctypes.c_int(-1)
        if hip.hipGetDevice(ctypes.byref(current)):raise RuntimeError('cannot query raw HIP device')
        device_observations.append({'phase':phase,'raw_hip_device':current.value,'torch_device':torch.cuda.current_device(),'expected':device})
        if hip.hipSetDevice(device):raise RuntimeError('cannot restore assigned HIP device')
        torch.cuda.set_device(device)
    restore_device('before_attachment')
    pointers=tuple(t.data_ptr() for t in runtime.main+runtime.index)
    torch.cuda.synchronize();world.barrier()
    hidden,logits=forward();torch.cuda.synchronize()
    expected_hidden=hidden.clone();expected_logits=logits.clone();del hidden,logits
    # Load comparison kernels while host registration is still small. Validate
    # all elements, including logits, with bounded temporary storage.
    compare(expected_hidden,expected_hidden);compare(expected_logits,expected_logits)
    torch.cuda.synchronize();world.barrier()
    pool=getattr(runtime,'execution_pool_owner',None)
    trimmed=pool.trim_native_cache() if pool is not None else None
    print('MINIMAX_ATTACH_NATIVE_CACHE_TRIM '+json.dumps({'rank':rank,'trim':trimmed}),flush=True)
    world.barrier()
    shape=(2,60,2378880,4,128);new=None;started=time.perf_counter()
    for turn in range(8):
        if turn==rank:
            print('MINIMAX_ATTACH_HISTORY_BEGIN '+json.dumps({'rank':rank}),flush=True)
            fd=open_history_lease(args.host_history_ready,rank,shape)
            try:
                new=MappedHostStorage(shape,layer_bytes=2378880*512,zero=True,
                    chunk_bytes=4*1024**3,registration_lock=args.task_root/'host_registration.lock',
                    existing_zero_backing_fd=fd)
            finally:os.close(fd)
            restore_device('after_full_registration')
            print('MINIMAX_ATTACH_HISTORY_REGISTERED '+json.dumps({'rank':rank,
                'bytes':new.registered_bytes,'seconds':new.registration_seconds}),flush=True)
        world.barrier()
    assert new is not None and new.storage.is_contiguous() and new.readonly_zero_history
    runtime.backing=new
    # Avoid live HCU USERPTR unmap while the much larger mapping is present.
    # It can spend minutes in TTM reservation even with GPU streams quiescent.
    # This extra readonly registration is small (one zero layer), holds no
    # additional physical CPU history, and remains owned until consumer exit.
    runtime.bootstrap_history_backing=old
    restore_device('after_bootstrap_retention')
    world.barrier()
    hidden,logits=forward();torch.cuda.synchronize()
    hidden_error=compare(hidden,expected_hidden);logits_error=compare(logits,expected_logits)
    if tuple(t.data_ptr() for t in runtime.main+runtime.index)!=pointers:raise RuntimeError('GPU caches changed during history switch')
    result={'rank':rank,'shared_zero_host_history':False,'registered_host_bytes':new.registered_bytes,
            'previous_registered_host_bytes':old.registered_bytes,
            'bootstrap_alias_retained_until_exit':True,'gpu_caches_preserved':True,
            'target_weights_reloaded':False,'physical_cpu_history_reused':True,
            'complete_output_comparison':True,'comparison_rtol':.02,'comparison_atol':.04,
            'hidden_max_abs':hidden_error,'logits_max_abs':logits_error,
            'device_observations':device_observations,
            'native_cache_trim':trimmed,
            'full_registration_seconds':new.registration_seconds,'wall_seconds':time.perf_counter()-started}
    destination.mkdir(parents=True,exist_ok=True)
    (destination/('rank%d.history.json'%rank)).write_text(json.dumps(result,indent=2)+'\n')
    print('MINIMAX_ATTACH_HISTORY_PASSED '+json.dumps(result),flush=True)
    world.barrier()
