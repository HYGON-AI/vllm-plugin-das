"""Quiescent, offline-only registration experiment retaining physical history."""
import ctypes
import hashlib
import json
import time
import torch


def reregister(*,runtime,root,rank,world,chunk_bytes):
    from hcu_vllm_das.parallel.minimax31_hisparse import loaded_hip_runtime,_unregister,_release_host_region
    backing=runtime.backing
    if torch.cuda.is_current_stream_capturing():raise RuntimeError('registration experiment requires eager mode')
    layer_bytes=backing.layer_bytes
    if type(chunk_bytes) is not int or chunk_bytes<layer_bytes:raise ValueError('complete layers required')
    chunk_bytes=chunk_bytes//layer_bytes*layer_bytes
    # The finalizer owns the mutable list of actual live registrations. Keep
    # that list, CPU storage and mmap lifetime together even on partial failure.
    owned=backing._finalizer.peek()
    if owned is None or owned[1] is not _release_host_region:raise RuntimeError('unknown backing owner')
    native,ranges,region=owned[2]
    if tuple(ranges)!=backing.ranges:raise RuntimeError('registration lifetime metadata differs')
    pointer=backing.storage.data_ptr();numel=backing.storage.numel()
    def sample():
        return hashlib.sha256(b''.join(ctypes.string_at(pointer+offset,64)
            for offset in range(0,numel,layer_bytes))).hexdigest()
    before=sample();old_ranges=len(ranges);started=time.perf_counter()
    # No graph exists yet. Every rank must finish all GPU access before any
    # mapping is removed; registration proceeds one rank at a time.
    torch.cuda.synchronize();world.barrier()
    for turn in range(8):
        if turn==rank:
            print('MINIMAX_REREGISTER_BEGIN '+json.dumps({'rank':rank,'old_ranges':old_ranges}),flush=True)
            ranges[:]=_unregister(native,ranges);backing.ranges=tuple(ranges)
            if ranges:raise RuntimeError('failed to unregister owned host range; abort consumer')
            try:
                for offset in range(0,numel,chunk_bytes):
                    size=min(chunk_bytes,numel-offset)
                    status=native.hipHostRegister(ctypes.c_void_p(pointer+offset),size,2)
                    if status:raise RuntimeError('chunk registration failed: status=%d offset=%d'%(status,offset))
                    ranges.append((pointer+offset,size))
            finally:backing.ranges=tuple(ranges)
            if pointer!=backing.storage.data_ptr() or before!=sample():
                raise RuntimeError('host storage identity or sampled contents changed')
            print('MINIMAX_REREGISTER_END '+json.dumps({'rank':rank,'ranges':len(ranges)}),flush=True)
        world.barrier()
    result={'rank':rank,'old_ranges':old_ranges,'new_ranges':len(ranges),'chunk_bytes':chunk_bytes,
            'host_pointer':hex(pointer),'sample_sha256':before,'physical_storage_reused':True,
            'total_experiment_wall_seconds':time.perf_counter()-started}
    (root/('rank%d.reregister.json'%rank)).write_text(json.dumps(result,indent=2)+'\n')
