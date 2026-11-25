"""Explicit mapped host backing and stable headwise verify8 page ownership.

No allocations or registrations happen at import. The caller keeps this state
alive through every graph replay and destroys graphs before releasing backing.
"""
from __future__ import annotations
import ctypes
import errno
import fcntl
import mmap
import os
from pathlib import Path
import time
import weakref
import torch


def loaded_hip_runtime():
    paths={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
           if ('libamdhip64' in line or 'libgalaxyhip' in line) and '/' in line}
    if not paths:raise RuntimeError('initialize the assigned HCU before mapped host allocation')
    # Bind an already-loaded runtime, never guess a second library by soname.
    runtime=ctypes.CDLL(sorted(paths)[0])
    runtime.hipHostRegister.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_uint]
    runtime.hipHostRegister.restype=ctypes.c_int
    runtime.hipHostUnregister.argtypes=[ctypes.c_void_p];runtime.hipHostUnregister.restype=ctypes.c_int
    return runtime


def _unregister(runtime,ranges):
    failed=[]
    for pointer,size in reversed(ranges):
        if runtime.hipHostUnregister(ctypes.c_void_p(pointer)):failed.append((pointer,size))
    return failed


def _release_host_region(runtime,ranges,region):
    failed=_unregister(runtime,ranges)
    if region is not None and not failed:
        libc,pointer,size=region
        libc.munmap(ctypes.c_void_p(pointer),size)


def _map_shared_hint(size,flags,prot,hint,prefault):
    # An address hint never replaces an existing process mapping. Keeping large
    # host reservations away from the high-address GPU/IPC heap is explicit.
    libc=ctypes.CDLL(None,use_errno=True)
    libc.mmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_long]
    libc.mmap.restype=ctypes.c_void_p
    libc.munmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t];libc.munmap.restype=ctypes.c_int
    pointer=libc.mmap(ctypes.c_void_p(hint),size,prot,flags,-1,0)
    if pointer==ctypes.c_void_p(-1).value:raise OSError(ctypes.get_errno(),'shared host mmap failed')
    if pointer!=hint:
        libc.munmap(ctypes.c_void_p(pointer),size)
        raise RuntimeError('requested host address hint is occupied; no mapping was replaced')
    if prefault:
        libc.madvise.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
        if libc.madvise(ctypes.c_void_p(pointer),size,getattr(mmap,'MADV_POPULATE_WRITE',23)):
            error=ctypes.get_errno();libc.munmap(ctypes.c_void_p(pointer),size)
            raise OSError(error,'shared host pre-fault failed')
    return (ctypes.c_uint8*size).from_address(pointer),(libc,pointer,size)


class MappedHostStorage:
    """Page-aligned USERPTR storage; registration boundaries preserve layers."""
    def __init__(self,shape,*,layer_bytes,chunk_bytes=4*1024**3,zero=False,
                 registration_lock=None,zero_threads=None,address_hint=None,existing_zero_backing_fd=None,
                 zero_slice_source_bytes=None):
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('register host backing before graph capture')
        if zero_slice_source_bytes is not None and (existing_zero_backing_fd is None or not zero):
            raise ValueError('zero slice requires an explicit existing initialized backing')
        numel=1
        for dimension in shape:
            if type(dimension) is not int or dimension<=0:raise ValueError('positive explicit host dimensions required')
            numel*=dimension
        if layer_bytes<=0 or layer_bytes%4096 or chunk_bytes<layer_bytes:
            raise ValueError('registration must contain complete page-aligned layers')
        chunk_bytes=chunk_bytes//layer_bytes*layer_bytes
        # Match the reference USERPTR allocator: shared, writable, pre-faulted
        # pages. Lazy private pages can race with COW or fail large pinning.
        flags=mmap.MAP_SHARED|mmap.MAP_ANONYMOUS
        prot=mmap.PROT_READ|mmap.PROT_WRITE
        probe=mmap.mmap(-1,4096,flags=flags,prot=prot)
        try:
            try:probe.madvise(getattr(mmap,'MADV_POPULATE_WRITE',23));prefault=True
            except OSError as exc:
                if exc.errno not in (errno.EINVAL,errno.ENOSYS):raise
                prefault=False
            except ValueError:prefault=False
        finally:probe.close()
        begin=time.monotonic()
        flags=flags if prefault else flags|getattr(mmap,'MAP_POPULATE',0x8000)
        runtime=loaded_hip_runtime();ranges=[];region=None
        if existing_zero_backing_fd is not None:
            # The offline caller validates the lease's generation, owner birth,
            # rank, inode and zero-history provenance before supplying this FD.
            expected_bytes=numel if zero_slice_source_bytes is None else zero_slice_source_bytes
            if (not zero or address_hint is not None or type(expected_bytes) is not int or
                    expected_bytes<numel or os.fstat(existing_zero_backing_fd).st_size!=expected_bytes):
                raise ValueError('explicit initialized zero backing must match the full shape')
            backing=mmap.mmap(existing_zero_backing_fd,numel,flags=mmap.MAP_SHARED,prot=prot)
        elif address_hint is None:
            backing=mmap.mmap(-1,numel,flags=flags,prot=prot)
            if prefault:backing.madvise(getattr(mmap,'MADV_POPULATE_WRITE',23))
        else:
            if type(address_hint) is not int or address_hint<=0 or address_hint%4096:
                raise ValueError('host address hint must be a positive aligned Linux address')
            backing,region=_map_shared_hint(numel,flags,prot,address_hint,prefault)
        finalizer=weakref.finalize(backing,_release_host_region,runtime,ranges,region)
        finalizer.atexit=False
        self.prefault_seconds=time.monotonic()-begin
        storage=torch.frombuffer(backing,dtype=torch.uint8,count=numel).view(shape)
        begin=time.monotonic()
        if zero and existing_zero_backing_fd is None:
            old_threads=torch.get_num_threads()
            if zero_threads is not None:torch.set_num_threads(zero_threads)
            try:storage.zero_()
            finally:
                if zero_threads is not None:torch.set_num_threads(old_threads)
        self.zero_seconds=time.monotonic()-begin
        lock=None
        if registration_lock is not None:
            lock=Path(registration_lock).open('a')
            fcntl.flock(lock,fcntl.LOCK_EX)
        begin=time.monotonic()
        try:
            for offset in range(0,numel,chunk_bytes):
                size=min(chunk_bytes,numel-offset);pointer=storage.data_ptr()+offset
                status=runtime.hipHostRegister(ctypes.c_void_p(pointer),size,2)
                if status:
                    free,total=torch.cuda.mem_get_info()
                    raise RuntimeError('hipHostRegisterMapped failed: status=%d pointer=%#x offset=%d bytes=%d completed=%d gpu_free=%d gpu_total=%d' % (status,pointer,offset,size,len(ranges),free,total))
                ranges.append((pointer,size))
        except BaseException:
            ranges[:]=_unregister(runtime,ranges);raise
        finally:
            self.registration_seconds=time.monotonic()-begin
            if lock is not None:lock.close()
        self.storage=storage;self.ranges=tuple(ranges);self.layer_bytes=layer_bytes
        self.registered_bytes=numel;self.zero_history_layer_alias=False
        self.zero_initialized=bool(zero)
        self.readonly_zero_history=existing_zero_backing_fd is not None
        # Torch's frombuffer storage retains mmap until its last view is gone.
        self._finalizer=finalizer

    def as_readonly_zero_layer_alias(self,layers):
        if (not self.readonly_zero_history or not self.zero_initialized or self.zero_history_layer_alias or
                self.storage.ndim!=3 or self.storage.shape[-2:]!=(4,128) or not self.storage.is_contiguous() or
                type(layers) is not int or layers<=0):
            raise ValueError('only explicit existing readonly zero history can alias layers')
        self.storage=self.storage.view(1,1,*self.storage.shape).expand(2,layers,*self.storage.shape)
        self.zero_history_layer_alias=True
        return self

    def complete_layer_ranges(self,layers):
        if self.zero_history_layer_alias:raise RuntimeError('readonly zero layer aliases cannot commit history')
        step=self.ranges[0][1]//self.layer_bytes
        return [(start,min(start+step,layers)) for start in range(0,layers,step)]


class HeadwisePageState:
    def __init__(self,*,native,host_map,device_map,reqs,real):
        b=reqs.numel();page=128;hot_pages=240
        if (device_map.shape!=(b,(hot_pages+1)*page) or host_map.shape[0]!=b or
                host_map.shape[1]%page or host_map.shape[1]//page>2048):
            raise ValueError('unsupported verify8 host/hot extent')
        self.native=native;self.host_map=host_map;self.device_map=device_map
        self.reqs=reqs;self.real=real
        opts={'device':reqs.device}
        self.tags=torch.full((b,4*hot_pages),-1,dtype=torch.int64,**opts)
        self.page_slots=torch.full((b*4,host_map.shape[1]//page),-1,dtype=torch.int32,**opts)
        self.hot=torch.empty((b,8*4*16),dtype=torch.int32,**opts)
        self.miss_src=torch.empty((b*4,128),dtype=torch.int64,**opts)
        self.miss_dst=torch.empty((b*4,128),dtype=torch.int32,**opts)
        self.miss_count=torch.empty((b*4,),dtype=torch.int32,**opts)
        self.verify_src=torch.empty((b,8),dtype=torch.int64,**opts)
        self.verify_dst=torch.empty((b,8),dtype=torch.int32,**opts)
        self.verify_count=torch.zeros((b,),dtype=torch.int32,**opts)
        self.indices=torch.empty((b,8,4,2048),dtype=torch.int32,**opts)

    def plan(self,topk,ends):
        self.native.plan_headwise(self.reqs,self.real,topk,ends,self.host_map,self.device_map,
            self.tags,self.hot,self.page_slots,self.miss_src,self.miss_dst,self.miss_count,
            self.verify_src,self.verify_dst,self.verify_count)
        from triton_tmp_dir.minimax31_hisparse_triton import expand_indices
        expand_indices(self,topk,ends)
        return self.indices

    def swap(self,host_k,host_v,device_k,device_v):
        self.native.swap_headwise(self.miss_src,self.miss_dst,self.miss_count,self.real,
                                  host_k,host_v,device_k,device_v)
