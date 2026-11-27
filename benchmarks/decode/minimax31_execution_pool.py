"""Opt-in execution pool retaining the native global allocator and IPC loader."""
import ctypes
from contextlib import contextmanager
import hashlib
from pathlib import Path
import torch


class ExecutionPool:
    def __init__(self,library,device):
        if torch.cuda.memory.get_allocator_backend()!='native':
            raise RuntimeError('preload import must retain the native global allocator')
        self.device=int(device);self.path=Path(library).resolve()
        self._active_context=None
        self.library=ctypes.CDLL(str(self.path))
        self.library.m31_pool_set_device.argtypes=[ctypes.c_int]
        self.library.m31_pool_set_device.restype=ctypes.c_int
        if self.library.m31_pool_set_device(self.device):raise RuntimeError('execution pool device changed')
        for name in ('allocations','frees','errors','total_requested_bytes'):
            function=getattr(self.library,'m31_pool_'+name);function.argtypes=[];function.restype=ctypes.c_ulonglong
        self.reserve=None
        if hasattr(self.library,'m31_pool_set_arena'):
            configure=self.library.m31_pool_set_arena
            configure.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
            configure.restype=ctypes.c_int
            self.reserve=torch.empty(512*1024**2+256,device=torch.device('cuda',self.device),dtype=torch.uint8)
            # Keep the native allocation start distinct from every pool block key.
            if configure(self.reserve.data_ptr()+256,self.reserve.numel()-256,self.device):
                raise RuntimeError('execution arena ownership configuration failed')
            for name in ('arena_capacity','arena_used'):
                function=getattr(self.library,'m31_pool_'+name);function.argtypes=[];function.restype=ctypes.c_ulonglong
        self.allocator=torch.cuda.memory.CUDAPluggableAllocator(str(self.path),'m31_pool_malloc','m31_pool_free')
        self.pool=torch.cuda.MemPool(self.allocator.allocator())
        if self.reserve is not None:self.pool._minimax31_native_reserve=self.reserve
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()

    @contextmanager
    def context(self):
        if self._active_context is not None:raise RuntimeError('execution pool context is already active')
        self._enter_context()
        try:yield
        finally:
            active=self._active_context;self._active_context=None
            if active is not None:active.__exit__(None,None,None)

    def _enter_context(self):
        active=torch.cuda.use_mem_pool(self.pool,device=self.device)
        active.__enter__();self._active_context=active

    @contextmanager
    def capture(self,graph,*,stream):
        # Graph capture registers its own allocation scope for this pool.
        # PyTorch rejects a simultaneous use_mem_pool scope for the same ID.
        active=self._active_context
        if active is not None:
            self._active_context=None;active.__exit__(None,None,None)
        try:
            with torch.cuda.graph(graph,stream=stream,pool=self.pool.id):yield
        finally:
            if active is not None:self._enter_context()

    def trim_native_cache(self):
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('trim only while GPU execution is quiescent')
        torch.cuda.synchronize()
        before=self.check();reserved_before=torch.cuda.memory_reserved(self.device)
        active=self._active_context
        if active is not None:
            self._active_context=None;active.__exit__(None,None,None)
        try:torch.cuda.empty_cache()
        finally:
            if active is not None:self._enter_context()
        after=self.check()
        if after['frees']!=before['frees']:
            raise RuntimeError('native cache trim unexpectedly released execution pool segments')
        return {'reserved_before':reserved_before,'reserved_after':torch.cuda.memory_reserved(self.device),
                'execution_pool_segments_preserved':True,'raw_frees_before':before['frees'],'raw_frees_after':after['frees']}

    def stats(self):
        values={name:int(getattr(self.library,'m31_pool_'+name)())
                for name in ('allocations','frees','errors','total_requested_bytes')}
        if self.reserve is not None:
            values.update(arena_capacity=int(self.library.m31_pool_arena_capacity()),
                          arena_used=int(self.library.m31_pool_arena_used()))
        return dict(values,library_sha256=self.sha256,global_allocator=torch.cuda.memory.get_allocator_backend(),
                    device=self.device,pool_id=self.pool.id)

    def check(self):
        result=self.stats()
        if result['errors'] or result['global_allocator']!='native':
            raise RuntimeError('execution pool violated its allocation/IPC scope: '+str(result))
        return result
