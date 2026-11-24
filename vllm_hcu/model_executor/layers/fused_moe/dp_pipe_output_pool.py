# SPDX-License-Identifier: Apache-2.0
"""Profile-accounted DP-pipe output storage for the bounded Step5 profile.

Only the private moe_full output/scratch is pooled. Returned result, quantized
inputs, gathers and expert workspaces retain their original owners. There is
one process-global pool per worker device, shared by target and MTP layers.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import os
import json
import threading


_STATES = {}
_LOCK = threading.RLock()
_FLAG = "HCU_STEP5_DP_PIPE_OUTPUT_POOL"

def _emit(record):
    record = dict(record, pid=os.getpid())
    print("[dp_moe_pipe_pool] " + json.dumps(record, sort_keys=True), flush=True)


@dataclass
class _Slot:
    buffer: object
    reader_done: object = None
    has_reader: bool = False
    busy: bool = False
    poisoned: bool = False
    fence_sequence: int = 0
    acquire_logged: bool = False
    release_logged: bool = False
    last_reader_stream: object = None


class _Pool:
    def __init__(self, backend, *, device, dtype, K, capacity_rows):
        self.backend, self.device, self.dtype = backend, device, dtype
        self.K, self.capacity_rows = K, capacity_rows
        self.slots = []
        self.prepared = False
        self.locked = False

    def prepare(self):
        if self.prepared or self.locked or self.slots:
            raise RuntimeError("Pool preparation is one-time")
        for _ in range(2):
            # Publish every strong tensor ref before priming/event creation,
            # including a partial OOM or event-init failure. Never retry/free.
            slot = _Slot(self.backend.allocate(self.capacity_rows, self.K, self.dtype))
            self.slots.append(slot)
            self.backend.prime(slot.buffer)
            slot.reader_done = self.backend.initialized_event()
        if self.backend.aliases(self.slots[0].buffer, self.slots[1].buffer):
            raise RuntimeError("DBO output lanes must not alias")
        self.prepared = True
        _emit(self.description("prepared"))

    def description(self, event):
        return dict(event=event, device=str(self.device), slots=2,
            capacity_rows=self.capacity_rows, K=self.K, dtype=str(self.dtype),
            total_bytes=2*self.capacity_rows*self.K*2,
            base_ptrs=[self.backend.pointer(s.buffer) for s in self.slots])

    def acquire(self, lane, rows, K, dtype, stream, *, runtime=False, dbo=False, sizes=None):
        if not self.prepared or lane not in (0, 1):
            raise RuntimeError("Pool is incomplete or lane is invalid")
        if K != self.K or dtype != self.dtype or not 0 < rows <= self.capacity_rows:
            raise RuntimeError("DP-pipe pool capacity/geometry exceeded; no runtime growth")
        slot = self.slots[lane]
        if slot.busy or slot.poisoned:
            raise RuntimeError("DP-pipe output lane still leased or reader is unknown")
        if slot.has_reader:
            self.backend.wait(stream, slot.reader_done)
        slot.busy = True
        if runtime and not slot.acquire_logged:
            slot.acquire_logged = True
            _emit(dict(event="first-acquire", device=str(self.device), lane=lane,
                dbo=bool(dbo), rows=rows, sizes=list(sizes),
                base_ptr=self.backend.pointer(slot.buffer),
                compute_stream=self.backend.stream_id(stream),
                previous_final_reader_stream=slot.last_reader_stream,
                previous_reader_fence_sequence=slot.fence_sequence))
        # Exact rows, never the full capacity: direct_scratch must have S rows.
        return self.backend.rows(slot.buffer, rows)

    def release(self, lane, stream, *, runtime=False, dbo=False, rows=None, sizes=None, compute_stream=None):
        slot = self.slots[lane]
        if not slot.busy or slot.poisoned:
            raise RuntimeError("Cannot release an unleased/poisoned lane")
        try:
            self.backend.record(slot.reader_done, stream)
        except BaseException:
            slot.poisoned = True
            raise
        slot.has_reader, slot.busy = True, False
        slot.fence_sequence += 1
        slot.last_reader_stream = self.backend.stream_id(stream)
        if runtime and not slot.release_logged:
            slot.release_logged = True
            _emit(dict(event="first-release", device=str(self.device), lane=lane,
                dbo=bool(dbo), rows=rows, sizes=list(sizes),
                base_ptr=self.backend.pointer(slot.buffer),
                compute_stream=self.backend.stream_id(compute_stream),
                final_reader_stream=slot.last_reader_stream,
                recorded_reader_fence_sequence=slot.fence_sequence))


    def finish_profile(self):
        if not self.prepared or any(s.busy or s.poisoned for s in self.slots):
            raise RuntimeError("Incomplete or poisoned DP-pipe measured profile")
        stream = self.backend.current_stream()
        for slot in self.slots:
            if slot.has_reader:
                self.backend.wait(stream, slot.reader_done)
        self.backend.finish_profile(stream)
        self.locked = True
        _emit(self.description("locked"))


class _TorchBackend:
    def __init__(self, device):
        import torch
        self.torch, self.device = torch, device

    def capturing(self):
        return self.torch.cuda.is_current_stream_capturing()

    def allocate(self, rows, K, dtype):
        return self.torch.empty((rows, K), device=self.device, dtype=dtype)

    def prime(self, tensor):
        tensor.zero_()

    def initialized_event(self):
        # torch CUDA/HIP Events initialize lazily at first record. Do this in
        # the measured profile, never on the first serving/capture call.
        event = self.torch.cuda.Event(enable_timing=False, blocking=False)
        event.record(self.current_stream())
        return event

    def aliases(self, a, b):
        return a.data_ptr() == b.data_ptr()

    def pointer(self, tensor):
        return int(tensor.data_ptr())

    def stream_id(self, stream):
        return int(stream.cuda_stream)

    def current_stream(self):
        return self.torch.cuda.current_stream(self.device)

    def rows(self, tensor, count):
        return tensor[:count]

    def wait(self, stream, event):
        stream.wait_event(event)

    def record(self, event, stream):
        event.record(stream)

    def finish_profile(self, stream):
        stream.synchronize()


_BACKEND_FACTORY = _TorchBackend


@dataclass
class _State:
    device: object
    backend: object
    capacity_rows: int
    profile_thread: int
    profiling: bool = True
    failed: bool = False
    pool: object = None


def _device_key(device):
    # The configured worker device must include its explicit visible ordinal.
    index = getattr(device, "index", None)
    if type(index) is not int or index < 0:
        raise RuntimeError("DP-pipe pool requires an explicit worker device")
    return str(device), index


def _enabled():
    if os.environ.get(_FLAG, "0") != "1":
        return False
    from vllm_hcu.platforms import envs as henvs
    return henvs.optional_custom_op_enabled(True)


def _validate_config(config):
    pc, sc = config.parallel_config, config.scheduler_config
    cc, spec = config.cache_config, config.speculative_config
    if not (pc.data_parallel_size == 16 and pc.tensor_parallel_size == 1
            and pc.prefill_context_parallel_size == 1 and pc.enable_expert_parallel
            and pc.all2all_backend == "allgather_reducescatter"
            and sc.max_num_batched_tokens == 8192
            and (cc.kv_cache_memory_bytes is None or (
                type(cc.kv_cache_memory_bytes) is int
                and cc.kv_cache_memory_bytes == 5100273664
                and os.environ.get("EP_FUSED_MOE_CHUNK") in ("32768", "65536")))
            and (spec is None or (spec.num_speculative_tokens == 3
            and spec.rejection_sample_method == "standard"))):
        raise RuntimeError("Step5 output pool requires Native TP1/DP16/EP16, batch8192, no speculation or standard MTP3, and the supported KV allocation")
    if os.environ.get("EP_FUSED_MOE_CHUNK") == "65536" and cc.kv_cache_memory_bytes != 5100273664:
        raise RuntimeError("EP64 requires the fixed 4.75GiB KV recipe")
    expected = {"HCU_DP_MOE_PIPE_MIN_TOKENS": "16384", "HCU_DBO_MIN_TOKENS": "4096",
                "HCU_EP_DIRECT_SUM": "16384", "HCU_DP_MOE_PIPE_UNIFORM": "1",
                "HCU_STEP5_DENSE_TUNING": "1"}
    if (any(os.environ.get(k) != v for k, v in expected.items())
            or os.environ.get("EP_FUSED_MOE_CHUNK") not in ("16384", "32768", "65536")):
        raise RuntimeError("Step5 output-pool settings differ from the supported profile")
    # Do not assume DPMetadata was halved for every mixed/uneven DBO edge.
    # W*B bounds M even for full per-DP batch metadata. S<=M for either path.
    return 2 * pc.data_parallel_size * sc.max_num_batched_tokens


def _reserve_ep64_workspace(config, device):
    if os.environ.get("EP_FUSED_MOE_CHUNK") != "65536":
        return
    import torch
    import torch.distributed as dist
    from vllm.distributed.parallel_state import get_ep_group
    from vllm.v1.worker.workspace import current_workspace_manager
    group, manager = get_ep_group(), current_workspace_manager()
    assert group.world_size == 16 and str(dist.get_backend(group.cpu_group)) == "gloo"
    assert not manager.is_locked()
    # The ordinary unsplit profile needs the two bounded scratch tensors plus
    # its distinct output. Reserve before the first model collective so that
    # later DBO/fallback paths never grow the workspace underneath a collective.
    shapes = ((530176, 4096), (530176, 4096), (131072, 4096))
    assert sum(a*b*2 for a,b in shapes) == 9760145408
    torch.cuda.synchronize(device)
    dist.barrier(group=group.cpu_group)
    views = manager.get_simultaneous(*[(shape, torch.bfloat16) for shape in shapes])
    del views
    dist.barrier(group=group.cpu_group)
    _emit(dict(event="ep64-workspace-reserved", total_bytes=9760145408))


@contextmanager
def measured_profile_scope(config, device):
    from vllm_hcu.runtime_compat.step5 import is_step5_config
    if not is_step5_config(config) or not _enabled():
        yield
        return
    capacity = _validate_config(config)
    key = _device_key(device)
    with _LOCK:
        if key in _STATES:
            raise RuntimeError("DP-pipe pool profile already exists; use a fresh worker")
        state = _State(device, _BACKEND_FACTORY(device), capacity, threading.get_ident())
        if state.backend.capturing():
            raise RuntimeError("Cannot prepare DP-pipe pool during graph capture")
        _STATES[key] = state
    try:
        _reserve_ep64_workspace(config, device)
        # The unchanged super.profile_run executes inside Worker's existing
        # memory_profiling context when no explicit KV byte override is used.
        yield
        if state.pool is None:
            raise RuntimeError("Measured profile never initialized the DP-pipe pool")
        state.pool.finish_profile()
    except BaseException:
        state.failed = True
        raise
    finally:
        state.profiling = False


class _Lease:
    def __init__(self, pool, lane, buffer, *, runtime, dbo, rows, sizes, compute_stream):
        self.pool, self.lane, self.buffer = pool, lane, buffer
        self.runtime, self.dbo, self.rows, self.sizes = runtime, dbo, rows, list(sizes)
        self.compute_stream = compute_stream
        self.released = False

    def release_after_last_reader(self, stream):
        self.pool.release(self.lane, stream, runtime=self.runtime, dbo=self.dbo,
                          rows=self.rows, sizes=self.sizes, compute_stream=self.compute_stream)
        self.released = True


@contextmanager
def output_lease(*, device, dtype, K, rows, sizes, dbo):
    key = _device_key(device)
    if not _enabled():
        if key in _STATES:
            raise RuntimeError("Output-pool flag changed after profiling")
        yield None
        return
    with _LOCK:
        state = _STATES.get(key)
        if state is None or state.failed:
            raise RuntimeError("DP-pipe pool needs a successful measured profile first")
        if state.backend.capturing():
            # This pipe is eager-only at graph64 and threshold16384. Capturing
            # event reuse/replay is intentionally not claimed by this patch.
            raise RuntimeError("DP-pipe pooled path is not qualified for graph capture")
        if (len(sizes) != 16 or any(type(n) is not int or n < 0 or n > 8192 for n in sizes)
                or K != 4096 or str(dtype) != "torch.bfloat16"
                or not 0 < rows <= state.capacity_rows):
            raise RuntimeError("DP-pipe output geometry exceeds the measured bound")
        if state.pool is None:
            if not state.profiling or state.profile_thread != threading.get_ident():
                raise RuntimeError("No first-use allocation outside measured profile")
            # Publish before prepare, so even a partial allocation stays owned.
            state.pool = _Pool(state.backend, device=device, dtype=dtype, K=K,
                               capacity_rows=state.capacity_rows)
            state.pool.prepare()
        if not state.profiling and not state.pool.locked:
            raise RuntimeError("DP-pipe pool was not locked after measured profile")
        if dbo:
            # Do not use workspace.dbo_current_ubatch_id: HCU patches it to0.
            from vllm.v1.worker.ubatching import dbo_current_ubatch_id
            lane = dbo_current_ubatch_id()
        else:
            lane = 0
        compute_stream = state.backend.current_stream()
        runtime = not state.profiling
        buffer = state.pool.acquire(lane, rows, K, dtype, compute_stream,
                                    runtime=runtime, dbo=dbo, sizes=sizes)
        lease = _Lease(state.pool, lane, buffer, runtime=runtime, dbo=dbo,
                       rows=rows, sizes=sizes, compute_stream=compute_stream)
    try:
        yield lease
        if not lease.released:
            raise RuntimeError("DP-pipe output returned without a final-reader fence")
    except BaseException:
        # Preserve refs and prohibit reuse after an unknown asynchronous reader.
        state.pool.slots[lane].poisoned = True
        raise
