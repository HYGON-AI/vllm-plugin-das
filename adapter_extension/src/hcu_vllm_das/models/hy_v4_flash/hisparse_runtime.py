# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Eager Core033 HiSparse ownership and DFly alias consumption.

Core allocates the target source/resident/hot caches and request state. A single
DFly reader omitted by Core's allocation reuses that entire lease and native
replacement state, but has private top-k workspace, stream and write staging.
Its real Core writer publishes draft-owned source slots and resident copies.
This is a synchronizing correctness slice, not engine admission or Graph support.
No global Core/provider/allocator patch is made. Reallocation requires teardown.
"""
from __future__ import annotations

from contextlib import contextmanager
import os


class HiSparseBindingError(RuntimeError):
    """Unsupported ownership/layout/batch; never fall back to a host KV kernel."""


def _layout(tensor):
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.dtype, tensor.device, tensor.storage_offset())


def same_cache_layout(left, right):
    return _layout(left) == _layout(right)


class HiSparseRuntimeBinding:
    """Fixed ownership lease over an already fully allocated Core cache handle.

    Runtime/source/hot/replacement tensors remain owned by Core. Query indices
    remain owned by each backend. Only explicitly bound reader aliases may
    share the full allocation lease; index-sharing follower groups fail closed.
    ``stream`` is Core's current_stream function, injected by the real backend.
    """

    _CACHE_FIELDS = (
        'view', 'block_table', 'residency', 'slot_mapping', 'source_block_table',
        'mirror_slot_mapping',
    )
    _RUNTIME_FIELDS = (
        'hot', 'hot_backing', '_host_cache', 'registered_host_pool',
        'hot_block_table', 'request_state_indices', 'shared_host_region',
    )
    _GROUP_FIELDS = (
        'row_indices', 'request_ids',
        'lru_init', 'device_global_indices', 'lru_slots', 'swap_stats',
        'shared_topk', 'copy_stream', 'logical_topk_ready',
    )
    _WORKSPACE_FIELDS = (
        'device_topk_rows', 'physical_topk_indices', 'swap_host_physical_rows',
        'swap_device_physical_rows', 'swap_counts', 'valid_topk_counts',
    )

    def __init__(self, cache, *, stream, owner=None):
        self._owner = owner
        self._readers = {}
        self.cache = cache
        self.runtime = cache.runtime
        self.group = cache.runtime.index_group
        self._pid = os.getpid()
        self._stream = stream
        self._failure = None
        self._topology()
        runtime = self.runtime
        for obj, names in ((cache, self._CACHE_FIELDS),
                           (runtime, self._RUNTIME_FIELDS)):
            for name in names:
                if not hasattr(obj, name) or (
                    getattr(obj, name) is None
                    and name != 'shared_host_region'
                ):
                    raise HiSparseBindingError(f'incomplete Core binding: {name}')
        if runtime.host_cache.device.type != 'cpu':
            raise HiSparseBindingError('source must be the Core pinned host pool')
        if (runtime.host_cache.dtype != runtime.kv_dtype
                or runtime.host_cache.shape[-1] != runtime.row_width):
            raise HiSparseBindingError('source row layout mismatch')
        if not same_cache_layout(cache.view.cache, runtime.hot.cache):
            raise HiSparseBindingError('Core resident/hot views must alias exactly')
        if (cache.view.block_size != runtime.hot.block_size
                or cache.view.attention_block_stride
                != runtime.hot.attention_block_stride):
            raise HiSparseBindingError('resident/hot block layout mismatch')
        if (runtime.host_cache.untyped_storage().data_ptr()
                != runtime.registered_host_pool.untyped_storage().data_ptr()):
            raise HiSparseBindingError('source is outside its registered host pool')
        if (runtime.hot.cache.untyped_storage().data_ptr()
                != runtime.hot_backing.untyped_storage().data_ptr()):
            raise HiSparseBindingError('hot view is outside its Core backing')
        for name, tensor in (
            ('source block table', cache.source_block_table),
            ('resident block table', cache.block_table),
            ('hot block table', runtime.hot_block_table),
        ):
            if (len(tensor.shape) != 2 or tensor.shape[0] != runtime.max_num_reqs
                    or tensor.device != runtime.device
                    or str(tensor.dtype).split('.')[-1] != 'int32'):
                raise HiSparseBindingError(f'invalid {name} layout')
        for name, tensor, dtypes in (
            ('resident slots', cache.slot_mapping, ('int64',)),
            ('source slots', cache.mirror_slot_mapping, ('int64',)),
            ('request state', runtime.request_state_indices, ('int32',)),
        ):
            if (len(tensor.shape) != 1 or tensor.device != runtime.device
                    or str(tensor.dtype).split('.')[-1] not in dtypes):
                raise HiSparseBindingError(f'invalid {name} layout')
        if (runtime.hot.cache.device != runtime.device
                or runtime.hot.cache.dtype != runtime.kv_dtype
                or runtime.hot.cache.shape[-1] != runtime.row_width):
            raise HiSparseBindingError('hot row layout mismatch')
        if (runtime.request_state_indices.numel() < runtime.max_num_reqs
                or runtime.hot_block_table.shape[1] * runtime.hot.block_size
                < runtime.region_stride):
            raise HiSparseBindingError('request/hot table capacity is incomplete')
        if (cache.residency.device != runtime.device
                or len(cache.residency.shape) != 3
                or not 0 <= runtime.resident_source_index < cache.residency.shape[0]
                or cache.residency.shape[1:] != cache.block_table.shape
                or str(cache.residency.dtype).split('.')[-1] != 'int32'
                or not same_cache_layout(
                    cache.block_table,
                    cache.residency[runtime.resident_source_index])):
            raise HiSparseBindingError('resident table is outside its ownership state')
        self._scalars = self._scalar_state()
        self._objects = []
        for obj, names in ((cache, self._CACHE_FIELDS),
                           (runtime, self._RUNTIME_FIELDS),
                           (self.group, self._GROUP_FIELDS),
                           (self.group.shared_topk, self._WORKSPACE_FIELDS),
                           (cache.view, ('cache', 'attention_cache')),
                           (runtime.hot, ('cache', 'attention_cache'))):
            for name in names:
                value = getattr(obj, name)
                layout = _layout(value) if hasattr(value, 'data_ptr') else None
                self._objects.append((obj, name, value, layout))

    def _scalar_state(self):
        r = self.runtime
        return (r.row_width, r.kv_dtype, r.device, r.max_num_reqs,
                r.region_stride, r.max_union_rows, r.resident_source_index,
                r.max_swap_rows, r.hot.block_size, r.hot.attention_block_stride)

    def _topology(self):
        r, g, c = self.runtime, self.group, self.cache
        if (c.runtime is not r or r.index_group is not g
                or not r.is_group_leader or g.leader is not r or g.followers
                or len(c.index_group_caches) != 1
                or c.index_group_caches[0] is not c):
            raise HiSparseBindingError('HiSparse binding requires a singleton leader')
        if c.draft_layer and self._owner is None:
            raise HiSparseBindingError(
                'DFly source/hot/slot alias binding is not qualified'
            )

    @property
    def writer_cache(self):
        return self.cache.view.cache

    def _participants(self):
        owner = self._owner or self
        return (owner, *owner._readers.values())

    def _fence_compute(self):
        compute = self._stream()
        for binding in self._participants():
            binding._healthy()
            if binding.runtime._swap_staged:
                raise HiSparseBindingError('participant has a staged swap')
            compute.wait_stream(binding.group.copy_stream)
        return compute

    def _poison(self, error):
        # Sharing replacement state also shares failure fate: never let the
        # target consume a half-published reader write (or vice versa).
        for binding in self._participants():
            binding._failure = error

    def _healthy(self):
        if os.getpid() != self._pid:
            raise HiSparseBindingError('HiSparse binding inherited across fork')
        if self._failure is not None:
            raise HiSparseBindingError(
                'HiSparse device operation failed; restart'
            ) from self._failure

    def check(self, *, source, layers=()):
        self._healthy()
        self._topology()
        for obj, name, value, layout in self._objects:
            current = getattr(obj, name, None)
            if current is not value or (
                layout is not None and _layout(current) != layout
            ):
                raise HiSparseBindingError(f'Core binding changed: {name}')
        if self._scalar_state() != self._scalars:
            raise HiSparseBindingError('Core cache capacity/layout changed')
        host = self.runtime.host_cache
        if not same_cache_layout(source.view(-1, self.runtime.row_width), host):
            raise HiSparseBindingError('layer source does not own the bound host rows')
        owner_layers = set()
        for layer in layers:
            other = getattr(layer, 'hisparse_cache', None)
            tensor = getattr(layer, 'kv_cache', None)
            if other is self.cache:
                owner_layers.add(id(layer))
                if len(owner_layers) > 1:
                    raise HiSparseBindingError('one cache cannot own two query layers')
                continue
            # Target/draft aliases have the same first row even after DFly clears
            # kv_sharing_target_layer_name. Distinct interleaved host layer views
            # share a backing pool, but NOT this data_ptr; do not reject the pool.
            if (tensor is not None and tensor.numel()
                    and tensor.data_ptr() == host.data_ptr()):
                reader = self._readers.get(id(layer))
                if reader is not None and reader.cache is other:
                    reader.check(source=tensor)
                    continue
                if self._owner is not None and other is self._owner.cache:
                    self._owner.check(source=tensor)
                    continue
                raise HiSparseBindingError(
                    'source-only KV alias lacks full HiSparse ownership'
                )

    def validate_batch(self, metadata, *, query_rows, topk_width,
                       allow_decode_prefix=False):
        """Validate actual request runs, not a uniform-decode approximation.

        Target prefill/chunked-prefill writes use resident slots and Core's
        connector mirrors them before they become host sources. Only decode
        rows consume the native swap/union scratch: applying those limits to
        the entire prompt incorrectly rejects ordinary engine startup/prefill.
        Core may route prefill queries through dense MHA; in that case the MQA
        reader sees only the decode prefix, while the writer still sees all KV.
        Padding is not a source slot and is never admitted as a live row here.
        """
        self._healthy()
        c, r = self.cache, self.runtime
        if metadata is None or c.dummy_batch:
            raise HiSparseBindingError(
                'real HiSparse operations need prepared metadata'
            )
        n, requests, width = (metadata.num_actual_tokens, metadata.num_reqs,
                              metadata.max_query_len)
        decodes, prefills, decode_rows = (metadata.num_decodes,
            metadata.num_prefills, metadata.num_decode_tokens)
        if (n <= 0 or requests <= 0 or width <= 0
                or not 0 <= decodes <= requests
                or prefills != requests - decodes
                or not 0 <= decode_rows <= n
                or (query_rows != n and not (
                    allow_decode_prefix and query_rows == decode_rows > 0))):
            raise HiSparseBindingError('query rows/request counts are inconsistent')
        if requests > r.max_num_reqs:
            raise HiSparseBindingError('request rows exceed Core swap capacity')
        if topk_width != self.group.shared_topk.physical_topk_indices.shape[1]:
            raise HiSparseBindingError('multi-query top-k union exceeds Core capacity')
        if metadata.block_size != r.hot.block_size:
            raise HiSparseBindingError('query/cache block size mismatch')
        decode_batch = decode_rows == n and width == 1 and requests == n
        if (c.num_actual_tokens != n or c.num_decode_tokens != decode_rows
                or c.req_id_per_token is not metadata.req_id_per_token
                or c.decode_batch != decode_batch):
            raise HiSparseBindingError('cache batch was not prepared for these queries')
        for tensor in (c.slot_mapping, c.mirror_slot_mapping, c.req_id_per_token):
            if tensor.numel() < n:
                raise HiSparseBindingError(
                    'query/source/resident slot rows are incomplete'
                )
        for tensor in (c.req_id_per_token, metadata.query_start_loc):
            if (len(tensor.shape) != 1 or tensor.device != r.device
                    or str(tensor.dtype).split('.')[-1] != 'int32'):
                raise HiSparseBindingError('query ids/boundaries layout mismatch')
        if metadata.query_start_loc.numel() < requests + 1:
            raise HiSparseBindingError('query boundaries do not cover every request')
        # Deliberately synchronizing eager validation; no graph/perf claim.
        boundaries = metadata.query_start_loc[:requests + 1].tolist()
        lengths = [right - left for left, right in
                   zip(boundaries[:-1], boundaries[1:])]
        request_ids = c.req_id_per_token[:n].tolist()
        if (boundaries[0] != 0 or boundaries[-1] != n
                or len(lengths) != requests or min(lengths) <= 0
                or max(lengths) != width or boundaries[decodes] != decode_rows
                or request_ids != [req for req, count in enumerate(lengths)
                                   for _ in range(count)]):
            raise HiSparseBindingError('queries must be contiguous request-major runs')
        decode_width = max(lengths[:decodes], default=0)
        if decode_rows > r.max_swap_rows:
            raise HiSparseBindingError('multi-query rows exceed Core swap capacity')
        if decode_width * topk_width > min(r.max_union_rows, r.region_stride):
            raise HiSparseBindingError('multi-query top-k union exceeds Core capacity')
        states = r.request_state_indices[:requests].tolist()
        if (len(set(states)) != requests
                or any(not 0 <= state < r.max_num_reqs for state in states)):
            raise HiSparseBindingError('request state rows are inactive or overlap')
        for slots, limit in ((c.mirror_slot_mapping, r.host_cache.shape[0]),
                             (c.slot_mapping, self.writer_cache.shape[0]
                              * self.writer_cache.shape[1])):
            live_slots = slots[:n].tolist()
            if (len(set(live_slots)) != n
                    or any(not 0 <= slot < limit for slot in live_slots)):
                raise HiSparseBindingError(
                    'live source/resident slots exceed bound storage'
                )
        self._validate_slot_ownership(request_ids, states, n)

    def _validate_slot_ownership(self, request_ids, states, rows):
        c, r = self.cache, self.runtime
        bs = r.hot.block_size
        source = c.source_block_table.tolist()
        resident = c.block_table.tolist()
        source_slots = c.mirror_slot_mapping[:rows].tolist()
        writer_slots = c.slot_mapping[:rows].tolist()
        for req, host_slot, writer_slot in zip(
                request_ids, source_slots, writer_slots, strict=True):
            pages = [page for page, block in enumerate(source[req])
                     if block == host_slot // bs]
            if host_slot // bs <= 0 or len(pages) != 1:
                raise HiSparseBindingError('source write has no unique owned page')
            if self._owner is None:
                page = pages[0]
                if (page >= len(resident[states[req]])
                        or resident[states[req]][page] <= 0
                        or writer_slot != resident[states[req]][page] * bs
                        + host_slot % bs):
                    raise HiSparseBindingError('resident/source write ownership mismatch')

    def write(self, writer, *, kv_cache, slot_mapping, rows):
        """Invalidate HOST slot ids, then write RESIDENT slots on the Core stream.

        The Core logical-ready event may have been recorded before the KV write.
        Explicitly fence the copy stream after the writer: waiting only on that
        event would race both invalidation and newly written resident data.
        """
        self._healthy()
        c, r = self.cache, self.runtime
        if rows != c.num_actual_tokens or slot_mapping.numel() != rows:
            raise HiSparseBindingError('Core write_target truncated multi-query rows')
        if (not same_cache_layout(kv_cache, self.writer_cache)
                or not same_cache_layout(slot_mapping, c.slot_mapping[:rows])):
            raise HiSparseBindingError('writer must use the bound resident slots')
        if r._swap_staged:
            raise HiSparseBindingError('cannot write while a layer swap is staged')
        try:
            compute = self._fence_compute()
            r.invalidate_written_slots(
                c.mirror_slot_mapping[:rows], c.req_id_per_token[:rows]
            )
            result = writer()
            self._publish_rows(rows)
            for binding in self._participants():
                binding.group.copy_stream.wait_stream(compute)
            return result
        except Exception as error:
            # A native writer/reset cannot be rolled back from Python.
            self._poison(error)
            raise

    def _publish_rows(self, rows):
        pass  # Core/connector owns ordinary target resident->host mirroring.

    def read(self, reader):
        """Latch failed Core conversion/swap/kernel calls; never retry stale LRU."""
        self._healthy()
        try:
            compute = self._fence_compute()
            self.group.copy_stream.wait_stream(compute)
            return reader()
        except Exception as error:
            self._poison(error)
            raise


class HiSparseReaderBinding(HiSparseRuntimeBinding):
    """Eager correctness binding: shared source/hot, private query/write scratch.

    A draft omitted by Core allocation has no resident writable-slot allocation.
    Never borrow the target step's slots. Pack with the real Core writer into a
    private GPU staging cache, synchronously publish those rows to the registered
    shared host source, and update any existing resident copies by page mapping.
    This is deliberately a correctness baseline, NOT asynchronous offload/perf.
    """

    def __init__(self, cache, owner, *, stream, torch, staging, dense_slots):
        self._torch = torch
        self._staging = staging
        self._query_source_slots = None
        self._resident_updates = None
        super().__init__(cache, stream=stream, owner=owner)
        for name in ('_staging',):
            value = getattr(self, name)
            self._objects.append((self, name, value, _layout(value)))
        if dense_slots is not cache.slot_mapping:
            raise HiSparseBindingError('reader does not own its write slots')

    @property
    def writer_cache(self):
        return self._staging

    def prepare_query(self, metadata, source_slots):
        self.check(source=self.layer.kv_cache)
        self._owner.check(source=self.runtime.host_cache)
        # A previous prepare must never authorize a later write on stale slots.
        self._query_source_slots = None
        self._resident_updates = None
        self._prepared_metadata = None
        n, reqs = metadata.num_actual_tokens, metadata.num_reqs
        if (n <= 0 or n > self.runtime.max_swap_rows or reqs <= 0
                or reqs > self.runtime.max_num_reqs
                or source_slots is None or source_slots.numel() != n
                or len(source_slots.shape) != 1
                or source_slots.device != self.runtime.device
                or str(source_slots.dtype).split('.')[-1] != 'int64'):
            raise HiSparseBindingError('reader query/source slots are incomplete')
        # finish_forward() precedes DFly propose() in pinned MRV2 and enqueues a
        # compute-stream wait on host_write_event. Drain that wait and both copy
        # streams before any CPU source publication / replacement-state sharing.
        compute = self._fence_compute()
        compute.synchronize()
        source_table = self.cache.source_block_table[:reqs].tolist()
        if metadata.block_table[:reqs].tolist() != source_table:
            raise HiSparseBindingError('reader and target request/page ordering differs')
        slots = source_slots[:n].tolist()
        ids = metadata.req_id_per_token[:n].tolist()
        if (len(ids) != n or len(set(slots)) != n
                or any(not 0 <= slot < self.runtime.host_cache.shape[0]
                       for slot in slots)):
            raise HiSparseBindingError('reader source slots overlap or exceed storage')
        states = self.runtime.request_state_indices[:reqs].tolist()
        resident_table = self.cache.block_table.tolist()
        bs = self.runtime.hot.block_size
        updates = []
        for row, (slot, req) in enumerate(zip(slots, ids)):
            if not 0 <= req < reqs or slot // bs <= 0:
                raise HiSparseBindingError('inactive/padded reader queries are not bound')
            pages = [page for page, block in enumerate(source_table[req])
                     if block == slot // bs]
            if len(pages) != 1 or not 0 <= states[req] < self.runtime.max_num_reqs:
                raise HiSparseBindingError('reader source slot has no unique owned page')
            page = pages[0]
            if page >= len(resident_table[states[req]]):
                raise HiSparseBindingError('reader resident page table is incomplete')
            block = resident_table[states[req]][page]
            if block < 0 or block >= self.cache.view.cache.shape[0]:
                raise HiSparseBindingError('reader resident page exceeds bound storage')
            if block:
                updates.append((row, block, slot % bs))
        self.cache.mirror_slot_mapping[:n].copy_(source_slots[:n])
        self._prepared_metadata = metadata
        self.cache.all_context_pages_resident = self._owner.cache.all_context_pages_resident
        self._query_source_slots = slots
        self._resident_updates = updates

    def validate_batch(self, metadata, **kwargs):
        if (self._query_source_slots is None
                or getattr(self, '_prepared_metadata', None) is not metadata):
            raise HiSparseBindingError('reader query was not prepared for this metadata')
        super().validate_batch(metadata, **kwargs)

    def write_target(self, num_input_rows, num_slot_rows):
        n = self.cache.num_actual_tokens
        if (self._query_source_slots is None or num_input_rows < n
                or num_slot_rows < n or n > self.runtime.max_swap_rows):
            raise HiSparseBindingError('reader was not prepared before write_target')
        return self._staging, self.cache.slot_mapping[:n], n

    def _publish_rows(self, rows):
        if self._query_source_slots is None or len(self._query_source_slots) != rows:
            raise HiSparseBindingError('reader source publication is not prepared')
        torch = self._torch
        staged = self._staging.view(-1, self.runtime.row_width)[:rows]
        # to(cpu) is intentionally blocking: the registered source must contain
        # fresh bytes before either target or reader can gather a hot miss.
        host_rows = staged.to(device='cpu')
        slots = torch.tensor(self._query_source_slots, dtype=torch.int64, device='cpu')
        self.runtime.host_cache.index_copy_(0, slots, host_rows)
        if self._resident_updates:
            row, block, offset = zip(*self._resident_updates)
            indices = [torch.tensor(values, dtype=torch.int64,
                                    device=self.runtime.device)
                       for values in (row, block, offset)]
            # Preserve the strided Core layer view; flattening interleaved slabs
            # would write another layer or require an unintended contiguous copy.
            self.cache.view.cache[indices[1], indices[2]] = staged[indices[0]]


def bind_hisparse_reader(owner, reader_cache, *, reader_layer, stream, torch):
    """Publish a complete per-instance binding after explicit stream quiescence.

    No Core allocator/class/global patch. No target source/resident slot mutation.
    The owner retains the reader until backend teardown; resize/detach is guarded.
    """
    owner._healthy()
    owner.check(source=owner.runtime.host_cache)
    if owner._owner is not None:
        raise HiSparseBindingError('reader chains are not supported')
    previous = owner._readers.get(id(reader_layer))
    if previous is not None:
        if previous.cache is not reader_cache:
            raise HiSparseBindingError('reader cache was replaced')
        return previous
    if owner._readers:
        raise HiSparseBindingError('one direct DFly reader per owner is supported')
    r, o, c = reader_cache.runtime, owner.runtime, reader_cache
    g = r.index_group
    if (r is o or not r.is_group_leader or g.leader is not r or g.followers
            or len(c.index_group_caches) != 1 or c.index_group_caches[0] is not c
            or r._swap_staged or o._swap_staged):
        raise HiSparseBindingError('reader binding needs distinct idle singleton leaders')
    if (r.row_width != o.row_width or r.kv_dtype != o.kv_dtype
            or r.device != o.device or r.max_num_reqs != o.max_num_reqs
            or r.max_union_rows > o.region_stride
            or o.shared_host_region is not None):
        raise HiSparseBindingError('reader layout/union or shared-host topology unsupported')
    if getattr(c, 'view', None) is not None or hasattr(r, 'hot'):
        raise HiSparseBindingError('reader already owns an allocation; no destructive rebind')
    if not same_cache_layout(reader_layer.kv_cache.view(-1, o.row_width), o.host_cache):
        raise HiSparseBindingError('reader does not alias the selected target source')
    compute = owner._fence_compute()
    compute.wait_stream(g.copy_stream)
    compute.synchronize()
    staging = torch.empty((r.max_swap_rows, 1, r.row_width),
                          dtype=r.kv_dtype, device=r.device)
    dense = torch.arange(r.max_swap_rows, dtype=torch.int64, device=r.device)
    mirror = torch.full((r.max_swap_rows,), -1, dtype=torch.int64, device=r.device)
    runtime_names = (*owner._RUNTIME_FIELDS, 'resident_source_index', 'region_stride')
    cache_names = (*owner._CACHE_FIELDS, 'draft_layer')
    state_names = ('lru_init', 'device_global_indices', 'lru_slots', 'swap_stats')
    saved = [(obj, name, hasattr(obj, name), getattr(obj, name, None))
             for obj, names in ((r, runtime_names), (c, cache_names), (g, state_names))
             for name in names]
    try:
        for name in runtime_names:
            setattr(r, name, getattr(o, name))
        for name in ('view', 'block_table', 'residency', 'source_block_table'):
            setattr(c, name, getattr(owner.cache, name))
        c.slot_mapping, c.mirror_slot_mapping, c.draft_layer = dense, mirror, True
        for name in state_names:
            setattr(g, name, getattr(owner.group, name))
        binding = HiSparseReaderBinding(c, owner, stream=stream, torch=torch,
                                       staging=staging, dense_slots=dense)
    except Exception:
        for obj, name, existed, value in saved:
            if existed:
                setattr(obj, name, value)
            elif hasattr(obj, name):
                delattr(obj, name)
        raise
    binding.layer = reader_layer
    owner._readers[id(reader_layer)] = binding
    return binding


_REPLACEMENT_FIELDS = ('lru_init', 'device_global_indices', 'lru_slots', 'swap_stats')


def bind_hisparse_resident_reader(owner, reader, source_cache, *, profiling=False):
    """Borrow a Core resident lease at KV initialization, before graph capture.

    Draft rows are transient and live in allocated lookahead slots. The target
    connector continues to own publication to the host source. Each reader
    keeps its query indices, staging events and copy stream.
    """
    target, draft = owner.runtime, reader.runtime
    source_group, draft_group = target.index_group, draft.index_group
    if (owner is reader or target is draft or owner.draft_layer
            or source_group is draft_group or not draft.is_group_leader
            or draft_group.leader is not draft or draft_group.followers
            or target._swap_staged or draft._swap_staged
            or reader.submit_layer_mirror is not None):
        raise HiSparseBindingError('invalid resident draft/target cache ownership')
    for name in ('kv_dtype', 'row_width', 'max_num_reqs', 'device'):
        if getattr(draft, name) != getattr(target, name):
            raise HiSparseBindingError(f'resident draft/target layout mismatch: {name}')
    if draft.max_union_rows > target.region_stride:
        raise HiSparseBindingError('resident draft top-k union exceeds target capacity')

    host = getattr(target, '_host_cache', None)
    source = source_cache if len(source_cache.shape) == 2 else source_cache.view(-1, target.row_width)
    if (host is None or not same_cache_layout(source, host) or owner.view is None
            or not same_cache_layout(owner.view.cache, target.hot.cache)):
        raise HiSparseBindingError('resident draft needs the bound target host and GPU views')

    aliases = [(reader, name, getattr(owner, name)) for name in HiSparseRuntimeBinding._CACHE_FIELDS]
    aliases.extend((draft, name, getattr(target, name)) for name in (
        *HiSparseRuntimeBinding._RUNTIME_FIELDS, 'resident_source_index', 'region_stride'))
    aliases.extend((draft_group, name, getattr(source_group, name)) for name in _REPLACEMENT_FIELDS)
    if any(value is None and name != 'shared_host_region' for _, name, value in aliases):
        raise HiSparseBindingError('incomplete target resident cache binding')

    current_owner = getattr(reader, '_hcu_resident_owner', None)
    if current_owner is not None and current_owner is not owner:
        raise HiSparseBindingError('resident draft already has a different target')
    if hasattr(draft, '_host_cache'):
        if current_owner is not owner or any(getattr(obj, name, None) is not value
                                              for obj, name, value in aliases):
            raise HiSparseBindingError('resident draft binding changed without teardown')
        return
    if profiling:
        reader._hcu_profile_resident_state = (
            draft.region_stride,
            tuple(getattr(draft_group, name) for name in _REPLACEMENT_FIELDS),
        )
    for obj, name, value in aliases:
        setattr(obj, name, value)
    reader.draft_layer = True
    reader._hcu_resident_owner = owner


def release_hisparse_resident_reader(reader):
    """Drop Plugin aliases to Core's throwaway KV allocation after profiling."""
    state = getattr(reader, '_hcu_profile_resident_state', None)
    if state is None:
        return
    draft = reader.runtime
    for name in HiSparseRuntimeBinding._CACHE_FIELDS:
        setattr(reader, name, None)
    for name in ('hot', 'hot_backing', 'hot_block_table', '_host_cache', 'registered_host_pool'):
        if hasattr(draft, name):
            delattr(draft, name)
    draft.request_state_indices = None
    draft.shared_host_region = None
    draft.resident_source_index = -1
    draft.region_stride, replacement = state
    for name, value in zip(_REPLACEMENT_FIELDS, replacement, strict=True):
        setattr(draft.index_group, name, value)
    reader.draft_layer = False
    del reader._hcu_resident_owner
    del reader._hcu_profile_resident_state


@contextmanager
def suspend_hisparse_mirrors(layers):
    """Synthetic graph capture has no request or connector lifecycle."""
    callbacks = []
    try:
        for layer in layers:
            cache = getattr(layer, 'hisparse_cache', None)
            callback = getattr(cache, 'submit_layer_mirror', None)
            if callback is not None:
                callbacks.append((cache, callback))
                cache.submit_layer_mirror = None
        yield
    finally:
        for cache, callback in callbacks:
            cache.submit_layer_mirror = callback


def install_hisparse_prewrite_hook(impl, cache):
    """Instance-local hook called by Core BEFORE write_target(), not after it."""
    installed = getattr(cache, '_hcu_prepare_owner', None)
    if installed is not None:
        if installed is not impl:
            raise HiSparseBindingError('HiSparse cache hook has another owner')
        return
    original_prepare = cache.prepare_group_for_batch
    original_target = cache.write_target

    def prepare(metadata):
        if metadata is None:
            # Core cannot safely select an unallocated draft's dummy writer.
            # It marks dummy_batch itself and skips the update entirely.
            return original_prepare(metadata)
        impl._prepare_hisparse_layer(metadata)
        return original_prepare(metadata)

    def write_target(num_input_rows, num_slot_rows):
        binding = getattr(impl, '_hcu_hisparse_binding', None)
        if isinstance(binding, HiSparseReaderBinding):
            return binding.write_target(num_input_rows, num_slot_rows)
        return original_target(num_input_rows, num_slot_rows)

    cache.prepare_group_for_batch = prepare
    cache.write_target = write_target
    cache._hcu_prepare_owner = impl
