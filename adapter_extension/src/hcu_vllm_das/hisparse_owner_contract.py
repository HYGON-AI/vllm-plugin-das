# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Unwired Core033 hot-cache owner contract; NOT HiSparse+DFly admission.

Explicit, process-local provider, never an import-time patch/registration. Only
hot views and replacement state are covered: host/resident/source slots, model
loading and speculative writes still require a separate binding integration.
Every lifecycle call requires caller-established quiescence of ALL participants.
The lock serializes lifecycle calls, not attention or device-stream execution.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import import_module
from inspect import signature
import os
from threading import RLock
from typing import Any


_STATE = ("lru_init", "device_global_indices", "lru_slots", "swap_stats")
_HOT = ("hot", "hot_backing", "hot_block_table")
_METADATA = ("row_width", "kv_dtype", "device", "max_num_reqs")


class OwnerContractError(RuntimeError):
    """Unsupported ABI, ownership or lifecycle; never a silent fallback."""


@dataclass(frozen=True)
class _Layout:
    raw: Any
    byte_offset: int
    block_stride: int
    num_blocks: int
    block_size: int
    block_table: Any

    def kwargs(self):
        return {name: getattr(self, name) for name in (
            "byte_offset", "block_stride", "num_blocks", "block_size", "block_table"
        )}


@dataclass
class _Reader:
    runtime: Any
    stride: int
    state: tuple


@dataclass
class _Owner:
    runtime: Any
    layout: _Layout
    readers: dict
    state: tuple
    stride: int
    views: dict
    groups: dict
    metadata: tuple


def get_owner_provider():
    """Resolve the real Core lazily; importing this module loads no Torch/Core.

    Deliberately not installed into any runtime seam. The caller retains one
    provider for the entire lifetime of its owner/reader runtimes.
    """
    core = import_module("vllm.v1.hisparse.runtime")
    return Core033OwnerProvider(core, import_module("torch"))


class Core033OwnerProvider:
    """Instance-owned owner leases over original Core033 bind/reset methods.

    Reader index groups remain distinct (top-k scratch/streams are NOT shared).
    Just the four replacement tensors and the logical region stride alias the
    owner, including its int16 reset template. Followers/owner chains and resize
    are intentionally unsupported. No native schema or global class is changed.
    """

    def __init__(self, core, torch):
        runtime_type = core.HiSparseRuntime
        expected = ("self", "raw_tensor", "byte_offset", "block_stride",
                    "num_blocks", "block_size", "block_table")
        parameters = signature(runtime_type.bind_hot_cache).parameters
        if tuple(parameters) != expected or any(
            parameters[name].kind != parameters[name].KEYWORD_ONLY
            for name in expected[2:]
        ):
            raise OwnerContractError("requires exact Core033 bind_hot_cache ABI")
        if not callable(getattr(runtime_type, "reset_hot_state", None)):
            raise OwnerContractError("Core033 reset_hot_state is missing")
        self._runtime_type = runtime_type
        self._bind = runtime_type.bind_hot_cache
        self._reset = runtime_type.reset_hot_state
        self._view = core.PagedCacheView
        self._torch = torch
        self._pid = os.getpid()
        self._lock = RLock()
        self._owners = {}
        self._readers = {}
        self._failure = None

    def _check(self, quiescent):
        # Check before taking a potentially fork-inherited lock.
        if os.getpid() != self._pid:
            raise OwnerContractError("owner provider inherited across fork")
        if self._failure is not None:
            raise OwnerContractError("owner reset failed; restart with fresh runtimes") from self._failure
        if quiescent is not True:
            raise OwnerContractError("caller must establish quiescence for all runtimes")

    def _idle(self, runtime):
        if type(runtime) is not self._runtime_type:
            raise OwnerContractError("runtime must be an exact Core033 HiSparseRuntime")
        group = runtime.index_group
        if (not runtime.is_group_leader or group.leader is not runtime
                or group.followers):
            raise OwnerContractError("requires a singleton leader group without followers")
        if runtime.device.type not in ("cpu", "cuda"):
            raise OwnerContractError("only Core033 CPU fixtures or CUDA/HIP runtimes are supported")
        if runtime._swap_staged:
            raise OwnerContractError("staged swaps must complete before lifecycle changes")

    def _state(self, runtime):
        self._idle(runtime)
        torch, group = self._torch, runtime.index_group
        capacity, count = runtime.region_stride, runtime.max_num_reqs
        if (type(capacity) is not int or not 0 < capacity <= 32768
                or type(count) is not int or count <= 0):
            raise OwnerContractError("Core033 int16 LRU capacity must be in [1, 32768]")
        if not 0 < runtime.max_union_rows <= capacity:
            raise ValueError("max_union_rows exceeds device_buffer_size")
        layouts = (
            (group.lru_init, (capacity,), torch.int16),
            (group.device_global_indices, (count, capacity), torch.int32),
            (group.lru_slots, (count, capacity), torch.int16),
            (group.swap_stats, (2,), torch.uint64),
        )
        for tensor, shape, dtype in layouts:
            if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                    or tensor.dtype != dtype or tensor.device != runtime.device
                    or not tensor.is_contiguous()):
                raise OwnerContractError("Core033 replacement tensor shape/dtype/device drift")
        pointers = {tensor.untyped_storage().data_ptr() for tensor, _, _ in layouts}
        if len(pointers) != len(layouts):
            raise OwnerContractError("replacement tensors must have distinct storage")
        if not torch.equal(group.lru_init, torch.arange(
            capacity, dtype=torch.int16, device=runtime.device
        )):
            raise OwnerContractError("Core033 lru_init is not the canonical int16 template")

    def _fresh(self, runtime):
        self._state(runtime)
        if runtime in self._owners or runtime in self._readers or any(
            hasattr(runtime, name) for name in _HOT
        ):
            raise OwnerContractError("runtime is already bound; detach before changing owner")
        if any(entry.runtime.index_group is runtime.index_group
               for entry in self._owners.values()):
            raise OwnerContractError("an index group cannot have two owners")
        pointers = {getattr(runtime.index_group, name).untyped_storage().data_ptr()
                    for name in _STATE}
        if any(pointers & {tensor.untyped_storage().data_ptr() for tensor in entry.state}
               for entry in self._owners.values()):
            raise OwnerContractError("unbound runtime already aliases another owner state")

    def _layout(self, runtime, layout):
        torch = self._torch
        raw, table = layout.raw, layout.block_table
        if (not isinstance(raw, torch.Tensor) or raw.dtype != torch.uint8
                or raw.ndim != 1 or not raw.is_contiguous() or raw.storage_offset() != 0
                or raw.device != runtime.device):
            raise OwnerContractError("hot backing must be a base contiguous 1D uint8 tensor")
        if (not isinstance(table, torch.Tensor) or table.dtype != torch.int32
                or table.ndim != 2 or not table.is_contiguous()
                or table.device != runtime.device
                or table.shape[0] != runtime.max_num_reqs):
            raise OwnerContractError("hot block table shape/dtype/device drift")
        if any(type(value) is not int or value <= 0 for value in (
            layout.block_stride, layout.num_blocks, layout.block_size
        )) or type(layout.byte_offset) is not int or layout.byte_offset < 0:
            raise OwnerContractError("invalid hot-cache layout dimensions")
        row_bytes = runtime.row_width * runtime.kv_dtype.itemsize
        if (row_bytes <= 0 or row_bytes % 16 or layout.byte_offset % runtime.kv_dtype.itemsize
                or layout.block_stride % (layout.block_size * row_bytes)
                or table.shape[1] * layout.block_size < runtime.region_stride):
            raise OwnerContractError("unsupported Core033 hot layout or insufficient table capacity")
        required = (layout.byte_offset + (layout.num_blocks - 1) * layout.block_stride
                    + layout.block_size * row_bytes)
        if required > raw.numel():
            # as_strided alone checks STORAGE, not the supplied tensor's extent.
            raise OwnerContractError("hot view exceeds the raw tensor byte extent")
        for entry in self._owners.values():
            if entry.runtime is runtime:
                continue
            other = entry.layout
            if raw.untyped_storage().data_ptr() == other.raw.untyped_storage().data_ptr():
                other_end = (other.byte_offset + (other.num_blocks - 1) * other.block_stride
                             + other.block_size * entry.runtime.row_width
                             * entry.runtime.kv_dtype.itemsize)
                if layout.byte_offset < other_end and other.byte_offset < required:
                    raise OwnerContractError("independent owner hot ranges overlap")
        # Use the original Core view implementation for bounds/stride validation.
        # This is allocation-free and does not change the runtime on failure.
        self._view.bind(raw, dtype=runtime.kv_dtype, row_width=runtime.row_width,
                        **{k: v for k, v in layout.kwargs().items() if k != "block_table"})

    def _entry(self, runtime):
        owner = self._readers.get(runtime, runtime)
        if owner not in self._owners:
            raise OwnerContractError("runtime is not bound by this provider")
        entry = self._owners[owner]
        self._state(owner)
        for member in (entry.runtime, *entry.readers):
            self._idle(member)
            if (member.index_group is not entry.groups[member]
                    or getattr(member, "hot", None) is not entry.views[member]
                    or getattr(member, "hot_backing", None) is not entry.layout.raw
                    or getattr(member, "hot_block_table", None) is not entry.layout.block_table
                    or member.region_stride != entry.stride
                    or tuple(getattr(member, name) for name in _METADATA) != entry.metadata
                    or any(getattr(member.index_group, name) is not tensor
                           for name, tensor in zip(_STATE, entry.state))):
                raise OwnerContractError("external hot-cache binding/state mutation detected")
        return entry

    def _bind_fresh(self, runtime, layout):
        try:
            self._bind(runtime, layout.raw, **layout.kwargs())
        except Exception:
            for name in _HOT:
                if hasattr(runtime, name):
                    delattr(runtime, name)
            raise

    def bind_owner(self, runtime, raw_tensor, *, byte_offset, block_stride,
                   num_blocks, block_size, block_table, quiescent=False):
        self._check(quiescent)
        with self._lock:
            self._check(quiescent)
            layout = _Layout(raw_tensor, byte_offset, block_stride, num_blocks,
                             block_size, block_table)
            if runtime in self._owners:
                entry = self._entry(runtime)
                old = entry.layout
                if (old.raw is raw_tensor and old.block_table is block_table
                        and (old.byte_offset, old.block_stride, old.num_blocks, old.block_size) ==
                        (byte_offset, block_stride, num_blocks, block_size)):
                    return  # Idempotence must not discard live replacement state.
                raise OwnerContractError("owner already bound; use atomic rebind_owner")
            self._fresh(runtime)
            self._layout(runtime, layout)
            self._bind_fresh(runtime, layout)
            entry = _Owner(
                runtime=runtime, layout=layout, readers={},
                state=tuple(getattr(runtime.index_group, name) for name in _STATE),
                stride=runtime.region_stride, views={runtime: runtime.hot},
                groups={runtime: runtime.index_group},
                metadata=tuple(getattr(runtime, name) for name in _METADATA),
            )
            self._owners[runtime] = entry
            self._reset_entry(entry)  # A new slab must never inherit old hits.

    def bind_reader(self, runtime, *, owner, quiescent=False):
        self._check(quiescent)
        with self._lock:
            self._check(quiescent)
            if owner not in self._owners:
                raise OwnerContractError("owner must be directly bound before its readers")
            entry = self._entry(owner)
            if runtime is owner:
                raise OwnerContractError("runtime cannot own itself as a reader")
            if self._readers.get(runtime) is owner:
                self._entry(runtime)
                return
            self._fresh(runtime)
            if runtime.max_union_rows > owner.region_stride:
                raise ValueError(
                    f"reader max_union_rows={runtime.max_union_rows} exceeds "
                    f"owner device_buffer_size={owner.region_stride}"
                )
            if (any(getattr(runtime, name) != getattr(owner, name) for name in _METADATA)
                    or runtime.index_group is owner.index_group):
                raise OwnerContractError("reader/owner layout or request capacity mismatch")
            # Validate first; original bind must succeed BEFORE publishing aliases.
            self._bind_fresh(runtime, entry.layout)
            reader = _Reader(runtime, runtime.region_stride, tuple(
                getattr(runtime.index_group, name) for name in _STATE
            ))
            runtime.region_stride = owner.region_stride
            for name in _STATE:
                setattr(runtime.index_group, name, getattr(owner.index_group, name))
            entry.views[runtime] = runtime.hot
            entry.groups[runtime] = runtime.index_group
            entry.readers[runtime] = reader
            self._readers[runtime] = owner

    def _reset_entry(self, entry):
        try:
            owner = entry.runtime
            if owner.device.type == "cpu":
                # Core033 reset uses CUDA streams even for a CPU fixture. Keep
                # its three in-place tensor operations; never fake a GPU stream.
                group = owner.index_group
                group.device_global_indices.fill_(-1)
                group.lru_slots.copy_(group.lru_init.expand_as(group.lru_slots))
                group.swap_stats.zero_()
            else:
                # Preserve Core033 stream ordering, not a Python CPU fallback.
                self._reset(owner)
        except Exception as exc:
            self._failure = exc  # Device reset is not safely rollbackable.
            raise

    def reset(self, runtime, *, quiescent=False):
        """Reset the owner in place, even when called through a reader."""
        self._check(quiescent)
        with self._lock:
            self._check(quiescent)
            self._reset_entry(self._entry(runtime))

    @staticmethod
    def _clear_hot(runtime):
        for name in _HOT:
            delattr(runtime, name)

    def unbind_reader(self, runtime, *, quiescent=False):
        self._check(quiescent)
        with self._lock:
            self._check(quiescent)
            entry = self._entry(runtime)
            if runtime not in entry.readers:
                raise OwnerContractError("runtime is not a reader")
            reader = entry.readers.pop(runtime)
            del entry.views[runtime]
            del entry.groups[runtime]
            self._clear_hot(runtime)
            runtime.region_stride = reader.stride
            for name, value in zip(_STATE, reader.state):
                setattr(runtime.index_group, name, value)
            del self._readers[runtime]

    def unbind_owner(self, runtime, *, quiescent=False):
        self._check(quiescent)
        with self._lock:
            self._check(quiescent)
            entry = self._entry(runtime)
            if entry.runtime is not runtime or entry.readers:
                raise OwnerContractError("detach all readers before unbinding the owner")
            self._clear_hot(runtime)
            del self._owners[runtime]

    def rebind_owner(self, runtime, raw_tensor, *, block_table, quiescent=False):
        """Replace fixed-layout storage for owner AND readers, then invalidate.

        No resize, capacity/config change or individual reader rebind. Caller
        supplies a new slab/table with the original shapes. Host ownership is
        outside this hot-only slice. Rebinding to the identical objects is a no-op.
        """
        self._check(quiescent)
        with self._lock:
            self._check(quiescent)
            entry = self._entry(runtime)
            if entry.runtime is not runtime:
                raise OwnerContractError("only the direct owner can rebind")
            old = entry.layout
            if raw_tensor is old.raw and block_table is old.block_table:
                return
            if (raw_tensor.shape != old.raw.shape or block_table.shape != old.block_table.shape):
                raise OwnerContractError("resize is not supported by the owner contract")
            new = replace(old, raw=raw_tensor, block_table=block_table)
            self._layout(runtime, new)
            members = (runtime, *entry.readers)
            saved = [tuple(getattr(member, name) for name in _HOT) for member in members]
            try:
                for member in members:
                    self._bind(member, raw_tensor, **new.kwargs())
            except Exception:
                for member, values in zip(members, saved):
                    for name, value in zip(_HOT, values):
                        setattr(member, name, value)
                raise
            entry.layout = new
            entry.views = {member: member.hot for member in members}
            self._reset_entry(entry)
