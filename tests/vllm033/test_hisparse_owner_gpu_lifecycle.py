# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Opt-in physical GPU2 HIP hot-state lifecycle, NOT native init/attention.

Actual Core033 runtime/group TYPES are allocated with __new__, not __init__.
Real Torch HIP tensors, streams, events and ORIGINAL Core reset are used. No
Core method or stream/event is replaced. Reader-private copy streams require an
explicit caller join: the provider promises quiescent lifecycle, not auto drain.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from importlib import import_module, util
import os
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "adapter_extension/src/hcu_vllm_das/hisparse_owner_contract.py"
TRACE_RECORDS = []
RESOURCE_RECORD = {}
STATE = ("lru_init", "device_global_indices", "lru_slots", "swap_stats")


@pytest.fixture(scope="module")
def hip():
    if os.environ.get("HCU_OWNER_GPU_LIFECYCLE") != "1":
        pytest.skip("requires explicit isolated physical GPU2 authorization")
    assert os.environ.get("ROCR_VISIBLE_DEVICES") == "2"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    # Establish the physical PCI identity before initializing a Torch context.
    library = ctypes.CDLL("libamdhip64.so")
    bus = ctypes.create_string_buffer(64)
    assert library.hipDeviceGetPCIBusId(bus, 64, 0) == 0
    assert bus.value.decode().lower() == "0000:5e:00.0"
    assert torch.version.hip and torch.cuda.device_count() == 1
    device = torch.device("cuda:0")  # This logical 0 is PHYSICAL GPU2 only.
    free, total = torch.cuda.mem_get_info(device)
    if free < 512 * 1024**2:
        pytest.fail(f"RESOURCE_STOP: only {free} bytes free, no test allocation")
    torch.cuda.set_per_process_memory_fraction(32 * 1024**2 / total, device)
    torch.cuda.reset_peak_memory_stats(device)
    core = import_module("vllm.v1.hisparse.runtime")
    original_bind = core.HiSparseRuntime.bind_hot_cache
    original_reset = core.HiSparseRuntime.reset_hot_state
    spec = util.spec_from_file_location("_owner_real_hip_slice", MODULE)
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    RESOURCE_RECORD.update(physical_gpu=2, logical_gpu=0, pci_bus=bus.value.decode(),
                           free_before=free, total=total, allocator_limit=32*1024**2)
    yield core, module, device
    assert core.HiSparseRuntime.bind_hot_cache is original_bind
    assert core.HiSparseRuntime.reset_hot_state is original_reset
    RESOURCE_RECORD.update(peak_allocated=torch.cuda.max_memory_allocated(device),
                           peak_reserved=torch.cuda.max_memory_reserved(device),
                           free_after=torch.cuda.mem_get_info(device)[0])
    assert RESOURCE_RECORD["peak_allocated"] < 8 * 1024**2
    assert RESOURCE_RECORD["peak_reserved"] <= 32 * 1024**2


def runtime(core, device, stride=4096, union=128):
    obj = core.HiSparseRuntime.__new__(core.HiSparseRuntime)
    obj.max_num_reqs, obj.region_stride, obj.max_union_rows = 2, stride, union
    obj.row_width, obj.kv_dtype, obj.device = 8, torch.float16, device
    obj.is_group_leader, obj._swap_staged = True, False
    obj._layer_ready_event, obj._swap_step = None, 0
    group = core.HiSparseIndexGroup.__new__(core.HiSparseIndexGroup)
    group.leader, group.followers = obj, []
    group.lru_init = torch.arange(stride, dtype=torch.int16, device=device)
    group.device_global_indices = torch.full((2, stride), -1, dtype=torch.int32, device=device)
    group.lru_slots = group.lru_init.repeat(2, 1).contiguous()
    group.swap_stats = torch.zeros(2, dtype=torch.uint64, device=device)
    group.shared_topk = core._create_shared_topk_state(device, 2, 128)
    group.copy_stream = torch.cuda.Stream(device=device)
    obj.index_group = group
    return obj


def slab(owner):
    blocks, stride, offset = (owner.region_stride + 15) // 16 + 1, 768, 64
    raw = torch.zeros(offset + blocks * stride, dtype=torch.uint8, device=owner.device)
    table = torch.arange(blocks - 1, dtype=torch.int32, device=owner.device).repeat(2, 1)
    return raw, dict(byte_offset=offset, block_stride=stride, num_blocks=blocks,
                     block_size=16, block_table=table)


def state(obj):
    return tuple(getattr(obj.index_group, name) for name in STATE)


def assert_shared(owner, reader):
    assert reader.region_stride == owner.region_stride
    assert all(a is b for a, b in zip(state(owner), state(reader)))
    assert reader.index_group is not owner.index_group
    assert reader.index_group.copy_stream != owner.index_group.copy_stream
    assert reader.hot.cache.data_ptr() == owner.hot.cache.data_ptr()
    assert reader.hot.cache.stride() == owner.hot.cache.stride() == (384, 8, 1)
    assert reader.hot.cache.storage_offset() == owner.hot.cache.storage_offset() == 32
    assert reader.index_group.lru_init.dtype == torch.int16
    assert reader.index_group.lru_slots.dtype == torch.int16


@pytest.fixture
def bound(hip):
    core, module, device = hip
    provider = module.get_owner_provider()
    assert provider._reset is core.HiSparseRuntime.reset_hot_state
    streams = [torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)]
    with torch.cuda.stream(streams[0]):
        owner = runtime(core, device)
        reader = runtime(core, device, 2176, 2176)
        private = state(reader)
        raw, layout = slab(owner)
        provider.bind_owner(owner, raw, **layout, quiescent=True)
        provider.bind_reader(reader, owner=owner, quiescent=True)
    streams[0].synchronize()
    yield core, provider, owner, reader, streams, raw, layout, private
    # Drain ONLY this test's real streams before removing storage references.
    for stream in (*streams, owner.index_group.copy_stream, reader.index_group.copy_stream):
        stream.synchronize()
    with torch.cuda.stream(streams[1]):
        if reader in provider._readers:
            provider.unbind_reader(reader, quiescent=True)
        if owner in provider._owners:
            provider.unbind_owner(owner, quiescent=True)
    streams[1].synchronize()


@contextmanager
def original_reset_trace(core, owner, caller, label):
    """Observe original Python call frames, never wrap/replace a Core method."""
    reset_code = core.HiSparseRuntime.reset_hot_state.__code__
    wait_code = torch.cuda.Stream.wait_stream.__code__
    record = dict(label=label, caller=int(caller.cuda_stream),
                  owner_copy=int(owner.index_group.copy_stream.cuda_stream),
                  calls=[], waits=[])
    depth = 0
    previous = sys.getprofile()
    def observe(frame, event, arg):
        nonlocal depth
        if frame.f_code is reset_code:
            if event == "call":
                depth += 1
                record["calls"].append(dict(self_is_owner=frame.f_locals["self"] is owner,
                                            current=int(torch.cuda.current_stream().cuda_stream)))
            elif event == "return":
                depth -= 1
        if depth and event == "call" and frame.f_code is wait_code:
            record["waits"].append([int(frame.f_locals["self"].cuda_stream),
                                    int(frame.f_locals["stream"].cuda_stream)])
        if previous is not None:
            previous(frame, event, arg)
    sys.setprofile(observe)
    try:
        yield record
    finally:
        sys.setprofile(previous)
        TRACE_RECORDS.append(record)
    assert record["calls"] == [dict(self_is_owner=True, current=int(caller.cuda_stream))]
    assert record["waits"] == [[record["caller"], record["owner_copy"]],
                               [record["owner_copy"], record["caller"]]]
    assert caller != torch.cuda.default_stream(owner.device)


def dirty(group):
    group.device_global_indices.fill_(73)
    group.lru_slots.zero_()
    group.swap_stats.fill_(9)


def sample_after_copy(owner):
    """Core's copy->compute->copy edges must order these real GPU snapshots."""
    with torch.cuda.stream(owner.index_group.copy_stream):
        values = [getattr(owner.index_group, name).clone() for name in STATE[1:]]
        done = torch.cuda.Event()
        done.record()
    done.synchronize()
    indices, lru, stats = [tensor.cpu() for tensor in values]
    assert indices.eq(-1).all().item()
    torch.testing.assert_close(lru, torch.arange(owner.region_stride, dtype=torch.int16).repeat(2, 1))
    assert stats.tolist() == [0, 0]


def test_real_hip_hot_bind_offset_stride_and_storage(bound):
    _, _, owner, reader, streams, _, _, _ = bound
    assert_shared(owner, reader)
    with torch.cuda.stream(streams[1]):
        reader.hot.cache[1, 2, 3] = 19
        seen = owner.hot.cache[1, 2, 3].clone()
        event = torch.cuda.Event()
        event.record()
    event.synchronize()
    assert seen.item() == 19


@pytest.mark.parametrize("target,index", [("owner", 0), ("reader", 1)])
def test_original_core_gpu_reset_on_distinct_nondefault_streams(bound, target, index):
    core, provider, owner, reader, streams, _, _, _ = bound
    with torch.cuda.stream(owner.index_group.copy_stream):
        dirty(owner.index_group)
    with torch.cuda.stream(streams[index]):
        with original_reset_trace(core, owner, streams[index], "reset-through-" + target):
            provider.reset(owner if target == "owner" else reader, quiescent=True)
    sample_after_copy(owner)
    assert_shared(owner, reader)


@pytest.mark.parametrize("target,index", [("owner", 0), ("reader", 1)])
def test_reader_private_copy_has_explicit_caller_event_join(bound, target, index):
    core, provider, owner, reader, streams, _, _, _ = bound
    with torch.cuda.stream(reader.index_group.copy_stream):
        dirty(reader.index_group)
        producer_done = torch.cuda.Event()
        producer_done.record()
    with torch.cuda.stream(streams[index]):
        # The utility does not auto-drain reader-private streams. This real join
        # is caller responsibility, in addition to no concurrent attention.
        streams[index].wait_event(producer_done)
        with original_reset_trace(core, owner, streams[index], "explicit-reader-join-" + target):
            provider.reset(owner if target == "owner" else reader, quiescent=True)
    sample_after_copy(owner)


def test_fixed_rebind_original_gpu_reset_and_old_slab_isolation(bound):
    core, provider, owner, reader, streams, old, layout, _ = bound
    with torch.cuda.stream(streams[0]):
        replacement = torch.zeros_like(old)
        table = layout["block_table"].clone()
    streams[0].synchronize()
    with torch.cuda.stream(owner.index_group.copy_stream):
        dirty(owner.index_group)
    with torch.cuda.stream(streams[1]):
        with original_reset_trace(core, owner, streams[1], "fixed-rebind"):
            provider.rebind_owner(owner, replacement, block_table=table, quiescent=True)
        reader.hot.cache[1, 0, 0] = 5
    sample_after_copy(owner)
    assert_shared(owner, reader)
    assert owner.hot_backing is reader.hot_backing is replacement
    assert owner.hot_block_table is reader.hot_block_table is table
    streams[1].synchronize()
    assert old.eq(0).all().item()
    assert owner.hot.cache[1, 0, 0].item() == 5


def test_teardown_restores_reader_and_reuse_clears_stale_gpu_state(bound):
    core, provider, owner, reader, streams, _, _, private = bound
    with torch.cuda.stream(streams[0]):
        dirty(owner.index_group)
    streams[0].synchronize()
    with torch.cuda.stream(streams[1]):
        provider.unbind_reader(reader, quiescent=True)
        provider.unbind_owner(owner, quiescent=True)
        assert not hasattr(owner, "hot") and not hasattr(reader, "hot")
        assert reader.region_stride == 2176
        assert all(a is b for a, b in zip(state(reader), private))
        raw, layout = slab(owner)
        with original_reset_trace(core, owner, streams[1], "teardown-then-fresh-bind"):
            provider.bind_owner(owner, raw, **layout, quiescent=True)
        provider.bind_reader(reader, owner=owner, quiescent=True)
    sample_after_copy(owner)
    assert_shared(owner, reader)


def test_independent_owner_and_request_isolation_on_gpu(bound, hip):
    core, provider, owner, reader, streams, _, _, _ = bound
    _, _, device = hip
    with torch.cuda.stream(streams[0]):
        other = runtime(core, device)
        raw, layout = slab(other)
        provider.bind_owner(other, raw, **layout, quiescent=True)
        other.index_group.device_global_indices[0, 1] = 29
        owner.index_group.device_global_indices[0, 1] = 31
    streams[0].synchronize()
    assert owner.index_group.device_global_indices[1, 1].item() == -1
    with torch.cuda.stream(streams[1]):
        with original_reset_trace(core, owner, streams[1], "independent-owner-isolation"):
            provider.reset(reader, quiescent=True)
    sample_after_copy(owner)
    assert other.index_group.device_global_indices[0, 1].item() == 29
    streams[1].synchronize()
    other.index_group.copy_stream.synchronize()
    with torch.cuda.stream(streams[1]):
        provider.unbind_owner(other, quiescent=True)
    streams[1].synchronize()


@pytest.mark.parametrize("capacity,accepted", [(4096, True), (2048, False), (2176, True)])
def test_real_hip_capacity_admission(hip, capacity, accepted):
    core, module, device = hip
    provider = module.get_owner_provider()
    caller = torch.cuda.Stream(device=device)
    with torch.cuda.stream(caller):
        owner, reader = runtime(core, device, capacity), runtime(core, device, 2176, 2176)
        raw, layout = slab(owner)
        provider.bind_owner(owner, raw, **layout, quiescent=True)
        if accepted:
            provider.bind_reader(reader, owner=owner, quiescent=True)
            assert_shared(owner, reader)
        else:
            with pytest.raises(ValueError, match="max_union_rows=2176.*device_buffer_size=2048"):
                provider.bind_reader(reader, owner=owner, quiescent=True)
            assert not hasattr(reader, "hot")
    caller.synchronize()
    owner.index_group.copy_stream.synchronize()
    reader.index_group.copy_stream.synchronize()
    with torch.cuda.stream(caller):
        if accepted:
            provider.unbind_reader(reader, quiescent=True)
        provider.unbind_owner(owner, quiescent=True)
    caller.synchronize()


def test_shared_group_followers_still_rejected_on_gpu(bound, hip):
    _, provider, owner, reader, _, _, _, _ = bound
    _, module, _ = hip
    owner.index_group.followers.append(reader)
    try:
        with pytest.raises(module.OwnerContractError, match="singleton leader"):
            provider.reset(owner, quiescent=True)
    finally:
        owner.index_group.followers.clear()
