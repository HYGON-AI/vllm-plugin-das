# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for IPC ownership and rank-consistent all-gather selection."""

from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.distributed.device_communicators import ipc_all_gather as ipc
from vllm_hcu.patch.worker.framework_opt import patch_cuda_communicator as adapter


class Launch:
    def __init__(self, callback):
        self.callback = callback

    def __getitem__(self, grid):
        return self.callback


def _native_fakes(monkeypatch, *, fail=None):
    events = []
    counts = {"allocate": 0, "open": 0, "exchange": 0, "sync": 0}

    def allocate(size):
        counts["allocate"] += 1
        if fail == "second_allocate" and counts["allocate"] == 2:
            raise RuntimeError("allocation failed")
        pointer = counts["allocate"] * 100
        events.append(("allocate", pointer, size))
        return pointer, pointer + 1

    def open_handle(handle):
        counts["open"] += 1
        if fail == "open":
            raise RuntimeError("open failed")
        events.append(("open", handle))
        return handle + 1

    def exchange(outputs, payload, group):
        counts["exchange"] += 1
        events.append(("exchange", counts["exchange"]))
        if fail == "exchange" or (fail == "exchange_close" and counts["exchange"] >= 6):
            raise RuntimeError("process group failed")
        outputs[:] = [payload, payload]
        if fail == "peer_allocate" and isinstance(payload, tuple):
            outputs[1] = ("RuntimeError: peer allocation failed", None)
        if fail == "peer_close" and counts["exchange"] == 7:
            outputs[1] = "RuntimeError: peer close failed"

    def close_handle(pointer):
        events.append(("close_import", pointer))
        if fail == "close":
            raise RuntimeError("close failed")

    def synchronize(device):
        counts["sync"] += 1
        events.append(("sync",))
        if fail == "quiesce" and counts["sync"] >= 2:
            raise RuntimeError("quiesce failed")

    def zero(*args, **kwargs):
        if fail == "zero":
            raise RuntimeError("zero failed")

    monkeypatch.setattr(ipc.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(ipc.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(ipc.dist, "all_gather_object", exchange)
    monkeypatch.setattr(torch.ops.hcu_ops, "allocate_shared_buffer_and_handle", allocate, raising=False)
    monkeypatch.setattr(torch.ops.hcu_ops, "open_mem_handle", open_handle, raising=False)
    monkeypatch.setattr(torch.ops.hcu_ops, "close_mem_handle", close_handle, raising=False)
    monkeypatch.setattr(torch.ops.hcu_ops, "free_shared_buffer", lambda pointer: events.append(("free", pointer)), raising=False)
    monkeypatch.setattr(ipc, "_zero_i64_kernel", Launch(zero))
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    return events, counts


def test_ipc_initialization_and_close_are_balanced(monkeypatch):
    events, counts = _native_fakes(monkeypatch)
    ag = ipc.IpcAllGather(object(), torch.device("cpu"))
    assert not ag.disabled
    assert counts["exchange"] == 5  # initialization only, never a decode collective
    assert ag._owned_buffers == [100, 200]
    ag.close()
    ag.close()
    assert [event for event in events if event[0] == "free"] == [("free", 100), ("free", 200)]
    assert ag.disabled and ag._bufs is None and ag._flags is None
    assert [e for e in events if e[0] == "close_import"] == [
        ("close_import", 102), ("close_import", 202)
    ]
    close_indices = [i for i, e in enumerate(events) if e[0] == "close_import"]
    free_indices = [i for i, e in enumerate(events) if e[0] == "free"]
    confirmed = events.index(("exchange", 7))
    assert max(close_indices) < confirmed < min(free_indices)
    assert counts["exchange"] == 7  # repeated close starts no collective
    assert ag._owned_buffers == [] and ag._imported_buffers == []


@pytest.mark.parametrize("failure", ["second_allocate", "peer_allocate", "open", "zero", "exchange"])
def test_initialization_failure_releases_all_returned_allocations(monkeypatch, failure):
    events, _ = _native_fakes(monkeypatch, fail=failure)
    with pytest.raises(RuntimeError):
        ipc.IpcAllGather(object(), torch.device("cpu"))
    allocated = [event[1] for event in events if event[0] == "allocate"]
    freed = [event[1] for event in events if event[0] == "free"]
    assert freed == ([] if failure == "exchange" else allocated)
    if failure == "exchange":
        assert not any(event[0] == "close_import" for event in events)
    if failure == "peer_allocate":
        assert not any(event[0] == "open" for event in events)


def test_close_attempts_both_buffers_even_if_first_free_fails():
    ag = _logical_gather()
    ag._owned_buffers = [11, 22]
    freed = []

    def free(pointer):
        freed.append(pointer)
        if pointer == 11 and freed.count(11) == 1:
            raise RuntimeError("free failed")

    ag._free_buffer = free
    with pytest.raises(RuntimeError, match="free failed"):
        ag.close()
    ag.close()
    assert freed == [11, 22, 11]
    assert ag._owned_buffers == []


def _logical_gather(rank=0, capacity=1 << 20):
    ag = object.__new__(ipc.IpcAllGather)
    ag.rank, ag.world_size, ag.disabled = rank, 2, False
    ag.device, ag.region_elems = torch.device("cpu"), capacity
    ag._owned_buffers = []
    ag._imported_buffers = []
    ag._needs_coordinated_close = False
    ag._collective_failed = False
    ag._data_ptrs = ag._flag_ptrs = ag._bufs = ag._flags = ag._cnt = None
    return ag


def _patched_communicator(monkeypatch, ag):
    import vllm.config

    class CudaCommunicator:
        def __init__(self, cpu_group, device=None, device_group=None, unique_name="",
                     global_ranks=None, global_world_size=None, tcp_store_group=None,
                     use_all2all=False):
            self.world_size = 2
            self.events = []

        def all_gatherv(self, input_, dim=0, sizes=None):
            self.events.append(("fallback", input_, sizes))
            return input_

        def destroy(self):
            self.events.append(("destroy",))
            return "destroyed"

    module = ModuleType(adapter.TARGET_MODULE)
    module.CudaCommunicator = CudaCommunicator
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(adapter, "_maybe_init_ep_custom_allreduce", lambda *args: None)
    monkeypatch.setattr(adapter, "_maybe_init_dp_ipc_all_gather", lambda obj, name: setattr(obj, "_hcu_ipc_ag", ag))
    assert adapter.apply_to_module(module)
    assert not adapter.apply_to_module(module)
    return module.CudaCommunicator(object())


def test_communicator_destroy_closes_ipc_once(monkeypatch):
    calls = []
    ag = SimpleNamespace(close=lambda: calls.append("closed"))
    comm = _patched_communicator(monkeypatch, ag)
    assert comm.destroy() == "destroyed"
    comm.destroy()
    assert calls == ["closed"] and comm._hcu_ipc_ag is None
    assert comm.events == [("destroy",), ("destroy",)]


def test_communicator_destroy_preserves_base_cleanup_on_ipc_error(monkeypatch):
    def close():
        raise RuntimeError("IPC free failed")

    comm = _patched_communicator(monkeypatch, SimpleNamespace(close=close))
    with pytest.raises(RuntimeError, match="IPC free failed"):
        comm.destroy()
    assert comm.events == [("destroy",)]
    assert comm._hcu_ipc_ag is None


def test_communicator_late_initialization_failure_closes_ipc(monkeypatch):
    import vllm.config

    calls = []
    comm = _patched_communicator(monkeypatch, SimpleNamespace(close=lambda: calls.append("closed")))
    calls.clear()

    def bad_config():
        raise RuntimeError("bad configuration")

    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", bad_config)
    with pytest.raises(RuntimeError, match="bad configuration"):
        type(comm)(object())
    assert calls == ["closed"]


@pytest.mark.parametrize("capacity, route", [(1 << 20, "ipc"), (1, "fallback")])
def test_uneven_rank_strides_choose_same_collective(monkeypatch, capacity, route):
    sizes = [1, 2]
    decisions = []
    for rank, rows in enumerate(sizes):
        ag = _logical_gather(rank, capacity)
        calls = []

        def gather(tensors, sizes):
            calls.append(tensors)
            assert all(t.is_contiguous() for t in tensors)
            return tensors

        ag.gather = gather
        comm = _patched_communicator(monkeypatch, ag)
        source = torch.arange(rows * 8).view(rows, 8)[:, :4]
        # M=1 is contiguous; M=2 is not, despite the same row-slice operation.
        assert source.is_contiguous() == (rank == 0)
        output = comm.all_gatherv(source, sizes=sizes)
        decisions.append("ipc" if calls else "fallback")
        assert output.is_contiguous()
        torch.testing.assert_close(output, source)
    assert decisions == [route, route]


@pytest.mark.parametrize("invalid", ["rows", "device", "sizes", "scalar"])
def test_invalid_collective_contract_raises_instead_of_falling_back(monkeypatch, invalid):
    ag = _logical_gather()
    comm = _patched_communicator(monkeypatch, ag)
    tensor, sizes = torch.empty(1, 4), [1, 2]
    if invalid == "rows":
        tensor = torch.empty(2, 4)
    elif invalid == "device":
        tensor = torch.empty(1, 4, device="meta")
    elif invalid == "sizes":
        sizes = [1]
    else:
        tensor = torch.tensor(1)
    with pytest.raises(ValueError):
        comm.all_gatherv(tensor, sizes=sizes)
    assert not comm.events


def test_closed_ipc_does_not_select_an_independent_fallback():
    ag = _logical_gather()
    ag.close()
    with pytest.raises(RuntimeError, match="closed"):
        ag.can_gather([torch.empty(1, 4)], [1, 1])


def test_direct_gather_packs_misaligned_one_row_view(monkeypatch):
    ag = _logical_gather()
    ag.slot_elems = 2 * ag.region_elems
    calls = []
    monkeypatch.setattr(ipc, "_ipc_all_gather_kernel", Launch(lambda *args, **kw: calls.append(args)))
    source = torch.arange(5, dtype=torch.bfloat16)[1:].view(1, 4)
    assert source.is_contiguous() and source.storage_offset() == 1
    assert ag.can_gather([source], [1, 1])
    output = ag.gather([source], [1, 1])
    assert output[0].shape == (2, 4)
    assert len(calls) == 1
    assert calls[0][0].storage_offset() == 0
    torch.testing.assert_close(calls[0][0].view(torch.bfloat16).view_as(source), source)


@pytest.mark.parametrize("failure", ["close", "peer_close", "quiesce", "exchange_close"])
def test_exports_are_not_freed_until_every_importer_is_closed(monkeypatch, failure):
    events, _ = _native_fakes(monkeypatch, fail=failure)
    ag = ipc.IpcAllGather(object(), torch.device("cpu"))
    with pytest.raises(RuntimeError):
        ag.close()
    assert ag._owned_buffers == [100, 200]
    assert not any(event[0] == "free" for event in events)
    # GC cannot retry a collective or free buffers still exported to peers.
    before = list(events)
    ag.__del__()
    assert events == before


def test_destructor_does_not_communicate_or_free_exported_memory(monkeypatch):
    events, _ = _native_fakes(monkeypatch)
    ag = ipc.IpcAllGather(object(), torch.device("cpu"))
    before = list(events)
    ag.__del__()
    assert events == before
    assert ag._owned_buffers == [100, 200]
    assert ag._imported_buffers == [102, 202]
    # The explicit owner can still close correctly after disabled/GC signaling.
    ag.close()
    assert ag._owned_buffers == [] and ag._imported_buffers == []
