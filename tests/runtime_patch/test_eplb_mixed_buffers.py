# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU transfer/completion regressions, Gloo P2P, and NIXL registration contracts."""

from __future__ import annotations

from datetime import timedelta
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.eplb import eplb_communicator as communicators
from vllm.distributed.eplb import eplb_state as native_state
from vllm.distributed.eplb import rebalance_execute as native_transfer
from vllm_hcu.distributed import eplb_runtime as runtime
from vllm_hcu.distributed import eplb_transfer_runtime as transfer
from vllm_hcu.distributed.eplb_buffer_runtime import (
    buffer_signature,
    create_communicator_with_buffers,
    get_buffer_pool,
)


class LocalCommunicator:
    needs_profile_buffer_reservation = False

    def __init__(self):
        self.sends = 0
        self.recvs = 0

    def set_transfer_context(self, old_indices, layer_idx):
        pass

    def add_send(self, tensors, dst_rank, expert_id):
        self.sends += 1

    def add_recv(self, tensors, src_rank, expert_id):
        self.recvs += 1

    def execute(self):
        pass


def weights_for(mapping, rank):
    ids = mapping[rank * 4 : (rank + 1) * 4].float().unsqueeze(1)
    fp8 = [
        (ids + torch.tensor([[0.5, 0.75]])).to(torch.float8_e4m3fn),
        (ids + torch.tensor([[1.0, 1.5, 2.0]])).to(torch.float8_e4m3fn),
        ids + 0.125,
        ids + 0.25,
    ]
    bf16 = [
        (ids + torch.tensor([[1.0078125, 1.0234375]])).to(torch.bfloat16),
        (ids + torch.tensor([[2.015625, 2.03125, 2.0625]])).to(torch.bfloat16),
    ]
    return [fp8, bf16]


def prepared_communicator(layers, communicator):
    seed = [torch.empty_like(w) for w in layers[0]]
    result = create_communicator_with_buffers(
        lambda **kwargs: communicator, None, "torch_gloo", layers, seed
    )
    return seed, result


def model_state(layers, seed, communicator, old):
    return SimpleNamespace(
        model=SimpleNamespace(expert_weights=layers, num_moe_layers=len(layers)),
        expert_buffer=seed,
        communicator=communicator,
        physical_to_logical_map=old.unsqueeze(0).repeat(len(layers), 1),
        logical_to_physical_map=torch.full((len(layers), 4, 8), -1, dtype=torch.long),
        logical_replica_count=torch.zeros((len(layers), 4), dtype=torch.long),
        expert_load_pass=torch.zeros((len(layers), 8), dtype=torch.int32),
        pending_result=None,
        rebalanced=True,
    )


def complete_layer(ms, layer_idx, metadata, new, rank):
    acknowledged = []

    def record():
        # The native async worker may proceed as soon as this is recorded.
        assert ms.pending_result is None
        acknowledged.append(layer_idx)

    ms.pending_result = native_transfer.AsyncEplbLayerResult(
        layer_idx, new, metadata, SimpleNamespace(record=record)
    )
    runtime._move_to_workspace(ms, rank)
    assert acknowledged == [layer_idx]


def assert_weights_equal(actual, expected):
    for layer, reference in zip(actual, expected):
        assert len(layer) == len(reference)
        for weight, value in zip(layer, reference):
            assert weight.dtype == value.dtype
            assert torch.equal(weight.float(), value.float())


def test_async_local_replica_moves_preserve_bf16_and_result_ownership(monkeypatch):
    monkeypatch.setattr(native_state, "PIN_MEMORY", False)
    old = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    new = torch.tensor([0, 0, 2, 0, 0, 1, 0, 3])
    layers = weights_for(old, 0)
    seed, communicator = prepared_communicator(layers, LocalCommunicator())
    ms = model_state(layers, seed, communicator, old)
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    monkeypatch.setattr(
        runtime, "get_ep_group", lambda: SimpleNamespace(device_group=group)
    )
    state = SimpleNamespace(
        model_states={"model": ms},
        is_async=True,
        expert_rearrangement_step=0,
        expert_rearrangement_step_interval=100,
        _all_ranks_result_ready=lambda model: True,
        _update_layer_should_record=lambda **kwargs: None,
    )
    for index, layer in enumerate(layers):
        metadata = transfer.transfer_layer(
            old, new, layer, seed, group, communicator, layer_idx=index
        )
        assert isinstance(metadata, transfer.BufferedTransferMetadata)
        assert buffer_signature(metadata.expert_buffer) == buffer_signature(layer)
        assert metadata.expert_buffer is get_buffer_pool(
            communicator, layers, seed
        ).for_layer(layer)
        assert metadata.layer_idx == index
        recorded = []

        def record():
            assert ms.pending_result is None
            recorded.append(True)

        ms.pending_result = native_transfer.AsyncEplbLayerResult(
            index, new, metadata, SimpleNamespace(record=record)
        )
        # Exercise the real main-thread step dispatch, not just the helper.
        runtime.step(state, is_dummy=True)
        assert recorded == [True]
    assert not ms.rebalanced
    assert communicator.sends == communicator.recvs == 0
    assert ms.expert_buffer is seed
    assert_weights_equal(layers, weights_for(new, 0))
    assert torch.equal(ms.physical_to_logical_map, new.unsqueeze(0).repeat(2, 1))


@pytest.mark.parametrize("wrong_layer", [False, True])
def test_async_completion_rejects_wrong_buffer_or_layer_before_copy(
    monkeypatch, wrong_layer
):
    monkeypatch.setattr(native_state, "PIN_MEMORY", False)
    old = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    layers = weights_for(old, 0)
    seed, comm = prepared_communicator(layers, LocalCommunicator())
    ms = model_state(layers, seed, comm, old)
    metadata = transfer.transfer_layer(
        old, old, layers[0], seed, SimpleNamespace(size=lambda: 2, rank=lambda: 0), comm
    )
    if not wrong_layer:
        metadata.layer_idx = 1
    recorded = []
    ms.pending_result = native_transfer.AsyncEplbLayerResult(
        1, old, metadata, SimpleNamespace(record=lambda: recorded.append(True))
    )
    with pytest.raises(RuntimeError, match="different layer|incompatible staging"):
        runtime._move_to_workspace(ms, 0)
    assert not recorded
    assert ms.pending_result is not None
    assert_weights_equal(layers, weights_for(old, 0))


@pytest.mark.parametrize("reserve_communication", [False, True])
def test_profile_and_transfer_use_only_preallocated_format_groups(
    monkeypatch, reserve_communication
):
    old = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]).repeat(2, 1)
    layers = [
        [torch.zeros(4, 32, dtype=torch.float8_e4m3fn) for _ in range(2)]
        + [torch.ones(4, 1) for _ in range(2)],
        [torch.zeros(4, 32, dtype=torch.bfloat16) for _ in range(2)],
    ]
    communicator = LocalCommunicator()
    communicator.needs_profile_buffer_reservation = reserve_communication
    seed, communicator = prepared_communicator(layers, communicator)
    pool = get_buffer_pool(communicator, layers, seed)
    pointers = [t.data_ptr() for t in pool.registration_buffers()]
    assert len(pointers) == 6  # Four FP8/scale tensors and two BF16 tensors.
    assert sum(t.nbytes for t in pool.for_layer(layers[1])) > sum(
        t.nbytes for t in seed
    )
    assert sum(t.nbytes for t in pool.registration_buffers()) == sum(
        t.nbytes for layer in layers for t in layer
    )
    profile_calls = []

    def all_gather(outputs, weight, group):
        assert all(t.data_ptr() in pointers for t in outputs)
        profile_calls.append((weight.dtype, tuple(weight.shape)))

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("EPLB allocated a new staging tensor after initialization")

    monkeypatch.setattr(transfer, "all_gather", all_gather)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(torch, "empty_like", unexpected_allocation)
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    transfer.rearrange_expert_weights_inplace(
        old, old, layers, seed, group, communicator, is_profile=True
    )
    transfer.rearrange_expert_weights_inplace(
        old, old, layers, seed, group, communicator
    )
    assert profile_calls == (
        [(w.dtype, tuple(w.shape)) for layer in layers for w in layer]
        if reserve_communication
        else []
    )
    assert [t.data_ptr() for t in pool.registration_buffers()] == pointers


class RegistrationCheckingNixl:
    """NIXL API stand-in; real native registration/descriptor methods call it."""

    def __init__(self):
        self.ranges = []
        self.next_handle = 0
        self.deregistered = []

    def get_reg_descs(self, tensors):
        return [(t.data_ptr(), t.nbytes) for t in tensors]

    def register_memory(self, descriptors):
        self.ranges.extend(descriptors)

    def deregister_memory(self, descriptors):
        self.deregistered.append(descriptors)

    def get_xfer_descs(self, descriptors, memory_type):
        return descriptors

    def prep_xfer_dlist(self, agent, descriptors):
        if agent == "NIXL_INIT_AGENT":
            for pointer, size, device in descriptors:
                assert any(
                    start <= pointer and pointer + size <= start + length
                    for start, length in self.ranges
                ), "unregistered receive address"
        self.next_handle += 1
        return self.next_handle

    def make_prepped_xfer(self, *args):
        self.next_handle += 1
        return self.next_handle

    def transfer(self, handle):
        pass

    def release_xfer_handle(self, handle):
        pass

    def release_dlist_handle(self, handle):
        pass

    def remove_remote_agent(self, agent):
        pass


def nixl_factory(**kwargs):
    comm = object.__new__(communicators.NixlEplbCommunicator)
    comm._all_expert_weights = kwargs["expert_weights"]
    comm._expert_buffer = kwargs["expert_buffer"]
    comm._nixl_wrapper = RegistrationCheckingNixl()
    comm._registered_descs = []
    comm._xfer_entries = []
    comm._remote_agents = {1: "peer"}
    comm._nixl_memory_type = "VRAM"
    comm._cuda_device_id = 0
    comm._world_size = 2
    comm._num_local_experts = 4
    comm._expert_to_src_row = None
    comm._layer_idx = None
    communicators.NixlEplbCommunicator._init_registered_buffers(comm)
    return comm


def test_nixl_registers_all_receive_formats_before_descriptor_preparation():
    old = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    layers = weights_for(old, 0)
    seed = [torch.empty_like(w) for w in layers[0]]
    comm = create_communicator_with_buffers(nixl_factory, None, "nixl", layers, seed)
    pool = get_buffer_pool(comm, layers, seed)
    assert len(comm._expert_buffer) == 6
    remote = weights_for(old, 1)
    comm._remote_send_meta = {
        1: {
            (layer, index): (tensor.data_ptr(), tensor[0].nbytes, 0)
            for layer, tensors in enumerate(remote)
            for index, tensor in enumerate(tensors)
        }
    }
    comm.set_transfer_context(old.numpy(), 1)
    received = [buffer[0] for buffer in pool.for_layer(layers[1])]
    comm.add_recv(received, 1, expert_id=2)
    assert len(comm._xfer_entries) == 1
    # Negative control: a late allocation is rejected by native descriptor setup.
    late = [torch.empty_like(w[0]) for w in layers[1]]
    with pytest.raises(AssertionError, match="unregistered receive address"):
        comm.add_recv(late, 1, expert_id=2)
    api = comm._nixl_wrapper
    comm.__del__()
    assert api.deregistered and not comm._registered_descs


@pytest.mark.parametrize("first_layer_only", [False, True])
def test_direct_mixed_nixl_transfer_fails_before_late_allocation(
    monkeypatch, first_layer_only
):
    old = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    layers = weights_for(old, 0)
    seed = [torch.empty_like(w) for w in layers[0]]
    comm = nixl_factory(expert_weights=layers, expert_buffer=seed)

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("attempted an unregistered allocation")

    monkeypatch.setattr(torch, "empty_like", unexpected_allocation)
    with pytest.raises(RuntimeError, match="before transfer"):
        get_buffer_pool(comm, layers[:1] if first_layer_only else layers, seed)


class GlooByteCommunicator(LocalCommunicator):
    """Real two-process Gloo P2P; bytes support FP8 tensors on CPU."""

    def __init__(self):
        super().__init__()
        self.ops = []

    def add_send(self, tensors, dst_rank, expert_id):
        super().add_send(tensors, dst_rank, expert_id)
        self.ops.append(("send", tensors, dst_rank, expert_id))

    def add_recv(self, tensors, src_rank, expert_id):
        super().add_recv(tensors, src_rank, expert_id)
        self.ops.append(("recv", tensors, src_rank, expert_id))

    def execute(self):
        manifests = [None, None]
        manifest = [
            (op, peer, expert, [(tuple(t.shape), str(t.dtype)) for t in tensors])
            for op, tensors, peer, expert in self.ops
        ]
        dist.all_gather_object(manifests, manifest)
        rank = dist.get_rank()
        for op, peer, expert, formats in manifest:
            opposite = "recv" if op == "send" else "send"
            assert (opposite, rank, expert, formats) in manifests[peer]
        p2p = [
            dist.P2POp(
                dist.isend if op == "send" else dist.irecv,
                tensor.view(torch.uint8),
                peer,
            )
            for op, tensors, peer, expert in self.ops
            for tensor in tensors
        ]
        for request in dist.batch_isend_irecv(p2p):
            request.wait()
        self.ops.clear()
        dist.barrier()


def run_rank(rank, rendezvous, use_async):
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        native_state.PIN_MEMORY = False
        old = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        new = torch.tensor([2, 2, 0, 1, 0, 0, 2, 3])
        layers = weights_for(old, rank)
        seed, comm = prepared_communicator(layers, GlooByteCommunicator())
        group = dist.group.WORLD
        if use_async:
            ms = model_state(layers, seed, comm, old)
            for index, layer in enumerate(layers):
                metadata = transfer.transfer_layer(
                    old, new, layer, seed, group, comm, layer_idx=index
                )
                complete_layer(ms, index, metadata, new, rank)
            assert not ms.rebalanced
            assert torch.equal(
                ms.physical_to_logical_map, new.unsqueeze(0).repeat(2, 1)
            )
        else:
            transfer.rearrange_expert_weights_inplace(
                old.unsqueeze(0).repeat(2, 1),
                new.unsqueeze(0).repeat(2, 1),
                layers,
                seed,
                group,
                comm,
            )
        assert comm.sends > 0 and comm.recvs > 0
        assert_weights_equal(layers, weights_for(new, rank))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("use_async", [False, True])
def test_two_rank_mixed_format_rearrangement(tmp_path, use_async):
    mp.spawn(
        run_rank, args=(str(tmp_path / "rendezvous"), use_async), nprocs=2, join=True
    )


def test_cold_worker_registration_covers_async_import_aliases():
    code = """
import vllm
from vllm_hcu.patch.worker import prepare_worker_patches
prepare_worker_patches()
from vllm.distributed.eplb import eplb_state, rebalance_execute, async_worker, eplb_communicator
from vllm_hcu.distributed import eplb_runtime, eplb_transfer_runtime
assert async_worker.transfer_layer is eplb_transfer_runtime.transfer_layer
assert rebalance_execute.transfer_layer is eplb_transfer_runtime.transfer_layer
assert eplb_state._move_to_workspace is eplb_runtime._move_to_workspace
assert eplb_state.EplbState.step is eplb_runtime.step
assert eplb_state.create_eplb_communicator is eplb_communicator.create_eplb_communicator
assert hasattr(eplb_communicator, '_vllm_hcu_step5_eplb_communicator_applied')
from vllm_hcu.patch.worker.framework_opt import patch_step5_eplb_communicator
assert patch_step5_eplb_communicator.apply_to_module(eplb_communicator) is False
print('mixed-eplb-cold-registration-ok')
"""
    env = dict(os.environ, VLLM_PLUGINS="__disabled__")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "mixed-eplb-cold-registration-ok" in result.stdout


def test_already_imported_async_worker_and_state_aliases_are_rebound():
    code = """
import vllm
from vllm.distributed.eplb import eplb_state, rebalance_execute, async_worker, eplb_communicator
from vllm_hcu.patch.worker.framework_opt import (
    patch_step5_eplb_communicator as communicator_adapter,
    patch_step5_eplb_transfer as transfer_adapter,
    patch_step5_eplb_async_worker as async_adapter,
    patch_step5_eplb_state as state_adapter,
)
for adapter, module in ((communicator_adapter, eplb_communicator),
                        (transfer_adapter, rebalance_execute),
                        (async_adapter, async_worker), (state_adapter, eplb_state)):
    assert adapter.apply_to_module(module)
    assert adapter.apply_to_module(module) is False
from vllm_hcu.distributed import eplb_runtime, eplb_transfer_runtime
assert async_worker.transfer_layer is eplb_transfer_runtime.transfer_layer
assert eplb_state._move_to_workspace is eplb_runtime._move_to_workspace
assert eplb_state.create_eplb_communicator is eplb_communicator.create_eplb_communicator
print('mixed-eplb-loaded-aliases-ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, VLLM_PLUGINS="__disabled__"),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "mixed-eplb-loaded-aliases-ok" in result.stdout
