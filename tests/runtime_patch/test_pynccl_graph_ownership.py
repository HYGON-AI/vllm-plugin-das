# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""RCCL eager/captured all-reduce ownership through the real adapter."""
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.framework_opt import patch_cuda_communicator as adapter


@pytest.fixture
def communicator_factory(monkeypatch):
    import vllm.config
    import vllm.distributed.device_communicators.pynccl as pynccl_module

    instances = []
    calls = []
    capture = [False]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capture[0])
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    class PyNccl:
        def __init__(self, group, device, library_path=None):
            self.group = group
            self.device = device
            self.world_size = 16
            self.available = True
            self.disabled = False
            self.index = len(instances)
            self.destroyed = 0
            instances.append(self)

        def all_reduce(self, in_tensor, out_tensor=None,
                       op=torch.distributed.ReduceOp.SUM, stream=None):
            if self.disabled:
                return None
            calls.append((self.index, in_tensor, out_tensor, op, stream))
            return out_tensor if out_tensor is not None else self.index

        def all_gather(self, output_tensor, input_tensor, stream=None):
            calls.append((self.index, input_tensor, output_tensor, None, stream))
            output_tensor.copy_(torch.cat([input_tensor + rank for rank in range(16)]))

        def all_gatherv(self, output_tensor, input_tensor, sizes, stream=None):
            calls.append((self.index, input_tensor, output_tensor, sizes, stream))

        def reduce_scatter(self, output_tensor, input_tensor,
                           op=torch.distributed.ReduceOp.SUM, stream=None):
            calls.append((self.index, input_tensor, output_tensor, op, stream))

        def reduce_scatterv(self, output_tensor, input_tensor, sizes,
                            op=torch.distributed.ReduceOp.SUM, stream=None):
            calls.append((self.index, input_tensor, output_tensor, (sizes, op), stream))

        def destroy(self):
            self.destroyed += 1
            self.disabled = True

    monkeypatch.setattr(pynccl_module, "PyNcclCommunicator", PyNccl)

    def factory(disabled=False, missing=False, unique_name="tp:0", graph_mode=None):
        from vllm.config.compilation import CUDAGraphMode

        config = SimpleNamespace(
            additional_config={},
            compilation_config=SimpleNamespace(
                cudagraph_mode=(
                    CUDAGraphMode.FULL_AND_PIECEWISE
                    if graph_mode is None else graph_mode
                )
            ),
        )
        monkeypatch.setattr(
            vllm.config, "get_current_vllm_config_or_none", lambda: config
        )
        class CudaCommunicator:
            def __init__(self, cpu_group, device=None, device_group=None,
                         unique_name="", global_ranks=None, global_world_size=None,
                         tcp_store_group=None, use_all2all=False):
                self.cpu_group = cpu_group
                self.device = device
                self.device_group = device_group
                self.world_size = 16
                self.use_all2all = use_all2all
                self.pynccl_comm = None if missing else PyNccl(cpu_group, device)
                if self.pynccl_comm is not None:
                    self.pynccl_comm.disabled = disabled

            def all_gather(self, input_, dim=-1):
                calls.append(("torch", input_, dim, None, None))
                return torch.cat([input_ + rank for rank in range(16)], dim=dim)

            def all_gatherv(self, input_, dim=0, sizes=None):
                return input_

            def destroy(self):
                if self.pynccl_comm is not None:
                    self.pynccl_comm.destroy()

        module = ModuleType(adapter.TARGET_MODULE)
        module.CudaCommunicator = CudaCommunicator
        assert adapter.apply_to_module(module)
        assert not adapter.apply_to_module(module)
        return CudaCommunicator("rank-group", device="cuda:0", unique_name=unique_name)

    return factory, instances, calls, capture


@pytest.mark.parametrize("capturing,expected", [(False, 1), (True, 0)])
def test_all_reduce_separates_eager_from_captured_plans(communicator_factory,
                                                       capturing, expected):
    factory, instances, calls, capture = communicator_factory
    owner = factory()
    capture[0] = capturing
    assert owner.pynccl_comm.all_reduce("input") == expected
    assert len(instances) == 2
    assert [(c.group, c.device) for c in instances] == [("rank-group", "cuda:0")] * 2
    assert calls[-1][0] == expected


def test_all_reduce_preserves_output_op_and_explicit_stream(communicator_factory):
    factory, _, calls, _ = communicator_factory
    owner = factory()
    out, stream = object(), object()
    op = torch.distributed.ReduceOp.MAX
    assert owner.pynccl_comm.all_reduce("input", out, op, stream) is out
    assert calls == [(1, "input", out, op, stream)]


def test_disabled_communicator_keeps_disabled_semantics(communicator_factory):
    factory, _, calls, _ = communicator_factory
    owner = factory()
    owner.pynccl_comm.disabled = True
    assert owner.pynccl_comm.all_reduce("input") is None
    assert calls == []


@pytest.mark.parametrize("missing", [False, True])
def test_unavailable_communicator_does_not_allocate_peer(communicator_factory, missing):
    factory, instances, _, _ = communicator_factory
    factory(disabled=True, missing=missing)
    assert len(instances) == (0 if missing else 1)


def test_teardown_destroys_both_communicators(communicator_factory):
    factory, instances, _, _ = communicator_factory
    owner = factory()
    owner.destroy()
    assert len(instances) == 2
    assert [c.destroyed for c in instances] == [1, 1]


def test_explicit_stream_capture_uses_original_communicator(
    communicator_factory, monkeypatch
):
    from contextlib import contextmanager

    factory, _, calls, capture = communicator_factory
    owner = factory()
    selected = object()

    @contextmanager
    def select_stream(stream):
        assert stream is selected
        capture[0] = True
        try:
            yield
        finally:
            capture[0] = False

    monkeypatch.setattr(torch.cuda, "stream", select_stream)
    assert owner.pynccl_comm.all_reduce("input", stream=selected) == 0
    assert calls[-1][0] == 0
    assert capture[0] is False


def test_initialization_failure_releases_eager_communicator(
    communicator_factory, monkeypatch
):
    import vllm_hcu.distributed.device_communicators.step5_dp_fabric as fabric

    factory, instances, _, _ = communicator_factory

    def fail(*args, **kwargs):
        raise RuntimeError("later initialization failed")

    monkeypatch.setattr(fabric, "maybe_install", fail)
    with pytest.raises(RuntimeError, match="later initialization failed"):
        factory()
    assert len(instances) == 2
    assert instances[1].destroyed == 1


def test_eager_communicator_cleanup_is_idempotent(communicator_factory):
    from vllm_hcu.distributed.device_communicators.pynccl_graph import (
        close_graph_ownership,
        install_graph_ownership,
    )

    factory, instances, _, _ = communicator_factory
    owner = factory()
    install_graph_ownership(owner)
    assert len(instances) == 2
    close_graph_ownership(owner)
    owner.destroy()
    assert [c.destroyed for c in instances] == [1, 1]


@pytest.mark.parametrize("group_name,graphs", [
    ("dp:0", True), ("ep:0", True), ("pp:0", True), ("tp:0", False),
])
def test_only_graph_enabled_tp_allocates_peer(
    communicator_factory, group_name, graphs
):
    from vllm.config.compilation import CUDAGraphMode

    factory, instances, calls, _ = communicator_factory
    mode = CUDAGraphMode.FULL_AND_PIECEWISE if graphs else CUDAGraphMode.NONE
    owner = factory(unique_name=group_name, graph_mode=mode)
    assert len(instances) == 1
    assert owner.pynccl_comm.all_reduce("input") == 0
    assert calls[-1][0] == 0


@pytest.mark.parametrize("capturing", [False, True])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_captured_gather_avoids_eager_torch_group(
    communicator_factory, capturing, dim
):
    factory, _, calls, capture = communicator_factory
    owner = factory()
    capture[0] = capturing
    value = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).t()
    result = owner.all_gather(value, dim=dim)
    torch.testing.assert_close(
        result, torch.cat([value + rank for rank in range(16)], dim=dim),
        rtol=0, atol=0,
    )
    assert calls[-1][0] == (0 if capturing else "torch")


@pytest.mark.parametrize("missing", [False, True])
def test_captured_gather_keeps_unavailable_fallback(communicator_factory, missing):
    factory, _, calls, capture = communicator_factory
    owner = factory(disabled=True, missing=missing)
    capture[0] = True
    owner.all_gather(torch.ones(1, 2))
    assert calls[-1][0] == "torch"


@pytest.mark.parametrize("capturing", [False, True])
@pytest.mark.parametrize("name", ["all_gather", "all_gatherv", "reduce_scatter", "reduce_scatterv"])
@pytest.mark.parametrize("keyword_stream", [False, True])
def test_other_collectives_preserve_stream_and_arguments(
    communicator_factory, capturing, name, keyword_stream
):
    factory, _, calls, capture = communicator_factory
    owner = factory()
    capture[0] = capturing
    value = torch.ones(1, 2)
    output = torch.empty(16, 2)
    stream = object()
    args = [output, value]
    if name in ("all_gatherv", "reduce_scatterv"):
        args.append([1] * 16)
    if name.startswith("reduce_scatter"):
        args.append(torch.distributed.ReduceOp.MAX)
    if keyword_stream:
        getattr(owner.pynccl_comm, name)(*args, stream=stream)
    else:
        getattr(owner.pynccl_comm, name)(*args, stream)
    assert calls[-1][0] == (0 if capturing else 1)
    assert calls[-1][-1] is stream
    assert calls[-1][1] is value
    assert calls[-1][2] is output


def test_cleanup_restores_original_routing(communicator_factory):
    from vllm_hcu.distributed.device_communicators.pynccl_graph import close_graph_ownership

    factory, _, calls, _ = communicator_factory
    owner = factory()
    close_graph_ownership(owner)
    assert owner.pynccl_comm.all_reduce("input") == 0
    assert calls[-1][0] == 0
    owner.all_gather(torch.ones(1, 2))
    assert calls[-1][0] == "torch"
