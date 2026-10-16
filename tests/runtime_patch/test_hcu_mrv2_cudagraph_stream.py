# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

from vllm_hcu.patch.worker.framework_opt.patch_mrv2_cudagraph_stream import (
    apply_to_module,
    use_torch_collectives_for_warmup,
)


TARGET_MODULE = "vllm.v1.worker.gpu.cudagraph_utils"


def _make_target_module():
    module = ModuleType(TARGET_MODULE)
    current_stream = object()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def graph(*args, **kwargs):
        calls.append((args, kwargs))
        return kwargs.get("stream")

    cuda = SimpleNamespace(
        CUDAGraph=object,
        graph=graph,
        current_stream=lambda: current_stream,
    )
    module.torch = SimpleNamespace(cuda=cuda)
    exec(
        "class CudaGraphManager:\n"
        "    def __init__(self):\n"
        "        self.pool = object()\n"
        "    def capture(self, create_forward_fn, "
        "progress_bar_desc='Capturing CUDA graphs'):\n"
        "        graph = torch.cuda.CUDAGraph()\n"
        "        return torch.cuda.graph(graph, self.pool)\n",
        module.__dict__,
    )
    return module, current_stream, calls


def test_mrv2_capture_reuses_current_graph_capture_stream() -> None:
    module, current_stream, calls = _make_target_module()

    assert apply_to_module(module) is True
    manager = module.CudaGraphManager()
    assert manager.capture(lambda: None) is current_stream

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[1] is manager.pool
    assert kwargs == {"stream": current_stream}
    assert apply_to_module(module) is False


def test_non_mrv2_graph_calls_keep_their_default_stream_behavior() -> None:
    module, _, calls = _make_target_module()
    original_graph = module.torch.cuda.graph
    assert apply_to_module(module) is True

    original_graph(object(), object())
    module.torch.cuda.graph(object(), object())

    assert len(calls) == 2
    assert [kwargs for _, kwargs in calls] == [{}, {}]


@pytest.mark.parametrize("disabled", [False, True])
def test_warmup_collective_scope_restores_communicator_on_error(monkeypatch, disabled):
    import vllm.distributed

    comm = SimpleNamespace(disabled=disabled)
    monkeypatch.setattr(
        vllm.distributed,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=SimpleNamespace(pynccl_comm=comm)),
    )
    with pytest.raises(RuntimeError, match="warmup failed"):
        with use_torch_collectives_for_warmup():
            assert comm.disabled is True
            raise RuntimeError("warmup failed")
    assert comm.disabled is disabled


def test_capture_uses_separate_collectives_only_in_eager_warmup(monkeypatch):
    import vllm.distributed

    comm = SimpleNamespace(disabled=False)
    monkeypatch.setattr(
        vllm.distributed,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=SimpleNamespace(pynccl_comm=comm)),
    )
    module, _, _ = _make_target_module()
    exec(
        "def capture(self, create_forward_fn, progress_bar_desc='Capturing CUDA graphs'):\n"
        "    create_forward_fn(4, True)('eager')\n"
        "    graph = torch.cuda.CUDAGraph()\n"
        "    torch.cuda.graph(graph, self.pool)\n"
        "    create_forward_fn(4, False)('capture')\n",
        module.__dict__,
    )
    module.CudaGraphManager.capture = module.capture
    observed = []

    def factory(desc, warmup):
        # Preparation and captured forward retain the original communicator.
        assert comm.disabled is False
        return lambda mode: observed.append((mode, comm.disabled))

    assert apply_to_module(module) is True
    module.CudaGraphManager().capture(factory)
    assert observed == [("eager", True), ("capture", False)]
    assert comm.disabled is False
