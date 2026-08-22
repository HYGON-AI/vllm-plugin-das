# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import ModuleType, SimpleNamespace

from vllm_hcu.patch.worker.framework_opt.patch_mrv2_cudagraph_stream import (
    apply_to_module,
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
