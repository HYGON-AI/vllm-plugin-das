# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep MRv2 capture streams and eager RCCL plan ownership separate."""

from __future__ import annotations

import functools
import inspect
import sys
from contextlib import contextmanager
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.worker.gpu.cudagraph_utils"
PATCH_ID = "worker.framework_opt.cudagraph.mrv2_current_stream"
TARGETS = (f"{TARGET_MODULE}.CudaGraphManager.capture",)
_MARKER = "_vllm_hcu_mrv2_cudagraph_stream_applied"
_WRAPPER = "_vllm_hcu_mrv2_cudagraph_stream_wrapper"
_CAPTURE_WRAPPER = "_vllm_hcu_mrv2_cudagraph_eager_comm_wrapper"


@contextmanager
def use_torch_collectives_for_warmup():
    """Keep eager warmup off the communicator holding captured RCCL plans.

    RCCL 2.22.3 frees non-persistent proxy ops from its HIP host callback
    into the same unlocked pool used by ncclLaunchPrepare on the caller
    thread. Once a graph holds persistent plans, eager launches use that
    callback and can corrupt the pool while subsequent ops are allocated.
    The existing torch process group has its own communicator, with no
    captured plans, and preserves the same TP collective semantics.
    Only synthetic eager warmup uses it; graph capture and requests keep
    their normal communicator and asynchronous execution.
    """
    from vllm.distributed import get_tp_group

    device_comm = get_tp_group().device_communicator
    pynccl = getattr(device_comm, "pynccl_comm", None)
    if pynccl is None:
        yield
        return
    disabled = pynccl.disabled
    pynccl.disabled = True
    try:
        yield
    finally:
        pynccl.disabled = disabled


def apply_to_module(module: ModuleType) -> bool:
    cudagraph_utils = load_exact_module(TARGET_MODULE, module)
    manager = require_class(
        cudagraph_utils,
        "CudaGraphManager",
        f"{TARGET_MODULE}.CudaGraphManager",
    )
    capture = require_callable(manager, "capture", TARGETS[0])

    if getattr(cudagraph_utils, _MARKER, False):
        graph = require_callable(
            cudagraph_utils.torch.cuda,
            "graph",
            "torch.cuda.graph",
        )
        if not getattr(graph, _WRAPPER, False) or not getattr(
            capture, _CAPTURE_WRAPPER, False
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale"
            )
        return False

    require_exact_signature(
        capture,
        TARGETS[0],
        positional=("self", "create_forward_fn", "progress_bar_desc"),
        defaults={"progress_bar_desc": "Capturing CUDA graphs"},
    )
    capture_impl = inspect.unwrap(capture)
    capture_code = getattr(capture_impl, "__code__", None)
    required_names = {"torch", "cuda", "CUDAGraph", "graph", "pool"}
    if capture_code is None or not required_names.issubset(capture_code.co_names):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} no longer contains the "
            "audited v0.25 full-graph capture call"
        )

    cuda = getattr(getattr(cudagraph_utils, "torch", None), "cuda", None)
    if cuda is None:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.torch.cuda is missing"
        )
    original_graph = require_callable(cuda, "graph", "torch.cuda.graph")
    current_stream = require_callable(
        cuda,
        "current_stream",
        "torch.cuda.current_stream",
    )

    @functools.wraps(original_graph)
    def graph_on_current_stream(*args, **kwargs):
        caller = sys._getframe(1)
        if (
            caller.f_code is capture_code
            and len(args) < 3
            and "stream" not in kwargs
        ):
            kwargs["stream"] = current_stream()
        return original_graph(*args, **kwargs)

    setattr(graph_on_current_stream, _WRAPPER, True)
    setattr(graph_on_current_stream, "_vllm_hcu_original_graph", original_graph)
    setattr(cuda, "graph", graph_on_current_stream)

    @functools.wraps(capture)
    def capture_with_separate_eager_comm(self, create_forward_fn, *args, **kwargs):
        def create_forward(desc, warmup):
            forward = create_forward_fn(desc, warmup)
            if not warmup:
                return forward

            @functools.wraps(forward)
            def eager_forward(*forward_args, **forward_kwargs):
                with use_torch_collectives_for_warmup():
                    return forward(*forward_args, **forward_kwargs)

            return eager_forward

        return capture(self, create_forward, *args, **kwargs)

    setattr(capture_with_separate_eager_comm, _CAPTURE_WRAPPER, True)
    setattr(manager, "capture", capture_with_separate_eager_comm)
    setattr(cudagraph_utils, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
