# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep captured TP collectives separate from eager RCCL launches."""
from __future__ import annotations

import functools
from contextlib import nullcontext

import torch


def _route_collective(pynccl, original, eager, stream_position):
    @functools.wraps(original)
    def collective(*args, **kwargs):
        if pynccl.disabled:
            return original(*args, **kwargs)
        stream = kwargs.get("stream")
        if stream is None and len(args) > stream_position:
            stream = args[stream_position]
        # Inspect the stream that will actually receive the collective.
        scope = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with scope:
            target = original if torch.cuda.is_current_stream_capturing() else eager
            return target(*args, **kwargs)

    return collective


def install_graph_ownership(communicator) -> None:
    """Initialize eager TP communication before any graph capture.

    Keep graph plans on the original PyNccl communicator and route eager
    PyNccl collectives to a separate communicator. Captured all-gather also
    uses the original PyNccl communicator, so FULL speculative graphs do not
    capture the Torch process group used by eager logits all-gather on ROCm.
    Custom all-reduce dispatch and the eager all-gather fast path are retained.
    """
    pynccl = getattr(communicator, "pynccl_comm", None)
    if pynccl is None or not pynccl.available or pynccl.disabled:
        return
    if getattr(communicator, "_hcu_eager_pynccl_comm", None) is not None:
        return

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    eager = PyNcclCommunicator(pynccl.group, device=pynccl.device)
    if not eager.available or eager.disabled:
        eager.destroy()
        raise RuntimeError("HCU eager communicator failed to initialize")
    communicator._hcu_eager_pynccl_comm = eager
    originals = {}
    communicator._hcu_graph_originals = originals
    # Positional indices exclude self. Forward all arguments unchanged.
    for name, stream_position in (
        ("all_reduce", 3), ("all_gather", 2), ("all_gatherv", 3),
        ("reduce_scatter", 3), ("reduce_scatterv", 4),
    ):
        original = getattr(pynccl, name, None)
        if original is None:
            continue
        originals[name] = original
        setattr(pynccl, name, _route_collective(
            pynccl, original, getattr(eager, name), stream_position
        ))

    original_gather = getattr(communicator, "all_gather", None)
    communicator._hcu_original_all_gather = original_gather
    if original_gather is None:
        return

    @functools.wraps(original_gather)
    def all_gather(input_, dim=-1):
        if pynccl.disabled or not torch.cuda.is_current_stream_capturing():
            return original_gather(input_, dim)
        if dim < 0:
            dim += input_.dim()
        input_size = input_.size()
        output = torch.empty(
            (input_size[0] * communicator.world_size,) + input_size[1:],
            dtype=input_.dtype, device=input_.device,
        )
        pynccl.all_gather(output, input_.contiguous())
        output = output.reshape((communicator.world_size,) + input_size)
        output = output.movedim(0, dim)
        return output.reshape(
            input_size[:dim] + (communicator.world_size * input_size[dim],)
            + input_size[dim + 1:]
        )

    communicator.all_gather = all_gather


def close_graph_ownership(communicator) -> None:
    eager = getattr(communicator, "_hcu_eager_pynccl_comm", None)
    communicator._hcu_eager_pynccl_comm = None
    pynccl = getattr(communicator, "pynccl_comm", None)
    originals = getattr(communicator, "_hcu_graph_originals", {})
    for name, original in originals.items():
        setattr(pynccl, name, original)
    communicator._hcu_graph_originals = {}
    original_gather = getattr(communicator, "_hcu_original_all_gather", None)
    if original_gather is not None:
        communicator.all_gather = original_gather
        communicator._hcu_original_all_gather = None
    if eager is not None:
        eager.destroy()
