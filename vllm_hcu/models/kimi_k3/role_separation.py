"""Kimi-K3 NCCL communicator role separation (auxiliary eager-AR comm).

DCU/RCCL requirement: one NCCL communicator must
host only ONE collective role — captured (CUDA-graph) collectives or eager
collectives, never both.  Mixing replayed decode-phase collectives with
eager prefill-phase collectives on the same communicator deadlocks the
RCCL LL protocol.

Ported from the vLLM functional branch (kimi-k3-dspark-liuwc, commit
877e6996) to the HCU plugin.  The reroute targets the paired v0.25.1
runtime while keeping the plugin boundary: no vLLM source tree is
modified, and the auxiliary communicator is an HCU-owned construction.

This module owns an auxiliary "eager" TP all_reduce communicator for the
Kimi-K3 target and DSpark draft models:

  - the TP pynccl communicator keeps only captured-graph all_reduces
    (the target decode graph and the DSpark draft graph);
  - the auxiliary communicator keeps only eager all_reduces
    (prefill / profile / warmup forwards).

The reroute is installed inside the plugin-owned Kimi-K3 model forwards
only (see ``eager_ar_scope`` usage in
``vllm_hcu/models/kimi_k3/amd/linear.py`` and
``vllm_hcu/models/kimi_k3/dspark.py``): each scope temporarily swaps the
tensor-parallel ``GroupCoordinator``'s ``_all_reduce_out_place`` at
INSTANCE level, so no other model, coordinator group (DP/PP/EP), or code
path observes it, and the original state is restored on scope exit.
During a CUDA-graph capture (or torch.compile tracing) the scope becomes
a no-op (and never builds the auxiliary communicator), so captured ARs
stay recorded on — and replayed from — the original communicator.

Rollback: set ``VLLM_ROLE_SEP=0`` to disable everything.

NOTE: the swap relies on the ``_all_reduce_out_place`` method name of
``GroupCoordinator`` (both the direct ``all_reduce`` call and the
``torch.ops.vllm.all_reduce`` custom op resolve through it).  If that
method is ever renamed, this scope silently stops rerouting; a strict
validation run (``VLLM_ROLE_SEP_STRICT=1``) plus a runtime check that the
auxiliary communicator exists detects it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)

# Factory default is on: this module ships with the Kimi-K3 DCU deployment,
# where mixed-role communicators deadlock the RCCL stack.  Set
# VLLM_ROLE_SEP=0 to restore the legacy single-communicator behavior.
_ROLE_SEP_ENV = "VLLM_ROLE_SEP"
_STRICT_ENV = "VLLM_ROLE_SEP_STRICT"

_aux_ar_comm: "PyNcclCommunicator | None" = None
_tp_group: "GroupCoordinator | None" = None
_disabled = False

_SWAP_ATTR = "_all_reduce_out_place"


def _role_separation_enabled() -> bool:
    return not _disabled and os.getenv(_ROLE_SEP_ENV, "1").lower() not in (
        "false",
        "0",
    )


def role_separation_strict() -> bool:
    """Strict mode for validation deployments (VLLM_ROLE_SEP_STRICT=1).

    The RCCL deadlock needs one communicator to carry two collective roles.
    Two silent degradations put a deployment
    back on that topology without any visible symptom:

      1. the auxiliary eager-AR communicator cannot be built (this module
         falls back to the shared TP communicator);
      2. the DSpark draft graph keeps its logits all_gather captured.

    Strict mode turns both into hard errors so a validation run cannot
    pass while the dangerous topology is in place.  Default off:
    production keeps the previous warn-and-continue behaviour.
    """
    return os.getenv(_STRICT_ENV, "0").lower() not in ("", "0", "false")


def _raise_or_warn(msg: str) -> None:
    if role_separation_strict():
        raise RuntimeError(msg)
    logger.warning_once(msg)


def _get_aux_ar_comm():
    """Construct the auxiliary eager-AR communicator once.

    Called from ``eager_ar_scope`` at the top of a Kimi-K3 model forward.
    Every rank of the TP group executes that forward symmetrically (SPMD),
    so the collective ``ncclCommInitRank`` stays rank-symmetric.  Returns
    None when the feature is off or the communicator cannot be built (with
    a one-time warning and a permanent fallback to the shared TP comm).
    """
    global _aux_ar_comm, _tp_group, _disabled
    if _aux_ar_comm is not None or _disabled:
        return _aux_ar_comm

    try:
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )
        from vllm.distributed.parallel_state import GroupCoordinator, get_tp_group

        if get_tensor_model_parallel_world_size() <= 1:
            _disabled = True
            return None
        tp = get_tp_group()
        comm = PyNcclCommunicator(
            group=tp.cpu_group, device=torch.cuda.current_device()
        )
        if not (comm.available and not comm.disabled):
            raise RuntimeError("auxiliary PyNcclCommunicator is unavailable")
        _aux_ar_comm = comm
        _tp_group = tp
        logger.info_once(
            "Kimi-K3 role separation: eager-phase TP all_reduce routed to a "
            "dedicated NCCL communicator; captured-graph ARs keep the "
            "original TP communicator."
        )
        return _aux_ar_comm
    except Exception as e:
        # Fall back to the legacy single-communicator behavior rather than
        # break the service; the captured-x-eager pairing then remains as
        # before this feature existed.  Strict validation runs must not
        # proceed on that topology.
        _disabled = True
        _raise_or_warn(
            "Kimi-K3 role separation disabled: failed to build the "
            "auxiliary all_reduce communicator (%s); the captured-x-eager "
            "same-communicator pair is back (set VLLM_ROLE_SEP_STRICT=1 to "
            "make this fatal)." % e
        )
        return None


def eager_all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Eager-role TP all_gather: the docking point for the DSpark
    draft-logits replay tail.

    The tail must NOT use ``tensor_model_parallel_all_gather``.  In the
    source branch the c10d TP communicator only ever carried eager gathers,
    so docking the eager tail there was safe; in this paired runtime the
    target decode graph CAPTURES its own logits all_gather on c10d, making
    c10d a captured-role comm.  An eager tail gather on it would re-create
    the captured-x-eager same-communicator pair that deadlocks RCCL.  The
    auxiliary eager communicator from ``_get_aux_ar_comm`` carries the tail
    instead: eager AR (``eager_ar_scope``) and eager AG share one comm, and
    that comm only ever sees eager collectives.

    ``VLLM_ROLE_SEP=0`` restores the legacy c10d gather verbatim (rollback
    contract); during a capture the legacy path is used as well because a
    capture must never construct or touch the auxiliary comm (the caller's
    capture guard is the real safety net there).
    """
    if (
        not _role_separation_enabled()
        or torch.cuda.is_current_stream_capturing()
    ):
        from vllm.distributed import tensor_model_parallel_all_gather

        return tensor_model_parallel_all_gather(tensor, dim=dim)
    comm = _aux_ar_comm if _aux_ar_comm is not None else _get_aux_ar_comm()
    if comm is None:
        # Feature disabled or the build failed (reported once by
        # _get_aux_ar_comm).  Keep serving on the legacy pairing.
        from vllm.distributed import tensor_model_parallel_all_gather

        return tensor_model_parallel_all_gather(tensor, dim=dim)
    from vllm.distributed import get_tensor_model_parallel_world_size

    if dim < 0:
        dim += tensor.dim()
    tensor = tensor.contiguous()
    world_size = get_tensor_model_parallel_world_size()
    input_size = tensor.size()
    # Layout ported byte-exact from DeviceCommunicatorBase.all_gather (the
    # official tensor_model_parallel_all_gather(dim=...) implementation).
    # ncclAllGather has NO dim semantics: rank r's chunk lands at flat
    # offset r*numel (concat-style along dim 0); the ``dim`` layout is
    # rebuilt afterwards with reshape/movedim/reshape.  The first version
    # of this function allocated a dim-scaled buffer and fed it to nccl
    # directly, so [B, V/ws] shards reassembled with WRONG row blocks
    # (shape-valid garbage) -- runtime evidence 20261010: DSpark per-
    # position acceptance ~0 while drafted throughput and GSM8K accuracy
    # stayed healthy.
    output = torch.empty(
        (input_size[0] * world_size,) + input_size[1:],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    comm.all_gather(output, tensor)
    output = output.reshape((world_size,) + input_size)
    output = output.movedim(0, dim)
    return output.reshape(
        input_size[:dim]
        + (world_size * input_size[dim],)
        + input_size[dim + 1 :]
    )


@contextmanager
def eager_ar_scope() -> Iterator[None]:
    """Route this rank's eager TP all_reduces to the auxiliary comm.

    Instance-level swap of the TP coordinator's ``_all_reduce_out_place``:
    both the direct call (``GroupCoordinator.all_reduce``) and the
    ``torch.ops.vllm.all_reduce`` custom op resolve through it, while every
    other coordinator (DP/PP/EP) and any other model is untouched.  Nested
    scopes restore in LIFO order.

    The scope is a no-op while a CUDA-graph capture or a torch.compile
    trace is in progress: capture-time enqueues must be recorded on the
    captured-role comm (the one replays will use), the auxiliary
    communicator must never be constructed inside a capture (its init
    synchronizes the stream), and traces must not bake the reroute in.
    """
    if not _role_separation_enabled():
        yield
        return
    if (
        torch.cuda.is_current_stream_capturing()
        or torch.compiler.is_compiling()
    ):
        yield
        return

    comm = (
        _aux_ar_comm
        if _aux_ar_comm is not None
        else _get_aux_ar_comm()
    )
    if comm is None:
        yield
        return
    tp = _tp_group
    assert tp is not None
    saved = tp.__dict__.get(_SWAP_ATTR)
    original = saved if saved is not None else tp._all_reduce_out_place

    def _shim(input_: torch.Tensor, _comm=comm, _original=original):
        # Belt-and-braces for capture boundaries opening inside the scope.
        if torch.cuda.is_current_stream_capturing():
            return _original(input_)
        return _comm.all_reduce(input_)

    tp.__dict__[_SWAP_ATTR] = _shim
    try:
        yield
    finally:
        if saved is None:
            tp.__dict__.pop(_SWAP_ATTR, None)
        else:
            tp.__dict__[_SWAP_ATTR] = saved
