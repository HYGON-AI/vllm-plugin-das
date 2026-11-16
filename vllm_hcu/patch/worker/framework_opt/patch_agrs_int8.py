# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Optional INT8 hidden-state transport for naive MoE AllGatherv dispatch."""

from __future__ import annotations

import functools
from contextlib import contextmanager
from contextvars import ContextVar
from types import ModuleType
from typing import Iterator

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.distributed.device_communicators.all2all"
PATCH_ID = "worker.framework_opt.communicator.agrs_int8_dispatch"
TARGETS = (
    f"{TARGET_MODULE}.AgRsAll2AllManager.dispatch_router_logits",
    f"{TARGET_MODULE}.AgRsAll2AllManager.dispatch",
)
_MARKER = "_vllm_hcu_agrs_int8_dispatch_applied"
_WRAPPER = "_vllm_hcu_agrs_int8_dispatch_wrapper"
_RUNTIME_ENABLED = False
_DIRECT_TRANSPORT = ContextVar(
    "vllm_hcu_int8_allgatherv_direct_transport",
    default=False,
)
_PREQUANTIZED_TRANSPORT = ContextVar(
    "vllm_hcu_int8_allgatherv_prequantized_transport",
    default=None,
)


def configure(enabled: bool) -> None:
    """Latch the effective worker policy resolved before model construction."""

    if not isinstance(enabled, bool):
        raise TypeError("INT8 AllGatherv worker policy must be bool")
    global _RUNTIME_ENABLED
    _RUNTIME_ENABLED = enabled


def _enabled() -> bool:
    return _RUNTIME_ENABLED


@contextmanager
def int8_transport(
    prequantized: tuple[object, object] | None = None,
) -> Iterator[None]:
    """Scope INT8 transport and optionally reuse an existing quantized pair."""

    direct_token = _DIRECT_TRANSPORT.set(True)
    prequantized_token = _PREQUANTIZED_TRANSPORT.set(prequantized)
    try:
        yield
    finally:
        _PREQUANTIZED_TRANSPORT.reset(prequantized_token)
        _DIRECT_TRANSPORT.reset(direct_token)


def _direct_transport_enabled() -> bool:
    return _DIRECT_TRANSPORT.get()


def _supports_transport_quantization(hidden_states: torch.Tensor) -> bool:
    import torch

    # Quantized MoE prepare paths already attach their activation scale through
    # extra_tensors. Re-quantizing those codes would corrupt that contract and
    # cannot reduce their one-byte communication payload.
    return hidden_states.dtype is torch.bfloat16


def _quantize_hidden_states(
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    if hidden_states.shape[0] == 0:
        return (
            torch.empty_like(hidden_states, dtype=torch.int8),
            torch.empty(
                (*hidden_states.shape[:-1], 1),
                dtype=torch.float32,
                device=hidden_states.device,
            ),
        )
    from lightop.quant import per_token_quant_int8

    quantized, scales = per_token_quant_int8(hidden_states)
    if quantized.dtype is not torch.int8 or scales.dtype is not torch.float32:
        raise RuntimeError(
            "LightOp INT8 AllGatherv quantization must return INT8 data and "
            "FP32 per-token scales"
        )
    expected_scale_shape = (*hidden_states.shape[:-1], 1)
    if quantized.shape != hidden_states.shape or scales.shape != expected_scale_shape:
        raise RuntimeError(
            "LightOp INT8 AllGatherv quantization returned incompatible shapes"
        )
    return quantized, scales


def _gather_quantized(
    manager,
    hidden_states: torch.Tensor,
    metadata: list[torch.Tensor],
    is_sequence_parallel: bool,
    extra_tensors: list[torch.Tensor] | None,
) -> list[torch.Tensor]:
    from vllm.logger import init_logger

    init_logger(__name__).info_once(
        "Using INT8 hidden-state AllGatherv for naive DP/EP MoE dispatch"
    )
    group = manager._get_comm_group(is_sequence_parallel)
    sizes = manager._get_sizes(hidden_states.shape[0], group)
    assert sizes[group.rank_in_group] == hidden_states.shape[0]
    prequantized = _PREQUANTIZED_TRANSPORT.get()
    if prequantized is None:
        quantized, scales = _quantize_hidden_states(hidden_states)
    else:
        quantized, scales = prequantized
    tensors = [quantized, scales, *metadata]
    if extra_tensors is not None:
        tensors.extend(extra_tensors)
    return group.all_gatherv(tensors, dim=0, sizes=sizes)


def apply_to_module(module: ModuleType) -> bool:
    all2all = load_exact_module(TARGET_MODULE, module)
    manager = require_class(
        all2all, "AgRsAll2AllManager", f"{TARGET_MODULE}.AgRsAll2AllManager"
    )
    wrapped = (
        (manager, "dispatch_router_logits", TARGETS[0], _WRAPPER),
        (manager, "dispatch", TARGETS[1], _WRAPPER),
    )
    if already_applied(all2all, _MARKER, wrapped):
        return False

    original_router = require_callable(manager, "dispatch_router_logits", TARGETS[0])
    require_exact_signature(
        original_router,
        TARGETS[0],
        positional=(
            "self",
            "hidden_states",
            "router_logits",
            "is_sequence_parallel",
            "extra_tensors",
        ),
        defaults={"is_sequence_parallel": False, "extra_tensors": None},
    )
    original_dispatch = require_callable(manager, "dispatch", TARGETS[1])
    require_exact_signature(
        original_dispatch,
        TARGETS[1],
        positional=(
            "self",
            "hidden_states",
            "topk_weights",
            "topk_ids",
            "is_sequence_parallel",
            "extra_tensors",
        ),
        defaults={"is_sequence_parallel": False, "extra_tensors": None},
    )

    @functools.wraps(original_router)
    def hcu_dispatch_router_logits(
        self,
        hidden_states,
        router_logits,
        is_sequence_parallel=False,
        extra_tensors=None,
    ):
        if (
            not _enabled()
            or not _direct_transport_enabled()
            or not _supports_transport_quantization(hidden_states)
        ):
            return original_router(
                self,
                hidden_states,
                router_logits,
                is_sequence_parallel,
                extra_tensors,
            )
        gathered = _gather_quantized(
            self,
            hidden_states,
            [router_logits],
            is_sequence_parallel,
            extra_tensors,
        )
        if extra_tensors is None:
            return gathered[0], gathered[2], gathered[1]
        return gathered[0], gathered[2], gathered[3:], gathered[1]

    @functools.wraps(original_dispatch)
    def hcu_dispatch(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        is_sequence_parallel=False,
        extra_tensors=None,
    ):
        if (
            not _enabled()
            or not _direct_transport_enabled()
            or not _supports_transport_quantization(hidden_states)
        ):
            return original_dispatch(
                self,
                hidden_states,
                topk_weights,
                topk_ids,
                is_sequence_parallel,
                extra_tensors,
            )
        gathered = _gather_quantized(
            self,
            hidden_states,
            [topk_weights, topk_ids],
            is_sequence_parallel,
            extra_tensors,
        )
        if extra_tensors is None:
            return gathered[0], gathered[2], gathered[3], gathered[1]
        return gathered[0], gathered[2], gathered[3], gathered[4:], gathered[1]

    setattr(hcu_dispatch_router_logits, _WRAPPER, True)
    setattr(hcu_dispatch, _WRAPPER, True)
    setattr(manager, "_vllm_hcu_original_dispatch_router_logits", original_router)
    setattr(manager, "_vllm_hcu_original_dispatch", original_dispatch)
    setattr(manager, "dispatch_router_logits", hcu_dispatch_router_logits)
    setattr(manager, "dispatch", hcu_dispatch)
    setattr(all2all, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
    "configure",
    "int8_transport",
]
