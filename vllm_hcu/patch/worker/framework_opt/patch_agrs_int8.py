# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Optional INT8 hidden-state transport for naive MoE AllGatherv dispatch."""

from __future__ import annotations

import functools
from types import ModuleType

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


def _enabled() -> bool:
    from vllm.config import get_current_vllm_config_or_none

    from vllm_hcu.patch.config import get_hcu_config
    from vllm_hcu.platforms import envs as henvs

    config = get_current_vllm_config_or_none()
    if config is not None:
        return get_hcu_config(config).dp_int8_allgather
    return henvs.dp_int8_allgather_enabled()


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


def _dequantize_hidden_states(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    import torch

    return (quantized.to(torch.float32) * scales).to(dtype)


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
    quantized, scales = _quantize_hidden_states(hidden_states)
    tensors = [quantized, scales, *metadata]
    if extra_tensors is not None:
        tensors.extend(extra_tensors)
    gathered = group.all_gatherv(tensors, dim=0, sizes=sizes)
    restored = _dequantize_hidden_states(gathered[0], gathered[1], hidden_states.dtype)
    return [restored, *gathered[2:]]


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
        if not _enabled():
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
            return gathered[0], gathered[1]
        return gathered[0], gathered[1], gathered[2:]

    @functools.wraps(original_dispatch)
    def hcu_dispatch(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        is_sequence_parallel=False,
        extra_tensors=None,
    ):
        if not _enabled():
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
            return gathered[0], gathered[1], gathered[2]
        return gathered[0], gathered[1], gathered[2], gathered[3:]

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


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
