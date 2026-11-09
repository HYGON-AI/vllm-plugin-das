# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Allow explicit AITER selection for the HCU WNA16 runtime adapter."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_parameter_names,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.oracle.int_wna16"
PATCH_ID = "worker.op_opt.moe.oracle.wna16_aiter"
TARGETS = (
    f"{TARGET_MODULE}.WNA16MoEBackend",
    f"{TARGET_MODULE}.map_wna16_backend",
)
_MARKER = "_vllm_hcu_wna16_aiter_oracle_applied"
_WRAPPER_MARKER = "_vllm_hcu_wna16_aiter_mapper"
_GROUPED_WRAPPER_MARKER = "_vllm_hcu_grouped_w4a16_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        mapper = require_callable(target, "map_wna16_backend", TARGETS[1])
        if not getattr(mapper, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"stale HCU WNA16 oracle marker for {TARGET_MODULE}; restart process"
            )
        return False

    backend_enum = require_class(target, "WNA16MoEBackend", TARGETS[0])
    triton_backend = getattr(backend_enum, "TRITON", None)
    if triton_backend is None:
        raise PatchCompatibilityError(f"{TARGETS[0]} has no TRITON member")
    original = require_callable(target, "map_wna16_backend", TARGETS[1])
    require_parameter_names(original, TARGETS[1], ("runner_backend",))

    @functools.wraps(original)
    def hcu_map_wna16_backend(runner_backend):
        # vLLM has no AITER member in its WNA16 oracle.  The HCU compressed-
        # tensors adapter dispatches the actual W4A16 call to AITER and needs
        # the official Triton experts object only as its validated fallback.
        if runner_backend == "aiter":
            return triton_backend
        return original(runner_backend)

    setattr(hcu_map_wna16_backend, _WRAPPER_MARKER, True)
    target._vllm_hcu_original_map_wna16_backend = original
    target.map_wna16_backend = hcu_map_wna16_backend
    _patch_grouped_w4a16_experts(target)
    setattr(target, _MARKER, True)
    return True


def _patch_grouped_w4a16_experts(target: ModuleType) -> None:
    """Route AutoGPTQ's WNA16 expert object through AITER when requested.

    QGPT checkpoints select ``oracle.int_wna16`` directly, so patching only
    ``experts.triton_moe`` is insufficient when that module was imported
    before the worker import coordinator armed its callback.
    """
    experts_cls = getattr(target, "TritonWNA16Experts", None)
    # The small oracle adapter test double only exposes the backend enum and
    # mapper; real vLLM int_wna16 always exports the experts class.
    if experts_cls is None:
        return
    if not isinstance(experts_cls, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.TritonWNA16Experts is missing"
        )
    if getattr(experts_cls, _GROUPED_WRAPPER_MARKER, False):
        return
    original = require_callable(experts_cls, "apply", f"{TARGET_MODULE}.TritonWNA16Experts.apply")
    signature = inspect.signature(original)
    expected = (
        "self", "output", "hidden_states", "w1", "w2", "topk_weights",
        "topk_ids", "activation", "global_num_experts", "expert_map",
        "a1q_scale", "a2_scale", "workspace13", "workspace2",
        "expert_tokens_meta", "apply_router_weight_on_input",
    )
    if tuple(signature.parameters) != expected:
        raise PatchCompatibilityError(
            "unexpected TritonWNA16Experts.apply signature: "
            f"{tuple(signature.parameters)}"
        )

    @functools.wraps(original)
    def hcu_grouped_apply(
        self, output, hidden_states, w1, w2, topk_weights, topk_ids,
        activation, global_num_experts, expert_map, a1q_scale, a2_scale,
        workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input,
    ):
        quant_config = getattr(self, "quant_config", None)
        moe_config = getattr(self, "moe_config", None)
        if (
            getattr(quant_config, "use_int4_w4a16", False)
            and not getattr(quant_config, "_hcu_channel_w4a16", False)
            and getattr(moe_config, "moe_backend", None) == "aiter"
        ):
            from vllm_hcu.model_executor.layers.quantization.grouped_w4a16_moe_runtime import (
                apply_grouped_w4a16_moe,
            )

            if apply_grouped_w4a16_moe(
                self, output, hidden_states, w1, w2, topk_weights, topk_ids,
                activation, global_num_experts, expert_map, a1q_scale, a2_scale,
                expert_tokens_meta, apply_router_weight_on_input,
            ):
                return None
        return original(
            self, output, hidden_states, w1, w2, topk_weights, topk_ids,
            activation, global_num_experts, expert_map, a1q_scale, a2_scale,
            workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input,
        )

    experts_cls._vllm_hcu_original_grouped_apply = original
    experts_cls.apply = hcu_grouped_apply
    setattr(experts_cls, _GROUPED_WRAPPER_MARKER, True)


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
