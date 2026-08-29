# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Metadata and validation for the cold HCU ``vllm._aiter_ops`` exchange."""

from __future__ import annotations

import importlib
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm._aiter_ops"
REPLACEMENT_MODULE = (
    "vllm_hcu.model_executor.layers.fused_moe.aiter_ops"
)
PATCH_ID = "worker.op_opt.aiter_ops.hcu_runtime"
TARGETS = (
    TARGET_MODULE,
    f"{TARGET_MODULE}.is_aiter_found_and_supported",
    f"{TARGET_MODULE}._get_aiter_w16a16_moe_solution_id",
    f"{TARGET_MODULE}._rocm_aiter_fused_moe_impl",
    f"{TARGET_MODULE}._rocm_aiter_topk_softmax_impl",
    f"{TARGET_MODULE}.rocm_aiter_ops.get_aiter_activation_type",
    f"{TARGET_MODULE}.rocm_aiter_ops.is_shuffled_per_token_w8a8_gemm_tuned",
    f"{TARGET_MODULE}.rocm_aiter_ops.is_per_token_w8a8_gemm_tuned",
    f"{TARGET_MODULE}.rocm_aiter_ops.is_fp8bmm_enabled",
    f"{TARGET_MODULE}._rocm_aiter_rmsnorm_fused_dynamic_quant_impl",
    f"{TARGET_MODULE}._rocm_aiter_rmsnorm_fused_add_dynamic_quant_impl",
)

_REPLACEMENT_MARKER = "_vllm_hcu_aiter_ops_replacement"
_VALIDATED_MARKER = "_vllm_hcu_aiter_ops_replacement_validated"
_WRAPPER_MARKER = "_vllm_hcu_aiter_ops_wrapper"

_FUSED_POSITIONAL = (
    "hidden_states",
    "w1",
    "w2",
    "topk_weight",
    "topk_ids",
    "expert_mask",
    "activation_method",
    "quant_method",
    "doweight_stage1",
    "w1_scale",
    "w2_scale",
    "a1_scale",
    "a2_scale",
    "num_local_tokens",
    "output_dtype",
    "hidden_pad",
    "intermediate_pad",
    "gate_mode",
    "bias1",
    "bias2",
    "moe_sorting_dispatch_policy",
    "swiglu_limit",
    "beta",
    "linear_beta",
)
_FUSED_DEFAULTS = {
    "expert_mask": None,
    "activation_method": 0,
    "quant_method": 0,
    "doweight_stage1": False,
    "w1_scale": None,
    "w2_scale": None,
    "a1_scale": None,
    "a2_scale": None,
    "num_local_tokens": None,
    "output_dtype": None,
    "hidden_pad": 0,
    "intermediate_pad": 0,
    "gate_mode": "",
    "bias1": None,
    "bias2": None,
    "moe_sorting_dispatch_policy": 0,
    "swiglu_limit": 0.0,
    "beta": None,
    "linear_beta": None,
}
_TOPK_POSITIONAL = (
    "topk_weights",
    "topk_indices",
    "token_expert_indices",
    "gating_output",
    "renormalize",
    "num_shared_experts",
    "shared_expert_scoring_func",
)
_TOPK_DEFAULTS = {
    "num_shared_experts": 0,
    "shared_expert_scoring_func": "",
}
_SOLUTION_POSITIONAL = (
    "M",
    "E",
    "N1",
    "N2",
    "K",
    "top_k",
    "dtype",
    "activation",
    "use_shuffle",
)
_RMSNORM_POSITIONAL = (
    "x",
    "weight",
    "epsilon",
    "quant_dtype",
)
_RMSNORM_ADD_POSITIONAL = (
    "x",
    "residual",
    "weight",
    "epsilon",
    "quant_dtype",
)


def _require_hcu_wrapper(owner: object, name: str, target: str):
    function = require_callable(owner, name, target)
    if not getattr(function, _WRAPPER_MARKER, False):
        raise PatchCompatibilityError(
            f"HCU AITER replacement target {target} is not HCU-owned"
        )
    return function


def _validate(module: ModuleType) -> None:
    if getattr(module, "__name__", None) != REPLACEMENT_MODULE:
        raise PatchCompatibilityError(
            f"{TARGET_MODULE} registers custom ops at import time and must be "
            f"cold-replaced by {REPLACEMENT_MODULE}; got "
            f"{getattr(module, '__name__', None)!r}"
        )
    if not getattr(module, _REPLACEMENT_MARKER, False):
        raise PatchCompatibilityError(
            "HCU AITER replacement marker is missing; official/HCU operator "
            "ownership cannot be proven"
        )
    if getattr(module, "_HCU_REGISTER_OPS_CALLS", None) != 1:
        raise PatchCompatibilityError(
            "HCU AITER replacement must make exactly one custom-op "
            "registration attempt"
        )
    if not isinstance(getattr(module, "_OPS_REGISTERED", None), bool):
        raise PatchCompatibilityError("HCU AITER registration latch is missing")

    aiter_class = require_class(
        module, "rocm_aiter_ops", f"{TARGET_MODULE}.rocm_aiter_ops"
    )
    supported = _require_hcu_wrapper(
        module, "is_aiter_found_and_supported", TARGETS[1]
    )
    fused = _require_hcu_wrapper(
        module, "_rocm_aiter_fused_moe_impl", TARGETS[3]
    )
    topk = _require_hcu_wrapper(
        module, "_rocm_aiter_topk_softmax_impl", TARGETS[4]
    )
    activation = _require_hcu_wrapper(
        aiter_class, "get_aiter_activation_type", TARGETS[5]
    )
    solution = require_callable(
        module, "_get_aiter_w16a16_moe_solution_id", TARGETS[2]
    )
    require_exact_signature(supported, TARGETS[1])
    require_exact_signature(
        fused,
        TARGETS[3],
        positional=_FUSED_POSITIONAL,
        defaults=_FUSED_DEFAULTS,
    )
    require_exact_signature(
        topk,
        TARGETS[4],
        positional=_TOPK_POSITIONAL,
        defaults=_TOPK_DEFAULTS,
    )
    require_exact_signature(
        activation, TARGETS[5], positional=("activation_str",)
    )
    require_exact_signature(
        solution, TARGETS[2], positional=_SOLUTION_POSITIONAL
    )
    rmsnorm = _require_hcu_wrapper(
        module,
        "_rocm_aiter_rmsnorm_fused_dynamic_quant_impl",
        TARGETS[9],
    )
    rmsnorm_add = _require_hcu_wrapper(
        module,
        "_rocm_aiter_rmsnorm_fused_add_dynamic_quant_impl",
        TARGETS[10],
    )
    require_exact_signature(
        rmsnorm,
        TARGETS[9],
        positional=_RMSNORM_POSITIONAL,
    )
    require_exact_signature(
        rmsnorm_add,
        TARGETS[10],
        positional=_RMSNORM_ADD_POSITIONAL,
    )
    for index, name in (
        (6, "is_shuffled_per_token_w8a8_gemm_tuned"),
        (7, "is_per_token_w8a8_gemm_tuned"),
    ):
        tuning_probe = require_callable(aiter_class, name, TARGETS[index])
        require_exact_signature(
            tuning_probe,
            TARGETS[index],
            positional=("N", "K", "q_dtype_w"),
        )
        if not isinstance(vars(aiter_class).get(name), staticmethod):
            raise PatchCompatibilityError(
                f"required HCU target {TARGETS[index]} must remain a staticmethod"
            )
    fp8_bmm_probe = require_callable(
        aiter_class, "is_fp8bmm_enabled", TARGETS[8]
    )
    require_exact_signature(fp8_bmm_probe, TARGETS[8], positional=())
    if not isinstance(
        vars(aiter_class).get("is_fp8bmm_enabled"), classmethod
    ):
        raise PatchCompatibilityError(
            f"required HCU target {TARGETS[8]} must remain a classmethod"
        )
    if not isinstance(
        vars(aiter_class).get("get_aiter_activation_type"), staticmethod
    ):
        raise PatchCompatibilityError(
            f"required HCU target {TARGETS[5]} must remain a staticmethod"
        )


def apply_to_module(module: ModuleType) -> bool:
    _validate(module)
    if getattr(module, _VALIDATED_MARKER, False):
        return False
    setattr(module, _VALIDATED_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    if module is None:
        module = importlib.import_module(REPLACEMENT_MODULE)
    return apply_to_module(module)


__all__ = [
    "PATCH_ID",
    "REPLACEMENT_MODULE",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
