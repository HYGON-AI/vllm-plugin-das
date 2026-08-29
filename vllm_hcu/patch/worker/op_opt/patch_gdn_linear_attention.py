# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""NN-layout adapter for vLLM v0.28's native Qwen GDN AITER path."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
)
from ._gdn_common import (
    normalize_nn_conv_weight,
    require_parameter_names,
    shape_dim,
    use_nn_layout,
)

TARGET_MODULE = (
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
)
PATCH_ID = "worker.op_opt.mamba.gdn.qwen_kernel_bindings"
TARGETS = (
    f"{TARGET_MODULE}.gdn_aiter_fused_reshape_causal_conv1d_update_single_token",
)
_MARKER = "_vllm_hcu_qwen_gdn_aiter_layout_applied"
_WRAPPER = "_vllm_hcu_qwen_gdn_aiter_layout_wrapper"

# This is the audited vLLM v0.28 AITER launcher contract.  Binding by name is
# intentional: the old HCU adapter rewrote args[10], which could silently
# transpose the wrong object if AITER inserted or reordered a parameter.
_AITER_UPDATE_PARAMETERS = (
    "x",
    "num_actual_tokens",
    "num_k_heads",
    "num_v_heads",
    "head_k_dim",
    "head_v_dim",
    "ba",
    "z_out",
    "core_attn_out",
    "conv_state",
    "weight",
    "bias",
    "activation",
    "conv_state_indices",
    "num_accepted_tokens",
    "query_start_loc",
    "max_query_len",
    "pad_slot_id",
    "block_idx_last_scheduled_token",
    "initial_state_idx",
    "validate_data",
    "qkvz_layout",
)


def apply_to_module(module: ModuleType) -> bool:
    qwen = load_exact_module(TARGET_MODULE, module)
    aiter_available = bool(getattr(qwen, "GDN_AITER_TRITON_AVAILABLE", False))
    wrapped = (
        (qwen, TARGETS[0].rsplit(".", 1)[-1], TARGETS[0], _WRAPPER),
    ) if aiter_available else ()
    if already_applied(qwen, _MARKER, wrapped):
        return False

    if not aiter_available:
        setattr(qwen, _MARKER, True)
        return True

    aiter_update = require_callable(
        qwen,
        "gdn_aiter_fused_reshape_causal_conv1d_update_single_token",
        TARGETS[0],
    )
    require_parameter_names(
        aiter_update,
        TARGETS[0],
        _AITER_UPDATE_PARAMETERS,
    )
    aiter_signature = inspect.signature(aiter_update)

    @functools.wraps(aiter_update)
    def hcu_aiter_update(*args, **kwargs):
        if not use_nn_layout():
            return aiter_update(*args, **kwargs)
        try:
            bound = aiter_signature.bind(*args, **kwargs)
        except TypeError as exc:
            raise PatchCompatibilityError(
                f"required HCU call for {TARGETS[0]} does not match the "
                f"audited vLLM v0.28 AITER contract {aiter_signature}"
            ) from exc
        conv_state = bound.arguments.get("conv_state")
        if conv_state is None or "weight" not in bound.arguments:
            raise PatchCompatibilityError(
                f"required HCU call for {TARGETS[0]} is missing conv_state or weight"
            )
        expected_dim = shape_dim(conv_state, -2)
        bound.arguments["weight"] = normalize_nn_conv_weight(
            bound.arguments["weight"], expected_dim, TARGETS[0]
        )
        return aiter_update(*bound.args, **bound.kwargs)

    setattr(hcu_aiter_update, _WRAPPER, True)
    setattr(qwen, "_vllm_hcu_original_gdn_aiter_update", aiter_update)
    setattr(
        qwen,
        "gdn_aiter_fused_reshape_causal_conv1d_update_single_token",
        hcu_aiter_update,
    )
    setattr(qwen, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
