# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU kernel bindings for vLLM v0.28.1 Qwen GDN."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_exact_signature,
)
from ._boltops_fla import make_boltops_gdn_resolver
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
    f"{TARGET_MODULE}.fused_sigmoid_gating_delta_rule_update",
    f"{TARGET_MODULE}.fused_recurrent_gated_delta_rule_packed_decode",
)
_MARKER = "_vllm_hcu_qwen_gdn_aiter_layout_applied"
_WRAPPER = "_vllm_hcu_qwen_gdn_aiter_layout_wrapper"
_SIGMOID_WRAPPER = "_vllm_hcu_qwen_gdn_sigmoid_wrapper"
_RECURRENT_WRAPPER = "_vllm_hcu_qwen_gdn_recurrent_wrapper"

# This is the audited vLLM v0.28.1 AITER launcher contract.  Binding by name is
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
    wrapped = [
        (qwen, TARGETS[1].rsplit(".", 1)[-1], TARGETS[1], _SIGMOID_WRAPPER),
        (qwen, TARGETS[2].rsplit(".", 1)[-1], TARGETS[2], _RECURRENT_WRAPPER),
    ]
    if aiter_available:
        wrapped.insert(
            0,
            (qwen, TARGETS[0].rsplit(".", 1)[-1], TARGETS[0], _WRAPPER),
        )
    if already_applied(qwen, _MARKER, wrapped):
        return False

    official_sigmoid = require_callable(
        qwen,
        "fused_sigmoid_gating_delta_rule_update",
        TARGETS[1],
    )
    require_exact_signature(
        official_sigmoid,
        TARGETS[1],
        positional=(
            "A_log",
            "a",
            "b",
            "dt_bias",
            "q",
            "k",
            "v",
            "beta",
            "threshold",
            "scale",
            "initial_state",
            "inplace_final_state",
            "cu_seqlens",
            "ssm_state_indices",
            "num_accepted_tokens",
            "use_qk_l2norm_in_kernel",
            "is_kda",
        ),
        defaults={
            "beta": 1.0,
            "threshold": 20.0,
            "scale": None,
            "initial_state": None,
            "inplace_final_state": True,
            "cu_seqlens": None,
            "ssm_state_indices": None,
            "num_accepted_tokens": None,
            "use_qk_l2norm_in_kernel": False,
            "is_kda": False,
        },
    )
    sigmoid_signature = inspect.signature(official_sigmoid)
    official_recurrent = require_callable(
        qwen,
        "fused_recurrent_gated_delta_rule_packed_decode",
        TARGETS[2],
    )
    require_exact_signature(
        official_recurrent,
        TARGETS[2],
        positional=(
            "mixed_qkv",
            "a",
            "b",
            "A_log",
            "dt_bias",
            "scale",
            "initial_state",
            "out",
            "ssm_state_indices",
            "use_qk_l2norm_in_kernel",
        ),
        defaults={"use_qk_l2norm_in_kernel": False},
    )
    recurrent_signature = inspect.signature(official_recurrent)
    resolve_boltops_sigmoid = make_boltops_gdn_resolver(
        "fused_sigmoid_gating_delta_rule_update"
    )
    resolve_boltops_recurrent = make_boltops_gdn_resolver(
        "fused_recurrent_gated_delta_rule_packed_decode"
    )

    @functools.wraps(official_sigmoid)
    def hcu_sigmoid_update(*args, **kwargs):
        bound = sigmoid_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if _boltops_enabled():
            boltops_kernel = resolve_boltops_sigmoid()
            if boltops_kernel is not None:
                return boltops_kernel(*bound.args, **bound.kwargs)
        return official_sigmoid(*bound.args, **bound.kwargs)

    @functools.wraps(official_recurrent)
    def hcu_recurrent_update(*args, **kwargs):
        bound = recurrent_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if _boltops_enabled():
            boltops_kernel = resolve_boltops_recurrent()
            if boltops_kernel is not None:
                return boltops_kernel(*bound.args, **bound.kwargs)
        return official_recurrent(*bound.args, **bound.kwargs)

    if aiter_available:
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
                    f"audited vLLM v0.28.1 AITER contract {aiter_signature}"
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

    setattr(hcu_sigmoid_update, _SIGMOID_WRAPPER, True)
    setattr(hcu_recurrent_update, _RECURRENT_WRAPPER, True)
    setattr(qwen, "_vllm_hcu_original_fused_sigmoid", official_sigmoid)
    setattr(qwen, "_vllm_hcu_original_fused_recurrent", official_recurrent)
    setattr(qwen, "fused_sigmoid_gating_delta_rule_update", hcu_sigmoid_update)
    setattr(
        qwen,
        "fused_recurrent_gated_delta_rule_packed_decode",
        hcu_recurrent_update,
    )
    if aiter_available:
        setattr(hcu_aiter_update, _WRAPPER, True)
        setattr(qwen, "_vllm_hcu_original_gdn_aiter_update", aiter_update)
        setattr(
            qwen,
            "gdn_aiter_fused_reshape_causal_conv1d_update_single_token",
            hcu_aiter_update,
        )
    setattr(qwen, _MARKER, True)
    return True


def _boltops_enabled() -> bool:
    from vllm_hcu.platforms import envs as henvs

    return bool(
        henvs.VLLM_HCU_USE_CUSTOM_OPS
        and henvs.VLLM_HCU_USE_CUSTOM_AITER_FLA
    )


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
