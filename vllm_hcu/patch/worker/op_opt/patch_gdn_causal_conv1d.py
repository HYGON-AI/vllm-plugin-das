# SPDX-License-Identifier: Apache-2.0
"""Qwen-local NN-layout deltas for vLLM v0.25.1 GDN causal-conv1d."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import already_applied, load_exact_module, require_callable
from ._gdn_common import (
    normalize_nn_conv_weight,
    require_parameter_names,
    shape_dim,
    use_nn_layout,
)

TARGET_MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
PATCH_ID = "worker.op_opt.mamba.gdn.causal_conv1d"
TARGETS = (
    f"{TARGET_MODULE}.causal_conv1d_fn",
    f"{TARGET_MODULE}.causal_conv1d_update",
)
_MARKER = "_vllm_hcu_gdn_causal_conv1d_applied"
_WRAPPER = "_vllm_hcu_gdn_causal_conv1d_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    qwen = load_exact_module(TARGET_MODULE, module)
    wrapped = (
        (qwen, "causal_conv1d_fn", TARGETS[0], _WRAPPER),
        (qwen, "causal_conv1d_update", TARGETS[1], _WRAPPER),
    )
    if already_applied(qwen, _MARKER, wrapped):
        return False

    causal = require_callable(qwen, "causal_conv1d_fn", TARGETS[0])
    causal_signature = inspect.signature(causal)
    require_parameter_names(
        causal,
        TARGETS[0],
        (
            "x",
            "weight",
            "bias",
            "conv_states",
            "query_start_loc",
            "cache_indices",
            "has_initial_state",
            "activation",
            "pad_slot_id",
            "null_block_id",
            "block_idx_first_scheduled_token",
            "block_idx_last_scheduled_token",
            "initial_state_idx",
            "num_computed_tokens",
            "block_size_to_align",
            "metadata",
            "validate_data",
        ),
    )
    causal_update = require_callable(
        qwen, "causal_conv1d_update", TARGETS[1]
    )
    causal_update_signature = inspect.signature(causal_update)
    require_parameter_names(
        causal_update,
        TARGETS[1],
        (
            "x",
            "conv_state",
            "weight",
            "bias",
            "activation",
            "conv_state_indices",
            "num_accepted_tokens",
            "query_start_loc",
            "max_query_len",
            "null_block_id",
            "block_idx_last_scheduled_token",
            "initial_state_idx",
            "validate_data",
        ),
    )

    @functools.wraps(causal)
    def hcu_causal_conv(*args, **kwargs):
        if not use_nn_layout():
            return causal(*args, **kwargs)
        bound = causal_signature.bind(*args, **kwargs)
        conv_states = bound.arguments.get("conv_states")
        x = bound.arguments["x"]
        expected_dim = shape_dim(conv_states, -2) or shape_dim(x, 0)
        bound.arguments["weight"] = normalize_nn_conv_weight(
            bound.arguments["weight"], expected_dim, TARGETS[0]
        )
        return causal(*bound.args, **bound.kwargs)

    @functools.wraps(causal_update)
    def hcu_causal_update(*args, **kwargs):
        if not use_nn_layout():
            return causal_update(*args, **kwargs)
        bound = causal_update_signature.bind(*args, **kwargs)
        x = bound.arguments["x"]
        conv_state = bound.arguments["conv_state"]
        expected_dim = shape_dim(conv_state, -2) or shape_dim(x, 1)
        bound.arguments["weight"] = normalize_nn_conv_weight(
            bound.arguments["weight"], expected_dim, TARGETS[1]
        )
        return causal_update(*bound.args, **bound.kwargs)

    for function in (hcu_causal_conv, hcu_causal_update):
        setattr(function, _WRAPPER, True)
    setattr(qwen, "_vllm_hcu_original_causal_conv1d_fn", causal)
    setattr(qwen, "_vllm_hcu_original_causal_conv1d_update", causal_update)
    setattr(qwen, "causal_conv1d_fn", hcu_causal_conv)
    setattr(qwen, "causal_conv1d_update", hcu_causal_update)
    setattr(qwen, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
