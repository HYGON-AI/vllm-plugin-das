# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU kernel bindings for vLLM v0.28.1 Qwen GDN."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_exact_signature,
)
from ._aiter_fla import make_aiter_gdn_resolver
from ._boltops_fla import make_boltops_gdn_resolver
from ._gdn_backend import (
    AITER,
    BOLTOPS,
    aiter_input_fallback,
    make_provider_selector,
    record_prefill_backend,
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
    f"{TARGET_MODULE}.fused_sigmoid_gating_delta_rule_update",
    f"{TARGET_MODULE}.fused_recurrent_gated_delta_rule_packed_decode",
)
_MARKER = "_vllm_hcu_qwen_gdn_aiter_layout_applied"
_WRAPPER = "_vllm_hcu_qwen_gdn_aiter_layout_wrapper"
_SIGMOID_WRAPPER = "_vllm_hcu_qwen_gdn_sigmoid_wrapper"
_RECURRENT_WRAPPER = "_vllm_hcu_qwen_gdn_recurrent_wrapper"
_PREFILL_WRAPPER = "_vllm_hcu_qwen_gdn_prefill_backend_wrapper"
_PREFILL_TARGET = f"{TARGET_MODULE}._resolve_gdn_prefill_backend"

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


def _aiter_state_indices(ssm_state_indices):
    """Map vLLM NULL_BLOCK_ID (0) padding to aiter's negative null index.

    vLLM and BoltOPs (null_state_index=0) skip state index <= 0; the aiter
    HIP kernels only skip < 0 and would read/write state block 0 for
    CUDA-graph padding rows.
    """
    if ssm_state_indices is None:
        return None
    return torch.where(
        ssm_state_indices > 0,
        ssm_state_indices,
        torch.full_like(ssm_state_indices, -1),
    )


_AITER_SIGMOID_HEAD_DIMS = (32, 64, 128, 256, 512)


def _aiter_sigmoid_unsupported(arguments) -> str | None:
    """Return why aiter's HIP sigmoid update rejects these inputs, or None."""
    q = arguments["q"]
    state = arguments["initial_state"]
    indices = arguments["ssm_state_indices"]
    cu_seqlens = arguments["cu_seqlens"]
    if state is None or state.dtype != torch.float32 or state.stride(-1) != 1:
        return "a missing or non-fp32 recurrent state"
    if indices is not None and indices.dtype != torch.int32:
        return f"{indices.dtype} state indices"
    if cu_seqlens is not None and cu_seqlens.dtype != torch.int32:
        return f"{cu_seqlens.dtype} cu_seqlens"
    if arguments["num_accepted_tokens"] is not None and (
        indices is None or indices.dim() != 2
    ):
        return "spec decode without 2D state indices"
    if q.shape[-1] not in _AITER_SIGMOID_HEAD_DIMS:
        return f"head_k_dim={q.shape[-1]}"
    activations = ("k", "v", "a", "b", "dt_bias")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32) or any(
        arguments[name].dtype != q.dtype for name in activations
    ) or arguments["A_log"].dtype != torch.float32:
        return "mixed activation dtypes or non-fp32 A_log"
    return None


def _aiter_recurrent_unsupported(arguments) -> str | None:
    """Return why aiter's HIP packed decode rejects these inputs, or None."""
    state = arguments["initial_state"]
    indices = arguments["ssm_state_indices"]
    if indices.dim() != 1 or indices.dtype not in (torch.int32, torch.int64):
        return "non-1D or non-integer state indices"
    if state.stride(-1) != 1 or state.stride(-2) != state.shape[-1]:
        return "a non-contiguous recurrent state"
    if not arguments["out"].is_contiguous():
        return "a non-contiguous output buffer"
    return None


def apply_to_module(module: ModuleType) -> bool:
    qwen = load_exact_module(TARGET_MODULE, module)
    aiter_available = bool(
        getattr(
            qwen,
            "_vllm_hcu_original_gdn_aiter_triton_available",
            getattr(qwen, "GDN_AITER_TRITON_AVAILABLE", False),
        )
    )
    wrapped = [
        (qwen, TARGETS[1].rsplit(".", 1)[-1], TARGETS[1], _SIGMOID_WRAPPER),
        (qwen, TARGETS[2].rsplit(".", 1)[-1], TARGETS[2], _RECURRENT_WRAPPER),
    ]
    if aiter_available:
        wrapped.insert(
            0,
            (qwen, TARGETS[0].rsplit(".", 1)[-1], TARGETS[0], _WRAPPER),
        )
    official_resolver = getattr(qwen, "_resolve_gdn_prefill_backend", None)
    if callable(official_resolver):
        wrapped.append(
            (qwen, "_resolve_gdn_prefill_backend", _PREFILL_TARGET,
             _PREFILL_WRAPPER)
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
    sigmoid_op = "fused_sigmoid_gating_delta_rule_update"
    recurrent_op = "fused_recurrent_gated_delta_rule_packed_decode"
    resolve_boltops_sigmoid = make_boltops_gdn_resolver(sigmoid_op)
    resolve_boltops_recurrent = make_boltops_gdn_resolver(recurrent_op)
    select_sigmoid = make_provider_selector(
        "decode",
        sigmoid_op,
        {
            BOLTOPS: resolve_boltops_sigmoid,
            AITER: make_aiter_gdn_resolver(sigmoid_op),
        },
    )
    select_recurrent = make_provider_selector(
        "decode",
        recurrent_op,
        {
            BOLTOPS: resolve_boltops_recurrent,
            AITER: make_aiter_gdn_resolver(recurrent_op),
        },
    )

    @functools.wraps(official_sigmoid)
    def hcu_sigmoid_update(*args, **kwargs):
        bound = sigmoid_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        provider, kernel = select_sigmoid()
        if provider == AITER:
            reason = _aiter_sigmoid_unsupported(bound.arguments)
            if reason is not None:
                provider, kernel = aiter_input_fallback(
                    "decode", sigmoid_op, reason, resolve_boltops_sigmoid
                )
        if provider == BOLTOPS:
            return kernel(*bound.args, **bound.kwargs, null_state_index=0)
        if provider == AITER:
            bound.arguments["ssm_state_indices"] = _aiter_state_indices(
                bound.arguments["ssm_state_indices"]
            )
            return kernel(*bound.args, **bound.kwargs)
        return official_sigmoid(*bound.args, **bound.kwargs)

    @functools.wraps(official_recurrent)
    def hcu_recurrent_update(*args, **kwargs):
        bound = recurrent_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        provider, kernel = select_recurrent()
        if provider == AITER:
            reason = _aiter_recurrent_unsupported(bound.arguments)
            if reason is not None:
                provider, kernel = aiter_input_fallback(
                    "decode", recurrent_op, reason, resolve_boltops_recurrent
                )
        if provider == BOLTOPS:
            return kernel(*bound.args, **bound.kwargs, null_state_index=0)
        if provider == AITER:
            bound.arguments["ssm_state_indices"] = _aiter_state_indices(
                bound.arguments["ssm_state_indices"]
            )
            return kernel(*bound.args, **bound.kwargs)
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
        # VLLM_HCU_GDN_DECODE_BACKEND owns GDN decode.  vLLM's ROCm AITER
        # fast path (taken only with VLLM_ROCM_USE_AITER=1) would bypass the
        # selected sigmoid/recurrent provider, so HCU never takes it.
        setattr(qwen, "_vllm_hcu_original_gdn_aiter_triton_available", True)
        setattr(qwen, "GDN_AITER_TRITON_AVAILABLE", False)

    if callable(official_resolver):

        @functools.wraps(official_resolver)
        def hcu_resolve_gdn_prefill_backend(vllm_config):
            # Capture --gdn-prefill-backend while layers are constructed so
            # the FLA kernel patches can select their provider later.
            record_prefill_backend(vllm_config)
            return official_resolver(vllm_config)

        setattr(hcu_resolve_gdn_prefill_backend, _PREFILL_WRAPPER, True)
        setattr(
            qwen,
            "_vllm_hcu_original_resolve_gdn_prefill_backend",
            official_resolver,
        )
        setattr(
            qwen,
            "_resolve_gdn_prefill_backend",
            hcu_resolve_gdn_prefill_backend,
        )
    setattr(qwen, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
