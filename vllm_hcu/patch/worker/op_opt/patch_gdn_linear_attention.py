# SPDX-License-Identifier: Apache-2.0
"""HCU runtime adapters for Gated DeltaNet attention."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.layers.mamba.gdn_linear_attn"
PATCH_ID = "worker.op_opt.mamba.gdn_linear_attention"
TARGETS = (
    f"{TARGET_MODULE}.causal_conv1d_fn",
    f"{TARGET_MODULE}.causal_conv1d_update",
    f"{TARGET_MODULE}.gdn_aiter_fused_reshape_causal_conv1d_update_single_token",
    f"{TARGET_MODULE}.fused_recurrent_gated_delta_rule_packed_decode",
    f"{TARGET_MODULE}.fused_sigmoid_gating_delta_rule_update",
    f"{TARGET_MODULE}.GatedDeltaNetAttention.get_state_dtype",
)
_MARKER = "_vllm_hcu_gdn_runtime_applied"
_WRAPPER = "_vllm_hcu_gdn_runtime_wrapper"


def _require_parameter_names(function, target: str, expected: tuple[str, ...]) -> None:
    try:
        actual = tuple(inspect.signature(function).parameters)
    except (TypeError, ValueError) as exc:
        raise PatchCompatibilityError(f"cannot inspect required target {target}") from exc
    if actual != expected:
        raise PatchCompatibilityError(
            f"required HCU target {target} has incompatible parameters {actual!r}"
        )


def _flags() -> tuple[bool, bool, bool]:
    from vllm_hcu.platforms import envs as henvs

    return (
        bool(henvs.VLLM_USE_NN),
        bool(henvs.VLLM_HCU_USE_CUSTOM_OPS),
        bool(henvs.VLLM_HCU_USE_CUSTOM_CAUSAL_CONV1D),
    )


def _shape_dim(tensor, index: int) -> int | None:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    try:
        return int(shape[index])
    except (IndexError, TypeError, ValueError):
        return None


def _normalize_nn_conv_weight(weight, expected_dim: int | None, target: str):
    use_nn, _, _ = _flags()
    if not use_nn:
        return weight
    ndim = getattr(weight, "ndim", None)
    if ndim != 2 or expected_dim is None:
        raise RuntimeError(
            f"HCU GDN NN-layout requires a 2D conv weight for {target}"
        )
    first = _shape_dim(weight, 0)
    second = _shape_dim(weight, 1)
    if second == expected_dim:
        return weight.transpose(0, 1).contiguous()
    if first == expected_dim:
        return weight.contiguous()
    raise RuntimeError(
        f"HCU GDN NN-layout conv weight for {target} is incompatible: "
        f"weight shape={tuple(weight.shape)!r}, expected dim={expected_dim}"
    )


def apply_to_module(module: ModuleType) -> bool:
    gdn = load_exact_module(TARGET_MODULE, module)
    cls = require_class(gdn, "GatedDeltaNetAttention", f"{TARGET_MODULE}.GatedDeltaNetAttention")
    aiter_available = bool(getattr(gdn, "GDN_AITER_TRITON_AVAILABLE", False))
    wrapped_aiter = (
        (gdn, "gdn_aiter_fused_reshape_causal_conv1d_update_single_token",
         TARGETS[2], _WRAPPER),
    ) if aiter_available else ()
    wrapped = (
        (gdn, "causal_conv1d_fn", TARGETS[0], _WRAPPER),
        (gdn, "causal_conv1d_update", TARGETS[1], _WRAPPER),
        *wrapped_aiter,
        (gdn, "fused_recurrent_gated_delta_rule_packed_decode", TARGETS[3], _WRAPPER),
        (gdn, "fused_sigmoid_gating_delta_rule_update", TARGETS[4], _WRAPPER),
        (cls, "get_state_dtype", TARGETS[5], _WRAPPER),
    )
    if already_applied(gdn, _MARKER, wrapped):
        return False
    causal = require_callable(gdn, "causal_conv1d_fn", TARGETS[0])
    causal_signature = inspect.signature(causal)
    _require_parameter_names(
        causal,
        TARGETS[0],
        ("x", "weight", "bias", "conv_states", "query_start_loc", "cache_indices",
         "has_initial_state", "activation", "pad_slot_id", "null_block_id",
         "block_idx_first_scheduled_token", "block_idx_last_scheduled_token",
         "initial_state_idx", "num_computed_tokens", "block_size_to_align",
         "metadata", "validate_data"),
    )
    causal_update = require_callable(gdn, "causal_conv1d_update", TARGETS[1])
    _require_parameter_names(
        causal_update,
        TARGETS[1],
        ("x", "conv_state", "weight", "bias", "activation", "conv_state_indices",
         "num_accepted_tokens", "query_start_loc", "max_query_len", "null_block_id",
         "block_idx_last_scheduled_token", "initial_state_idx", "validate_data"),
    )
    aiter_update = (
        require_callable(
            gdn,
            "gdn_aiter_fused_reshape_causal_conv1d_update_single_token",
            TARGETS[2],
        )
        if aiter_available else None
    )
    recurrent = require_callable(gdn, "fused_recurrent_gated_delta_rule_packed_decode", TARGETS[3])
    sigmoid = require_callable(gdn, "fused_sigmoid_gating_delta_rule_update", TARGETS[4])
    state_dtype = require_callable(cls, "get_state_dtype", TARGETS[5])
    require_exact_signature(state_dtype, TARGETS[5], positional=("self",))

    @functools.wraps(causal)
    def hcu_causal_conv(x, weight, bias=None, *args, **kwargs):
        use_nn, use_custom_ops, use_custom_conv = _flags()
        conv_states = args[0] if args else kwargs.get("conv_states")
        expected_dim = _shape_dim(conv_states, -2) or _shape_dim(x, 0)
        weight = _normalize_nn_conv_weight(weight, expected_dim, TARGETS[0])
        if not (use_custom_ops and use_custom_conv):
            return causal(x, weight, bias, *args, **kwargs)
        bound = causal_signature.bind(x, weight, bias, *args, **kwargs)
        bound.apply_defaults()
        metadata = bound.arguments["metadata"]
        nums_dict = getattr(metadata, "nums_dict", None)
        if not isinstance(nums_dict, dict) or "seqlens" not in nums_dict:
            raise RuntimeError(
                "HCU causal-conv1d requires metadata.nums_dict['seqlens']; "
                "the causal metadata callback was not applied"
            )
        try:
            from causal_conv1d import causal_conv1d_fn_dcu
        except ImportError as exc:
            raise RuntimeError("HCU custom causal-conv1d is enabled but unavailable") from exc
        return causal_conv1d_fn_dcu(
            bound.arguments["x"], weight, bound.arguments["bias"],
            activation=bound.arguments["activation"],
            initial_states=bound.arguments["conv_states"],
            has_initial_state=bound.arguments["has_initial_state"],
            cache_indices=bound.arguments["cache_indices"],
            query_start_loc=bound.arguments["query_start_loc"],
            seq_lens_cpu=nums_dict["seqlens"],
        )

    @functools.wraps(causal_update)
    def hcu_causal_update(x, conv_state, weight, *args, **kwargs):
        expected_dim = _shape_dim(conv_state, -2) or _shape_dim(x, 1)
        weight = _normalize_nn_conv_weight(weight, expected_dim, TARGETS[1])
        return causal_update(x, conv_state, weight, *args, **kwargs)

    if aiter_update is not None:
        @functools.wraps(aiter_update)
        def hcu_aiter_update(*args, **kwargs):
            if len(args) < 11:
                raise RuntimeError(
                    "HCU GDN AITER update wrapper requires positional conv_state "
                    "and conv weight arguments"
                )
            mutable_args = list(args)
            expected_dim = _shape_dim(mutable_args[9], -2)
            mutable_args[10] = _normalize_nn_conv_weight(
                mutable_args[10], expected_dim, TARGETS[2]
            )
            return aiter_update(*mutable_args, **kwargs)
    else:
        hcu_aiter_update = None

    @functools.wraps(recurrent)
    def hcu_recurrent(*args, **kwargs):
        _, custom, _ = _flags()
        if not custom:
            return recurrent(*args, **kwargs)
        try:
            from aiter.ops.triton.fla.fused_recurrent import (
                fused_recurrent_gated_delta_rule_packed_decode,
            )
        except ImportError as exc:
            raise RuntimeError("HCU GDN recurrent AITER kernel is enabled but unavailable") from exc
        return fused_recurrent_gated_delta_rule_packed_decode(*args, **kwargs)

    @functools.wraps(sigmoid)
    def hcu_sigmoid(*args, **kwargs):
        _, custom, _ = _flags()
        if not custom:
            return sigmoid(*args, **kwargs)
        try:
            from aiter.ops.triton.fla.fused_sigmoid_gating import (
                fused_sigmoid_gating_delta_rule_update,
            )
        except ImportError as exc:
            raise RuntimeError("HCU GDN sigmoid AITER kernel is enabled but unavailable") from exc
        return fused_sigmoid_gating_delta_rule_update(*args, **kwargs)

    @functools.wraps(state_dtype)
    def hcu_state_dtype(self):
        from vllm_hcu.platforms import envs as henvs

        if not (
            henvs.VLLM_HCU_MAMBA_SSM_CACHE_DTYPE
            and henvs.VLLM_HCU_USE_CUSTOM_OPS
        ):
            # Keep the feature-off contract exactly owned by upstream.  This
            # matters if vLLM later adds another dtype validation/default in
            # this method while retaining the audited public signature.
            return state_dtype(self)
        return gdn.MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            "auto",
        )

    functions = [hcu_causal_conv, hcu_causal_update, hcu_recurrent, hcu_sigmoid, hcu_state_dtype]
    if hcu_aiter_update is not None:
        functions.append(hcu_aiter_update)
    for function in functions:
        setattr(function, _WRAPPER, True)
    setattr(gdn, "_vllm_hcu_original_causal_conv1d_fn", causal)
    setattr(gdn, "_vllm_hcu_original_causal_conv1d_update", causal_update)
    if aiter_update is not None:
        setattr(gdn, "_vllm_hcu_original_gdn_aiter_fused_reshape_causal_conv1d_update_single_token", aiter_update)
    setattr(gdn, "_vllm_hcu_original_fused_recurrent", recurrent)
    setattr(gdn, "_vllm_hcu_original_fused_sigmoid", sigmoid)
    setattr(cls, "_vllm_hcu_original_get_state_dtype", state_dtype)
    setattr(gdn, "causal_conv1d_fn", hcu_causal_conv)
    setattr(gdn, "causal_conv1d_update", hcu_causal_update)
    if hcu_aiter_update is not None:
        setattr(gdn, "gdn_aiter_fused_reshape_causal_conv1d_update_single_token", hcu_aiter_update)
    setattr(gdn, "fused_recurrent_gated_delta_rule_packed_decode", hcu_recurrent)
    setattr(gdn, "fused_sigmoid_gating_delta_rule_update", hcu_sigmoid)
    setattr(cls, "get_state_dtype", hcu_state_dtype)
    setattr(gdn, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
