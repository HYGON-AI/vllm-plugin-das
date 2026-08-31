# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU-owned AITER compatibility implementations.

This module does not register Torch operators.  The HCU-owned cold replacement
for ``vllm._aiter_ops`` installs these functions before its sole
``rocm_aiter_ops.register_ops_once`` call.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import import_module
from typing import Any

import torch

from vllm_hcu.model_executor.layers.fused_moe.aiter_moe_dispatch import (
    AiterMoeProblem,
    execute_aiter_moe,
    prepare_aiter_moe_weights,
    select_aiter_moe_config,
)


class HcuAiterRuntimeError(RuntimeError):
    """An explicitly requested HCU AITER path cannot be provided."""


def aiter_gate_mode_kwargs(
    gate_mode: str,
    *,
    supports_gate_mode: bool,
) -> dict[str, str]:
    """Return the AITER ABI argument or reject an unsupported layout."""

    if not gate_mode:
        return {}
    if not supports_gate_mode:
        raise HcuAiterRuntimeError(
            "HCU AITER fused_moe ABI does not support "
            f"non-default gate_mode={gate_mode!r}"
        )
    return {"gate_mode": gate_mode}


_EXPLICIT_AITER_MOE: ContextVar[bool] = ContextVar(
    "vllm_hcu_explicit_aiter_moe", default=False
)
_AITER_MOE_GLOBAL_NUM_EXPERTS: ContextVar[int | None] = ContextVar(
    "vllm_hcu_aiter_moe_global_num_experts", default=None
)
_AITER_ASM_BOLTOPS_INT8_QUANT: ContextVar[bool] = ContextVar(
    "vllm_hcu_aiter_asm_boltops_int8_quant", default=False
)
_AITER_ASM_BOLTOPS_FP8_QUANT: ContextVar[bool] = ContextVar(
    "vllm_hcu_aiter_asm_boltops_fp8_quant", default=False
)
_AITER_ASM_INT8_QUANT_WRAPPER_MARKER = (
    "_vllm_hcu_aiter_asm_int8_quant_wrapper"
)
_AITER_ASM_FP8_QUANT_WRAPPER_MARKER = "_vllm_hcu_aiter_asm_fp8_quant_wrapper"
_AITER_ASM_FP8_QUANT_PARAMETERS = (
    "x",
    "scale",
    "quant_dtype",
    "num_rows",
    "num_rows_factor",
)


def _boltops_per_token_quant_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        module = import_module("boltops.fused_moe.triton.moe_compat")
    except ImportError as exc:
        raise HcuAiterRuntimeError(
            "AITER ASM INT8 MoE requires the BoltOps per-token quantizer"
        ) from exc
    operation = getattr(module, "per_token_quant_hip", None)
    if not callable(operation):
        raise HcuAiterRuntimeError(
            "BoltOps exposes no callable per_token_quant_hip"
        )
    return operation(x)


def _boltops_per_token_quant_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        module = import_module("boltops.fused_moe.triton.moe_compat")
    except ImportError as exc:
        raise HcuAiterRuntimeError(
            "AITER ASM FP8 MoE requires the BoltOps per-token quantizer"
        ) from exc
    operation = getattr(module, "per_token_quant_hip", None)
    if not callable(operation):
        raise HcuAiterRuntimeError(
            "BoltOps exposes no callable per_token_quant_hip"
        )
    quantized, scale = operation(x, quant_dtype=torch.float8_e4m3fn)
    zero_scale = scale == 0
    quantized = torch.where(
        zero_scale.expand_as(quantized),
        torch.zeros_like(quantized),
        quantized,
    )
    minimum_scale = torch.full_like(scale, 1.0e-10) * (
        1.0 / torch.finfo(torch.float8_e4m3fn).max
    )
    scale = torch.where(
        zero_scale,
        minimum_scale,
        scale,
    )
    return quantized, scale


def _install_aiter_asm_int8_quant_wrapper() -> None:
    module = import_module("aiter.fused_moe_asm_wna16")
    current = getattr(module, "per_token_quant_int8", None)
    if not callable(current):
        raise HcuAiterRuntimeError(
            "AITER ASM exposes no callable per_token_quant_int8"
        )
    if bool(getattr(current, _AITER_ASM_INT8_QUANT_WRAPPER_MARKER, False)):
        return
    try:
        signature = inspect.signature(current)
    except (TypeError, ValueError) as exc:
        raise HcuAiterRuntimeError(
            "AITER ASM per_token_quant_int8 has no inspectable ABI"
        ) from exc
    if tuple(signature.parameters) != ("x",):
        raise HcuAiterRuntimeError(
            "AITER ASM per_token_quant_int8 exposes unsupported arguments "
            f"{tuple(signature.parameters)!r}"
        )

    original = current

    @functools.wraps(original)
    def wrapped_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if _AITER_ASM_BOLTOPS_INT8_QUANT.get():
            return _boltops_per_token_quant_int8(x)
        return original(x)

    setattr(wrapped_quant, _AITER_ASM_INT8_QUANT_WRAPPER_MARKER, True)
    module.per_token_quant_int8 = wrapped_quant


def _install_aiter_asm_fp8_quant_wrapper() -> None:
    module = import_module("aiter.fused_moe_asm_wna16")
    current = getattr(module, "per_token_quant_hip", None)
    if not callable(current):
        raise HcuAiterRuntimeError(
            "AITER ASM exposes no callable per_token_quant_hip"
        )
    if bool(getattr(current, _AITER_ASM_FP8_QUANT_WRAPPER_MARKER, False)):
        return
    try:
        signature = inspect.signature(current)
    except (TypeError, ValueError) as exc:
        raise HcuAiterRuntimeError(
            "AITER ASM per_token_quant_hip has no inspectable ABI"
        ) from exc
    parameters = signature.parameters
    expected_defaults = {
        "x": inspect.Parameter.empty,
        "scale": None,
        "quant_dtype": torch.int8,
        "num_rows": None,
        "num_rows_factor": 1,
    }
    if tuple(parameters) != _AITER_ASM_FP8_QUANT_PARAMETERS or any(
        parameters[name].kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        or parameters[name].default != expected_default
        for name, expected_default in expected_defaults.items()
    ):
        raise HcuAiterRuntimeError(
            "AITER ASM per_token_quant_hip exposes unsupported arguments "
            f"{signature}"
        )

    original = current

    @functools.wraps(original)
    def wrapped_quant(*args: Any, **kwargs: Any) -> Any:
        if _AITER_ASM_BOLTOPS_FP8_QUANT.get():
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if (
                bound.arguments["quant_dtype"] == torch.float8_e4m3fn
                and bound.arguments.get("scale") is None
                and bound.arguments.get("num_rows") is None
                and bound.arguments.get("num_rows_factor", 1) == 1
            ):
                return _boltops_per_token_quant_fp8(bound.arguments["x"])
        return original(*args, **kwargs)

    setattr(wrapped_quant, _AITER_ASM_FP8_QUANT_WRAPPER_MARKER, True)
    module.per_token_quant_hip = wrapped_quant


@contextmanager
def aiter_asm_boltops_int8_quant_context(enabled: bool):
    """Align AITER ASM's dynamic INT8 quantization with BoltOps Triton."""

    if not enabled:
        yield
        return
    _install_aiter_asm_int8_quant_wrapper()
    token = _AITER_ASM_BOLTOPS_INT8_QUANT.set(True)
    try:
        yield
    finally:
        _AITER_ASM_BOLTOPS_INT8_QUANT.reset(token)


@contextmanager
def aiter_asm_boltops_fp8_quant_context(enabled: bool):
    """Use BoltOps for both dynamic per-token FP8 quantization stages."""

    if enabled:
        _install_aiter_asm_fp8_quant_wrapper()
    token = _AITER_ASM_BOLTOPS_FP8_QUANT.set(bool(enabled))
    try:
        yield
    finally:
        _AITER_ASM_BOLTOPS_FP8_QUANT.reset(token)


def is_aiter_moe_requested(moe_config: object | None = None) -> bool:
    """Keep explicit ``moe_backend=aiter`` independent of auto env gates."""

    if getattr(moe_config, "moe_backend", None) == "aiter":
        return True
    if _EXPLICIT_AITER_MOE.get():
        return True
    try:
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        if (
            config is not None
            and getattr(getattr(config, "kernel_config", None), "moe_backend", None)
            == "aiter"
        ):
            return True
    except (AttributeError, ImportError):
        pass

    import vllm.envs as envs

    return bool(envs.VLLM_ROCM_USE_AITER) and bool(
        envs.VLLM_ROCM_USE_AITER_MOE
    )


@contextmanager
def aiter_moe_request_context(moe_config: object):
    request_token = _EXPLICIT_AITER_MOE.set(
        getattr(moe_config, "moe_backend", None) == "aiter"
    )
    global_num_experts = getattr(moe_config, "num_experts", None)
    if not isinstance(global_num_experts, int) or global_num_experts <= 0:
        global_num_experts = None
    experts_token = _AITER_MOE_GLOBAL_NUM_EXPERTS.set(global_num_experts)
    try:
        yield
    finally:
        _AITER_MOE_GLOBAL_NUM_EXPERTS.reset(experts_token)
        _EXPLICIT_AITER_MOE.reset(request_token)


def _import_optional_aiter_module(module_name: str) -> object | None:
    """Import one optional AITER capability without hiding ABI failures.

    A missing requested module (or one of its package parents) means that the
    optional capability is unavailable.  Missing transitive dependencies and
    other errors raised while executing an existing module are not capability
    misses and must remain visible.
    """

    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_name = exc.name
        if isinstance(missing_name, str) and (
            missing_name == module_name
            or module_name.startswith(f"{missing_name}.")
        ):
            return None
        raise


def get_w8a8_tuned_config_path(
    runtime_symbol: str,
    config_attribute: str,
) -> str | None:
    """Return a usable target-style AITER W8A8 tuning-config path.

    AITER linear kernels are optional candidates.  Some locked HCU AITER
    builds import successfully but expose neither the target runtime callable
    nor the target ``AITER_CONFIGS`` contract.  Treat those expected capability
    gaps as an unavailable candidate so vLLM's native selector can continue;
    propagate other import/runtime failures instead of hiding an ABI fault.
    """

    aiter = _import_optional_aiter_module("aiter")
    if aiter is None or not callable(getattr(aiter, runtime_symbol, None)):
        return None
    gemm_ops = _import_optional_aiter_module("aiter.ops.gemm_op_a8w8")
    if gemm_ops is None:
        return None
    configs = getattr(gemm_ops, "AITER_CONFIGS", None)
    path = getattr(configs, config_attribute, None)
    return path if isinstance(path, str) and path else None


_AITER_TRITON_FP8_BMM_MODULE = (
    "aiter.ops.triton."
    "batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant"
)
_AITER_TRITON_FP8_BMM_SYMBOL = (
    "batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant"
)


def has_triton_fp8_bmm() -> bool:
    """Whether the locked AITER build exposes vLLM's optional MLA FP8 BMM."""

    module = _import_optional_aiter_module(_AITER_TRITON_FP8_BMM_MODULE)
    return module is not None and callable(
        getattr(module, _AITER_TRITON_FP8_BMM_SYMBOL, None)
    )


def is_triton_fp8_bmm_enabled(
    aiter_enabled: bool,
    feature_enabled: bool,
) -> bool:
    """Apply environment gates before probing the optional AITER module."""

    return bool(
        aiter_enabled
        and feature_enabled
        and has_triton_fp8_bmm()
    )


def triton_rope_and_cache_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    layer_slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    flash_layout: bool,
    apply_scale: bool,
) -> None:
    """Compose the RoPE and cache APIs exposed by HCU AITER."""

    from aiter.ops.cache import reshape_and_cache
    from vllm_hcu.v1.attention.backends.fa_utils import (
        reshape_and_cache_flash,
    )
    from aiter.ops.triton.rope import (
        rope_cached_thd_positions_2c_fwd_inplace,
    )

    num_tokens = positions.numel()
    cos, sin = cos_sin_cache.chunk(2, dim=-1)
    head_size = cos.shape[-1]
    query_view = query.view(num_tokens, -1, head_size)
    key_view = key.view(num_tokens, -1, head_size)
    rope_cached_thd_positions_2c_fwd_inplace(
        query_view,
        key_view,
        cos,
        sin,
        positions.view(num_tokens),
        0 if is_neox else 1,
        reuse_freqs_front_part=True,
        nope_first=False,
    )

    kv_cache_dtype = "fp8" if apply_scale else "auto"
    if flash_layout:
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            layer_slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
        return

    reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        layer_slot_mapping,
        kv_cache_dtype,
        float(k_scale.item()),
        float(v_scale.item()),
        False,
    )


_AITER_RMSNORM_DYNAMIC_QUANT_ARGUMENTS = {
    "rmsnorm2d_fwd_with_dynamicquant": (
        "out",
        "input",
        "yscale",
        "weight",
        "epsilon",
    ),
    "rmsnorm2d_fwd_with_add_dynamicquant": (
        "out",
        "input",
        "residual_in",
        "residual_out",
        "yscale",
        "weight",
        "epsilon",
    ),
}
_AITER_MODEL_SENSITIVE_RMSNORM_ARGUMENT = "use_model_sensitive_rmsnorm"
_VLLM_NATIVE_RMSNORM_DYNAMIC_QUANT_ARGUMENTS = (
    "result",
    "input",
    "weight",
    "scale",
    "epsilon",
    "scale_ub",
    "residual",
)


def _schema_argument_names(schema: object, owner: str) -> tuple[str, ...]:
    arguments = getattr(schema, "arguments", None)
    if not isinstance(arguments, (list, tuple)) or not arguments:
        raise HcuAiterRuntimeError(f"{owner} has no readable operator schema")
    names = tuple(getattr(argument, "name", None) for argument in arguments)
    if any(not isinstance(name, str) or not name for name in names):
        raise HcuAiterRuntimeError(
            f"{owner} exposes an operator schema with unnamed arguments"
        )
    return names  # type: ignore[return-value]


def _aiter_rmsnorm_dynamic_quant_abi(op_name: str) -> str:
    """Return the exact supported ABI profile for one AITER RMSNorm op."""

    expected = _AITER_RMSNORM_DYNAMIC_QUANT_ARGUMENTS.get(op_name)
    if expected is None:
        raise HcuAiterRuntimeError(
            f"unsupported HCU AITER RMSNorm operator {op_name!r}"
        )
    packet = getattr(getattr(torch.ops, "aiter", None), op_name, None)
    overload = getattr(packet, "default", None)
    schema = getattr(overload, "_schema", None)
    names = _schema_argument_names(schema, f"aiter::{op_name}")
    if names == expected:
        return "legacy-default"
    if names == expected + (_AITER_MODEL_SENSITIVE_RMSNORM_ARGUMENT,):
        return "model-sensitive"
    raise HcuAiterRuntimeError(
        f"aiter::{op_name} exposes unsupported arguments {names!r}; "
        f"expected {expected!r} with or without the trailing "
        f"{_AITER_MODEL_SENSITIVE_RMSNORM_ARGUMENT!r}"
    )


def _vllm_native_rmsnorm_dynamic_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Use vLLM v0.25.1's native fused op when legacy AITER lacks FP8."""

    import_module("vllm._C_stable_libtorch")
    packet = getattr(
        getattr(torch.ops, "_C", None),
        "rms_norm_dynamic_per_token_quant",
        None,
    )
    operation = getattr(packet, "default", None)
    schema = getattr(operation, "_schema", None)
    names = _schema_argument_names(
        schema, "_C::rms_norm_dynamic_per_token_quant"
    )
    if names != _VLLM_NATIVE_RMSNORM_DYNAMIC_QUANT_ARGUMENTS:
        raise HcuAiterRuntimeError(
            "vLLM native RMSNorm dynamic-quant fallback exposes unsupported "
            f"arguments {names!r}"
        )
    if not callable(operation):
        raise HcuAiterRuntimeError(
            "vLLM native RMSNorm dynamic-quant fallback is not callable"
        )

    scale = torch.empty(x.shape[0], 1, dtype=torch.float32, device=x.device)
    output = torch.empty(x.shape, dtype=quant_dtype, device=x.device)
    residual_out = residual.clone() if residual is not None else None
    operation(output, x, weight, scale, epsilon, None, residual_out)
    return output, scale, residual_out


def rmsnorm_dynamic_quant_impl(
    aiter_operation: Callable[..., None],
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the target AITER ABI, with a native FP8 legacy fallback."""

    op_name = "rmsnorm2d_fwd_with_dynamicquant"
    abi = _aiter_rmsnorm_dynamic_quant_abi(op_name)
    if abi == "legacy-default" and quant_dtype != torch.int8:
        output, scale, _ = _vllm_native_rmsnorm_dynamic_quant(
            x, weight, epsilon, quant_dtype
        )
        return output, scale

    scale = torch.empty(x.shape[0], 1, dtype=torch.float32, device=x.device)
    output = torch.empty(x.shape, dtype=quant_dtype, device=x.device)
    arguments = (output, x, scale, weight, epsilon)
    if abi == "model-sensitive":
        aiter_operation(
            *arguments,
            use_model_sensitive_rmsnorm=0,
        )
    else:
        aiter_operation(*arguments)
    return output, scale


def rmsnorm_add_dynamic_quant_impl(
    aiter_operation: Callable[..., None],
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call the fused-add target ABI without mutating either caller input."""

    op_name = "rmsnorm2d_fwd_with_add_dynamicquant"
    abi = _aiter_rmsnorm_dynamic_quant_abi(op_name)
    if abi == "legacy-default" and quant_dtype != torch.int8:
        output, scale, residual_out = _vllm_native_rmsnorm_dynamic_quant(
            x, weight, epsilon, quant_dtype, residual
        )
        if residual_out is None:  # pragma: no cover - internal invariant
            raise HcuAiterRuntimeError(
                "vLLM native fused-add RMSNorm fallback lost residual output"
            )
        return output, residual_out, scale

    scale = torch.empty(x.shape[0], 1, dtype=torch.float32, device=x.device)
    output = torch.empty(x.shape, dtype=quant_dtype, device=x.device)
    residual_out = torch.empty_like(x)
    arguments = (
        output,
        x,
        residual,
        residual_out,
        scale,
        weight,
        epsilon,
    )
    if abi == "model-sensitive":
        aiter_operation(
            *arguments,
            use_model_sensitive_rmsnorm=0,
        )
    else:
        aiter_operation(*arguments)
    return output, residual_out, scale


def is_aiter_found_and_supported(
    original: Callable[[], bool],
    current_platform: object,
    is_aiter_found: bool,
) -> bool:
    """Extend the upstream ROCm capability probe to HCU gfx93x devices."""

    if bool(current_platform.is_rocm()) and bool(is_aiter_found):
        try:
            from vllm_hcu.platforms.hcu import on_gfx93x
        except Exception as exc:
            raise HcuAiterRuntimeError(
                "HCU AITER capability check could not load on_gfx93x"
            ) from exc
        if on_gfx93x():
            return True
    return bool(original())


def _activation_name(activation_method: int) -> str:
    try:
        return {
            0: "silu",
            1: "gelu",
            2: "swiglu",
            3: "gelu_tanh",
        }[activation_method]
    except KeyError as exc:
        raise HcuAiterRuntimeError(
            f"HCU AITER MoE received unsupported activation_method={activation_method}"
        ) from exc


def _aiter_asm_expert_mask_contract(
    expert_mask: torch.Tensor | None,
    local_num_experts: int,
) -> tuple[int, torch.Tensor | None]:
    """Validate vLLM's AITER EP mask before it reaches the ASM sorter.

    vLLM uses two different EP tensors at the MoE layer boundary:

    * a global-to-local map of shape ``[global_num_experts]`` containing local
      expert ids and ``-1``;
    * an AITER mask of shape
      ``[global_num_experts + fused_shared_experts + 1]`` whose final element
      is a sentinel.

    The proprietary ASM sorter accepts only the latter.  Passing the former
    can make global expert ids index local weights and cause a device VMFault,
    so reject the ambiguous layout before launching a kernel.
    """

    global_num_experts = _AITER_MOE_GLOBAL_NUM_EXPERTS.get()
    if expert_mask is None:
        if (
            global_num_experts is not None
            and global_num_experts != local_num_experts
        ):
            raise HcuAiterRuntimeError(
                "AITER W16A16 ASM MoE requires an expert mask for EP: "
                f"global_num_experts={global_num_experts}, "
                f"local_num_experts={local_num_experts}"
            )
        return global_num_experts or local_num_experts, None

    if expert_mask.dim() != 1 or expert_mask.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise HcuAiterRuntimeError(
            "unexpected AITER expert mask layout: "
            f"shape={tuple(expert_mask.shape)}, dtype={expert_mask.dtype}"
        )
    if global_num_experts is None:
        raise HcuAiterRuntimeError(
            "AITER W16A16 ASM MoE received an EP tensor without the global "
            "expert count from FusedMoEConfig"
        )

    # AITER's mask contains all routed experts followed by any fused shared
    # experts and a final sentinel.  A plain vLLM global-to-local map has
    # exactly global_num_experts entries and must not reach the ASM sorter.
    if expert_mask.numel() < global_num_experts + 1:
        raise HcuAiterRuntimeError(
            "AITER W16A16 ASM MoE expected a 0/1 expert mask with a trailing "
            "sentinel, but received a global-to-local expert map or truncated "
            f"mask: shape={tuple(expert_mask.shape)}, "
            f"global_num_experts={global_num_experts}, "
            f"local_num_experts={local_num_experts}"
        )
    return global_num_experts, expert_mask


def _aiter_mask_to_vllm_expert_map(
    expert_mask: torch.Tensor | None,
    global_num_experts: int,
) -> torch.Tensor | None:
    if expert_mask is None:
        return None
    routed_mask = expert_mask[:global_num_experts].to(torch.bool)
    local_ids = torch.cumsum(
        routed_mask.to(torch.int32),
        dim=0,
        dtype=torch.int32,
    ) - 1
    return torch.where(
        routed_mask,
        local_ids,
        torch.full_like(local_ids, -1),
    )


def fused_moe_impl(
    original: Callable[..., torch.Tensor],
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: torch.Tensor | None = None,
    activation_method: int = 0,
    quant_method: int = 0,
    doweight_stage1: bool = False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    num_local_tokens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    hidden_pad: int = 0,
    intermediate_pad: int = 0,
    gate_mode: str = "",
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    moe_sorting_dispatch_policy: int = 0,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    """Route unquantized HCU MoE through AITER, otherwise preserve upstream."""

    from aiter import QuantType

    use_w16a16_aiter = (
        is_aiter_moe_requested()
        and QuantType(quant_method) == QuantType.No
        and w1_scale is None
        and w2_scale is None
        and a1_scale is None
        and a2_scale is None
    )
    if not use_w16a16_aiter:
        if activation_method != 3:
            return original(
                hidden_states,
                w1,
                w2,
                topk_weight,
                topk_ids,
                expert_mask,
                activation_method,
                quant_method,
                doweight_stage1,
                w1_scale,
                w2_scale,
                a1_scale,
                a2_scale,
                num_local_tokens,
                output_dtype,
                hidden_pad,
                intermediate_pad,
                gate_mode,
                bias1,
                bias2,
                moe_sorting_dispatch_policy,
                swiglu_limit,
            )
        # ActivationType in older HCU AITER builds has no numeric GELU-tanh
        # member, while fused_moe accepts its stable string spelling.
        try:
            from aiter.fused_moe import fused_moe
        except Exception as exc:
            raise HcuAiterRuntimeError(
                "HCU AITER GELU-tanh MoE was selected, but "
                "aiter.fused_moe.fused_moe is unavailable"
            ) from exc
        unsupported_arguments = {
            "num_local_tokens": (num_local_tokens, None),
            "output_dtype": (output_dtype, None),
            "hidden_pad": (hidden_pad, 0),
            "intermediate_pad": (intermediate_pad, 0),
            "gate_mode": (gate_mode, ""),
            "bias1": (bias1, None),
            "bias2": (bias2, None),
            "moe_sorting_dispatch_policy": (moe_sorting_dispatch_policy, 0),
            "swiglu_limit": (swiglu_limit, 0.0),
        }
        for name, (value, default) in unsupported_arguments.items():
            if value != default:
                raise HcuAiterRuntimeError(
                    "HCU AITER fused_moe ABI does not "
                    f"support non-default {name}={value!r}"
                )
        return fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            expert_mask,
            "gelu_tanh",
            QuantType(quant_method),
            doweight_stage1,
            w1_scale,
            w2_scale,
            a1_scale,
            a2_scale,
        )

    from vllm_hcu.platforms import envs as henvs

    activation = _activation_name(activation_method)
    use_shuffle = bool(henvs.VLLM_HCU_USE_AITER_MOE_SHUFFLE)
    global_num_experts, expert_mask = _aiter_asm_expert_mask_contract(
        expert_mask,
        int(w1.shape[0]),
    )
    unsupported = {
        "doweight_stage1": (doweight_stage1, False),
        "num_local_tokens": (num_local_tokens, None),
        "hidden_pad": (hidden_pad, 0),
        "intermediate_pad": (intermediate_pad, 0),
        "bias1": (bias1, None),
        "bias2": (bias2, None),
    }
    for name, (value, default) in unsupported.items():
        if value != default:
            raise HcuAiterRuntimeError(
                "HCU AITER W16A16 MoE does not support "
                f"non-default {name}={value!r}"
            )
    if gate_mode:
        raise HcuAiterRuntimeError(
            "HCU AITER W16A16 MoE has no gate_mode ABI; received "
            f"gate_mode={gate_mode!r}"
        )
    if moe_sorting_dispatch_policy:
        raise HcuAiterRuntimeError(
            "HCU AITER W16A16 MoE has no sorting-dispatch ABI; "
            "received moe_sorting_dispatch_policy="
            f"{moe_sorting_dispatch_policy}"
        )

    # HCU AITER uses w1=[E, 2N, K] and w2=[E, K, N]. Its N2
    # configuration argument is GEMM2's output dimension, i.e. w2.shape[1].
    if (
        w1.dim() != 3
        or w2.dim() != 3
        or int(w1.shape[0]) != int(w2.shape[0])
        or int(w1.shape[1]) != 2 * int(w2.shape[2])
        or int(w1.shape[2]) != int(w2.shape[1])
    ):
        raise ValueError(
            f"unexpected MoE weight layout: w1.shape={tuple(w1.shape)}, "
            f"w2.shape={tuple(w2.shape)}"
        )

    problem = AiterMoeProblem(
        M=int(hidden_states.shape[0]),
        E=global_num_experts,
        N1=int(w1.shape[1]),
        N2=int(w2.shape[1]),
        K=int(w1.shape[2]),
        top_k=int(topk_ids.shape[1]),
        block_size=0,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
        quant_type="w16a16",
        activation=activation,
        use_shuffle=use_shuffle,
    )
    aiter_config = select_aiter_moe_config(problem, cache_owner=w1)
    native_expert_map = _aiter_mask_to_vllm_expert_map(
        expert_mask,
        global_num_experts,
    )
    if aiter_config is None:
        if swiglu_limit:
            raise HcuAiterRuntimeError(
                "vLLM Triton W16A16 fallback does not support swiglu_limit"
            )
        if output_dtype not in (None, hidden_states.dtype):
            raise HcuAiterRuntimeError(
                "vLLM Triton W16A16 fallback cannot override output_dtype"
            )
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_experts_impl,
        )

        return fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            activation=activation,
            apply_router_weight_on_input=False,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            global_num_experts=global_num_experts,
            expert_map=native_expert_map,
        )

    prepared_w1, prepared_w2 = prepare_aiter_moe_weights(
        w1,
        w2,
        aiter_config,
        cache_owner=w1,
    )
    solution = getattr(aiter_config, "solution_type", None)
    solution = getattr(solution, "value", solution)
    aiter_expert_map = (
        expert_mask
        if str(solution).rsplit(".", 1)[-1].upper() == "ASM"
        else native_expert_map
    )
    return execute_aiter_moe(
        aiter_config,
        hidden_states=hidden_states,
        w1=prepared_w1,
        w2=prepared_w2,
        topk_weights=topk_weight,
        topk_ids=topk_ids,
        inplace=False,
        activation=activation,
        global_num_experts=global_num_experts,
        expert_map=aiter_expert_map,
        use_weight_shuffle=bool(
            getattr(aiter_config, "need_shuffle", False)
        ),
        output_dtype=output_dtype or hidden_states.dtype,
        gemm1_limit=swiglu_limit or None,
    )


def _topk_supports_extended_arguments(topk_softmax: Callable[..., Any]) -> bool:
    try:
        return "num_shared_experts" in inspect.signature(topk_softmax).parameters
    except (TypeError, ValueError):
        schema = getattr(
            getattr(getattr(torch.ops, "aiter", None), "topk_softmax", None),
            "default",
            None,
        )
        return "num_shared_experts" in str(getattr(schema, "_schema", ""))


def topk_softmax_impl(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    num_shared_experts: int = 0,
    shared_expert_scoring_func: str = "",
) -> None:
    """Call the installed AITER ABI without assuming its argument count."""

    try:
        from aiter import topk_softmax
    except Exception as exc:
        raise HcuAiterRuntimeError(
            "HCU AITER top-k was selected, but aiter.topk_softmax is unavailable"
        ) from exc

    common = (
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )
    if _topk_supports_extended_arguments(topk_softmax):
        topk_softmax(*common, num_shared_experts, shared_expert_scoring_func)
    else:
        topk_softmax(*common)


def get_gelu_tanh_activation_type() -> object | None:
    """Return AITER's GELU-tanh enum when the installed build exposes it."""

    try:
        from aiter import ActivationType
    except ImportError:
        return None
    return getattr(ActivationType, "GeluTanh", None)


def get_aiter_activation_type(
    original: Callable[[str], object | None], activation_str: str
) -> object | None:
    """Add the two stable GELU-tanh spellings and preserve all upstream ones."""

    if isinstance(activation_str, str) and activation_str.strip().lower() in {
        "gelu_tanh",
        "gelu_pytorch_tanh",
    }:
        return get_gelu_tanh_activation_type()
    return original(activation_str)


__all__ = [
    "HcuAiterRuntimeError",
    "aiter_asm_boltops_fp8_quant_context",
    "aiter_asm_boltops_int8_quant_context",
    "aiter_gate_mode_kwargs",
    "fused_moe_impl",
    "get_aiter_activation_type",
    "get_gelu_tanh_activation_type",
    "is_aiter_found_and_supported",
    "rmsnorm_add_dynamic_quant_impl",
    "rmsnorm_dynamic_quant_impl",
    "triton_rope_and_cache_impl",
    "topk_softmax_impl",
]
