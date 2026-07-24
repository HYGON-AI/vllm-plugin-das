# SPDX-License-Identifier: Apache-2.0
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


class HcuAiterRuntimeError(RuntimeError):
    """An explicitly requested HCU AITER path cannot be provided."""


_EXPLICIT_AITER_MOE: ContextVar[bool] = ContextVar(
    "vllm_hcu_explicit_aiter_moe", default=False
)


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
    token = _EXPLICIT_AITER_MOE.set(
        getattr(moe_config, "moe_backend", None) == "aiter"
    )
    try:
        yield
    finally:
        _EXPLICIT_AITER_MOE.reset(token)


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


@functools.cache
def get_w16a16_moe_solution_id(
    M: int,
    E: int,
    N1: int,
    N2: int,
    K: int,
    top_k: int,
    dtype: torch.dtype,
    activation: str,
    use_shuffle: int,
) -> str:
    try:
        from aiter.moe import MoeQuantType, MoeSolutionType, get_aiter_moe_config
    except Exception as exc:
        raise HcuAiterRuntimeError(
            "HCU AITER W16A16 ASM MoE configuration was requested, but "
            "aiter.moe.get_aiter_moe_config is unavailable"
        ) from exc

    status, moe_config = get_aiter_moe_config(
        M=M,
        E=E,
        N1=N1,
        N2=N2,
        K=K,
        top_k=top_k,
        block_size=0,
        dtype=dtype,
        quant_type=MoeQuantType.W16A16,
        activation=activation,
        use_shuffle=use_shuffle,
    )
    if not status or moe_config.solution_type != MoeSolutionType.ASM:
        raise HcuAiterRuntimeError(
            "AITER W16A16 MoE did not find an ASM solution for "
            f"M={M}, E={E}, N1={N1}, N2={N2}, K={K}, top_k={top_k}, "
            f"dtype={dtype}, activation={activation}, use_shuffle={use_shuffle}"
        )

    config = moe_config.config or {}
    try:
        return f"{config['SOL_ID1']}+{config['SOL_ID2']}"
    except KeyError as exc:
        raise HcuAiterRuntimeError(
            "AITER W16A16 ASM configuration is missing SOL_ID1/SOL_ID2: "
            f"{config}"
        ) from exc


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
    """Select HCU's W16A16 ASM path, otherwise preserve upstream exactly."""

    from aiter import QuantType

    use_w16a16_asm = (
        is_aiter_moe_requested()
        and QuantType(quant_method) == QuantType.No
        and w1_scale is None
        and w2_scale is None
        and a1_scale is None
        and a2_scale is None
    )
    if not use_w16a16_asm:
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
        parameters = inspect.signature(fused_moe).parameters
        optional_arguments = {
            "num_local_tokens": (num_local_tokens, None),
            "dtype": (output_dtype, None),
            "hidden_pad": (hidden_pad, 0),
            "intermediate_pad": (intermediate_pad, 0),
            "gate_mode": (gate_mode, ""),
            "bias1": (bias1, None),
            "bias2": (bias2, None),
            "moe_sorting_dispatch_policy": (moe_sorting_dispatch_policy, 0),
            "swiglu_limit": (swiglu_limit, 0.0),
        }
        supported_arguments: dict[str, object] = {}
        for name, (value, default) in optional_arguments.items():
            if name in parameters:
                supported_arguments[name] = value
            elif value != default:
                raise HcuAiterRuntimeError(
                    "the installed proprietary AITER fused_moe ABI does not "
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
            **supported_arguments,
        )

    try:
        from aiter.fused_moe_asm_wna16 import fused_experts_asm_impl
    except Exception as exc:
        raise HcuAiterRuntimeError(
            "VLLM_ROCM_USE_AITER_MOE is enabled for W16A16, but "
            "aiter.fused_moe_asm_wna16.fused_experts_asm_impl is unavailable"
        ) from exc

    from vllm_hcu.platforms import envs as henvs

    activation = _activation_name(activation_method)
    use_shuffle = int(bool(henvs.VLLM_HCU_USE_AITER_W16A16_MOE_SHUFFLE))
    global_num_experts = (
        int(expert_mask.numel()) if expert_mask is not None else int(w1.shape[0])
    )
    kwargs: dict[str, Any] = {
        "activation": activation,
        "global_num_experts": global_num_experts,
        "expert_map": expert_mask,
        "use_shuffle": use_shuffle,
    }
    if gate_mode:
        raise HcuAiterRuntimeError(
            "HCU W16A16 ASM MoE cannot represent vLLM v0.25.1 "
            f"gate_mode={gate_mode!r}"
        )
    if moe_sorting_dispatch_policy:
        raise HcuAiterRuntimeError(
            "HCU W16A16 ASM MoE cannot represent vLLM v0.25.1 "
            "moe_sorting_dispatch_policy="
            f"{moe_sorting_dispatch_policy}"
        )
    if swiglu_limit:
        # The proprietary HCU ASM ABI names vLLM's SwiGLU limit gemm1_limit.
        kwargs["gemm1_limit"] = swiglu_limit
    if bool(henvs.VLLM_HCU_USE_AITER_MOE_CONFIG):
        kwargs["solution_id"] = get_w16a16_moe_solution_id(
            M=int(hidden_states.shape[0]),
            E=int(w1.shape[0]),
            N1=int(w1.shape[1]),
            N2=int(w2.shape[1]),
            K=int(w1.shape[2]),
            top_k=int(topk_ids.shape[1]),
            dtype=hidden_states.dtype,
            activation=activation,
            use_shuffle=use_shuffle,
        )

    return fused_experts_asm_impl(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        output_dtype or hidden_states.dtype,
        **kwargs,
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
    "fused_moe_impl",
    "get_aiter_activation_type",
    "get_gelu_tanh_activation_type",
    "get_w16a16_moe_solution_id",
    "is_aiter_found_and_supported",
    "rmsnorm_add_dynamic_quant_impl",
    "rmsnorm_dynamic_quant_impl",
    "topk_softmax_impl",
]
