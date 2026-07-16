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
from typing import Any

import torch


class HcuAiterRuntimeError(RuntimeError):
    """An explicitly requested HCU AITER path cannot be provided."""


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
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select HCU's W16A16 ASM path, otherwise preserve upstream exactly."""

    import vllm.envs as envs
    from aiter import QuantType

    use_w16a16_asm = (
        bool(envs.VLLM_ROCM_USE_AITER)
        and bool(envs.VLLM_ROCM_USE_AITER_MOE)
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
                bias1,
                bias2,
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
            num_local_tokens=num_local_tokens,
            dtype=output_dtype,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            bias1=bias1,
            bias2=bias2,
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
    "topk_softmax_impl",
]
