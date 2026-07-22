# SPDX-License-Identifier: Apache-2.0
"""Guard target-owned compressed-tensors Channel-FP8 MoE routing."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8"
)
PATCH_ID = "worker.op_opt.compressed_tensors.moe_w8a8_fp8"
TARGETS = (
    f"{TARGET_MODULE}.CompressedTensorsW8A8Fp8MoEMethod.__init__",
)
_CLASS_MARKER = "_vllm_hcu_moe_w8a8_fp8_applied"
_WRAPPER_MARKER = "_vllm_hcu_moe_w8a8_fp8_wrapper"
_REQUIRED_BACKEND = "triton"
_REQUIRED_SELECTED_BACKEND = "TRITON"


def _aiter_moe_state() -> tuple[bool, bool]:
    try:
        import vllm.envs as target_envs
        from vllm_hcu.platforms import envs as henvs

        return (
            bool(target_envs.VLLM_ROCM_USE_AITER_MOE),
            bool(henvs.VLLM_HCU_USE_AITER_W8A8_FP8_MOE),
        )
    except (AttributeError, ImportError) as exc:
        raise PatchCompatibilityError(
            "required target/HCU FP8-MoE routing flags are unavailable"
        ) from exc


def _selected_backend_name(method) -> str | None:
    backend = getattr(method, "fp8_backend", None)
    value = getattr(backend, "value", backend)
    return value if isinstance(value, str) else None


def _is_channel_token_route(fp8_moe_module, weight_quant, input_quant) -> bool:
    strategy = getattr(fp8_moe_module, "QuantizationStrategy", None)
    try:
        return bool(
            weight_quant.strategy == strategy.CHANNEL
            and input_quant.strategy == strategy.TOKEN
        )
    except AttributeError as exc:
        raise PatchCompatibilityError(
            "vLLM compressed-tensors FP8 MoE quantization strategy contract "
            "is unavailable"
        ) from exc


def _require_target_triton_policy(moe) -> None:
    target_aiter, hcu_aiter = _aiter_moe_state()
    if target_aiter or hcu_aiter:
        raise RuntimeError(
            "Channel-FP8 MoE requires VLLM_ROCM_USE_AITER_MOE=0 and "
            "VLLM_HCU_USE_AITER_W8A8_FP8_MOE=0 before model construction"
        )

    requested_backend = getattr(moe, "moe_backend", None)
    if requested_backend != _REQUIRED_BACKEND:
        raise RuntimeError(
            "Channel-FP8 MoE requires the explicit vLLM v0.25 target route "
            "--moe-backend triton"
        )


def apply_to_module(module: ModuleType) -> bool:
    fp8_moe_module = load_exact_module(TARGET_MODULE, module)
    method_class = require_class(
        fp8_moe_module,
        "CompressedTensorsW8A8Fp8MoEMethod",
        f"{TARGET_MODULE}.CompressedTensorsW8A8Fp8MoEMethod",
    )
    wrapped = ((method_class, "__init__", TARGETS[0], _WRAPPER_MARKER),)
    if already_applied(method_class, _CLASS_MARKER, wrapped):
        return False

    original_init = require_callable(method_class, "__init__", TARGETS[0])
    require_exact_signature(
        original_init,
        TARGETS[0],
        positional=("self", "weight_quant", "input_quant", "moe", "layer_name"),
        defaults={"layer_name": None},
    )

    @functools.wraps(original_init)
    def hcu_init(self, weight_quant, input_quant, moe, layer_name=None):
        channel_token = _is_channel_token_route(
            fp8_moe_module, weight_quant, input_quant
        )
        if channel_token:
            _require_target_triton_policy(moe)
        original_init(self, weight_quant, input_quant, moe, layer_name)
        if channel_token:
            selected_backend = _selected_backend_name(self)
            if selected_backend != _REQUIRED_SELECTED_BACKEND:
                raise RuntimeError(
                    "vLLM v0.25 did not select the required target TRITON "
                    f"Channel-FP8 MoE backend (selected={selected_backend!r})"
                )

    setattr(hcu_init, _WRAPPER_MARKER, True)
    setattr(method_class, "_vllm_hcu_original_init", original_init)
    setattr(method_class, "__init__", hcu_init)
    setattr(method_class, _CLASS_MARKER, True)
    setattr(method_class, "_vllm_hcu_fp8_moe_owner", "target-triton")
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
