# SPDX-License-Identifier: Apache-2.0
"""Select the Qwen3.5 Mamba SSM cache dtype for HCU custom kernels."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_classmethod,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.models.qwen3_5"
PATCH_ID = "worker.core_fix.qwen3_5.mamba_state_dtype"
TARGET_SYMBOL = (
    f"{TARGET_MODULE}.Qwen3_5ForConditionalGeneration."
    "get_mamba_state_dtype_from_config"
)
_CLASS_MARKER = "_vllm_hcu_qwen3_5_mamba_dtype_applied"
_WRAPPER_MARKER = "_vllm_hcu_qwen3_5_mamba_dtype_wrapper"


def _auto_dtype_enabled() -> bool:
    try:
        from vllm_hcu.platforms import envs as henvs

        return bool(
            henvs.VLLM_HCU_MAMBA_SSM_CACHE_DTYPE
            and henvs.VLLM_HCU_USE_CUSTOM_OPS
        )
    except (AttributeError, ImportError) as exc:
        raise PatchCompatibilityError(
            "required Qwen3.5 HCU Mamba feature flags are unavailable"
        ) from exc


def apply_to_module(module: ModuleType) -> bool:
    qwen3_5 = load_exact_module(TARGET_MODULE, module)
    model_class = require_class(
        qwen3_5,
        "Qwen3_5ForConditionalGeneration",
        f"{TARGET_MODULE}.Qwen3_5ForConditionalGeneration",
    )
    descriptor, original_function = require_classmethod(
        model_class,
        "get_mamba_state_dtype_from_config",
        TARGET_SYMBOL,
        positional=("cls", "vllm_config"),
    )
    if getattr(model_class, _CLASS_MARKER, False):
        if not getattr(descriptor.__func__, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return False

    calculator = require_class(
        qwen3_5,
        "MambaStateDtypeCalculator",
        f"{TARGET_MODULE}.MambaStateDtypeCalculator",
    )
    calculate = require_callable(
        calculator,
        "gated_delta_net_state_dtype",
        f"{TARGET_MODULE}.MambaStateDtypeCalculator.gated_delta_net_state_dtype",
    )
    require_exact_signature(
        calculate,
        f"{TARGET_MODULE}.MambaStateDtypeCalculator.gated_delta_net_state_dtype",
        positional=(
            "model_dtype",
            "mamba_cache_dtype",
            "mamba_ssm_cache_dtype",
        ),
        defaults={"mamba_ssm_cache_dtype": "auto"},
    )
    @functools.wraps(original_function)
    def hcu_get_mamba_state_dtype_from_config(cls, vllm_config):
        if not _auto_dtype_enabled():
            return original_function(cls, vllm_config)
        try:
            model_config = vllm_config.model_config
            cache_config = vllm_config.cache_config
            model_dtype = model_config.dtype
            cache_dtype = cache_config.mamba_cache_dtype
        except AttributeError as exc:
            raise PatchCompatibilityError(
                f"required HCU patch runtime input for {TARGET_SYMBOL} is incomplete"
            ) from exc
        return calculate(model_dtype, cache_dtype, "auto")

    setattr(hcu_get_mamba_state_dtype_from_config, _WRAPPER_MARKER, True)
    setattr(
        model_class,
        "_vllm_hcu_original_get_mamba_state_dtype_from_config",
        descriptor,
    )
    setattr(
        model_class,
        "get_mamba_state_dtype_from_config",
        classmethod(hcu_get_mamba_state_dtype_from_config),
    )
    setattr(model_class, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
