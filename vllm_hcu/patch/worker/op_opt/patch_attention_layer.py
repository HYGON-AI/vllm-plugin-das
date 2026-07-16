# SPDX-License-Identifier: Apache-2.0
"""Runtime migration of the v0.21 dense attention HCU fragments."""

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

TARGET_MODULE = "vllm.model_executor.layers.attention.attention"
PATCH_ID = "worker.op_opt.attention.hcu_layout_and_fused_qkv"
TARGETS = (
    f"{TARGET_MODULE}._init_kv_cache_quant",
    f"{TARGET_MODULE}.Attention.forward",
    f"{TARGET_MODULE}.FusedQkvSplitRmsNormRopeAttention",
)
_MODULE_MARKER = "_vllm_hcu_attention_runtime_applied"
_WRAPPER_MARKER = "_vllm_hcu_attention_runtime_wrapper"


def _feature_flags() -> tuple[bool, bool]:
    try:
        from vllm_hcu.platforms import envs as henvs

        return (
            bool(henvs.VLLM_HCU_USE_CUSTOM_FLASH_ATTN),
            bool(henvs.VLLM_HCU_USE_FUSED_QKV_SPLIT_RMS_ROPE_KVSTORE),
        )
    except (AttributeError, ImportError) as exc:
        raise PatchCompatibilityError("required HCU attention flags are unavailable") from exc


def apply_to_module(module: ModuleType) -> bool:
    attention = load_exact_module(TARGET_MODULE, module)
    attention_class = require_class(
        attention, "Attention", f"{TARGET_MODULE}.Attention"
    )
    wrapped = (
        (attention, "_init_kv_cache_quant", TARGETS[0], _WRAPPER_MARKER),
        (attention_class, "forward", TARGETS[1], _WRAPPER_MARKER),
    )
    if already_applied(attention, _MODULE_MARKER, wrapped):
        return False

    original_init_quant = require_callable(attention, "_init_kv_cache_quant", TARGETS[0])
    require_exact_signature(
        original_init_quant,
        TARGETS[0],
        positional=("layer", "quant_config", "prefix"),
    )
    original_forward = require_callable(attention_class, "forward", TARGETS[1])
    require_exact_signature(
        original_forward,
        TARGETS[1],
        positional=("self", "query", "key", "value", "output_shape"),
        defaults={"output_shape": None},
    )
    if "FusedQkvSplitRmsNormRopeAttention" in vars(attention):
        raise PatchCompatibilityError(
            f"required HCU-owned target {TARGETS[2]} unexpectedly already exists"
        )

    # Import only after the official module is complete.  This custom-op module
    # is then loaded exactly once by the import coordinator's latched callback.
    from vllm_hcu.model_executor.layers import attention_runtime

    @functools.wraps(original_init_quant)
    def hcu_init_kv_cache_quant(layer, quant_config, prefix):
        if getattr(layer, "kv_cache_dtype", None) == "fp8_e5m2":
            return attention_runtime.init_kv_cache_quant_e5m2(
                attention, layer, quant_config, prefix
            )
        return original_init_quant(layer, quant_config, prefix)

    @functools.wraps(original_forward)
    def hcu_forward(self, query, key, value, output_shape=None):
        custom_flash, _ = _feature_flags()
        if custom_flash or getattr(self, "kv_cache_dtype", None) == "fp8_e5m2":
            return attention_runtime.attention_forward(
                attention, self, query, key, value, output_shape
            )
        return original_forward(self, query, key, value, output_shape)

    setattr(hcu_init_kv_cache_quant, _WRAPPER_MARKER, True)
    setattr(hcu_forward, _WRAPPER_MARKER, True)
    setattr(attention, "_vllm_hcu_original_init_kv_cache_quant", original_init_quant)
    setattr(attention_class, "_vllm_hcu_original_forward", original_forward)
    setattr(attention, "_init_kv_cache_quant", hcu_init_kv_cache_quant)
    setattr(attention_class, "forward", hcu_forward)
    setattr(
        attention,
        "FusedQkvSplitRmsNormRopeAttention",
        attention_runtime.FusedQkvSplitRmsNormRopeAttention,
    )
    setattr(attention, _MODULE_MARKER, True)
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
