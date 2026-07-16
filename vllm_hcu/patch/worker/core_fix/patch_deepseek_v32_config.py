# SPDX-License-Identifier: Apache-2.0
"""Audit and retire the broken DeepSeek-V3.2 verifier source patch.

The legacy patch changed the official ``is_v32`` verifier to include
``not VLLM_HCU_DISABLE_DSA`` and then asserted that value.  Enabling the HCU
flag therefore made every valid V3.2 config fail verification.  The official
vLLM verifier is intentionally preserved.

DSA selection belongs to the HCU model implementation.  The behavior guarded
by :func:`is_hcu_dsa_enabled` is already used by
``vllm_hcu.models.deepseek_v2.DeepseekV2MLAAttention`` and
``vllm_hcu.models.deepseek_mtp.DeepSeekMultiTokenPredictorLayer``.  This module
only checks that the official verifier API is still the audited v0.21 shape;
it never replaces that method.
"""

from __future__ import annotations

from types import ModuleType
from weakref import WeakKeyDictionary

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_class,
    require_classmethod,
)

TARGET_MODULE = "vllm.model_executor.models.config"
PATCH_ID = "worker.core_fix.deepseek_v32.retire_broken_verifier_patch"
TARGET_SYMBOL = f"{TARGET_MODULE}.DeepseekV32ForCausalLM.verify_and_update_config"

# A weak identity map gives the no-op audit callback real idempotency without
# mutating the official verifier class or module.
_AUDITED_MODULES: WeakKeyDictionary[ModuleType, classmethod] = WeakKeyDictionary()


def is_hcu_dsa_enabled(hf_config: object, *, disable_dsa: bool) -> bool:
    """Return the HCU model-side DSA decision kept out of the verifier."""

    return hasattr(hf_config, "index_topk") and not disable_dsa


def apply_to_module(module: ModuleType) -> bool:
    """Validate the official verifier and record its audited retirement."""

    models_config = load_exact_module(TARGET_MODULE, module)
    config_class = require_class(
        models_config,
        "DeepseekV32ForCausalLM",
        f"{TARGET_MODULE}.DeepseekV32ForCausalLM",
    )
    descriptor, _ = require_classmethod(
        config_class,
        "verify_and_update_config",
        TARGET_SYMBOL,
        positional=("cls", "vllm_config"),
    )

    audited = _AUDITED_MODULES.get(models_config)
    if audited is not None:
        if audited is not descriptor:
            raise PatchCompatibilityError(
                f"audited target {TARGET_SYMBOL} changed after validation; "
                "restart the process"
            )
        return False

    _AUDITED_MODULES[models_config] = descriptor
    return True


def apply(module: ModuleType | None = None) -> bool:
    """Explicit direct entry point; no source or method is modified."""

    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "apply",
    "apply_to_module",
    "is_hcu_dsa_enabled",
]
