# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Make the target vLLM mHC backend honor the HCU backend switch."""

from __future__ import annotations

from types import ModuleType

from ._common import PatchCompatibilityError, load_exact_module

TARGET_MODULE = "vllm.model_executor.layers.mhc"
PATCH_ID = "worker.core_fix.mhc_backend_switch"
_MARKER = "_vllm_hcu_mhc_backend_switch_applied"


def apply_to_module(module: ModuleType) -> bool:
    mhc = load_exact_module(TARGET_MODULE, module)
    if getattr(mhc, _MARKER, False):
        if not hasattr(mhc, "_vllm_hcu_original_has_aiter_mhc"):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_MODULE} is stale"
            )
        return False

    has_aiter_mhc = getattr(mhc, "HAS_AITER_MHC", None)
    if not isinstance(has_aiter_mhc, bool):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.HAS_AITER_MHC is missing"
        )

    import vllm_hcu.platforms.envs as henvs

    # The target implementation otherwise always prefers AITER whenever it is
    # installed, making the plugin's documented switch ineffective.  Preserve
    # target capability detection and only mask it when the user opts out; the
    # existing forward_hip methods then select TileLang, followed by native.
    mhc._vllm_hcu_original_has_aiter_mhc = has_aiter_mhc
    mhc.HAS_AITER_MHC = bool(has_aiter_mhc and henvs.VLLM_HCU_USE_AITER_MHC)
    setattr(mhc, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
