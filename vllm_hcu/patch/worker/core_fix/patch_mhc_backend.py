# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Make the target vLLM mHC backend honor HCU runtime capabilities."""

from __future__ import annotations

from types import ModuleType

from ._common import PatchCompatibilityError, load_exact_module

TARGET_MODULE = "vllm.model_executor.layers.mhc"
PATCH_ID = "worker.core_fix.mhc_backend_switch"
_MARKER = "_vllm_hcu_mhc_backend_switch_applied"


def _tilelang_runtime_available() -> bool:
    """Require a usable TileLang runtime, not merely an installed package."""

    try:
        import tilelang  # noqa: F401
    except Exception:
        return False
    return True


def apply_to_module(module: ModuleType) -> bool:
    mhc = load_exact_module(TARGET_MODULE, module)
    if getattr(mhc, _MARKER, False):
        if not hasattr(mhc, "_vllm_hcu_original_has_aiter_mhc") or not hasattr(
            mhc, "_vllm_hcu_original_has_tilelang_mhc"
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_MODULE} is stale"
            )
        return False

    has_aiter_mhc = getattr(mhc, "HAS_AITER_MHC", None)
    if not isinstance(has_aiter_mhc, bool):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.HAS_AITER_MHC is missing"
        )
    has_tilelang_mhc = getattr(mhc, "HAS_TILELANG_MHC", None)
    if not isinstance(has_tilelang_mhc, bool):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.HAS_TILELANG_MHC is missing"
        )

    import vllm_hcu.platforms.envs as henvs

    # Preserve target capability detection while honoring the HCU AITER switch.
    # Upstream's TileLang probe only checks package discovery; the DTK image can
    # contain an installed TileLang wheel whose tvm_ffi ABI is incompatible.
    # Probe the real import once so official forward_hip methods safely fall
    # back to their native Torch/Triton implementations instead of failing on
    # the first profile run.
    mhc._vllm_hcu_original_has_aiter_mhc = has_aiter_mhc
    mhc._vllm_hcu_original_has_tilelang_mhc = has_tilelang_mhc
    mhc.HAS_AITER_MHC = bool(has_aiter_mhc and henvs.VLLM_HCU_USE_AITER_MHC)
    mhc.HAS_TILELANG_MHC = bool(
        has_tilelang_mhc and _tilelang_runtime_available()
    )
    setattr(mhc, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
