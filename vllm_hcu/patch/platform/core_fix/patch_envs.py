# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Runtime migration of the HCU environment handling fragments.

vLLM main moved the unknown-``VLLM_*`` check from ``vllm.envs`` to the
platform extension boundary (``vllm.platforms.interface.Platform.validate_environ``,
upstream #48599).  The HCU adapter therefore keeps only the environment
registry override here and publishes the ``VLLM_HCU_*`` namespace through
``vllm_hcu.platforms.hcu.HCUPlatform.validate_environ``.
"""

from __future__ import annotations

import os
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    apply_once,
    load_exact_module,
)

TARGET_MODULE = "vllm.envs"
PATCH_ID = "platform.core_fix.envs"
TARGETS = (
    "vllm.envs.environment_variables.VLLM_ROCM_USE_AITER_MOE",
)
_MARKER = "_vllm_hcu_envs_patch_applied"


def apply_to_module(module: ModuleType) -> bool:
    """Apply to an exact module from the import coordinator, without reporting."""

    envs = load_exact_module(TARGET_MODULE, module)
    if getattr(envs, _MARKER, False):
        return False

    environment_variables = getattr(envs, "environment_variables", None)
    if not isinstance(environment_variables, dict):
        raise PatchCompatibilityError(
            "required HCU patch target vllm.envs.environment_variables must be a dict"
        )
    old_aiter_getter = environment_variables.get("VLLM_ROCM_USE_AITER_MOE")
    if not callable(old_aiter_getter):
        raise PatchCompatibilityError(
            "required HCU patch target "
            "vllm.envs.environment_variables['VLLM_ROCM_USE_AITER_MOE'] is missing"
        )

    environment_variables["VLLM_ROCM_USE_AITER_MOE"] = lambda: os.getenv(
        "VLLM_ROCM_USE_AITER_MOE", "False"
    ).lower() in ("true", "1")
    # vLLM can cache module-level environment reads.  Clear stale values so a
    # pre-plugin lookup cannot preserve the upstream True default.
    env_getattr = getattr(envs, "__getattr__", None)
    cache_clear = getattr(env_getattr, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()

    setattr(envs, "_vllm_hcu_original_aiter_moe_getter", old_aiter_getter)
    setattr(envs, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    """Disable AITER MoE by default until the user opts in."""

    envs = load_exact_module(TARGET_MODULE, module)

    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=envs,
        marker=_MARKER,
        callback=lambda: apply_to_module(envs),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
