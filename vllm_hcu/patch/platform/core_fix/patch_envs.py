# SPDX-License-Identifier: Apache-2.0
"""Runtime migration of the two HCU environment handling fragments."""

from __future__ import annotations

import functools
import os
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    apply_once,
    load_exact_module,
    require_callable,
    require_positional_signature,
)

TARGET_MODULE = "vllm.envs"
PATCH_ID = "platform.core_fix.envs"
TARGETS = (
    "vllm.envs.validate_environ",
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

    original_validate = require_callable(
        envs, "validate_environ", "vllm.envs.validate_environ"
    )
    require_positional_signature(
        original_validate,
        "vllm.envs.validate_environ",
        ("hard_fail",),
    )
    env_logger = getattr(envs, "logger", None)
    if not callable(getattr(env_logger, "warning", None)):
        raise PatchCompatibilityError(
            "required HCU patch target vllm.envs.logger.warning is missing"
        )

    @functools.wraps(original_validate)
    def hcu_validate_environ(hard_fail: bool) -> None:
        for name in os.environ:
            if not name.startswith("VLLM_"):
                continue
            if name in environment_variables or name.startswith("VLLM_HCU_"):
                continue
            if hard_fail:
                raise ValueError(
                    f"Unknown vLLM environment variable detected: {name}"
                )
            env_logger.warning("Unknown vLLM environment variable detected: %s", name)

    environment_variables["VLLM_ROCM_USE_AITER_MOE"] = lambda: os.getenv(
        "VLLM_ROCM_USE_AITER_MOE", "False"
    ).lower() in ("true", "1")
    # vLLM can cache module-level environment reads.  Clear stale values so a
    # pre-plugin lookup cannot preserve the upstream True default.
    env_getattr = getattr(envs, "__getattr__", None)
    cache_clear = getattr(env_getattr, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()

    setattr(envs, "_vllm_hcu_original_validate_environ", original_validate)
    setattr(envs, "_vllm_hcu_original_aiter_moe_getter", old_aiter_getter)
    setattr(envs, "validate_environ", hcu_validate_environ)
    setattr(envs, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    """Allow ``VLLM_HCU_*`` variables and disable AITER MoE by default."""

    envs = load_exact_module(TARGET_MODULE, module)

    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=envs,
        marker=_MARKER,
        callback=lambda: apply_to_module(envs),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
