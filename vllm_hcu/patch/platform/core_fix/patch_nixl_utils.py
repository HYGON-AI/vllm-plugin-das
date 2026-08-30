# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Use the standard NIXL Python package for HCU runtimes."""

from __future__ import annotations

import functools
import importlib.util
from types import ModuleType

from ._common import (
    apply_once,
    load_exact_module,
    require_callable,
    require_positional_signature,
)

TARGET_MODULE = "vllm.distributed.nixl_utils"
PATCH_ID = "platform.core_fix.nixl.package_name"
TARGETS = (
    f"{TARGET_MODULE}._get_nixl_module_name",
    f"{TARGET_MODULE}.is_nixl_available",
)
_MARKER = "_vllm_hcu_nixl_package_name_applied"


def apply_to_module(module: ModuleType) -> bool:
    """Replace vLLM's ROCm package selection with HCU's ``nixl`` package."""

    nixl_utils = load_exact_module(TARGET_MODULE, module)
    if getattr(nixl_utils, _MARKER, False):
        return False

    original_module_name = require_callable(
        nixl_utils,
        "_get_nixl_module_name",
        TARGETS[0],
    )
    require_positional_signature(original_module_name, TARGETS[0], ("name",))
    original_is_available = require_callable(
        nixl_utils,
        "is_nixl_available",
        TARGETS[1],
    )
    require_positional_signature(original_is_available, TARGETS[1], ())

    @functools.wraps(original_module_name)
    def hcu_get_nixl_module_name(name: str) -> str:
        pkg = "nixl"
        if name == "nixlXferTelemetry":
            return f"{pkg}._bindings"
        return f"{pkg}._api"

    @functools.wraps(original_is_available)
    def hcu_is_nixl_available() -> bool:
        pkg = "nixl"
        return importlib.util.find_spec(pkg) is not None

    setattr(nixl_utils, "_vllm_hcu_original_get_nixl_module_name", original_module_name)
    setattr(nixl_utils, "_vllm_hcu_original_is_nixl_available", original_is_available)
    setattr(nixl_utils, "_get_nixl_module_name", hcu_get_nixl_module_name)
    setattr(nixl_utils, "is_nixl_available", hcu_is_nixl_available)
    setattr(nixl_utils, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    """Install the HCU NIXL package-name override once per process."""

    nixl_utils = load_exact_module(TARGET_MODULE, module)
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=nixl_utils,
        marker=_MARKER,
        callback=lambda: apply_to_module(nixl_utils),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
