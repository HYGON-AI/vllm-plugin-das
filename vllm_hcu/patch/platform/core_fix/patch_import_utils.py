# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Runtime migration of HCU DeepGEMM package detection."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    apply_once,
    load_exact_module,
    require_callable,
    require_positional_signature,
)

TARGET_MODULE = "vllm.utils.import_utils"
PATCH_ID = "platform.core_fix.import_utils.deep_gemm"
TARGETS = ("vllm.utils.import_utils.has_deep_gemm",)
_MARKER = "_vllm_hcu_deep_gemm_detection_applied"


def apply_to_module(module: ModuleType) -> bool:
    """Apply to an exact module from the import coordinator, without reporting."""

    import_utils = load_exact_module(TARGET_MODULE, module)
    if getattr(import_utils, _MARKER, False):
        return False

    has_module = require_callable(
        import_utils, "_has_module", "vllm.utils.import_utils._has_module"
    )
    require_positional_signature(
        has_module,
        "vllm.utils.import_utils._has_module",
        ("module_name",),
    )
    original = require_callable(
        import_utils,
        "has_deep_gemm",
        "vllm.utils.import_utils.has_deep_gemm",
    )
    require_positional_signature(
        original, "vllm.utils.import_utils.has_deep_gemm", ()
    )

    @functools.wraps(original)
    def hcu_has_deep_gemm() -> bool:
        return any(
            has_module(name)
            for name in (
                "deepgemm",
                "deep_gemm",
                "vllm.third_party.deep_gemm",
            )
        )

    setattr(import_utils, "_vllm_hcu_original_has_deep_gemm", original)
    setattr(import_utils, "has_deep_gemm", hcu_has_deep_gemm)
    setattr(import_utils, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    """Recognize ``deepgemm``, ``deep_gemm``, and vLLM's vendored copy."""

    import_utils = load_exact_module(TARGET_MODULE, module)

    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=import_utils,
        marker=_MARKER,
        callback=lambda: apply_to_module(import_utils),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
