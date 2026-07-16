# SPDX-License-Identifier: Apache-2.0
"""Bind the ROCm fused all-reduce pass to HCU's communicator class."""

from __future__ import annotations

import importlib
from types import ModuleType

from ._common import PatchCompatibilityError, load_exact_module, require_class

TARGET_MODULE = "vllm.compilation.passes.fusion.allreduce_rms_fusion"
CUSTOM_ALLREDUCE_MODULE = (
    "vllm.distributed.device_communicators.custom_all_reduce"
)
PATCH_ID = "platform.framework_opt.allreduce_rms_fusion"
TARGETS = (f"{TARGET_MODULE}.CustomAllreduce",)
_MARKER = "_vllm_hcu_custom_allreduce_identity_applied"


def _load_canonical_custom_allreduce() -> type:
    """Resolve the coordinator-owned canonical communicator identity.

    The fusion module imports ``CustomAllreduce`` from the canonical vLLM
    module.  Once the cold module exchange is armed that canonical import is
    already the HCU implementation, even though the class' ``__module__``
    truthfully remains under ``vllm_hcu``.  Identity, not the implementation
    module string, is therefore the compatibility contract.
    """

    canonical = importlib.import_module(CUSTOM_ALLREDUCE_MODULE)
    return require_class(
        canonical,
        "CustomAllreduce",
        f"{CUSTOM_ALLREDUCE_MODULE}.CustomAllreduce",
    )


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    canonical = _load_canonical_custom_allreduce()
    if getattr(target, _MARKER, False):
        if target.CustomAllreduce is not canonical:
            raise PatchCompatibilityError(
                "HCU all-reduce fusion marker is stale; restart the process"
            )
        return False

    current = require_class(target, "CustomAllreduce", TARGETS[0])
    if current is not canonical:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} is not the class exposed "
            f"by canonical alias {CUSTOM_ALLREDUCE_MODULE!r}"
        )
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "CUSTOM_ALLREDUCE_MODULE",
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
