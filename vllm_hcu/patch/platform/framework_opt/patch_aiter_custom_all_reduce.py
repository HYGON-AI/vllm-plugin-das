# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Size vLLM's AITER wrapper for an explicit OpenDAS AITER AR limit."""

from __future__ import annotations

import functools
import inspect
import os
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
)

TARGET_MODULE = (
    "vllm.distributed.device_communicators.aiter_custom_all_reduce"
)
PATCH_ID = "platform.framework_opt.aiter_custom_allreduce_buffer"
TARGETS = (
    f"{TARGET_MODULE}.AiterCustomAllreduce",
    f"{TARGET_MODULE}.AiterCustomAllreduce.__init__",
    f"{TARGET_MODULE}.AiterCustomAllreduce.effective_max_size",
)
_MARKER = "_vllm_hcu_aiter_custom_allreduce_buffer_applied"
_WRAPPER = "_vllm_hcu_aiter_custom_allreduce_buffer_wrapper"
_MIB = 1024 * 1024


def _configured_buffer_size(default_size: int) -> int:
    value = os.environ.get("AITER_AR_MAX_SIZE_MB")
    if value is None:
        return default_size
    try:
        limit_mb = int(value)
    except ValueError:
        raise ValueError(
            "AITER_AR_MAX_SIZE_MB must be a positive integer"
        ) from None
    if limit_mb <= 0:
        raise ValueError("AITER_AR_MAX_SIZE_MB must be a positive integer")
    # OpenDAS AITER's two-stage ordinary all-reduce admits at most half of
    # ``max_size``.  Keep vLLM's audited default unless the explicit AITER
    # admission limit needs a larger staging allocation.
    return max(default_size, limit_mb * 2 * _MIB)


def apply_to_module(module: ModuleType) -> bool:
    aiter_wrapper = load_exact_module(TARGET_MODULE, module)
    cls = require_class(aiter_wrapper, "AiterCustomAllreduce", TARGETS[0])
    if getattr(aiter_wrapper, _MARKER, False):
        init = getattr(cls, "__init__", None)
        if not callable(init) or not getattr(init, _WRAPPER, False):
            raise PatchCompatibilityError(
                "HCU AITER custom-allreduce buffer marker is stale; restart the process"
            )
        return False

    init = require_callable(cls, "__init__", TARGETS[1])
    if tuple(inspect.signature(init).parameters) != (
        "self",
        "group",
        "device",
        "max_size",
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[1]} has incompatible signature "
            f"{inspect.signature(init)}"
        )
    effective_max_size = require_callable(
        cls, "effective_max_size", TARGETS[2]
    )
    if tuple(inspect.signature(effective_max_size).parameters):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[2]} has incompatible signature "
            f"{inspect.signature(effective_max_size)}"
        )
    default_size = getattr(cls, "MAX_SIZE", None)
    if not isinstance(default_size, int) or default_size <= 0:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]}.MAX_SIZE must be a "
            "positive integer"
        )

    configured_size = _configured_buffer_size(default_size)

    @functools.wraps(init)
    def hcu_init(self, group, device, max_size=None):
        if max_size is None:
            max_size = configured_size
        return init(self, group, device, max_size)

    setattr(hcu_init, _WRAPPER, True)
    setattr(cls, "_vllm_hcu_original_init", init)
    setattr(cls, "__init__", hcu_init)
    setattr(aiter_wrapper, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
