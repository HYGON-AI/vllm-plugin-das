# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Install the HCU-owned hybrid KV page-size helper."""

from __future__ import annotations

from types import ModuleType

from ._common import PatchCompatibilityError, load_exact_module, require_callable, require_signature_prefix

TARGET_MODULE = "vllm.v1.core.kv_cache_utils"
PATCH_ID = "platform.framework_opt.hybrid_kv_page_size"
TARGETS = (f"{TARGET_MODULE}.unify_kv_cache_spec_page_size",)
_MARKER = "_vllm_hcu_kv_page_size_applied"
_WRAPPER = "_vllm_hcu_kv_page_size_function"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    current = require_callable(target, "unify_kv_cache_spec_page_size", TARGETS[0])
    if getattr(target, _MARKER, False):
        if not getattr(current, _WRAPPER, False):
            raise PatchCompatibilityError(
                "HCU KV page-size marker is stale; restart the process"
            )
        return False
    require_signature_prefix(current, TARGETS[0], ("kv_cache_spec",))
    from vllm_hcu.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size

    setattr(unify_kv_cache_spec_page_size, _WRAPPER, True)
    target._vllm_hcu_original_unify_kv_cache_spec_page_size = current
    target.unify_kv_cache_spec_page_size = unify_kv_cache_spec_page_size
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
