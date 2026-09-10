# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Delegate validated HCU PCP+MTP configs to the plugin PCP manager."""

from __future__ import annotations

import copy
import functools
from types import ModuleType

from vllm_hcu.patch.platform.core_fix.patch_vllm_config import (
    _validate_hcu_pcp_scope,
)

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.worker.gpu.pcp_manager"
PATCH_ID = "worker.framework_opt.pcp.speculative_validation"
TARGETS = (f"{TARGET_MODULE}.PCPManager.validate_config",)
_MARKER = "_vllm_hcu_pcp_spec_validation_applied"
_WRAPPER = "_vllm_hcu_pcp_spec_validation_wrapper"
_UPSTREAM_SPEC_ERROR = "MRV2 PCP does not support speculative decoding yet."


def apply_to_module(module: ModuleType) -> bool:
    pcp = load_exact_module(TARGET_MODULE, module)
    manager = require_class(pcp, "PCPManager", f"{TARGET_MODULE}.PCPManager")
    wrapped = ((manager, "validate_config", TARGETS[0], _WRAPPER),)
    if already_applied(pcp, _MARKER, wrapped):
        return False

    original = require_callable(manager, "validate_config", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("vllm_config", "supports_mm_inputs"),
    )
    code = getattr(original, "__code__", None)
    if code is None or _UPSTREAM_SPEC_ERROR not in code.co_consts:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} no longer contains the "
            "audited speculative-decoding guard"
        )

    @functools.wraps(original)
    def hcu_validate_config(vllm_config, supports_mm_inputs):
        if getattr(vllm_config, "speculative_config", None) is None:
            return original(vllm_config, supports_mm_inputs)

        # The plugin-owned HcuPCPManager implements replicated MTP metadata.
        # Validate that narrow scope first, then run every current upstream PCP
        # check against an otherwise identical config with only its blanket
        # speculative-decoding rejection hidden.
        _validate_hcu_pcp_scope(vllm_config)
        validation_config = copy.copy(vllm_config)
        validation_config.speculative_config = None
        return original(validation_config, supports_mm_inputs)

    setattr(hcu_validate_config, _WRAPPER, True)
    setattr(manager, "_vllm_hcu_original_validate_config", original)
    setattr(manager, "validate_config", staticmethod(hcu_validate_config))
    setattr(pcp, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
