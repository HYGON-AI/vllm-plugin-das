# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Guard DeepGEMM SM control against local packages missing its SMS API."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.worker.gpu_ubatch_wrapper"
PATCH_ID = "worker.framework_opt.dbo.deep_gemm_sms_capability"
TARGETS = (
    f"{TARGET_MODULE}.deep_gemm_set_num_sms",
    f"{TARGET_MODULE}.UBatchWrapper._create_sm_control_context",
)
_MARKER = "_vllm_hcu_ubatch_sms_guard_applied"
_WRAPPER = "_vllm_hcu_ubatch_sms_guard_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    ubatch = load_exact_module(TARGET_MODULE, module)
    wrapper_class = require_class(
        ubatch, "UBatchWrapper", f"{TARGET_MODULE}.UBatchWrapper"
    )
    wrapped = (
        (wrapper_class, "_create_sm_control_context", TARGETS[1], _WRAPPER),
    )
    if already_applied(ubatch, _MARKER, wrapped):
        return False
    descriptor = vars(wrapper_class).get("_create_sm_control_context")
    if not isinstance(descriptor, staticmethod):
        raise PatchCompatibilityError(f"required target {TARGETS[1]} must be staticmethod")
    original = require_callable(wrapper_class, "_create_sm_control_context", TARGETS[1])
    require_exact_signature(original, TARGETS[1], positional=("vllm_config",))
    require_callable(ubatch, "deep_gemm_set_num_sms", TARGETS[0])

    @functools.wraps(original)
    def hcu_create_sm_control_context(vllm_config):
        from vllm_hcu.v1 import worker_framework_runtime

        if worker_framework_runtime.deep_gemm_has_sms_api(ubatch):
            return original(vllm_config)
        return worker_framework_runtime.create_sm_control_context_without_compute(
            ubatch, vllm_config
        )

    setattr(hcu_create_sm_control_context, _WRAPPER, True)
    setattr(wrapper_class, "_vllm_hcu_original_create_sm_control_context", descriptor)
    setattr(
        wrapper_class,
        "_create_sm_control_context",
        staticmethod(hcu_create_sm_control_context),
    )
    setattr(ubatch, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
