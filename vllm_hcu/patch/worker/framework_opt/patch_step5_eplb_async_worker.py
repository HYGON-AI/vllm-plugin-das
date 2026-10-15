# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Bind the async worker's cached transfer alias to the typed-buffer path."""

from __future__ import annotations

from types import ModuleType

from vllm_hcu.runtime_compat.step5_patch_contract import (
    already_applied,
    remember_targets,
    require_signatures,
)

from ._common import load_exact_module
from .patch_step5_eplb_transfer import SIGNATURES as _TRANSFER_SIGNATURES

TARGET_MODULE = "vllm.distributed.eplb.async_worker"
PATCH_ID = "worker.framework_opt.step5_eplb_async_worker"
TARGETS = (f"{TARGET_MODULE}.transfer_layer",)
_MARKER = "_vllm_hcu_step5_eplb_async_worker_applied"
SIGNATURES = ((TARGETS[0], _TRANSFER_SIGNATURES[0][1]),)
REVIEWED_SOURCES = (
    (TARGETS[0], ("ae3aa7ab3c8152ab1bd328c4651d54297dca6426cd672ef9e13a2ffd7ac93cd7",)),
)


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    from vllm_hcu.distributed.eplb_transfer_runtime import transfer_layer

    target.transfer_layer = transfer_layer
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
