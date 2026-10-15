# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Prepare all EPLB staging formats before communicator registration."""

from __future__ import annotations

from functools import wraps
from types import ModuleType

from vllm_hcu.runtime_compat.step5_patch_contract import (
    already_applied,
    remember_targets,
    require_signatures,
)

from ._common import load_exact_module

TARGET_MODULE = "vllm.distributed.eplb.eplb_communicator"
PATCH_ID = "worker.framework_opt.step5_eplb_communicator"
TARGETS = (f"{TARGET_MODULE}.create_eplb_communicator",)
_MARKER = "_vllm_hcu_step5_eplb_communicator_applied"
SIGNATURES = (
    (TARGETS[0], ("group_coordinator", "backend", "expert_weights", "expert_buffer")),
)
REVIEWED_SOURCES = (
    (TARGETS[0], ("40072a27544d91da0fde0eedec61ba3686fd2726457397110b387581bcd051c3",)),
)


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    original = target.create_eplb_communicator
    from vllm_hcu.distributed.eplb_buffer_runtime import (
        create_communicator_with_buffers,
    )

    @wraps(original)
    def create_eplb_communicator(
        group_coordinator, backend, expert_weights, expert_buffer
    ):
        return create_communicator_with_buffers(
            original, group_coordinator, backend, expert_weights, expert_buffer
        )

    target.create_eplb_communicator = create_eplb_communicator
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
