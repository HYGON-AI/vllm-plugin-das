# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Validate the upstream scheduler interfaces consumed by HCU patches."""

from __future__ import annotations

from types import ModuleType

from ._common import (
    load_exact_module,
    require_callable,
    require_class,
    require_signature_prefix,
)

TARGET_MODULE = "vllm.v1.core.sched.scheduler"
PATCH_ID = "platform.framework_opt.hcu_scheduler"
TARGETS = (
    f"{TARGET_MODULE}.Scheduler",
    f"{TARGET_MODULE}.Scheduler.update_draft_token_ids",
    f"{TARGET_MODULE}.Scheduler.update_draft_token_ids_in_output",
    f"{TARGET_MODULE}.Scheduler.__init__",
)
_MARKER = "_vllm_hcu_scheduler_contract_validated"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    scheduler = require_class(target, "Scheduler", TARGETS[0])
    if getattr(target, _MARKER, False):
        return False

    require_signature_prefix(
        require_callable(scheduler, "update_draft_token_ids", TARGETS[1]),
        TARGETS[1],
        ("self", "draft_token_ids"),
    )
    require_signature_prefix(
        require_callable(
            scheduler,
            "update_draft_token_ids_in_output",
            TARGETS[2],
        ),
        TARGETS[2],
        ("self", "draft_token_ids", "scheduler_output"),
    )
    require_signature_prefix(
        require_callable(scheduler, "__init__", TARGETS[3]),
        TARGETS[3],
        (
            "self",
            "vllm_config",
            "kv_cache_config",
            "structured_output_manager",
            "block_size",
        ),
    )
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
