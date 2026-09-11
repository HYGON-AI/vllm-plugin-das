# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Validate the upstream scheduler interfaces consumed by HCU patches."""

from __future__ import annotations

import functools
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
    f"{TARGET_MODULE}.Scheduler._update_after_schedule",
)
_MARKER = "_vllm_hcu_scheduler_contract_validated"


def _install_pp_spec_decode_cadence(scheduler: type) -> None:
    """Backport upstream PP speculative decode cadence fix (#52179)."""
    original = scheduler._update_after_schedule

    @functools.wraps(original)
    def _update_after_schedule(self, scheduler_output):
        original(self, scheduler_output)
        if not (
            self.use_v2_model_runner
            and self.use_pp
            and self.vllm_config.speculative_config is not None
        ):
            return

        next_step = self.current_step + self.parallel_config.pipeline_parallel_size
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if not request.is_prefill_chunk:
                request.next_decode_eligible_step = next_step

    scheduler._update_after_schedule = _update_after_schedule


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
    require_signature_prefix(
        require_callable(scheduler, "_update_after_schedule", TARGETS[4]),
        TARGETS[4],
        ("self", "scheduler_output"),
    )
    _install_pp_spec_decode_cadence(scheduler)
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
