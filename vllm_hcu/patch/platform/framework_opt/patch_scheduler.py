# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Validate and adapt upstream scheduler interfaces consumed by HCU patches."""

from __future__ import annotations

import functools
import importlib
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
    f"{TARGET_MODULE}.Scheduler._mamba_block_aligned_split",
    f"{TARGET_MODULE}.MambaSpec",
    "vllm.v1.kv_cache_interface.iter_layer_specs",
)
_MARKER = "_vllm_hcu_scheduler_contract_validated"
_MAMBA_BLOCK_SIZE = "_vllm_hcu_mamba_block_size"


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


def _install_hybrid_mamba_split_block_size(
    scheduler: type, mamba_spec_type: type, iter_layer_specs
) -> None:
    """Run the official Mamba split with the actual Mamba block size.

    HCU attention keeps ``cache_config.block_size`` at the 64-token vendor
    kernel page. ``Scheduler.block_size`` is the LCM of all effective group
    sizes and can exceed the Mamba page under DCP, so retain the concrete
    ``MambaSpec.block_size`` from the official KV groups. Scope the correction
    to the upstream split and restore the kernel page immediately afterwards.
    """

    original_init = scheduler.__init__
    original = scheduler._mamba_block_aligned_split

    @functools.wraps(original_init)
    def __init__(
        self,
        vllm_config,
        kv_cache_config,
        structured_output_manager,
        block_size,
        *args,
        **kwargs,
    ):
        original_init(
            self,
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            *args,
            **kwargs,
        )
        mamba_spec = next(
            (
                spec
                for group in kv_cache_config.kv_cache_groups
                for spec in iter_layer_specs(
                    getattr(group, "kv_cache_spec", None)
                )
                if isinstance(spec, mamba_spec_type)
            ),
            None,
        )
        if mamba_spec is not None:
            setattr(self, _MAMBA_BLOCK_SIZE, mamba_spec.block_size)

    @functools.wraps(original)
    def _mamba_block_aligned_split(self, request, num_new_tokens, *args, **kwargs):
        cache_config = self.cache_config
        kernel_block_size = cache_config.block_size
        mamba_block_size = getattr(self, _MAMBA_BLOCK_SIZE, kernel_block_size)
        if kernel_block_size == mamba_block_size:
            return original(self, request, num_new_tokens, *args, **kwargs)

        cache_config.block_size = mamba_block_size
        try:
            return original(self, request, num_new_tokens, *args, **kwargs)
        finally:
            cache_config.block_size = kernel_block_size

    scheduler.__init__ = __init__
    scheduler._mamba_block_aligned_split = _mamba_block_aligned_split


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    scheduler = require_class(target, "Scheduler", TARGETS[0])
    mamba_spec_type = require_class(target, "MambaSpec", TARGETS[6])
    kv_cache_interface = importlib.import_module("vllm.v1.kv_cache_interface")
    iter_layer_specs = require_callable(
        kv_cache_interface, "iter_layer_specs", TARGETS[7]
    )
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
    require_signature_prefix(
        require_callable(scheduler, "_mamba_block_aligned_split", TARGETS[5]),
        TARGETS[5],
        ("self", "request", "num_new_tokens"),
    )
    require_signature_prefix(
        iter_layer_specs,
        TARGETS[7],
        ("kv_cache_spec",),
    )
    _install_pp_spec_decode_cadence(scheduler)
    _install_hybrid_mamba_split_block_size(
        scheduler, mamba_spec_type, iter_layer_specs
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
