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
    f"{TARGET_MODULE}.Scheduler.schedule",
    f"{TARGET_MODULE}.MambaSpec",
    "vllm.v1.kv_cache_interface.iter_layer_specs",
)
_MARKER = "_vllm_hcu_scheduler_contract_validated"
_MAMBA_BLOCK_SIZE = "_vllm_hcu_mamba_block_size"
_ISOLATE_PREFILL = "_vllm_hcu_pcp_isolate_prefill"
_ISOLATE_BACKOFF = "_vllm_hcu_pcp_isolate_backoff"
_ISOLATE_STREAK = "_vllm_hcu_pcp_isolate_streak"


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


def _prefill_remaining(request) -> int:
    # A preempted request must recompute generated history as well as its
    # original prompt. The last known token is its next decode input.
    prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
    return max(0, prefill_end - request.num_computed_tokens)


def _waiting_prefill_ready(scheduler, request) -> bool:
    from vllm.v1.request import RequestStatus

    status = request.status
    if status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
        structured = request.structured_output_request
        grammar = None if structured is None else structured.grammar
        if grammar is None or isinstance(grammar, Exception):
            return False
    elif status == RequestStatus.WAITING_FOR_REMOTE_KVS:
        if request.request_id not in scheduler.finished_recving_kv_req_ids:
            return False
    elif status not in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
        return False
    return not (request.num_stale_output_tokens > 0 and not request.drop_stale_output)


def _pending_prefill_tokens(scheduler, limit: int) -> int:
    """Count prefills eligible under this step's admission and request state."""
    from vllm.v1.core.sched.interface import PauseState

    if scheduler._pause_state == PauseState.PAUSED_ALL:
        return 0
    pending = 0
    step = scheduler.current_step + 1
    for request in scheduler.running:
        if request.next_decode_eligible_step <= step:
            pending += _prefill_remaining(request)
        if pending >= limit:
            return pending
    num_running = len(scheduler.running) + scheduler.num_waiting_for_streaming_input
    if (
        scheduler._pause_state == PauseState.UNPAUSED
        and num_running < scheduler.max_num_running_reqs
    ):
        free_slots = scheduler.max_num_running_reqs - num_running
        # FCFS resumes skipped requests before admitting new waiting rows.
        # Even a ready row with no remaining prefill occupies a request slot.
        for queue in (scheduler.skipped_waiting, scheduler.waiting):
            for request in queue:
                if _waiting_prefill_ready(scheduler, request):
                    pending += _prefill_remaining(request)
                    free_slots -= 1
                    if pending >= limit or free_slots == 0:
                        return pending
    return pending


def _install_pcp_prefill_isolation(scheduler: type) -> None:
    """Schedule DeepSeek V4 PCP prefills without concurrent decode rows.

    A PCP prefill keeps rank-local queries only when every row of the step is
    a prefill; one decode row sends the whole step through global attention,
    which doubles a 12K-token prefill (~0.6 s -> ~1.2 s). SGLang likewise runs
    prefill and decode as separate batches. Rows are held back with the
    existing ``next_decode_eligible_step`` gate, so:

    - while a long prefill is pending, decode rows sit out its steps;
    - after ``VLLM_HCU_PCP_ISOLATE_PREFILL_MAX_STEPS`` such steps in a row
      (bursty arrivals at high concurrency), one decode-only step runs before
      prefills resume, bounding the decode stall;
    - if no prefill gets scheduled (e.g. no KV room), the next step is a
      normal one, so decode can never starve.
    """
    original = scheduler.schedule

    def isolation_enabled(self) -> bool:
        enabled = getattr(self, _ISOLATE_PREFILL, None)
        if enabled is None:
            import vllm_hcu.platforms.envs as henvs
            from vllm_hcu.deepseek_v4_runtime import is_deepseek_v4

            enabled = bool(
                henvs.VLLM_HCU_PCP_ISOLATE_PREFILL
                and henvs.optional_custom_op_enabled(
                    henvs.VLLM_HCU_DEEPSEEK_V4_PCP_LOCAL_QUERIES
                )
                and self.vllm_config.parallel_config.prefill_context_parallel_size > 1
                and is_deepseek_v4(self.vllm_config)
            )
            setattr(self, _ISOLATE_PREFILL, enabled)
        return enabled

    def run_deferring(self, rows, args, kwargs, hide_waiting=False):
        """Run ``schedule`` with ``rows`` held out of this step."""
        # schedule() increments current_step before checking eligibility.
        step = self.current_step + 1
        deferred = [(r, r.next_decode_eligible_step) for r in rows]
        for request, _ in deferred:
            request.next_decode_eligible_step = step + 1
        queues = (self.waiting, self.skipped_waiting)
        if hide_waiting:
            self.waiting = type(queues[0])()
            self.skipped_waiting = type(queues[1])()
        try:
            return original(self, *args, **kwargs)
        finally:
            for request, eligible_step in deferred:
                # A deferred row was not scheduled, so nothing else moved it.
                if request.next_decode_eligible_step == step + 1:
                    request.next_decode_eligible_step = eligible_step
            if hide_waiting:
                # Requests preempted during the step were queued on the
                # stand-ins; put them back in front of the real queues.
                for real, stand_in in zip(queues, (self.waiting, self.skipped_waiting)):
                    # FCFS prepend_requests uses extendleft and would reverse
                    # the already ordered temporary queue a second time.
                    for request in reversed(list(stand_in)):
                        real.prepend_request(request)
                self.waiting, self.skipped_waiting = queues

    @functools.wraps(original)
    def schedule(self, *args, **kwargs):
        from vllm.v1.core.sched.request_queue import SchedulingPolicy

        # Phase isolation is an FCFS optimization. Explicit priorities must
        # retain the upstream ordering and token-budget allocation.
        if (
            self.policy != SchedulingPolicy.FCFS
            or not isolation_enabled(self)
            or getattr(self, _ISOLATE_BACKOFF, False)
        ):
            setattr(self, _ISOLATE_BACKOFF, False)
            return original(self, *args, **kwargs)

        import vllm_hcu.platforms.envs as henvs

        min_tokens = henvs.VLLM_HCU_PCP_ISOLATE_PREFILL_MIN_TOKENS
        step = self.current_step + 1
        decodes = [
            r for r in self.running
            if _prefill_remaining(r) == 0
            and r.next_decode_eligible_step <= step
        ]
        if not decodes or _pending_prefill_tokens(self, min_tokens) < min_tokens:
            setattr(self, _ISOLATE_STREAK, 0)
            return original(self, *args, **kwargs)

        max_steps = henvs.VLLM_HCU_PCP_ISOLATE_PREFILL_MAX_STEPS
        streak = getattr(self, _ISOLATE_STREAK, 0)
        if max_steps > 0 and streak >= max_steps:
            setattr(self, _ISOLATE_STREAK, 0)
            prefills = [
                r for r in self.running
                if _prefill_remaining(r) > 0
            ]
            return run_deferring(self, prefills, args, kwargs, hide_waiting=True)

        output = run_deferring(self, decodes, args, kwargs)
        if output.num_scheduled_tokens:
            setattr(self, _ISOLATE_STREAK, streak + 1)
        else:
            setattr(self, _ISOLATE_BACKOFF, True)
            setattr(self, _ISOLATE_STREAK, 0)
        return output

    scheduler.schedule = schedule


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    scheduler = require_class(target, "Scheduler", TARGETS[0])
    mamba_spec_type = require_class(target, "MambaSpec", TARGETS[7])
    kv_cache_interface = importlib.import_module("vllm.v1.kv_cache_interface")
    iter_layer_specs = require_callable(
        kv_cache_interface, "iter_layer_specs", TARGETS[8]
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
        TARGETS[8],
        ("kv_cache_spec",),
    )
    _install_pp_spec_decode_cadence(scheduler)
    _install_hybrid_mamba_split_block_size(
        scheduler, mamba_spec_type, iter_layer_specs
    )
    require_callable(scheduler, "schedule", TARGETS[6])
    _install_pcp_prefill_isolation(scheduler)
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
