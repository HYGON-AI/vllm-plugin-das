# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Guarded single-token scheduling with a proven no-allocation fast path.

Derived from MR281's steady-decode scheduling. Every rejecting check happens
before scheduler/KV mutation. Page boundaries use the full split-P/D policy.
"""

from __future__ import annotations

from collections import Counter

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

# vLLM's default logging configuration only installs a handler below 'vllm'.
logger = init_logger("vllm.hcu.steady_decode_scheduler")


def supported_executor(backend) -> bool:
    if backend in (
        "mp",
        "uni",
        "vllm.v1.executor.multiproc_executor.MultiprocExecutor",
        "vllm.v1.executor.uniproc_executor.UniProcExecutor",
        "vllm_hcu.v1.executor.multiproc_executor.HcuMultiprocExecutor",
    ):
        return True
    if not isinstance(backend, type):
        return False
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    from vllm_hcu.v1.executor.multiproc_executor import HcuMultiprocExecutor

    return backend in (MultiprocExecutor, UniProcExecutor, HcuMultiprocExecutor)


def async_scope_error(scheduler) -> str | None:
    """Limit the new async scheduler to the validated single-token lane."""
    parallel = scheduler.parallel_config
    if not scheduler.scheduler_config.async_scheduling:
        return "async_scheduling must be true"
    if not supported_executor(parallel.distributed_executor_backend):
        return "only MP and uniprocess executors are supported"
    if (
        parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or scheduler.dcp_world_size != 1
        or scheduler.pcp_world_size != 1
    ):
        return "only TP=PP=DCP=PCP=1 is supported"
    if (
        scheduler.use_v2_model_runner
        or scheduler.use_pp
        or scheduler.is_encoder_decoder
        or scheduler.num_sampled_tokens_per_step != 1
    ):
        return "only the V1 autoregressive decoder runner is supported"
    if (
        scheduler.num_spec_tokens
        or scheduler.num_lookahead_tokens
        or scheduler.use_eagle
        or scheduler.dynamic_sd_lookup is not None
    ):
        return "speculative decoding is not supported"
    if scheduler.connector is not None or scheduler.ec_connector is not None:
        return "KV/EC connectors are not supported"
    if scheduler.lora_config is not None or scheduler.enable_return_routed_experts:
        return "LoRA and routed-expert outputs are not supported"
    if scheduler.kv_cache_manager.enable_caching:
        return "prefix caching is not supported"
    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    if not managers or any(type(m) is not FullAttentionManager for m in managers):
        return "only FullAttentionManager is supported"
    return None


def _scheduler_reason(scheduler) -> str | None:
    if not getattr(scheduler, "_vllm_hcu_steady_decode_supported", False):
        return "scheduler_not_supported"
    if not scheduler.scheduler_config.async_scheduling:
        return "synchronous_scheduler"
    if scheduler._pause_state.name != "UNPAUSED":
        return "paused"
    if scheduler.policy.name not in ("FCFS", "PRIORITY"):
        return "policy"
    if scheduler.num_waiting_for_streaming_input:
        return "streaming_input"
    if scheduler.finished_req_ids or scheduler.reset_preempted_req_ids:
        return "request_lifecycle"
    running = scheduler.running
    if (
        not 0
        < len(running)
        <= min(scheduler.max_num_running_reqs, scheduler.max_num_scheduled_tokens)
    ):
        return "batch_budget"
    # HCU's waiting-first loop exits before examining either queue when all
    # request slots are held. This is essential for c256 / max_num_seqs=64.
    if (scheduler.waiting or scheduler.skipped_waiting) and len(
        running
    ) != scheduler.max_num_running_reqs:
        return "waiting_admission"
    return scheduler._hcu_steady_decode_fallback_reason()


def _request_eligible(request, max_model_len: int) -> bool:
    computed = request.num_computed_tokens
    if (
        request.status.name != "RUNNING"
        or request.is_prefill_chunk
        or computed < request.num_prompt_tokens
        or computed >= max_model_len - 1
        or request.spec_token_ids
        or request.use_structured_output
        or request.lora_request is not None
        or request.pooling_params is not None
        or request.async_tokens_to_discard
        or request.resumable
        or request.num_tokens_with_spec + request.num_output_placeholders - computed
        != 1
    ):
        return False
    if (
        request.num_output_placeholders > 0
        and computed + 2 - request.num_output_placeholders
        >= request.num_prompt_tokens + request.max_tokens
    ):
        return False
    if request.has_encoder_inputs and any(
        f.mm_position.offset + f.mm_position.length > computed
        for f in request.mm_features
    ):
        return False
    return True


def remember_steady_decode_state(scheduler, output) -> None:
    scheduler._vllm_hcu_steady_decode_state = None
    if _scheduler_reason(scheduler) is not None:
        return
    cached = output.scheduled_cached_reqs
    running = scheduler.running
    if (
        output.scheduled_new_reqs
        or cached.resumed_req_ids
        or output.finished_req_ids
        or output.preempted_req_ids
        or output.scheduled_spec_decode_tokens
        or output.scheduled_encoder_inputs
        or output.free_encoder_mm_hashes
        or len(cached.req_ids) != len(running)
        or len(output.num_scheduled_tokens) != len(running)
        or any(n != 1 for n in output.num_scheduled_tokens.values())
        or any(r.request_id != rid for r, rid in zip(running, cached.req_ids))
    ):
        return
    scheduler._vllm_hcu_steady_decode_state = tuple(
        (r, r.request_id, r.num_computed_tokens) for r in running
    )


def _fallback(scheduler, reason):
    scheduler._vllm_hcu_steady_decode_state = None
    scheduler._vllm_hcu_steady_decode_last_reason = reason
    counts = getattr(scheduler, "_vllm_hcu_steady_decode_fallbacks", None)
    if counts is None:
        counts = scheduler._vllm_hcu_steady_decode_fallbacks = Counter()
    counts[reason] += 1
    return None


def try_steady_decode_schedule(scheduler, throttle_prefills: bool = False):
    reason = _scheduler_reason(scheduler)
    if reason:
        return _fallback(scheduler, reason)
    state = getattr(scheduler, "_vllm_hcu_steady_decode_state", None)
    running = scheduler.running
    if state is None or len(state) != len(running):
        return _fallback(scheduler, "no_stable_state")
    if any(
        r is not old
        or r.request_id != rid
        or r.num_computed_tokens != computed
        or not _request_eligible(r, scheduler.max_model_len)
        for r, (old, rid, computed) in zip(running, state)
    ):
        return _fallback(scheduler, "request_state_changed")
    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    for manager in managers:
        if (
            type(manager) is not FullAttentionManager
            or manager.enable_caching
            or manager.block_size <= 0
        ):
            return _fallback(scheduler, "unsupported_kv_manager")
        for request in running:
            blocks = manager.req_to_blocks.get(request.request_id)
            required = (
                request.num_computed_tokens + manager.block_size
            ) // manager.block_size
            if not blocks or len(blocks) < required:
                return _fallback(scheduler, "allocation_required")
    # Everything below is a committed schedule. All admission/allocation
    # checks have completed; there is no partially-mutated fallback path.
    scheduler.current_step += 1
    scheduler.kv_cache_manager.new_step_starts()
    if not (throttle_prefills and not scheduler.prefill_capacity_bound):
        scheduler.prefill_capacity_bound = bool(scheduler.waiting)
    scheduled = {r.request_id: 1 for r in running}
    blocks = {
        r.request_id: scheduler.kv_cache_manager.empty_kv_cache_blocks for r in running
    }
    cached = scheduler._make_cached_request_data(running, [], scheduled, {}, blocks)
    common_prefix = scheduler.kv_cache_manager.get_num_common_prefix_blocks(
        running[0].request_id
    )
    scheduler.prev_step_scheduled_req_ids.clear()
    scheduler.prev_step_scheduled_req_ids.update(scheduled)
    new_blocks_to_zero = (
        scheduler.kv_cache_manager.take_new_block_ids() or None
        if scheduler.needs_kv_cache_zeroing
        else None
    )
    output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=len(scheduled),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=common_prefix,
        preempted_req_ids=scheduler.reset_preempted_req_ids,
        finished_req_ids=scheduler.finished_req_ids,
        free_encoder_mm_hashes=scheduler.encoder_cache_manager.get_freed_mm_hashes(),
        new_block_ids_to_zero=new_blocks_to_zero,
        num_spec_tokens_to_schedule=0,
    )
    # Keep current vLLM's asynchronous placeholder and cache bookkeeping.
    scheduler._update_after_schedule(output)
    remember_steady_decode_state(scheduler, output)
    scheduler._vllm_hcu_steady_decode_last_reason = None
    hits = getattr(scheduler, "_vllm_hcu_steady_decode_hits", 0) + 1
    scheduler._vllm_hcu_steady_decode_hits = hits
    if hits == 1 or hits % 1000 == 0:
        logger.info(
            "HCU steady decode fastpath: hits=%d batch=%d waiting=%d fallbacks=%s",
            hits,
            len(running),
            len(scheduler.waiting),
            dict(getattr(scheduler, "_vllm_hcu_steady_decode_fallbacks", {})),
        )
    return output
