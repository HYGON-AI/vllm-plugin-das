# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import inspect
import os
import textwrap
from types import SimpleNamespace

import pytest
import torch

# These tests activate the adapter explicitly and do not need plugin discovery.
os.environ.setdefault("VLLM_PLUGINS", "__disabled__")

from vllm.sampling_params import SamplingParams
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm_hcu.platforms import envs as henvs
from vllm_hcu.v1.core.sched.scheduler import HcuScheduler


class _NoMultimodalRegistry:
    def supports_multimodal_inputs(self, model_config) -> bool:
        return False


class _NoStructuredOutput:
    def should_advance(self, request) -> bool:
        return False


def _make_scheduler(*, enable_prefix_caching: bool = False) -> HcuScheduler:
    scheduler_config = SimpleNamespace(
        max_num_seqs=4,
        max_num_scheduled_tokens=None,
        max_num_batched_tokens=64,
        policy="fcfs",
        long_prefill_token_threshold=0,
        enable_chunked_prefill=True,
        watermark=0.0,
        scheduler_reserve_full_isl=False,
    )
    cache_config = SimpleNamespace(
        num_gpu_blocks=128,
        enable_prefix_caching=enable_prefix_caching,
        mamba_cache_mode="none",
        block_size=16,
    )
    model_config = SimpleNamespace(
        is_encoder_decoder=False,
        max_model_len=64,
        is_diffusion=False,
        enable_return_routed_experts=False,
    )
    parallel_config = SimpleNamespace(
        data_parallel_index=0,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        pipeline_parallel_size=1,
    )
    observability_config = SimpleNamespace(
        kv_cache_metrics=None,
        kv_cache_metrics_sample=0.0,
        enable_mfu_metrics=False,
    )
    vllm_config = SimpleNamespace(
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        lora_config=None,
        kv_events_config=None,
        parallel_config=parallel_config,
        observability_config=observability_config,
        model_config=model_config,
        kv_transfer_config=None,
        ec_transfer_config=None,
        speculative_config=None,
        num_speculative_tokens=0,
        use_v2_model_runner=False,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    return HcuScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=_NoStructuredOutput(),
        block_size=16,
        mm_registry=_NoMultimodalRegistry(),
    )


def _request(request_id: str, num_tokens: int) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )


def _prime_decode(scheduler: HcuScheduler) -> Request:
    request = _request("running", 4)
    scheduler.add_request(request)
    output = scheduler.schedule()
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[99]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    assert not request.is_prefill_chunk
    return request


@pytest.fixture
def split_pd_enabled(monkeypatch):
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_PD_SPLIT", True, raising=False)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True, raising=False)


def test_feature_off_delegates_throttle_prefills_to_target(monkeypatch):
    calls: list[bool] = []

    def target_schedule(self, throttle_prefills: bool = False):
        calls.append(throttle_prefills)
        return "target"

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_PD_SPLIT", False, raising=False)
    monkeypatch.setattr(HcuScheduler.__mro__[1], "schedule", target_schedule)
    scheduler = object.__new__(HcuScheduler)
    assert scheduler.schedule(throttle_prefills=True) == "target"
    assert calls == [True]


def test_waiting_first_does_not_mix_existing_decode(split_pd_enabled):
    scheduler = _make_scheduler()
    running = _prime_decode(scheduler)
    waiting = _request("waiting", 8)
    scheduler.add_request(waiting)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[waiting.request_id] == 8
    assert running.request_id not in output.num_scheduled_tokens
    assert [req.req_id for req in output.scheduled_new_reqs] == [waiting.request_id]
    assert scheduler.current_step == 2


def test_throttled_prefill_defers_waiting_but_runs_decode(split_pd_enabled):
    scheduler = _make_scheduler()
    running = _prime_decode(scheduler)
    waiting = _request("waiting", 8)
    scheduler.add_request(waiting)

    output = scheduler.schedule(throttle_prefills=True)

    assert waiting.request_id not in output.num_scheduled_tokens
    assert waiting.status == RequestStatus.WAITING
    assert output.num_scheduled_tokens[running.request_id] == 1


class _AsyncLoadConnector:
    def __init__(self) -> None:
        self.allocations: list[tuple[str, int]] = []

    def on_new_request(self, request) -> None:
        return None

    def get_num_new_matched_tokens(self, request, num_local_tokens):
        return 16, True

    def update_state_after_alloc(self, request, blocks, num_external_tokens) -> None:
        self.allocations.append((request.request_id, num_external_tokens))

    def build_connector_meta(self, scheduler_output):
        return SimpleNamespace(requests=[])


def test_async_external_load_owns_blocks_and_inflight_state(
    split_pd_enabled, monkeypatch
):
    scheduler = _make_scheduler(enable_prefix_caching=True)
    connector = _AsyncLoadConnector()
    scheduler.connector = connector
    scheduler.num_lookahead_tokens = 2
    scheduler.need_mamba_block_aligned_split = True
    monkeypatch.setattr(
        scheduler,
        "_mamba_block_aligned_split",
        lambda *args, **kwargs: pytest.fail(
            "async receive must not run local Mamba block alignment"
        ),
    )
    allocate_kwargs = {}
    target_allocate_slots = scheduler.kv_cache_manager.allocate_slots

    def tracked_allocate_slots(*args, **kwargs):
        allocate_kwargs.update(kwargs)
        return target_allocate_slots(*args, **kwargs)

    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "allocate_slots",
        tracked_allocate_slots,
    )
    request = _request("remote", 32)
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {}
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_computed_tokens == 16
    assert request in scheduler._inflight_prefills
    assert connector.allocations == [(request.request_id, 16)]
    assert output.kv_connector_metadata.requests == []
    assert allocate_kwargs["delay_cache_blocks"] is True
    assert allocate_kwargs["num_lookahead_tokens"] == 0
    assert allocate_kwargs["reserved_blocks"] == 0
    assert allocate_kwargs["has_scheduled_reqs"] is False


def test_dynamic_spec_and_deferred_free_fence_are_v025_owned(split_pd_enabled):
    scheduler = _make_scheduler()
    scheduler.dynamic_sd_lookup = [0, 3, 2, 1, 1]
    scheduler.defer_block_free = True
    request = _request("new", 8)
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_spec_tokens_to_schedule == 3
    assert scheduler.sched_step_seq == 1
    assert request.last_sched_seq == 1


def _attributes(function, owner: str) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == owner
    }


def _call_keyword_contracts(function) -> dict[str, list[tuple[str, ...]]]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    contracts: dict[str, list[tuple[str, ...]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            # Positional-only builtins (for example len/set) naturally vary
            # with the HCU control-flow ordering and are not API contracts.
            if not node.keywords:
                continue
            name = node.func.id
        else:
            continue
        contracts.setdefault(name, []).append(
            tuple(sorted(keyword.arg or "**" for keyword in node.keywords))
        )
    return {name: sorted(items) for name, items in contracts.items()}


def test_split_pd_schedule_tracks_the_complete_target_state_contract():
    target_schedule = HcuScheduler.__mro__[1].schedule
    assert tuple(inspect.signature(target_schedule).parameters) == (
        "self",
        "throttle_prefills",
    )
    assert tuple(inspect.signature(HcuScheduler.schedule_split_pd).parameters) == (
        "self",
        "throttle_prefills",
    )

    for owner in ("self", "request"):
        missing = _attributes(target_schedule, owner) - _attributes(
            HcuScheduler.schedule_split_pd, owner
        )
        assert not missing, (
            f"target Scheduler {owner} ownership was not rebased: {missing}"
        )

    target_calls = _call_keyword_contracts(target_schedule)
    split_pd_calls = _call_keyword_contracts(HcuScheduler.schedule_split_pd)
    # All scheduler/API calls and their keyword contracts must remain at least
    # as complete as the matching target. Positional-only builtins are ignored.
    missing_calls = {
        name: keywords
        for name, keywords in target_calls.items()
        if split_pd_calls.get(name) != keywords
    }
    assert not missing_calls, (
        f"target Scheduler call contracts were not rebased: {missing_calls}"
    )
