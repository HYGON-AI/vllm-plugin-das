# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fast admission/encoder boundaries with real request and cache accounting."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import subprocess
import sys

import pytest


def scheduler_fixture(encoder_budget=128, encoder_cache=128):
    from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.core.sched.interface import PauseState
    from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
    from vllm_hcu.v1.locateanything.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    settings = dict(
        current_step=0, max_num_scheduled_tokens=256, max_model_len=4096,
        fast_max_model_len=4090, max_num_running_reqs=16,
        max_num_encoder_input_tokens=encoder_budget, num_waiting_for_streaming_input=0,
        _pause_state=PauseState.UNPAUSED, prefill_capacity_bound=False,
        running=[], requests={}, lora_config=None, connector=None, ec_connector=None,
        num_prefill_lookahead=0, num_lookahead_tokens=0, num_spec_tokens=0,
        num_sampled_tokens_per_step=1, dynamic_sd_lookup=None,
        need_mamba_block_aligned_split=False, is_encoder_decoder=False,
        scheduler_reserve_full_isl=True, block_size=16, use_v2_model_runner=True,
        model_uses_mrope=False, model_uses_xdrope=False, defer_block_free=False,
        log_stats=False, needs_kv_cache_zeroing=False, sched_step_seq=0,
        enable_return_routed_experts=False, spec_decode_metrics_level='none',
        finished_req_ids=set(), reset_preempted_req_ids=set(), _inflight_prefills=set(),
        policy=SchedulingPolicy.FCFS,
        vllm_config=SimpleNamespace(speculative_config=None),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256,
            long_prefill_token_threshold=0, enable_chunked_prefill=False,
            disable_chunked_mm_input=True, async_scheduling=False),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]),
    )
    for name, value in settings.items():
        setattr(scheduler, name, value)
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    scheduler.encoder_cache_manager = EncoderCacheManager(encoder_cache)
    # Only device KV allocation is replaced; queues, admission, encoder budget,
    # encoder cache, schedule(), request progress and output construction are real.
    cache = Mock(spec=KVCacheManager)
    cache.allocate_slots.return_value = KVCacheBlocks(((),))
    cache.get_num_common_prefix_blocks.return_value = [0]
    cache.take_boundary_state_offloads.return_value = {}
    cache.take_kv_cache_block_copies.return_value = ([], [])
    cache.take_new_block_ids.return_value = []
    scheduler.kv_cache_manager = cache
    scheduler._get_local_prefix_cache_hit = lambda request: (KVCacheBlocks(((),)), 0, 0, False)
    return scheduler


def request(prompt=80, budget=16, image_offset=None):
    from vllm import SamplingParams
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
    from vllm.v1.request import Request
    features = None if image_offset is None else [
        MultiModalFeatureSpec(data=None, modality='image', identifier='test-image',
                             mm_position=PlaceholderRange(offset=image_offset, length=32))]
    return Request('edge', prompt_token_ids=[9]*prompt, pooling_params=None,
                   sampling_params=SamplingParams(max_tokens=budget), mm_features=features)


@pytest.mark.parametrize('via_core', [False, True])
@pytest.mark.parametrize('prompt,budget', [(4095,1), (4090,1), (100,3996), (4089,2)])
def test_offline_admission_rejects_oversize_before_any_queue_or_allocation(via_core, prompt, budget):
    from vllm.v1.engine.core import EngineCore
    scheduler = scheduler_fixture()
    req = request(prompt, budget)
    with pytest.raises(ValueError, match='Fast logical context is 4090'):
        if via_core:
            EngineCore.add_request(SimpleNamespace(scheduler=scheduler), req)
        else:
            scheduler.add_request(req)
    assert not scheduler.requests and not scheduler.waiting and not scheduler.running
    scheduler.kv_cache_manager.allocate_slots.assert_not_called()


@pytest.mark.parametrize('prompt,budget', [(4089,1), (100,3990), (1,1)])
def test_offline_admission_accepts_logical_boundary(prompt, budget):
    scheduler = scheduler_fixture()
    req = request(prompt, budget)
    scheduler.add_request(req)
    assert scheduler.waiting.peek_request() is req and scheduler.requests == {'edge':req}


@pytest.mark.parametrize('reason', ['encoder_budget', 'encoder_cache', 'token_threshold'])
@pytest.mark.parametrize('image_offset', [0, 8, 32])
def test_waiting_prefill_defers_whole_query_until_encoder_and_token_budget_fit(reason, image_offset):
    from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
    from vllm.v1.request import RequestStatus
    scheduler = scheduler_fixture(encoder_budget=0 if reason=='encoder_budget' else 128,
                                  encoder_cache=0 if reason=='encoder_cache' else 128)
    if reason == 'token_threshold':
        scheduler.scheduler_config.long_prefill_token_threshold = 8
    req = request(image_offset=image_offset)
    scheduler.add_request(req)
    for _ in range(2):
        output = scheduler.schedule()
        assert output.total_num_scheduled_tokens == 0
        assert not output.scheduled_new_reqs and not output.scheduled_encoder_inputs
        assert req.status == RequestStatus.WAITING and req.num_computed_tokens == 0
        assert req.num_output_tokens == 0 and not scheduler.running
        assert scheduler.waiting.peek_request() is req
        assert scheduler.encoder_cache_manager.cached == {}
        scheduler.kv_cache_manager.allocate_slots.assert_not_called()
    scheduler.max_num_encoder_input_tokens = 128
    scheduler.encoder_cache_manager = EncoderCacheManager(128)
    scheduler.scheduler_config.long_prefill_token_threshold = 0
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {'edge':86}
    assert output.scheduled_encoder_inputs == {'edge':[0]}
    assert req.status == RequestStatus.RUNNING and req.num_computed_tokens == 80
    assert req.num_output_tokens == 0 and not req.is_prefill_chunk
    assert scheduler.encoder_cache_manager.cached == {'test-image':{'edge'}}
    assert scheduler.kv_cache_manager.allocate_slots.call_args.args[1] == 86


def test_fast_warmup_uses_real_zeroer_integer_contract(monkeypatch):
    import torch
    import vllm.v1.worker.gpu.warmup as base_warmup
    from vllm.v1.worker.utils import KVBlockZeroer
    from vllm_hcu.v1.locateanything.fast_warmup import warmup

    monkeypatch.setattr(base_warmup, '_warmup_block_counter', lambda runner: lambda tokens,spec: (tokens+15)//16)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.zero_block_ids = Mock()
    runner = SimpleNamespace(max_num_reqs=1, max_num_tokens=256,
        model_config=SimpleNamespace(max_model_len=256),
        kv_cache_config=SimpleNamespace(num_blocks=128,
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))]),
        kv_block_zeroer=zeroer, kv_connector=Mock())
    class Planned(Exception):
        pass
    def execute(output):
        if output.scheduled_new_reqs:
            raise Planned
        assert output.finished_req_ids == {'_locate_fast_warmup_0'}
    with pytest.raises(Planned):
        warmup(runner, execute, lambda _:pytest.fail('No device sampling in a planning test'))
    zeroer.zero_block_ids.assert_called_once_with([0])
    assert [call.args for call in runner.kv_connector.set_disabled.call_args_list] == [(True,), (False,)]


def test_fast_rejects_legacy_guided_choice_at_real_chat_boundary():
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm_hcu.v1.locateanything.request_contract import validate_request
    req = ChatCompletionRequest(model='test', messages=[dict(role='user',content='test')],
                                guided_choice=['cat','dog'])
    # This vLLM version ignores the removed field; Fast must not silently imply
    # that the requested constraint was honored (or defer rejection to a worker).
    assert 'guided_choice' in validate_request(req)
    plain = ChatCompletionRequest(model='test', messages=[dict(role='user',content='test')])
    assert validate_request(plain) is None


@pytest.mark.parametrize('option', ['help', 'missing', 'required'])
def test_reference_oracle_cli_is_portable_and_explains_missing_sources(tmp_path, option):
    script = Path(__file__).resolve().parents[2]/'examples/locateanything/validate_fast_rules.py'
    args = {'help':['--help'], 'missing':['--model-dir',str(tmp_path/'checkpoint')], 'required':[]}[option]
    result = subprocess.run([sys.executable,str(script),*args], cwd=tmp_path,
                            capture_output=True,text=True,timeout=30)
    if option == 'help':
        assert result.returncode == 0 and '--model-dir' in result.stdout
    elif option == 'missing':
        assert result.returncode == 2 and 'Missing trusted checkpoint reference source(s)' in result.stderr
        assert 'generate_utils.py' in result.stderr and 'mask_sdpa_utils.py' in result.stderr
    else:
        assert result.returncode == 2 and '--model-dir' in result.stderr
