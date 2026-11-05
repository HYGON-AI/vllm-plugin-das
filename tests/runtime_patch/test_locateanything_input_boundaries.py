# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Unsupported Fast sessions and custom processors must fail before mutation."""
from collections import deque
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import inspect

import pytest

from tests.runtime_patch.test_locateanything_scheduler_edges import scheduler_fixture


def request(prompt=80, **flags):
    from vllm import SamplingParams
    from vllm.v1.request import Request
    return Request('edge', [9]*prompt, SamplingParams(max_tokens=16), None, **flags)


def frontend():
    from vllm.v1.engine.input_processor import InputProcessor
    from vllm_hcu.patch.locateanything import _patch_fast_session_inputs
    class Target(InputProcessor):
        process_inputs = InputProcessor.process_inputs
    module = ModuleType('vllm.v1.engine.input_processor')
    module.InputProcessor = Target
    signature = inspect.signature(Target.process_inputs)
    assert _patch_fast_session_inputs(module) is True
    assert _patch_fast_session_inputs(module) is False
    processor = Target.__new__(Target)
    processor.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type='locateanything'))
    return processor, signature


SESSION_FLAGS = [dict(resumable=True), dict(session_id='session'),
                 dict(session_id=''), dict(resumable=True,session_id='session')]


@pytest.mark.parametrize('flags', SESSION_FLAGS)
@pytest.mark.parametrize('style', ['keyword','positional'])
def test_real_frontend_rejects_session_flags_before_rendering_or_ipc(flags, style):
    from vllm import SamplingParams
    from vllm.exceptions import VLLMValidationError
    processor, signature = frontend()
    params = SamplingParams()
    args = ('edge', {'type':'token','prompt_token_ids':[9]}, params, ('generate',))
    with pytest.raises(VLLMValidationError, match='does not support resumable or session input'):
        if style == 'keyword':
            processor.process_inputs(*args, **flags)
        else:
            bound = signature.bind(processor, *args, **flags)
            bound.apply_defaults()
            processor.process_inputs(*bound.args[1:])
    assert processor.__dict__.keys() == {'model_config'}
    assert params.max_tokens == 16


@pytest.mark.parametrize('flags', SESSION_FLAGS)
@pytest.mark.parametrize('via_core', [False,True])
def test_scheduler_rejects_session_flags_before_any_state_is_registered(flags, via_core):
    from vllm.v1.engine.core import EngineCore
    scheduler = scheduler_fixture()
    req = request(**flags)
    scheduler.connector = SimpleNamespace(on_new_request=Mock())
    with pytest.raises(ValueError, match='does not support resumable or session input'):
        if via_core:
            EngineCore.add_request(SimpleNamespace(scheduler=scheduler), req)
        else:
            scheduler.add_request(req)
    assert not scheduler.requests and not scheduler.waiting and not scheduler.running
    scheduler.kv_cache_manager.allocate_slots.assert_not_called()
    scheduler.connector.on_new_request.assert_not_called()


@pytest.mark.parametrize('state', ['WAITING','RUNNING','WAITING_FOR_STREAMING_REQ'])
@pytest.mark.parametrize('lengths', [(80,80),(3000,2000),(3500,700)])
def test_duplicate_active_ids_cannot_merge_individually_valid_chunks(state, lengths):
    from vllm.v1.request import RequestStatus
    from vllm_hcu.v1.locateanything.limits import context_error
    scheduler = scheduler_fixture()
    old = request(lengths[0],resumable=True)
    old.status = getattr(RequestStatus,state)
    old.streaming_queue = deque()
    scheduler.requests[old.request_id] = old
    incoming = request(lengths[1])  # Omitting session flags must not bypass the guard.
    assert context_error(4096,old.num_tokens,16) is None
    assert context_error(4096,incoming.num_tokens,16) is None
    before = list(old.all_token_ids)
    with pytest.raises(ValueError, match='active request ID reuse'):
        scheduler.add_request(incoming)
    assert scheduler.requests == {'edge':old} and list(old.all_token_ids) == before
    assert not old.streaming_queue and old.status == getattr(RequestStatus,state)
    assert not scheduler.waiting and not scheduler.running
    scheduler.kv_cache_manager.allocate_slots.assert_not_called()


@pytest.mark.parametrize('lengths', [(80,80),(3000,2000)])
def test_legacy_session_merge_entry_cannot_modify_retained_history(lengths):
    from vllm.v1.request import RequestStatus, StreamingUpdate
    scheduler = scheduler_fixture()
    old = request(lengths[0],resumable=True)
    old.append_output_token_ids([10]*6)
    old.num_computed_tokens = lengths[0]+2
    old.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    new = request(lengths[1],resumable=True)
    update = StreamingUpdate.from_request(new)
    before = (list(old.all_token_ids),list(old.prompt_token_ids),list(old.output_token_ids),old.sampling_params)
    with pytest.raises(ValueError, match='does not support resumable or session input'):
        scheduler._update_request_as_session(old,update)
    assert (list(old.all_token_ids),list(old.prompt_token_ids),list(old.output_token_ids),old.sampling_params) == before
    assert old.status == RequestStatus.WAITING_FOR_STREAMING_REQ


def test_resumable_stop_cannot_queue_another_chunk():
    scheduler = scheduler_fixture()
    req = request(resumable=True)
    req.streaming_queue = deque([object()])
    with pytest.raises(ValueError, match='does not support resumable or session input'):
        scheduler._handle_stopped_request(req)
    assert len(req.streaming_queue)==1 and not scheduler.waiting


@pytest.mark.parametrize('family', ['locateanything','qwen2'])
def test_session_callback_preserves_normal_calls_and_other_models(family):
    from vllm_hcu.patch.locateanything import _patch_fast_session_inputs
    calls=[]
    class Target:
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=family))
        def process_inputs(self, request_id, prompt, params, supported_tasks, resumable=False, session_id=None):
            calls.append((request_id,prompt,params,supported_tasks,resumable,session_id))
            return 'delegated'
    module=ModuleType('vllm.v1.engine.input_processor');module.InputProcessor=Target
    assert _patch_fast_session_inputs(module)
    flags=dict(resumable=True,session_id='session') if family=='qwen2' else {}
    params=object()
    assert Target().process_inputs('id','prompt',params,('generate',),**flags)=='delegated'
    assert calls==[('id','prompt',params,('generate',),flags.get('resumable',False),flags.get('session_id'))]


def test_current_sampling_constructor_or_contract_rejects_custom_processor():
    from vllm import SamplingParams
    from vllm_hcu.v1.locateanything.request_contract import validate_sampling_params
    processor=Mock()
    try:
        params=SamplingParams(logits_processors=[processor])
    except TypeError as exc:
        assert 'logits_processors' in str(exc)
    else:
        assert 'logits_processors' in validate_sampling_params(params)
    processor.assert_not_called()


@pytest.mark.parametrize('via_core', [False,True])
def test_extended_sampling_layout_rejects_processor_at_all_boundaries(via_core, monkeypatch):
    from vllm import SamplingParams
    from vllm.exceptions import VLLMValidationError
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.engine.input_processor import InputProcessor
    from vllm_hcu.patch.locateanything import _patch_fast_input_sampling
    from vllm_hcu.v1.locateanything.core import FastSampler, Sampler
    class LegacySamplingParams(SamplingParams):
        logits_processors: list | None = None
    processor=Mock()
    params=LegacySamplingParams(logits_processors=[processor])
    class Target(InputProcessor):
        _validate_params=InputProcessor._validate_params
    module=ModuleType('vllm.v1.engine.input_processor');module.InputProcessor=Target
    assert _patch_fast_input_sampling(module)
    front=Target.__new__(Target)
    front.model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type='locateanything'))
    with pytest.raises(VLLMValidationError,match='logits_processors'):
        front.process_inputs('id',{'type':'token','prompt_token_ids':[9]},params,('generate',))
    scheduler=scheduler_fixture();req=request();req.sampling_params=params
    with pytest.raises(ValueError,match='logits_processors'):
        if via_core:EngineCore.add_request(SimpleNamespace(scheduler=scheduler),req)
        else:scheduler.add_request(req)
    assert not scheduler.requests and not scheduler.waiting
    scheduler.kv_cache_manager.allocate_slots.assert_not_called()
    monkeypatch.setattr(Sampler,'add_request',lambda *args:pytest.fail('Base sampler state must remain untouched'))
    with pytest.raises(ValueError,match='logits_processors'):
        FastSampler.__new__(FastSampler).add_request(0,80,params)
    processor.assert_not_called()


@pytest.mark.parametrize('value', [None,[]])
def test_empty_legacy_processor_field_keeps_supported_sampling(value):
    from vllm import SamplingParams
    from vllm_hcu.v1.locateanything.request_contract import validate_sampling_params
    class LegacySamplingParams(SamplingParams):
        logits_processors: list | None = None
    assert validate_sampling_params(LegacySamplingParams(logits_processors=value)) is None


@pytest.mark.parametrize('flags,key', [(dict(resumable=True),'resumable'),
    (dict(session_id='session'),'session'),(dict(logits_processors=['unsupported']),'logits_processors')])
def test_fast_chat_rejects_unsupported_session_and_processor_fields(flags,key):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm_hcu.v1.locateanything.request_contract import validate_request
    req=ChatCompletionRequest(model='test',messages=[],**flags)
    assert key in validate_request(req)
