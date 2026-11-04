# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for detection API format and Fast request RNG ownership."""
import asyncio
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize('kind', ['chat', 'completion'])
@pytest.mark.parametrize('mode', ['slow', 'fast'])
@pytest.mark.parametrize('flags', [{}, {'skip_special_tokens': True},
                                  {'skip_special_tokens': False, 'spaces_between_special_tokens': False}])
def test_detection_api_preserves_default_and_explicit_special_tokens(monkeypatch, kind, mode, flags):
    from tokenizers import Tokenizer, decoders, models
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm_hcu.patch.locateanything import register_locateanything_api_callbacks

    monkeypatch.setenv('VLLM_HCU_LOCATEANYTHING_MODE', mode)
    calls = []
    coordinator = SimpleNamespace(register_callback=lambda *args, **kwargs: calls.append(args))
    register_locateanything_api_callbacks(coordinator)
    _, name, patch = calls[0 if kind == 'chat' else 1]
    module = ModuleType(name)

    class Endpoint:
        model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type='locateanything'))

        async def create(self, request, raw_request=None):
            self.seen_request = request
            self.seen_raw = raw_request
            return request.to_sampling_params(64, {})

    cls_name = 'OpenAIServingChat' if kind == 'chat' else 'OpenAIServingCompletion'
    method = 'create_chat_completion' if kind == 'chat' else 'create_completion'
    setattr(Endpoint, method, Endpoint.create)
    setattr(module, cls_name, Endpoint)
    assert patch(module) is True
    assert patch(module) is False  # A plugin reentry must not stack wrappers.
    request = (ChatCompletionRequest(model='model', messages=[], **flags) if kind == 'chat'
               else CompletionRequest(model='model', prompt='detect', **flags))
    original_flags = (request.skip_special_tokens, request.spaces_between_special_tokens)
    endpoint = Endpoint()
    raw = object()
    params = asyncio.run(getattr(endpoint, method)(request, raw_request=raw))
    assert endpoint.seen_raw is raw and endpoint.seen_request is not request
    assert (request.skip_special_tokens, request.spaces_between_special_tokens) == original_flags
    assert params.skip_special_tokens is False and params.spaces_between_special_tokens is False

    # A real tokenizer with special BOX/coordinate tokens: the old API default
    # strips the entire detection. The normalized real SamplingParams keeps it.
    tokens = ['<box>', '<0>', '<1>', '<2>', '<3>', '</box>']
    tokenizer = Tokenizer(models.WordLevel({s: i for i, s in enumerate(tokens)}))
    tokenizer.add_special_tokens(tokens)
    tokenizer.decoder = decoders.Fuse()
    assert tokenizer.decode(list(range(6)), skip_special_tokens=True) == ''
    assert tokenizer.decode(list(range(6)), skip_special_tokens=params.skip_special_tokens) == ''.join(tokens)

    endpoint.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type='qwen2'))
    other = asyncio.run(getattr(endpoint, method)(request))
    assert endpoint.seen_request is request
    assert (other.skip_special_tokens, other.spaces_between_special_tokens) == original_flags


def test_fast_completions_keeps_the_existing_image_route_rejection():
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm_hcu.v1.locateanything.completion_serving import OpenAIServingCompletion

    endpoint = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
    endpoint.create_error_response = lambda message: message
    error = asyncio.run(endpoint._create_completion(CompletionRequest(model='model', prompt='detect')))
    assert error == 'LocateAnything Fast serves image requests through /v1/chat/completions'


@pytest.fixture
def fast_sampler(monkeypatch):
    import torch
    from vllm_hcu.v1.locateanything.core import FastSampler, Sampler

    monkeypatch.delenv('LOCATE_FAST_TEACHER', raising=False)

    # Exercise the real FastSampler and torch.Generator on CPU; base sampling
    # buffers belong to unrelated device sampling tests.
    monkeypatch.setattr(Sampler, 'add_request', lambda self, *args: None)
    base = Sampler.__new__(Sampler)
    base.req_states = SimpleNamespace(index_to_req_id={}, req_id_to_index={},
                                      last_sampled_tokens=torch.empty(1, dtype=torch.long))
    return FastSampler.from_base(base)


def add(sampler, req_id, slot, seed):
    from vllm import SamplingParams
    sampler.req_states.index_to_req_id[slot] = req_id
    sampler.req_states.req_id_to_index[req_id] = slot
    sampler.add_request(slot, 5, SamplingParams(seed=seed))
    return sampler.fast_generators[slot]


def runner_for(sampler):
    from vllm_hcu.v1.locateanything.model_runner import GPUModelRunner

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.sampler = sampler
    runner.req_states = sampler.req_states

    def remove(req_id):
        slot = runner.req_states.req_id_to_index.pop(req_id, None)
        if slot is not None:
            runner.req_states.index_to_req_id.pop(slot)
        return slot

    runner.req_states.remove_request = remove
    runner.model_state = SimpleNamespace(remove_request=lambda req_id: None)
    runner.lora_state = SimpleNamespace(remove_request=lambda req_id: None)
    runner.pooling_runner = runner.pp_handler = runner.encoder_cache = runner.prompt_logprobs_worker = None
    return runner


def finish(runner, finished=(), preempted=()):
    runner.finish_requests(SimpleNamespace(finished_req_ids=set(finished), preempted_req_ids=set(preempted)))


def test_finish_bounds_generator_and_diagnostic_state(fast_sampler):
    runner = runner_for(fast_sampler)
    for i in range(100):
        req_id = f'request-{i}'
        add(fast_sampler, req_id, 0, i)
        fast_sampler.fast_teachers[0] = {'step': i}
        finish(runner, finished=[req_id])
        finish(runner, finished=[req_id])  # Duplicate finish is harmless.
        assert not fast_sampler.fast_request_generators
        assert not fast_sampler.fast_generators and not fast_sampler.fast_teachers


def test_reused_request_id_uses_new_seed(fast_sampler):
    import torch
    runner = runner_for(fast_sampler)
    old = add(fast_sampler, 'same-id', 0, 11)
    torch.rand(7, generator=old)
    finish(runner, finished=['same-id'])
    new = add(fast_sampler, 'same-id', 0, 29)
    expected = torch.Generator().manual_seed(29)
    assert new is not old
    torch.testing.assert_close(torch.rand(8, generator=new), torch.rand(8, generator=expected), rtol=0, atol=0)


def test_preemption_preserves_rng_across_slot_reuse_and_resume(fast_sampler):
    import torch
    runner = runner_for(fast_sampler)
    gen = add(fast_sampler, 'preempted', 0, 73)
    expected = torch.Generator().manual_seed(73)
    torch.testing.assert_close(torch.rand(6, generator=gen), torch.rand(6, generator=expected))
    fast_sampler.fast_teachers[0] = {'step': 1}
    finish(runner, preempted=['preempted'])
    assert fast_sampler.fast_request_generators['preempted'] is gen
    assert not fast_sampler.fast_generators and not fast_sampler.fast_teachers
    other = add(fast_sampler, 'other', 0, 12)
    resumed = add(fast_sampler, 'preempted', 7, 73)
    assert resumed is gen and fast_sampler.fast_generators[0] is other
    torch.testing.assert_close(torch.rand(6, generator=resumed), torch.rand(6, generator=expected), rtol=0, atol=0)
    finish(runner, finished=['preempted', 'other'])
    assert not fast_sampler.fast_generators and not fast_sampler.fast_request_generators


def test_abort_while_preempted_and_finish_preemption_overlap(fast_sampler):
    runner = runner_for(fast_sampler)
    add(fast_sampler, 'abort-while-preempted', 0, 19)
    finish(runner, preempted=['abort-while-preempted'])
    assert not runner.req_states.req_id_to_index
    finish(runner, finished=['abort-while-preempted'])
    assert not fast_sampler.fast_request_generators
    add(fast_sampler, 'finish-wins', 0, 37)
    finish(runner, finished=['finish-wins'], preempted=['finish-wins'])
    assert not fast_sampler.fast_generators and not fast_sampler.fast_request_generators
