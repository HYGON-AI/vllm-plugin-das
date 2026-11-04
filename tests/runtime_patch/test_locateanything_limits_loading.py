# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Real request stop boundaries and offline HF ID/image-config regressions."""
import json
from types import SimpleNamespace

import pytest

from vllm_hcu.v1.locateanything.limits import context_error, logical_context_limit
from vllm_hcu.v1.locateanything.warmup_budget import warmup_prompts


@pytest.mark.parametrize('physical,logical', [(8,2),(32,26),(4096,4090),(16390,16384),(32768,32762)])
def test_context_and_api_budget_share_probe_reservation(physical, logical):
    assert logical_context_limit(physical) == logical
    assert context_error(physical, logical-1, 1) is None
    assert context_error(physical, logical-1, None) is None
    assert str(logical) in context_error(physical, logical, None)
    assert context_error(physical, logical-1, 2) is not None


@pytest.mark.parametrize('physical', [0,1,6,7])
def test_invalid_context_fails_clearly(physical):
    with pytest.raises(ValueError, match='at least 8'):
        logical_context_limit(physical)


def request_with_budget(prompt, budget):
    from vllm import SamplingParams
    from vllm.v1.request import Request
    return Request('boundary', prompt_token_ids=[9]*prompt,
                   sampling_params=SamplingParams(max_tokens=budget), pooling_params=None)


@pytest.mark.parametrize('physical', [32,4096,16390,32768])
def test_real_stop_method_finishes_before_next_query_can_be_clipped(physical):
    from vllm.v1.request import RequestStatus
    from vllm_hcu.v1.locateanything.scheduler import Scheduler

    logical = logical_context_limit(physical)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.fast_max_model_len = logical
    request = request_with_budget(min(100, logical-1), physical)
    stopped = False
    for _ in range(physical):
        required = request.num_tokens - request.num_computed_tokens + 6
        physical_cap = physical - request.num_computed_tokens - 1
        assert required <= physical_cap  # Every live request can fit its whole block.
        request.num_computed_tokens = request.num_tokens
        output, stopped = scheduler._update_request_with_output(request, [9]*6)
        assert output == [9]*6
        if stopped:
            break
    assert stopped and request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert logical <= request.num_tokens <= physical-1
    assert request.num_output_tokens < request.max_tokens


def test_review_4096_counterexample_stops_with_six_output_tokens_left():
    from vllm.v1.request import RequestStatus
    from vllm_hcu.v1.locateanything.scheduler import Scheduler
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.fast_max_model_len = logical_context_limit(4096)
    request = request_with_budget(100, 3996)
    request.append_output_token_ids([9]*3984)
    request.num_computed_tokens = 4084
    _, stopped = scheduler._update_request_with_output(request, [9]*6)
    assert request.num_tokens == 4090 and request.num_output_tokens == 3990
    assert stopped and request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert context_error(4096, 100, 3996) is not None


def test_default_output_budget_excludes_probe_slots():
    from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
    logical = logical_context_limit(4096)
    assert get_max_tokens(logical, None, 100, {}, None) == 3990


@pytest.mark.parametrize('budget', range(1,7))
def test_original_complete_block_budget_semantics_are_preserved(budget):
    from vllm_hcu.v1.locateanything.scheduler import Scheduler
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.fast_max_model_len = logical_context_limit(16390)
    request = request_with_budget(20, budget)
    out, stopped = scheduler._update_request_with_output(request, [8,151645,9,10,11,12])
    assert stopped and len(out) == 6 and request.num_output_tokens == 6


@pytest.mark.parametrize('physical', [8,9,16,20,32,256,4096,16390])
@pytest.mark.parametrize('capacity', [1,16,256])
def test_warmup_fits_configured_context_including_small_limits(physical, capacity):
    logical = logical_context_limit(physical)
    prompts = warmup_prompts(capacity, 32768, logical)
    assert all(1 <= len(p) < logical for p in prompts)
    assert sum(len(p)+6 for p in prompts) <= 32768
    assert len(prompts)*min(12,logical+5) <= 32768
    assert all(len(p)+min(128,logical-len(p))+6 <= physical for p in prompts)


def test_default_warmup_profile_is_unchanged():
    assert warmup_prompts(128,32768,16384) == warmup_prompts(128,32768)
    assert warmup_prompts(256,32768,16384) == warmup_prompts(256,32768)


def test_hub_id_resolves_two_revisions_from_real_offline_cache(tmp_path):
    from vllm_hcu.models.locateanything.processing import LocateAnythingImageProcessor
    repo = 'nvidia/LocateAnything-3B'
    for revision, limit in [('a'*40,64), ('b'*40,128)]:
        snapshot = tmp_path/'models--nvidia--LocateAnything-3B'/'snapshots'/revision
        snapshot.mkdir(parents=True)
        (snapshot/'preprocessor_config.json').write_text(json.dumps({'patch_size':14,'in_token_limit':limit}))
        processor = LocateAnythingImageProcessor.from_pretrained(
            repo, revision=revision, cache_dir=tmp_path, local_files_only=True)
        assert processor.patch_size == 14 and processor.in_token_limit == limit
    with pytest.raises(OSError):
        LocateAnythingImageProcessor.from_pretrained(repo, revision='c'*40, cache_dir=tmp_path, local_files_only=True)


def test_resolver_preserves_auth_cache_subfolder_and_pixel_overrides(monkeypatch, tmp_path):
    from vllm_hcu.models.locateanything import processing
    path = tmp_path/'preprocessor_config.json'
    path.write_text(json.dumps({'in_token_limit':64,'patch_size':14}))
    seen = []
    def resolve(*args, **kwargs):
        seen.append((args,kwargs))
        return str(path)
    monkeypatch.setattr(processing, 'cached_file', resolve)
    token = object()
    processor = processing.LocateAnythingImageProcessor.from_pretrained(
        'private/model', revision='review-branch', token=token, cache_dir=tmp_path,
        local_files_only=True, subfolder='vision', in_token_limit=96)
    assert processor.in_token_limit == 96
    assert seen == [(('private/model','preprocessor_config.json'),dict(
        revision='review-branch',token=token,cache_dir=tmp_path,local_files_only=True,subfolder='vision'))]


def test_local_path_and_cached_settings_do_not_mutate_original_config(tmp_path):
    from vllm_hcu.models.locateanything.processing import LocateAnythingImageProcessor
    settings = dict(patch_size=14,in_token_limit=64,image_mean=[.1,.2,.3],merge_kernel_size=[2,2])
    (tmp_path/'preprocessor_config.json').write_text(json.dumps(settings))
    processor = LocateAnythingImageProcessor.from_pretrained(tmp_path,local_files_only=True)
    assert processor.image_mean == (.1,.2,.3) and processor.in_token_limit == 64
    override = LocateAnythingImageProcessor.from_config(settings,in_token_limit=96)
    assert override.in_token_limit == 96 and settings['in_token_limit'] == 64


def processing_info(settings, token=None):
    from vllm_hcu.models.locateanything.configuration import LocateAnythingConfig
    from vllm_hcu.models.locateanything.model import LocateAnythingProcessingInfo
    info = LocateAnythingProcessingInfo.__new__(LocateAnythingProcessingInfo)
    info.ctx = SimpleNamespace(
        model_config=SimpleNamespace(model='private/model',revision='pinned-revision',hf_token=token),
        get_hf_image_processor_config=lambda:settings,
        get_hf_config=lambda typ:LocateAnythingConfig(), get_tokenizer=lambda:object(),
        get_merged_mm_kwargs=lambda kwargs,**extra:dict(kwargs))
    return info


def test_processing_info_uses_vllm_resolved_hub_config_for_default_and_overrides(monkeypatch):
    from vllm_hcu.models.locateanything.processing import LocateAnythingImageProcessor
    def unexpected(*args, **kwargs):
        pytest.fail('vLLM already resolved the config: no local/Hub lookup should occur')
    monkeypatch.setattr(LocateAnythingImageProcessor, 'from_pretrained', unexpected)
    settings = dict(patch_size=14,in_token_limit=64,merge_kernel_size=[2,2])
    info = processing_info(settings)
    assert info.get_hf_processor().image_processor.in_token_limit == 64
    assert info.get_hf_processor(in_token_limit=96).image_processor.in_token_limit == 96
    assert settings['in_token_limit'] == 64


def test_processing_info_fallback_keeps_model_revision_and_auth(monkeypatch):
    from vllm_hcu.models.locateanything.processing import LocateAnythingImageProcessor
    seen=[]
    def load(model, **kwargs):
        seen.append((model,kwargs))
        return LocateAnythingImageProcessor()
    monkeypatch.setattr(LocateAnythingImageProcessor, 'from_pretrained', load)
    token=object()
    info=processing_info({},token)
    info.get_hf_processor()
    assert seen == [('private/model',dict(revision='pinned-revision',token=token))]
