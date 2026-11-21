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
    assert seen == [('private/model',dict(revision='pinned-revision',token=token,cache_dir=None))]


@pytest.mark.parametrize("use_snapshot_path", [False, True])
def test_processing_info_fallback_uses_offline_download_cache(monkeypatch, tmp_path, use_snapshot_path):
    import torch

    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx936"),
    )
    from vllm.config import LoadConfig
    from vllm.config.compilation import CompilationConfig, CUDAGraphMode
    from vllm_hcu.patch.platform.core_fix import patch_vllm_config
    from vllm_hcu.patch.platform.framework_opt import patch_multiproc_executor
    from vllm_hcu.platforms.hcu import HCUPlatform

    info = processing_info({})
    model = info.ctx.model_config
    model.hf_config = SimpleNamespace(model_type="locateanything")
    model.revision = "d" * 40
    if use_snapshot_path:
        default_snapshot = tmp_path / "default" / "models--private--model" / "snapshots" / model.revision
        default_snapshot.mkdir(parents=True)
        model.model = str(default_snapshot)
    for name, limit in (("download", 64), ("explicit", 128)):
        snapshot = tmp_path / name / "models--private--model" / "snapshots" / model.revision
        snapshot.mkdir(parents=True)
        (snapshot / "preprocessor_config.json").write_text(
            json.dumps({"in_token_limit": limit}), encoding="utf-8",
        )
    monkeypatch.setattr(patch_vllm_config, "validate_and_update_hcu_config", lambda config: None)
    monkeypatch.setattr(patch_multiproc_executor, "select_hcu_multiproc_executor", lambda config: False)
    HCUPlatform.check_and_update_config(SimpleNamespace(
        additional_config={},
        model_config=model,
        load_config=LoadConfig(download_dir=str(tmp_path / "download")),
        cache_config=None,
        attention_config=SimpleNamespace(backend=None),
        compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.PIECEWISE),
        parallel_config=SimpleNamespace(worker_cls="auto"),
    ))

    assert info._image_processor(local_files_only=True).in_token_limit == 64
    assert info._image_processor(
        local_files_only=True, cache_dir=tmp_path / "explicit",
    ).in_token_limit == 128
    if use_snapshot_path:
        (default_snapshot / "preprocessor_config.json").write_text(
            json.dumps({"in_token_limit": 256}), encoding="utf-8",
        )
        assert info._image_processor(local_files_only=True).in_token_limit == 256


def real_block_counter():
    from vllm_hcu.v1.locateanything.warmup import _reserved_block_count
    def count(tokens, spec):
        return _reserved_block_count(tokens, spec, num_lookahead_tokens=0,
                                     max_model_len=4096, max_encoder_len=0)
    return count


@pytest.mark.parametrize('block_sizes', [(16,),(16,32),(32,64)])
def test_warmup_plan_fits_exact_last_valid_block_and_all_groups(block_sizes):
    from vllm_hcu.v1.locateanything.warmup_budget import warmup_request_plan
    prompts=warmup_prompts(64,32768,4090)
    specs=[SimpleNamespace(block_size=size) for size in block_sizes]
    count=real_block_counter()
    first_two=sum(count(len(p)+134,s) for p in prompts[:2] for s in specs)
    plan=warmup_request_plan(prompts,4090,first_two+1,specs,count)
    assert len(plan)==2 and [p for p,_,_ in plan]==prompts[:2]
    ids=[];next_block=1
    for _,_,counts in plan:
        assert len(counts)==len(specs)
        for size in counts:
            ids.extend(range(next_block,next_block+size));next_block+=size
    assert min(ids)==1 and max(ids)==first_two and len(ids)==len(set(ids))
    assert next_block==first_two+1  # Last valid ID = num_blocks - 1.


def test_warmup_64_slots_is_reduced_for_295_block_pool():
    from vllm_hcu.v1.locateanything.warmup_budget import warmup_request_plan
    prompts=warmup_prompts(64,32768,4090)
    plan=warmup_request_plan(prompts,4090,295,[SimpleNamespace(block_size=16)],real_block_counter())
    assert len(plan)==28 and sum(sum(counts) for _,_,counts in plan)==294


@pytest.mark.parametrize('blocks',[0,1,5])
def test_warmup_too_small_for_one_request_has_clear_error(blocks):
    from vllm_hcu.v1.locateanything.warmup_budget import warmup_request_plan
    with pytest.raises(ValueError,match='insufficient KV blocks'):
        warmup_request_plan(warmup_prompts(64,32768,4090),4090,blocks,
                            [SimpleNamespace(block_size=16)],real_block_counter())


@pytest.mark.parametrize('capacity',[16,128,256])
def test_abundant_kv_keeps_original_synthetic_requests(capacity):
    from vllm_hcu.v1.locateanything.warmup_budget import warmup_request_plan
    prompts=warmup_prompts(capacity,32768,16384)
    plan=warmup_request_plan(prompts,16384,100000,[SimpleNamespace(block_size=16)],real_block_counter())
    assert [p for p,_,_ in plan]==prompts
    assert all(budget==128 for _,budget,_ in plan)
    assert plan[0][2]==(10,)


@pytest.mark.parametrize('spec',[{},False,SimpleNamespace(method='ngram'),SimpleNamespace(method='eagle')])
def test_fast_rejects_nonnull_speculation_before_scheduler_and_worker_setup(monkeypatch,spec):
    import torch
    from vllm_hcu.v1.locateanything.scheduler import Scheduler
    from vllm_hcu.v1.locateanything.model_runner import GPUModelRunner
    def unexpected(*args,**kwargs):pytest.fail('A rejected config must not create a device stream')
    monkeypatch.setattr(torch.cuda,'Stream',unexpected)
    config=SimpleNamespace(speculative_config=spec)
    with pytest.raises(ValueError,match='does not support vLLM speculative decoding'):
        Scheduler(config,None,None,16)
    with pytest.raises(ValueError,match='does not support vLLM speculative decoding'):
        GPUModelRunner(config,torch.device('cpu'))


def test_no_speculation_is_accepted_by_config_guard():
    from vllm_hcu.v1.locateanything.limits import reject_speculative_decoding
    assert reject_speculative_decoding(None) is None


@pytest.mark.parametrize('extra',[
    ['--speculative-config','{}'],['--speculative-config={}'],
    ['--speculative_config','{}'],['--speculative-config.method','ngram'],
    ['--speculative_config.method=ngram'],
])
def test_fast_cli_rejects_speculative_flags_and_slow_retains_them(extra):
    from vllm_hcu.models.locateanything.serve import build_command
    with pytest.raises(ValueError,match='does not support --speculative-config'):
        build_command('fast','/model',16,extra)
    assert build_command('slow','/model',16,extra)[-len(extra):]==extra
