# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Public input/output ownership and the complete qualified Fast profile."""
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import pytest


def qualified_config():
    from vllm.config.compilation import CUDAGraphMode
    return NS(speculative_config=None,
        parallel_config=NS(pipeline_parallel_size=1,tensor_parallel_size=1,data_parallel_size=1,
            prefill_context_parallel_size=1,decode_context_parallel_size=1,enable_dbo=False,ubatch_size=0),
        model_config=NS(hf_config=NS(model_type='locateanything'),dtype='bfloat16',max_model_len=4096,
            runner_type='generate',enforce_eager=False),
        quant_config=None,lora_config=None,kv_transfer_config=None,ec_transfer_config=None,
        cache_config=NS(enable_prefix_caching=False,cache_dtype='auto'),use_v2_model_runner=True,
        attention_config=NS(backend='TRITON_ATTN'),
        compilation_config=NS(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE),
        scheduler_config=NS(async_scheduling=False,enable_chunked_prefill=False,long_prefill_token_threshold=0,
            max_num_batched_tokens=8192,max_num_scheduled_tokens=None))


BAD_CONFIGS = [
    ('parallel_config.tensor_parallel_size',2,'tensor_parallel_size=1'),
    ('parallel_config.pipeline_parallel_size',2,'pipeline_parallel_size=1'),
    ('parallel_config.data_parallel_size',2,'data_parallel_size=1'),
    ('parallel_config.prefill_context_parallel_size',2,'prefill_context_parallel_size=1'),
    ('parallel_config.decode_context_parallel_size',2,'decode_context_parallel_size=1'),
    ('parallel_config.enable_dbo',True,'DBO'),('parallel_config.ubatch_size',2,'microbatching'),
    ('model_config.hf_config.model_type','qwen2','LocateAnything checkpoint'),
    ('model_config.dtype','float16','bfloat16 weights'),('model_config.enforce_eager',True,'graph execution'),
    ('model_config.runner_type','pooling','generate runner'),
    ('model_config.logits_processors',['unimplemented'],'logits_processors'),
    ('model_config.return_sampling_mask',True,'return_sampling_mask'),
    ('model_config.enable_trace_replay',True,'enable_trace_replay'),
    ('model_config.enable_return_routed_experts',True,'enable_return_routed_experts'),
    ('model_config.enable_prompt_embeds',True,'enable_prompt_embeds'),
    ('quant_config',object(),'quantization'),('lora_config',object(),'LoRA'),
    ('kv_transfer_config',object(),'transfer'),('ec_transfer_config',object(),'transfer'),
    ('cache_config.enable_prefix_caching',True,'enable_prefix_caching=False'),
    ('cache_config.cache_dtype','fp8','bfloat16 KV cache'),('use_v2_model_runner',False,'Runner V2'),
    ('attention_config.backend',None,'TRITON_ATTN'),('attention_config.backend','FLASH_ATTN','TRITON_ATTN'),
    ('compilation_config.cudagraph_mode','NONE','FULL decode graphs'),
    ('compilation_config.cudagraph_mode','PIECEWISE','FULL decode graphs'),
    ('scheduler_config.async_scheduling',True,'synchronous scheduling'),
    ('scheduler_config.enable_chunked_prefill',True,'chunked prefill'),
    ('scheduler_config.long_prefill_token_threshold',8,'long_prefill_token_threshold=0'),
    ('scheduler_config.max_num_batched_tokens',4000,'complete prompt'),
    ('scheduler_config.max_num_scheduled_tokens',1024,'complete prompt'),
    ('scheduler_config.max_num_scheduled_tokens',0,'complete prompt'),
]


@pytest.mark.parametrize('path,value,message',BAD_CONFIGS)
@pytest.mark.parametrize('entry',['scheduler','worker'])
def test_all_unqualified_config_branches_fail_before_device_or_queue_setup(path,value,message,entry,monkeypatch):
    import torch
    from vllm_hcu.v1.locateanything.scheduler import Scheduler
    from vllm_hcu.v1.locateanything.model_runner import GPUModelRunner
    config=qualified_config()
    node=config
    parts=path.split('.')
    for name in parts[:-1]:node=getattr(node,name)
    setattr(node,parts[-1],value)
    monkeypatch.setattr(torch.cuda,'Stream',lambda *args,**kwargs:pytest.fail('No device setup for rejected configs'))
    with pytest.raises(ValueError,match=message):
        if entry=='scheduler':Scheduler(config,None,None,16)
        else:GPUModelRunner(config,torch.device('cpu'))


@pytest.mark.parametrize('cache_dtype',['auto','bfloat16'])
def test_qualified_config_accepts_resolved_backend_and_graph_enums(cache_dtype):
    from vllm.config.compilation import CUDAGraphMode
    from vllm_hcu.v1.locateanything.limits import validate_fast_config
    config=qualified_config()
    config.cache_config.cache_dtype=cache_dtype
    config.attention_config.backend=NS(name='TRITON_ATTN')
    for mode in (CUDAGraphMode.FULL,CUDAGraphMode.FULL_AND_PIECEWISE,CUDAGraphMode.FULL_DECODE_ONLY):
        config.compilation_config.cudagraph_mode=mode
        assert validate_fast_config(config) is None


def input_processor(*,fast,family='locateanything'):
    from vllm.v1.engine.input_processor import InputProcessor
    from vllm_hcu.patch.locateanything import _patch_input_contract
    class Target(InputProcessor):
        process_inputs=InputProcessor.process_inputs
        def _validate_params(self,params,supported_tasks):
            self.validated.append(params)
    module=ModuleType('vllm.v1.engine.input_processor');module.InputProcessor=Target
    assert _patch_input_contract(module,fast=fast)
    assert not _patch_input_contract(module,fast=fast)
    processor=Target.__new__(Target)
    processor.model_config=NS(hf_config=NS(model_type=family),max_model_len=4096)
    processor.vllm_config=NS(parallel_config=NS(data_parallel_size=1,data_parallel_size_local=1,local_engines_only=False))
    processor.validated=[]
    processor._validate_lora=Mock()
    processor._validate_model_inputs=Mock()
    processor.generation_config_fields={}
    processor.renderer=NS(get_eos_token_id=lambda:151645,tokenizer=None)
    return processor


def process(processor,params,prompt=100):
    return processor.process_inputs('matrix',{'type':'token','prompt_token_ids':[9]*prompt},params,('generate',))


@pytest.mark.parametrize('fast',[False,True])
@pytest.mark.parametrize('flags',[{},dict(skip_special_tokens=True,spaces_between_special_tokens=True),
                                 dict(skip_special_tokens=False,spaces_between_special_tokens=False)])
@pytest.mark.parametrize('skip_clone',[False,True])
def test_actual_input_processor_final_params_preserve_detection_format_without_changing_caller_flags(fast,flags,skip_clone):
    from vllm import SamplingParams
    processor=input_processor(fast=fast)
    params=SamplingParams(skip_clone=skip_clone,**flags)
    original=(params.skip_special_tokens,params.spaces_between_special_tokens)
    result=process(processor,params)
    assert result.sampling_params is not params
    assert (params.skip_special_tokens,params.spaces_between_special_tokens)==original
    assert not result.sampling_params.skip_special_tokens and not result.sampling_params.spaces_between_special_tokens
    assert result.sampling_params.max_tokens==16


@pytest.mark.parametrize('fast',[False,True])
def test_unrelated_models_keep_native_format_and_sampling_options(fast):
    from vllm import SamplingParams
    processor=input_processor(fast=fast,family='qwen2')
    params=SamplingParams(stop=['x'])
    result=process(processor,params)
    assert result.sampling_params.skip_special_tokens and result.sampling_params.spaces_between_special_tokens
    assert result.sampling_params.stop==['x']


@pytest.mark.parametrize('skip_clone',[False,True])
@pytest.mark.parametrize('prompt',[100,4089])
def test_none_output_budget_is_capped_before_ipc_including_skip_clone(skip_clone,prompt):
    from vllm import SamplingParams
    processor=input_processor(fast=True)
    params=SamplingParams(max_tokens=None,skip_clone=skip_clone)
    result=process(processor,params,prompt)
    assert result.sampling_params.max_tokens==4090-prompt
    assert params.skip_special_tokens


@pytest.mark.parametrize('prompt,budget',[(4090,1),(4095,1),(100,3996),(4089,2)])
def test_final_expanded_prompt_budget_is_rejected_in_the_client(prompt,budget):
    from vllm import SamplingParams
    from vllm.exceptions import VLLMValidationError
    processor=input_processor(fast=True)
    with pytest.raises(VLLMValidationError,match='4090'):
        process(processor,SamplingParams(max_tokens=budget),prompt)


def test_generation_config_cannot_inject_unsupported_stop_ids_after_initial_validation():
    from vllm import SamplingParams
    from vllm.exceptions import VLLMValidationError
    processor=input_processor(fast=True)
    processor.generation_config_fields={'eos_token_id':[151645,151643]}
    params=SamplingParams()
    with pytest.raises(VLLMValidationError,match='stop_token_ids'):
        process(processor,params)
    assert len(processor.validated)==1 and not params.stop_token_ids


@pytest.mark.parametrize('field,value',[
    ('thinking_token_budget',0),('repetition_detection',object()),('trace_decode_token_ids',[9]),
    ('logprob_token_ids',[9]),('extra_args',{'custom':True}),
])
def test_ignored_sampling_features_rejected_before_native_validation(field,value):
    from vllm import SamplingParams
    from vllm.exceptions import VLLMValidationError
    processor=input_processor(fast=True)
    params=SamplingParams()
    setattr(params,field,value)
    with pytest.raises(VLLMValidationError,match=field):process(processor,params)
    assert not processor.validated


@pytest.mark.parametrize('field,value',[
    ('thinking_token_budget',0),('repetition_detection',object()),('trace_decode_token_ids',[9]),
    ('logprob_token_ids',[9]),('extra_args',{'custom':True}),
])
def test_http_contract_rejects_the_same_nondefault_sampling_features(field,value):
    from vllm_hcu.v1.locateanything.request_contract import validate_request
    assert field in validate_request(NS(**{field:value}))


@pytest.mark.parametrize('kind',['embeds','multi_image','lora'])
def test_finalized_fast_request_rejects_unqualified_input_types_without_mutation(kind):
    from vllm import SamplingParams
    from vllm_hcu.v1.locateanything.request_contract import finalize_input_request
    params=SamplingParams()
    request=NS(sampling_params=params,prompt_token_ids=[9]*20,prompt_embeds=None,mm_features=[],lora_request=None)
    if kind=='embeds':request.prompt_embeds=object()
    elif kind=='multi_image':request.mm_features=[object(),object()]
    else:request.lora_request=object()
    with pytest.raises(ValueError):finalize_input_request(request,NS(max_model_len=4096),fast=True)
    assert request.sampling_params is params and params.skip_special_tokens


def test_slow_retains_native_stop_and_logprob_features_after_format_normalization():
    from vllm import SamplingParams
    params=SamplingParams(stop=['x'],logprobs=2,presence_penalty=.1)
    result=process(input_processor(fast=False),params)
    assert result.sampling_params.stop==['x'] and result.sampling_params.logprobs==2
    assert result.sampling_params.presence_penalty==.1
    assert not result.sampling_params.skip_special_tokens and params.skip_special_tokens


def test_explicit_token_only_output_is_preserved():
    from vllm import SamplingParams
    params=SamplingParams(detokenize=False)
    result=process(input_processor(fast=True),params)
    assert result.sampling_params.detokenize is False


@pytest.mark.parametrize('extra',[
    ['--tensor-parallel-size','2'],['--tensor_parallel_size=2'],['-tp','2'],
    ['--data-parallel-size','2'],['-dp=2'],['--prefill-context-parallel-size','2'],
    ['--decode-context-parallel-size','2'],
])
def test_fast_launcher_rejects_all_distributed_variants_and_slow_keeps_them(extra):
    from vllm_hcu.models.locateanything.serve import build_command
    with pytest.raises(ValueError,match='requires .*_size=1'):build_command('fast','/model',16,extra)
    assert build_command('slow','/model',16,extra)[-len(extra):]==extra
