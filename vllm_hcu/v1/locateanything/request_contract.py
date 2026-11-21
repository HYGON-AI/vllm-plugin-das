# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fast API and sampling contracts checked before request state is mutated."""
from copy import copy

DETECTION_FORMAT_FLAGS = ('skip_special_tokens', 'spaces_between_special_tokens')

FAST_SESSION_ERROR = 'LocateAnything Fast does not support resumable or session input'


def validate_session_input(resumable=False, session_id=None):
    if resumable or session_id is not None:
        return FAST_SESSION_ERROR
    return None


def validate_sampling_params(params):
    """Shared by offline admission and the worker; preserve the Fast contract."""
    if params is None:
        return 'LocateAnything Fast requires generation sampling parameters'
    allowed = {
        'n': (1,), 'logprobs': (None,), 'prompt_logprobs': (None,),
        'presence_penalty': (0.0,), 'frequency_penalty': (0.0,),
        'min_p': (0.0,), 'min_tokens': (0,), 'ignore_eos': (False,),
    }
    bad = [key for key, values in allowed.items() if getattr(params, key) not in values]
    for key in ('logit_bias', 'bad_words', 'allowed_token_ids', 'structured_outputs',
                'stop', 'stop_token_ids'):
        if getattr(params, key):
            bad.append(key)
    # The qualified vLLM rejects this removed constructor field already. Keep
    # the check for older layouts/subclasses instead of silently ignoring it.
    if getattr(params, 'logits_processors', None):
        bad.append('logits_processors')
    for key in ('thinking_token_budget', 'repetition_detection', 'trace_decode_token_ids',
                'logprob_token_ids'):
        if getattr(params, key, None) is not None:
            bad.append(key)
    if getattr(params, 'extra_args', None):
        bad.append('extra_args')
    if bad:
        return 'Fast supports the original sampling contract only; unsupported options: ' + ', '.join(bad)
    return None


def finalize_input_request(request, model_config, *, fast, default_max_tokens=False):
    """Own the finalized params used by both detokenization and EngineCore IPC."""
    params = request.sampling_params
    if params is None:
        return request
    # clone() can return self when skip_clone=True. A separate struct copy is
    # necessary even then: normalization must not mutate the caller's flags.
    params = copy(params)
    for name in DETECTION_FORMAT_FLAGS:
        setattr(params, name, False)
    if fast:
        from vllm_hcu.v1.locateanything.limits import context_error, logical_context_limit
        error = validate_sampling_params(params)
        if error is not None:
            raise ValueError(error)
        if request.prompt_embeds is not None:
            raise ValueError('LocateAnything Fast requires token IDs; prompt embeddings are unsupported')
        if len(getattr(request, 'mm_features', None) or ()) > 1:
            raise ValueError('LocateAnything Fast supports at most one image per request')
        if getattr(request, 'lora_request', None) is not None:
            raise ValueError('LocateAnything Fast does not support LoRA requests')
        prompt_tokens = len(request.prompt_token_ids or ())
        if prompt_tokens == 0:
            raise ValueError('LocateAnything Fast requires a nonempty token prompt')
        error = context_error(model_config.max_model_len, prompt_tokens, None)
        if error is not None:
            raise ValueError(error)
        if default_max_tokens:
            available = logical_context_limit(model_config.max_model_len) - prompt_tokens
            params.max_tokens = min(params.max_tokens, available) if params.max_tokens is not None else available
        error = context_error(model_config.max_model_len, prompt_tokens, params.max_tokens)
        if error is not None:
            raise ValueError(error)
    request.sampling_params = params
    return request


def validate_request(request):
    error = validate_session_input(getattr(request, 'resumable', False), getattr(request, 'session_id', None))
    if error is not None:
        return error
    allowed={
        'stream':(None,False),'n':(None,1),'logprobs':(None,False,0),
        'top_logprobs':(None,0),'prompt_logprobs':(None,),
        'presence_penalty':(None,0.0),'frequency_penalty':(None,0.0),
        'min_p':(None,0.0),'min_tokens':(None,0),'ignore_eos':(None,False),
        'use_beam_search':(None,False),
    }
    bad=[key for key,values in allowed.items() if getattr(request,key,None) not in values]
    for key in ('stop','stop_token_ids','logit_bias','bad_words','allowed_token_ids',
                'structured_outputs','tools','guided_json','guided_regex','guided_grammar','guided_choice','logits_processors',
                'extra_args','repetition_detection','trace_decode_token_ids','logprob_token_ids'):
        if getattr(request,key,None):bad.append(key)
    if getattr(request,'thinking_token_budget',None) is not None:bad.append('thinking_token_budget')
    fmt=getattr(request,'response_format',None)
    if fmt is not None and getattr(fmt,'type','text')!='text':bad.append('response_format')
    if bad:
        return 'LocateAnything Fast uses non-streaming, single-output block generation; unsupported options: '+', '.join(bad)
    return None
