# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fast API and sampling contracts checked before request state is mutated."""

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
    if bad:
        return 'Fast supports the original sampling contract only; unsupported options: ' + ', '.join(bad)
    return None


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
                'structured_outputs','tools','guided_json','guided_regex','guided_grammar','guided_choice','logits_processors'):
        if getattr(request,key,None):bad.append(key)
    fmt=getattr(request,'response_format',None)
    if fmt is not None and getattr(fmt,'type','text')!='text':bad.append('response_format')
    if bad:
        return 'LocateAnything Fast uses non-streaming, single-output block generation; unsupported options: '+', '.join(bad)
    return None
