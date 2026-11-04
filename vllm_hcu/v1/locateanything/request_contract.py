# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Reject unsupported Fast API options before they reach the engine worker."""
def validate_request(request):
    allowed={
        'stream':(None,False),'n':(None,1),'logprobs':(None,False,0),
        'top_logprobs':(None,0),'prompt_logprobs':(None,),
        'presence_penalty':(None,0.0),'frequency_penalty':(None,0.0),
        'min_p':(None,0.0),'min_tokens':(None,0),'ignore_eos':(None,False),
        'use_beam_search':(None,False),
    }
    bad=[key for key,values in allowed.items() if getattr(request,key,None) not in values]
    for key in ('stop','stop_token_ids','logit_bias','bad_words','allowed_token_ids',
                'structured_outputs','tools','guided_json','guided_regex','guided_grammar'):
        if getattr(request,key,None):bad.append(key)
    fmt=getattr(request,'response_format',None)
    if fmt is not None and getattr(fmt,'type','text')!='text':bad.append('response_format')
    if bad:
        return 'LocateAnything Fast uses non-streaming, single-output block generation; unsupported options: '+', '.join(bad)
    return None
