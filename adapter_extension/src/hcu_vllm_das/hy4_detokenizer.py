# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Bounded ByteLevel prompt priming; Core retains all output/stop handling."""
from functools import wraps
import inspect

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel, DecodeStream

_OWNER = '_hcu_bytelevel_tail_constructor'


def choose_prompt_tail(tokenizer, prompt_token_ids, skip_special_tokens):
    if (not prompt_token_ids or len(prompt_token_ids) <= 32
            or type(tokenizer.decoder) is not ByteLevel):
        return prompt_token_ids
    # Search adjacent UTF-8 boundaries with work independent of prompt length.
    for size in (32, 33, 34, 35, 64, 65, 66, 67):
        if size >= len(prompt_token_ids):
            break
        tail = prompt_token_ids[-size:]
        try:
            decoded = tokenizer.decode(tail, skip_special_tokens=skip_special_tokens)
        except (OverflowError, TypeError, ValueError):
            return prompt_token_ids
        if decoded and '\ufffd' not in decoded:
            return tail
    return prompt_token_ids


def install(module):
    """Install once at frontend construction, without replacing DecodeStream."""
    cls = module.FastIncrementalDetokenizer
    original = cls.__init__
    owner = vars(cls).get(_OWNER)
    if owner is not None:
        if original is not owner:
            raise RuntimeError('ByteLevel constructor ownership changed; restart')
        return False
    params = tuple(inspect.signature(original).parameters)
    if params != ('self', 'tokenizer', 'request'):
        raise ValueError('Unsupported FastIncrementalDetokenizer constructor')

    @wraps(original)
    def constructor(self, tokenizer, request):
        original(self, tokenizer, request)
        if (type(self.tokenizer) is not Tokenizer
                or type(self.stream) is not DecodeStream
                or DecodeStream.__module__ != 'tokenizers.decoders'):
            return
        ids = request.prompt_token_ids
        tail = choose_prompt_tail(self.tokenizer, ids, self.skip_special_tokens)
        if tail is not ids:
            self.stream = DecodeStream(
                ids=tail, skip_special_tokens=self.skip_special_tokens)

    setattr(cls, _OWNER, constructor)
    cls.__init__ = constructor
    return True
