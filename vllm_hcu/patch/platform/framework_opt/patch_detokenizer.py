# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Opt-in bounded ByteLevel context for native frontend detokenization."""
from __future__ import annotations

import functools
from types import ModuleType

from vllm_hcu.platforms import envs as henvs

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_signature_prefix,
)

TARGET_MODULE = "vllm.v1.engine.detokenizer"
PATCH_ID = "platform.framework_opt.bytelevel_detokenizer_tail"
TARGETS = (f"{TARGET_MODULE}.FastIncrementalDetokenizer.__init__",)
_MARKER = "_vllm_hcu_bytelevel_tail_applied"
_WRAPPER = "_vllm_hcu_bytelevel_tail_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if not henvs.VLLM_HCU_ENABLE_BYTELEVEL_DETOKENIZER_TAIL:
        return False
    cls = require_class(target, "FastIncrementalDetokenizer", TARGETS[0])
    original = require_callable(cls, "__init__", TARGETS[0])
    if getattr(cls, _MARKER, False):
        if not getattr(original, _WRAPPER, False):
            raise PatchCompatibilityError("HCU ByteLevel tail marker is stale; restart the process")
        return False
    require_signature_prefix(original, TARGETS[0], ("self", "tokenizer", "request"))

    from tokenizers import Tokenizer
    from tokenizers.decoders import DecodeStream
    from vllm_hcu.runtime_compat.bytelevel_detokenizer import choose_prompt_tail

    @functools.wraps(original)
    def hcu_init(self, tokenizer, request):
        original(self, tokenizer, request)
        # A replacement DecodeStream implementation may have a different
        # context contract. Preserve it rather than overriding its state.
        if (
            type(self.tokenizer) is not Tokenizer
            or type(self.stream) is not DecodeStream
            or DecodeStream.__module__ != "tokenizers.decoders"
        ):
            return
        prime_ids = choose_prompt_tail(
            self.tokenizer, request.prompt_token_ids, self.skip_special_tokens
        )
        if prime_ids is not request.prompt_token_ids:
            self.stream = DecodeStream(
                ids=prime_ids, skip_special_tokens=self.skip_special_tokens
            )

    setattr(hcu_init, _WRAPPER, True)
    cls.__init__ = hcu_init
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
