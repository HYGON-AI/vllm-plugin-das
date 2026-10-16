# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations
from types import ModuleType
from vllm_hcu.runtime_compat.step5_patch_contract import (
    already_applied,
    remember_targets,
    require_signatures,
)
from ._common import load_exact_module, require_callable

TARGET_MODULE = "vllm.compilation.decorators"
PATCH_ID = "platform.core_fix.step5_compile_prefix"
TARGETS = (
    "vllm.compilation.decorators._support_torch_compile",
    "vllm.compilation.decorators._model_hash_key",
    "vllm.compilation.decorators.TorchCompileWithNoGuardsWrapper.__init__",
)
_MARKER = "_vllm_hcu_step5_compile_prefix_applied"

SIGNATURES = (
    (
        "vllm.compilation.decorators._support_torch_compile",
        ("cls", "dynamic_arg_dims", "mark_unbacked_dims", "enable_if", "is_encoder"),
    ),
    ("vllm.compilation.decorators._model_hash_key", ("fn",)),
    (
        "vllm.compilation.decorators.TorchCompileWithNoGuardsWrapper.__init__",
        ("self", "compile_prefix", "is_encoder"),
    ),
)
REVIEWED_SOURCES = ()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    import functools
    import hashlib
    from contextvars import ContextVar

    prefix_scope = ContextVar("step5_compile_prefix", default=None)
    support = require_callable(target, "_support_torch_compile", TARGETS[0])
    model_key = require_callable(target, "_model_hash_key", TARGETS[1])
    wrapper_cls = getattr(target, "TorchCompileWithNoGuardsWrapper", None)
    wrapper_init = require_callable(wrapper_cls, "__init__", TARGETS[2])

    @functools.wraps(wrapper_init)
    def initialize(self, compile_prefix="", is_encoder=False):
        scoped = prefix_scope.get()
        return wrapper_init(
            self, compile_prefix=scoped or compile_prefix, is_encoder=is_encoder
        )

    @functools.wraps(support)
    def support_compile(cls, *args, **kwargs):
        result = support(cls, *args, **kwargs)
        if not cls.__module__.startswith("vllm_hcu.models.step5") or getattr(
            cls, "_vllm_hcu_step5_prefix_init", False
        ):
            return result
        original_init = cls.__init__

        @functools.wraps(original_init)
        def initialize_model(self, *args, prefix="", **kwargs):
            token = prefix_scope.set(prefix)
            try:
                return original_init(self, *args, prefix=prefix, **kwargs)
            finally:
                prefix_scope.reset(token)

        cls.__init__ = initialize_model
        cls._vllm_hcu_step5_prefix_init = True
        return result

    @functools.wraps(model_key)
    def model_hash_key(fn):
        key = model_key(fn)
        owner = getattr(fn, "__self__", None)
        if owner is not None and type(owner).__module__.startswith(
            "vllm_hcu.models.step5"
        ):
            return hashlib.sha256(
                (key + "|" + getattr(owner, "_compile_prefix", "")).encode()
            ).hexdigest()
        return key

    wrapper_cls.__init__ = initialize
    target._support_torch_compile = support_compile
    target._model_hash_key = model_hash_key
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
