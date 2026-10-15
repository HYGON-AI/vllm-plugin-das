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

TARGET_MODULE = "vllm.envs"
PATCH_ID = "platform.core_fix.step5_envs"
TARGETS = ("vllm.envs.environment_variables", "vllm.envs.compile_factors")
_MARKER = "_vllm_hcu_step5_envs_applied"

SIGNATURES = (("vllm.envs.compile_factors", ()),)
REVIEWED_SOURCES = ()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    import functools
    import os
    from vllm_hcu.platforms.step5_envs import ENVIRONMENT_VARIABLES

    variables = getattr(target, "environment_variables", None)
    if not isinstance(variables, dict):
        raise TypeError("vllm.envs.environment_variables must be a dictionary")
    original = require_callable(target, "compile_factors", TARGETS[1])
    variables.update(ENVIRONMENT_VARIABLES)
    cache_clear = getattr(getattr(target, "__getattr__", None), "cache_clear", None)
    if callable(cache_clear):
        cache_clear()

    @functools.wraps(original)
    def compile_factors():
        factors = original().copy()
        for name in (
            "VLLM_STEP5_JIT_CACHE_DIR",
            "VLLM_STEP5_TOOL_PARSER_MAX_HEADER_CHARS",
        ):
            factors.pop(name, None)
        for name in sorted(os.environ):
            if name.startswith("VLLM_HCU_"):
                factors[name] = os.environ[name]
        return factors

    target.compile_factors = compile_factors
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
