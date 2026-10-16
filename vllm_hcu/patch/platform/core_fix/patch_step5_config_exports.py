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
from ._common import load_exact_module

TARGET_MODULE = "vllm.transformers_utils.configs"
PATCH_ID = "platform.core_fix.step5_config_exports"
TARGETS = ("vllm.transformers_utils.configs._CLASS_TO_MODULE",)
_MARKER = "_vllm_hcu_step5_config_exports_applied"

SIGNATURES = ()
REVIEWED_SOURCES = ()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    mapping = getattr(target, "_CLASS_TO_MODULE", None)
    if not isinstance(mapping, dict):
        raise TypeError("vllm config exports must be a dictionary")
    for name in (
        "PerceptionEncoderConfig",
        "Step5Config",
        "Step5VLConfig",
        "Step5MTPConfig",
    ):
        mapping[name] = "vllm_hcu.models.step5_config"
        if name not in target.__all__:
            target.__all__.append(name)
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
