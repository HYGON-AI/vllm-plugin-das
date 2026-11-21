# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

from types import ModuleType

from vllm_hcu.runtime_compat.step5_patch_contract import (
    already_applied,
    remember_targets,
    require_signatures,
)
from ._common import load_exact_module

TARGET_MODULE = "vllm.transformers_utils.config"
PATCH_ID = "platform.core_fix.hy_v4_flash_config_registry"
TARGETS = (f"{TARGET_MODULE}._CONFIG_REGISTRY",)
_MARKER = "_vllm_hcu_hy_v4_flash_config_registry_applied"
SIGNATURES = ()
REVIEWED_SOURCES = ()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    mapping = getattr(target, "_CONFIG_REGISTRY", None)
    if not isinstance(mapping, dict):
        raise TypeError("vLLM model config registry must be a dictionary")
    from vllm_hcu.models.hy_v4_flash.configs import (
        HYV4FlashConfig,
        HYV4FlashVLConfig,
    )

    mapping.update(
        hy_v4_flash=HYV4FlashConfig,
        hy_v4_flash_vl=HYV4FlashVLConfig,
    )
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
