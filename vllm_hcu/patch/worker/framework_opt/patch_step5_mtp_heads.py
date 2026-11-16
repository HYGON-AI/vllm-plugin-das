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

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.eagle.utils"
PATCH_ID = "worker.framework_opt.step5_mtp_heads"
TARGETS = ("vllm.v1.worker.gpu.spec_decode.eagle.utils.load_eagle_model",)
_MARKER = "_vllm_hcu_step5_mtp_heads_applied"

SIGNATURES = (
    (
        "vllm.v1.worker.gpu.spec_decode.eagle.utils.load_eagle_model",
        ("target_model", "vllm_config"),
    ),
)
REVIEWED_SOURCES = (
    (
        "vllm.v1.worker.gpu.spec_decode.eagle.utils.load_eagle_model",
        (
            "056ab3734e5c854537e08852e92747beaee3bf0c89aa0c609894512ec04a77f7",
            "8c3d2a1d4a2adcdf313bf15f1349d70044d734b91f2914ab90bf3301ded4c3a5",
        ),
    ),
)


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    import functools
    from vllm_hcu.runtime_compat.step5 import is_step5_config

    original = require_callable(target, "load_eagle_model", TARGETS[0])

    @functools.wraps(original)
    def load_eagle_model(target_model, vllm_config):
        if is_step5_config(vllm_config):
            from vllm_hcu.v1.step5_mtp_loading import load_eagle_model as load_step5

            return load_step5(target_model, vllm_config)
        return original(target_model, vllm_config)

    target.load_eagle_model = load_eagle_model
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
