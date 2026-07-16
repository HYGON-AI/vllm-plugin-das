# SPDX-License-Identifier: Apache-2.0
"""Multi-layer MTP top-k buffer sharing for the Eagle loader."""

from __future__ import annotations

import functools
from types import ModuleType

from vllm_hcu.patch.config import get_hcu_config

from ._common import already_applied, load_exact_module, require_callable, require_exact_signature

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.eagle.utils"
PATCH_ID = "worker.framework_opt.spec_decode.eagle_topk_buffer"
TARGETS = (f"{TARGET_MODULE}.load_eagle_model",)
_MARKER = "_vllm_hcu_eagle_topk_buffer_applied"
_WRAPPER = "_vllm_hcu_eagle_topk_buffer_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    eagle = load_exact_module(TARGET_MODULE, module)
    wrapped = ((eagle, "load_eagle_model", TARGETS[0], _WRAPPER),)
    if already_applied(eagle, _MARKER, wrapped):
        return False
    original = require_callable(eagle, "load_eagle_model", TARGETS[0])
    require_exact_signature(
        original, TARGETS[0], positional=("target_model", "vllm_config")
    )

    @functools.wraps(original)
    def hcu_load_eagle_model(target_model, vllm_config):
        eagle_model = original(target_model, vllm_config)
        if not get_hcu_config(vllm_config).enable_multi_layers_mtp:
            return eagle_model
        from vllm_hcu.v1.worker_framework_runtime import share_eagle_topk_buffer

        return share_eagle_topk_buffer(target_model, eagle_model)

    setattr(hcu_load_eagle_model, _WRAPPER, True)
    setattr(eagle, "_vllm_hcu_original_load_eagle_model", original)
    setattr(eagle, "load_eagle_model", hcu_load_eagle_model)
    setattr(eagle, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
