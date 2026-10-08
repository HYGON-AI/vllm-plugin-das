# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU compatibility fixes for the Eagle model loader."""

from __future__ import annotations

import functools
from copy import copy
from types import ModuleType

from vllm_hcu.patch.config import get_hcu_config

from ._common import already_applied, load_exact_module, require_callable, require_exact_signature

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.eagle.utils"
PATCH_ID = "worker.framework_opt.spec_decode.eagle_topk_buffer"
TARGETS = (
    f"{TARGET_MODULE}.load_eagle_model",
    f"{TARGET_MODULE}._should_share",
)
_MARKER = "_vllm_hcu_eagle_topk_buffer_applied"
_WRAPPER = "_vllm_hcu_eagle_topk_buffer_wrapper"
_SHARE_WRAPPER = "_vllm_hcu_eagle_step_mtp_head_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    eagle = load_exact_module(TARGET_MODULE, module)
    wrapped = (
        (eagle, "load_eagle_model", TARGETS[0], _WRAPPER),
        (eagle, "_should_share", TARGETS[1], _SHARE_WRAPPER),
    )
    if already_applied(eagle, _MARKER, wrapped):
        return False
    original = require_callable(eagle, "load_eagle_model", TARGETS[0])
    original_should_share = require_callable(eagle, "_should_share", TARGETS[1])
    require_exact_signature(
        original, TARGETS[0], positional=("target_model", "vllm_config")
    )
    require_exact_signature(
        original_should_share,
        TARGETS[1],
        positional=("eagle", "flag", "draft", "target"),
    )

    @functools.wraps(original_should_share)
    def hcu_should_share(eagle_model, flag, draft, target):
        config = getattr(eagle_model, "config", None)
        if (
            flag == "has_own_lm_head"
            and getattr(config, "model_type", None) == "step3p5_mtp"
        ):
            # Step MTP checkpoints train a distinct output head for every
            # prediction layer. Replacing them with the target lm_head leaves
            # speculative decoding correct but collapses draft acceptance.
            return False
        return original_should_share(eagle_model, flag, draft, target)

    @functools.wraps(original)
    def hcu_load_eagle_model(target_model, vllm_config):
        speculative_config = vllm_config.speculative_config
        draft_model_config = speculative_config.draft_model_config
        # Draft construction needs its own model context, while the engine
        # configs and runtime state still belong to the initialized target.
        # replace() reruns VllmConfig hooks and mutates those shared configs.
        draft_vllm_config = copy(vllm_config)
        draft_vllm_config.model_config = draft_model_config
        eagle_model = original(target_model, draft_vllm_config)
        if not get_hcu_config(vllm_config).enable_multi_layers_mtp:
            return eagle_model
        from vllm_hcu.v1.worker_framework_runtime import share_eagle_topk_buffer

        return share_eagle_topk_buffer(target_model, eagle_model)

    setattr(hcu_load_eagle_model, _WRAPPER, True)
    setattr(hcu_should_share, _SHARE_WRAPPER, True)
    setattr(eagle, "_vllm_hcu_original_load_eagle_model", original)
    setattr(eagle, "_vllm_hcu_original_should_share", original_should_share)
    setattr(eagle, "_should_share", hcu_should_share)
    setattr(eagle, "load_eagle_model", hcu_load_eagle_model)
    setattr(eagle, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
