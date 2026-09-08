# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Identify Qwen hybrid MTP KV groups without classifying target Mamba groups."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from vllm.logger import init_logger

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
)


TARGET_MODULE = "vllm.v1.core.kv_cache_utils"
PATCH_ID = "platform.framework_opt.qwen4_exp_mtp_kv_cache_groups"
TARGETS = (
    f"{TARGET_MODULE}._annotate_eagle_groups",
    f"{TARGET_MODULE}.get_kv_cache_groups",
)
_MARKER = "_vllm_hcu_qwen4_exp_mtp_kv_groups_applied"
_WRAPPER = "_vllm_hcu_qwen4_exp_mtp_kv_groups_wrapper"
_GROUPS_WRAPPER = "_vllm_hcu_qwen4_exp_mtp_get_kv_groups_wrapper"
logger = init_logger(__name__)
_SUPPORTED_MODEL_TYPES = frozenset(
    {
        "qwen4_exp",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
)


def _is_qwen_hybrid_mtp(vllm_config: object) -> bool:
    spec_config = getattr(vllm_config, "speculative_config", None)
    use_block_drop = getattr(spec_config, "use_eagle_block_drop", None)
    if not callable(use_block_drop) or not use_block_drop():
        return False
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "model_type", None) in _SUPPORTED_MODEL_TYPES


def _is_mtp_layer(name: object) -> bool:
    return isinstance(name, str) and "mtp" in name.lower().split(".")


def _annotate_qwen_mtp_groups(vllm_config, kv_cache_groups) -> None:
    if not _is_qwen_hybrid_mtp(vllm_config):
        return

    # These Qwen architectures register their draft model below the stable
    # ``mtp`` prefix.  Mark every group containing such a layer and leave
    # target Mamba groups untouched so align-mode checkpoints remain reusable.
    for group in kv_cache_groups:
        if any(_is_mtp_layer(name) for name in group.layer_names):
            group.is_eagle_group = True


def apply_to_module(module: ModuleType) -> bool:
    kv_cache_utils = load_exact_module(TARGET_MODULE, module)
    if already_applied(
        kv_cache_utils,
        _MARKER,
        (
            (kv_cache_utils, "_annotate_eagle_groups", _WRAPPER),
            (kv_cache_utils, "get_kv_cache_groups", _GROUPS_WRAPPER),
        ),
    ):
        return False

    original = require_callable(
        kv_cache_utils, "_annotate_eagle_groups", TARGETS[0]
    )
    signature = inspect.signature(original)
    if (
        tuple(signature.parameters)
        != (
            "vllm_config",
            "kv_cache_spec",
            "kv_cache_groups",
            "use_deepseek_v4_fallback",
        )
        or signature.parameters["use_deepseek_v4_fallback"].default is not False
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible "
            f"signature {signature}"
        )
    original_get_groups = require_callable(
        kv_cache_utils, "get_kv_cache_groups", TARGETS[1]
    )
    get_groups_signature = inspect.signature(original_get_groups)
    if tuple(get_groups_signature.parameters) != ("vllm_config", "kv_cache_spec"):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[1]} has incompatible "
            f"signature {get_groups_signature}"
        )

    @functools.wraps(original)
    def hcu_annotate_eagle_groups(
        vllm_config,
        kv_cache_spec,
        kv_cache_groups,
        use_deepseek_v4_fallback=False,
    ) -> None:
        original(
            vllm_config,
            kv_cache_spec,
            kv_cache_groups,
            use_deepseek_v4_fallback,
        )
        _annotate_qwen_mtp_groups(vllm_config, kv_cache_groups)

    @functools.wraps(original_get_groups)
    def hcu_get_kv_cache_groups(vllm_config, kv_cache_spec):
        groups = original_get_groups(vllm_config, kv_cache_spec)
        _annotate_qwen_mtp_groups(vllm_config, groups)
        if _is_qwen_hybrid_mtp(vllm_config):
            logger.info(
                "Qwen hybrid MTP KV cache groups after Eagle annotation: %s",
                [
                    {
                        "index": index,
                        "layers": tuple(group.layer_names),
                        "spec": type(group.kv_cache_spec).__name__,
                        "is_eagle_group": group.is_eagle_group,
                    }
                    for index, group in enumerate(groups)
                ],
            )
        return groups

    setattr(hcu_annotate_eagle_groups, _WRAPPER, True)
    setattr(hcu_get_kv_cache_groups, _GROUPS_WRAPPER, True)
    setattr(kv_cache_utils, "_vllm_hcu_original_annotate_eagle_groups", original)
    setattr(
        kv_cache_utils,
        "_vllm_hcu_original_get_kv_cache_groups",
        original_get_groups,
    )
    setattr(kv_cache_utils, "_annotate_eagle_groups", hcu_annotate_eagle_groups)
    setattr(kv_cache_utils, "get_kv_cache_groups", hcu_get_kv_cache_groups)
    setattr(kv_cache_utils, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
