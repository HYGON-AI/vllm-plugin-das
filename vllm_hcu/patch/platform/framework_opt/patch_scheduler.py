# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Align Qwen hybrid MTP scheduling with the upstream vLLM scheduler."""

from __future__ import annotations

from functools import wraps
from inspect import signature
from types import ModuleType

from vllm.logger import init_logger

from ._common import (
    load_exact_module,
    require_callable,
    require_class,
    require_signature_prefix,
)
from .patch_qwen4_exp_mtp_kv_cache_groups import (
    _is_mtp_layer,
    _is_qwen_hybrid_mtp,
)

TARGET_MODULE = "vllm.v1.core.sched.scheduler"
PATCH_ID = "platform.framework_opt.hcu_scheduler"
logger = init_logger(__name__)
TARGETS = (
    f"{TARGET_MODULE}.Scheduler",
    f"{TARGET_MODULE}.Scheduler.update_draft_token_ids",
    f"{TARGET_MODULE}.Scheduler.update_draft_token_ids_in_output",
    f"{TARGET_MODULE}.Scheduler.__init__",
)
_MARKER = "_vllm_hcu_scheduler_contract_validated"


def _mark_qwen_hybrid_mtp_groups(
    vllm_config: object, kv_cache_config: object
) -> None:
    if not _is_qwen_hybrid_mtp(vllm_config):
        return

    marked_group_ids = []
    for index, group in enumerate(
        getattr(kv_cache_config, "kv_cache_groups", ())
    ):
        layer_names = getattr(group, "layer_names", ())
        if any(_is_mtp_layer(layer_name) for layer_name in layer_names):
            group.is_eagle_group = True
            marked_group_ids.append(index)
    if marked_group_ids:
        logger.info(
            "Marked Qwen hybrid MTP KV cache groups as Eagle at scheduler "
            "boundary: %s",
            marked_group_ids,
        )


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    scheduler = require_class(target, "Scheduler", TARGETS[0])
    if getattr(target, _MARKER, False):
        return False

    require_signature_prefix(
        require_callable(scheduler, "update_draft_token_ids", TARGETS[1]),
        TARGETS[1],
        ("self", "draft_token_ids"),
    )
    require_signature_prefix(
        require_callable(
            scheduler,
            "update_draft_token_ids_in_output",
            TARGETS[2],
        ),
        TARGETS[2],
        ("self", "draft_token_ids", "scheduler_output"),
    )
    scheduler_init = require_callable(scheduler, "__init__", TARGETS[3])
    require_signature_prefix(
        scheduler_init,
        TARGETS[3],
        (
            "self",
            "vllm_config",
            "kv_cache_config",
            "structured_output_manager",
            "block_size",
        ),
    )
    init_signature = signature(scheduler_init)
    @wraps(scheduler_init)
    def scheduler_init_with_qwen_mtp_groups(self, *args, **kwargs):
        bound = init_signature.bind(self, *args, **kwargs)
        _mark_qwen_hybrid_mtp_groups(
            bound.arguments["vllm_config"],
            bound.arguments["kv_cache_config"],
        )
        return scheduler_init(self, *args, **kwargs)

    scheduler.__init__ = scheduler_init_with_qwen_mtp_groups
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
