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

TARGET_MODULE = "vllm.v1.worker.gpu.model_states.default"
PATCH_ID = "worker.framework_opt.step5_model_state"
TARGETS = ("vllm.v1.worker.gpu.model_states.default.DefaultModelState.prepare_attn",)
_MARKER = "_vllm_hcu_step5_model_state_applied"

SIGNATURES = (
    (
        "vllm.v1.worker.gpu.model_states.default.DefaultModelState.prepare_attn",
        (
            "self",
            "input_batch",
            "cudagraph_mode",
            "block_tables",
            "slot_mappings",
            "attn_groups",
            "kv_cache_config",
            "for_capture",
            "ubatch_idx",
        ),
    ),
)
REVIEWED_SOURCES = (
    (
        "vllm.v1.worker.gpu.model_states.default.DefaultModelState.prepare_attn",
        (
            "10d3722dbbed9652003fbc634b5183d344d5b9464bef8e2bcf35a615cae324c6",
            "81017edcd386afc0200784e14e376deecaea6b9e623c0d035b112ca8dd46dde9",
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

    cls = getattr(target, "DefaultModelState", None)
    original = require_callable(cls, "prepare_attn", TARGETS[0])

    @functools.wraps(original)
    def prepare_attn(self, *args, **kwargs):
        if (
            is_step5_config(self.vllm_config)
            and self.vllm_config.parallel_config.prefill_context_parallel_size == 1
        ):
            from vllm_hcu.v1.step5_metadata import prepare_step5_attn

            return prepare_step5_attn(self, *args, **kwargs)
        return original(self, *args, **kwargs)

    cls.prepare_attn = prepare_attn
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
