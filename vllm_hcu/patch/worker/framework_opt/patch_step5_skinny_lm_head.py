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

TARGET_MODULE = "vllm.model_executor.layers.logits_processor"
PATCH_ID = "worker.framework_opt.step5_skinny_lm_head"
TARGETS = ("vllm.model_executor.layers.logits_processor.LogitsProcessor._apply_head",)
_MARKER = "_vllm_hcu_step5_skinny_lm_head_applied"

SIGNATURES = (
    (
        "vllm.model_executor.layers.logits_processor.LogitsProcessor._apply_head",
        ("self", "lm_head", "hidden_states", "embedding_bias"),
    ),
)
REVIEWED_SOURCES = ()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    import functools

    cls = getattr(target, "LogitsProcessor", None)
    original = require_callable(cls, "_apply_head", TARGETS[0])

    @functools.wraps(original)
    def apply_head(self, lm_head, hidden_states, embedding_bias=None):
        # Only processors created by the Step5 models opt into this kernel.
        if getattr(self, "_step5_skinny_lm_head", False) and (
            self.head_dtype is None or self.head_dtype == hidden_states.dtype
        ):
            from vllm_hcu.model_executor.layers import skinny_lm_head

            if (
                embedding_bias is None
                and type(lm_head.quant_method)
                in (target.UnquantizedEmbeddingMethod, target.UnquantizedLinearMethod)
                and skinny_lm_head.usable(hidden_states, lm_head.weight)
            ):
                target.logger.info_once(
                    "lm_head: using the skinny Triton GEMM for M <= %d",
                    skinny_lm_head.MAX_M,
                )
                return skinny_lm_head.skinny_linear(hidden_states, lm_head.weight)
        return original(self, lm_head, hidden_states, embedding_bias)

    cls._apply_head = apply_head
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
