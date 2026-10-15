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

TARGET_MODULE = "vllm.config.compilation"
PATCH_ID = "platform.core_fix.step5_splitting_ops"
TARGETS = ("vllm.config.compilation.CompilationConfig.set_splitting_ops_for_v1",)
_MARKER = "_vllm_hcu_step5_splitting_ops_applied"

SIGNATURES = (
    (
        "vllm.config.compilation.CompilationConfig.set_splitting_ops_for_v1",
        ("self", "all2all_backend", "data_parallel_size"),
    ),
)
REVIEWED_SOURCES = ()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    import functools
    import os

    cls = getattr(target, "CompilationConfig", None)
    original = require_callable(cls, "set_splitting_ops_for_v1", TARGETS[0])

    @functools.wraps(original)
    def set_splitting_ops_for_v1(self, *args, **kwargs):
        automatic = self.splitting_ops is None
        result = original(self, *args, **kwargs)
        if automatic and self.splitting_ops:
            for flag, op in (
                ("VLLM_HCU_STEP5_FUSED_GATE", "vllm::step5_swa_attention_gate_quant"),
                ("VLLM_HCU_STEP5_SWA_FUSED_QK", "vllm::step5_swa_qk_norm_rope_cache"),
            ):
                if os.environ.get(flag, "0") == "1" and op not in self.splitting_ops:
                    self.splitting_ops.append(op)
        return result

    cls.set_splitting_ops_for_v1 = set_splitting_ops_for_v1
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
