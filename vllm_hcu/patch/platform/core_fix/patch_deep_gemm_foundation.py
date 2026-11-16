# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Arm shared DeepGEMM exports before cold MoE consumers bind their aliases."""

from __future__ import annotations
from types import ModuleType
from vllm_hcu.runtime_compat.step5_patch_contract import (
    already_applied,
    remember_targets,
)
from ._common import load_exact_module

TARGET_MODULE = "vllm.utils.deep_gemm"
PATCH_ID = "platform.core_fix.deep_gemm_foundation"
TARGETS = (
    f"{TARGET_MODULE}._import_deep_gemm",
    f"{TARGET_MODULE}.get_mk_alignment_for_contiguous_layout",
    f"{TARGET_MODULE}.set_mk_alignment_for_contiguous_layout",
    f"{TARGET_MODULE}.m_grouped_fp8_gemm_nt_contiguous",
    f"{TARGET_MODULE}.m_grouped_fp8_fp4_gemm_nt_contiguous",
    f"{TARGET_MODULE}.fp8_m_grouped_gemm_nt_masked",
    f"{TARGET_MODULE}.transform_sf_into_required_layout",
)
_MARKER = "_vllm_hcu_deep_gemm_foundation_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    # The same adapters are reused by Worker dispatch. Their callable identity
    # markers make the later registration idempotent, rather than wrapping twice.
    from vllm_hcu.patch.worker.framework_opt import patch_step5_deep_gemm_api
    from vllm_hcu.patch.worker.op_opt import patch_deep_gemm_transform

    patch_step5_deep_gemm_api.apply_to_module(target)
    patch_deep_gemm_transform.apply_to_module(target)
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
