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

TARGET_MODULE = "vllm.utils.deep_gemm"
PATCH_ID = "worker.framework_opt.step5_deep_gemm_api"
TARGETS = (
    "vllm.utils.deep_gemm._import_deep_gemm",
    "vllm.utils.deep_gemm._lazy_init",
    "vllm.utils.deep_gemm.get_mk_alignment_for_contiguous_layout",
    "vllm.utils.deep_gemm.set_mk_alignment_for_contiguous_layout",
    "vllm.utils.deep_gemm.m_grouped_fp8_gemm_nt_contiguous",
    "vllm.utils.deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous",
    "vllm.utils.deep_gemm.fp8_m_grouped_gemm_nt_masked",
)
_MARKER = "_vllm_hcu_step5_deep_gemm_api_applied"

SIGNATURES = (
    ("vllm.utils.deep_gemm._import_deep_gemm", ()),
    ("vllm.utils.deep_gemm._lazy_init", ()),
    ("vllm.utils.deep_gemm.get_mk_alignment_for_contiguous_layout", ()),
    ("vllm.utils.deep_gemm.set_mk_alignment_for_contiguous_layout", ("value",)),
    ("vllm.utils.deep_gemm.m_grouped_fp8_gemm_nt_contiguous", ("args", "kwargs")),
    ("vllm.utils.deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous", ("args", "kwargs")),
    ("vllm.utils.deep_gemm.fp8_m_grouped_gemm_nt_masked", ("args", "kwargs")),
)
REVIEWED_SOURCES = ()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    for symbol in TARGETS:
        require_callable(target, symbol.rsplit(".", 1)[1], symbol)
    from vllm_hcu.model_executor.layers.fused_moe.deep_gemm_api_runtime import (
        install_api,
    )

    install_api(target)
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
