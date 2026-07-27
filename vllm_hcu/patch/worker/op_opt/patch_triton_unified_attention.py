# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Force the HCU Triton unified-attention launch to a single pipeline stage."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType
from typing import Any

from ._common import PatchCompatibilityError, load_exact_module, require_callable

TARGET_MODULE = "vllm.v1.attention.ops.triton_unified_attention"
PATCH_ID = "worker.op_opt.attention.triton_unified_single_stage"
TARGETS = (
    f"{TARGET_MODULE}.unified_attention",
    f"{TARGET_MODULE}.kernel_unified_attention",
)
_MARKER = "_vllm_hcu_triton_unified_single_stage_applied"
_PROXY_MARKER = "_vllm_hcu_single_stage_kernel_proxy"
_EXPECTED_UNIFIED_PARAMETERS = (
    "q",
    "k",
    "v",
    "out",
    "cu_seqlens_q",
    "max_seqlen_q",
    "seqused_k",
    "max_seqlen_k",
    "softmax_scale",
    "causal",
    "window_size",
    "block_table",
    "softcap",
    "q_descale",
    "k_descale",
    "v_descale",
    "seq_threshold_3D",
    "num_par_softmax_segments",
    "softmax_segm_output",
    "softmax_segm_max",
    "softmax_segm_expsum",
    "alibi_slopes",
    "output_scale",
    "qq_bias",
    "sinks",
    "mm_prefix_range",
    "rswa_prefix_lens",
    "rswa_window",
    "use_alibi_sqrt",
    "kv_quant_mode",
    "k_scale_cache",
    "v_scale_cache",
    "chunk_lookback",
    "use_td",
    "mm_prefix_clamp_sliding_window",
)


class _SingleStageKernelProxy:
    """Preserve Triton's JIT object surface while narrowing launch kwargs."""

    _vllm_hcu_single_stage_kernel_proxy = True

    def __init__(self, kernel: object) -> None:
        self._kernel = kernel

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kernel, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._kernel(*args, **kwargs)  # type: ignore[operator]

    def __getitem__(self, grid: object):
        launcher = self._kernel[grid]  # type: ignore[index]

        @functools.wraps(launcher)
        def launch(*args: Any, **kwargs: Any) -> Any:
            # HCU code generation is not qualified for multi-stage pipelining.
            # Override target tuning (including the B200-only stage=2 branch)
            # while preserving every other target-owned launch argument.
            kwargs["num_stages"] = 1
            return launcher(*args, **kwargs)

        return launch


def apply_to_module(module: ModuleType) -> bool:
    unified_module = load_exact_module(TARGET_MODULE, module)
    kernel = getattr(unified_module, "kernel_unified_attention", None)
    if getattr(unified_module, _MARKER, False):
        if not getattr(kernel, _PROXY_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[1]} is stale; restart the process"
            )
        return False

    unified = require_callable(unified_module, "unified_attention", TARGETS[0])
    signature = inspect.signature(unified)
    if tuple(signature.parameters) != _EXPECTED_UNIFIED_PARAMETERS:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible "
            f"signature {signature}"
        )
    if kernel is None or not callable(getattr(kernel, "__getitem__", None)):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[1]} is missing Triton launch semantics"
        )

    setattr(unified_module, "_vllm_hcu_original_kernel_unified_attention", kernel)
    setattr(unified_module, "kernel_unified_attention", _SingleStageKernelProxy(kernel))
    setattr(unified_module, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
