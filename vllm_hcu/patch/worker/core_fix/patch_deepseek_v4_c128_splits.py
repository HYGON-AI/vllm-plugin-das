# SPDX-License-Identifier: Apache-2.0
"""Tune C128 head tiles while retaining the upstream compressor kernels."""
from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError, load_exact_module, require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache"
PATCH_ID = "worker.core_fix.deepseek_v4.c128_head_splits"
TARGET_SYMBOL = f"{TARGET_MODULE}._pick_compress_num_splits"
_MARKER = "_vllm_hcu_c128_head_splits_applied"
_WRAPPER_MARKER = "_vllm_hcu_c128_head_splits_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    original = require_callable(target, "_pick_compress_num_splits", TARGET_SYMBOL)
    if getattr(target, _MARKER, False):
        if not getattr(original, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(f"stale HCU patch marker for {TARGET_SYMBOL}")
        return False
    require_exact_signature(original, TARGET_SYMBOL,
                            positional=("num_actual", "compress_ratio", "head_dim"))
    from vllm.logger import init_logger
    from vllm.platforms import current_platform
    from vllm_hcu.platforms import envs as henvs
    logger = init_logger("vllm.hcu.c128")

    @functools.wraps(original)
    def pick_splits(num_actual: int, compress_ratio: int, head_dim: int) -> int:
        if (not current_platform.is_rocm() or compress_ratio != 128
                or head_dim != 512 or num_actual < 4096
                or not henvs.custom_ops_enabled()):
            return original(num_actual, compress_ratio, head_dim)
        splits = henvs.VLLM_HCU_C128_HEAD_SPLITS
        if splits not in (0, 1, 2, 4, 8, 16):
            raise ValueError("VLLM_HCU_C128_HEAD_SPLITS must be 0, 1, 2, 4, 8 or 16")
        if splits == 0:
            return original(num_actual, compress_ratio, head_dim)
        logger.info_once("C128 HCU prefill uses %d head splits (head_tile=%d).",
                         splits, head_dim // splits)
        return splits

    setattr(pick_splits, _WRAPPER_MARKER, True)
    target._vllm_hcu_original_pick_compress_num_splits = original
    target._pick_compress_num_splits = pick_splits
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
