# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep Kimi vision patch projection usable without AITER's Triton Conv2d."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.models.kimi_k25_vit"
PATCH_ID = "worker.core_fix.kimi_k25.vision_conv"
TARGET_SYMBOL = f"{TARGET_MODULE}.MoonVision3dPatchEmbed._proj"
_CONV_MODULE = "aiter.ops.triton.conv.conv2d"
_MISSING_CONV_MODULES = frozenset(("aiter.ops.triton.conv", _CONV_MODULE))
_MARKER = "_vllm_hcu_kimi_k25_vision_conv_applied"
_WRAPPER_MARKER = "_vllm_hcu_kimi_k25_vision_conv_wrapper"
_ORIGINAL = "_vllm_hcu_original_kimi_k25_vision_proj"


def _load_aiter_conv2d():
    from aiter.ops.triton.conv.conv2d import conv2d

    return conv2d


def apply_to_module(module: ModuleType) -> bool:
    """Wrap only the audited patch projection, without rewriting vLLM source."""

    vision = load_exact_module(TARGET_MODULE, module)
    patch_embed = require_class(
        vision,
        "MoonVision3dPatchEmbed",
        f"{TARGET_MODULE}.MoonVision3dPatchEmbed",
    )
    current = require_callable(patch_embed, "_proj", TARGET_SYMBOL)
    if getattr(patch_embed, _MARKER, False):
        stored_original = getattr(patch_embed, _ORIGINAL, None)
        if (
            not callable(stored_original)
            or not getattr(current, _WRAPPER_MARKER, False)
            or getattr(current, "__wrapped__", None) is not stored_original
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return False
    if getattr(current, _WRAPPER_MARKER, False) or hasattr(patch_embed, _ORIGINAL):
        raise PatchCompatibilityError(
            f"required HCU patch marker for {TARGET_SYMBOL} is incomplete"
        )

    require_exact_signature(current, TARGET_SYMBOL, positional=("self", "x"))
    platform = getattr(vision, "current_platform", None)
    is_rocm = require_callable(
        platform, "is_rocm", f"{TARGET_MODULE}.current_platform.is_rocm"
    )
    torch = getattr(vision, "torch", None)
    if not all(hasattr(torch, name) for name in ("float16", "bfloat16")):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.torch half dtypes are missing"
        )
    half_dtypes = (torch.float16, torch.bfloat16)
    logger = getattr(vision, "logger", None)
    warning_once = require_callable(
        logger, "warning_once", f"{TARGET_MODULE}.logger.warning_once"
    )
    assume_constant_result = require_callable(
        getattr(torch, "compiler", None),
        "assume_constant_result",
        f"{TARGET_MODULE}.torch.compiler.assume_constant_result",
    )
    original = current

    @assume_constant_result
    def has_aiter_conv2d() -> bool:
        # Operator packages are fixed for a serving process. Keep imports and
        # diagnostics outside captured graphs; restart after changing packages.
        try:
            _load_aiter_conv2d()
        except ModuleNotFoundError as exc:
            # A missing nested dependency is not an absent Conv2d capability.
            if exc.name not in _MISSING_CONV_MODULES:
                raise
            warning_once(
                "AITER Triton Conv2d is unavailable; using the PyTorch "
                "vision patch projection."
            )
            return False
        return True

    @functools.wraps(original)
    def hcu_vision_proj(self, x):
        if is_rocm() and x.dtype in half_dtypes and not has_aiter_conv2d():
            return self.proj(x)
        # Keep upstream behavior, including native AITER/kernel errors, intact.
        return original(self, x)

    setattr(hcu_vision_proj, _WRAPPER_MARKER, True)
    setattr(patch_embed, _ORIGINAL, original)
    patch_embed._proj = hcu_vision_proj
    setattr(patch_embed, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    """Explicit direct entry point for tests and preloaded modules."""

    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
