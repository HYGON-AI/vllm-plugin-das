# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Bridge MiniMax-M3's legacy RGB argument after Transformers consumes it."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    apply_once,
    load_exact_module,
    require_callable,
)

TARGET_MODULE = "vllm.transformers_utils.processors.minimax_m3"
PATCH_ID = "platform.core_fix.minimax_m3.video_processor_rgb"
TARGETS = (f"{TARGET_MODULE}.MiniMaxM3VLVideoProcessor._preprocess",)
_MARKER = "_vllm_hcu_minimax_m3_video_processor_rgb_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False

    processor = getattr(target, "MiniMaxM3VLVideoProcessor", None)
    if not isinstance(processor, type):
        raise PatchCompatibilityError(f"required HCU patch target {TARGETS[0]} is missing")
    original = require_callable(processor, "_preprocess", TARGETS[0])
    signature = inspect.signature(original)
    rgb = signature.parameters.get("do_convert_rgb")
    if rgb is None or rgb.default is not inspect.Parameter.empty:
        # Upstream or a deployment may already have removed/defaulted the
        # unused argument. Keep that implementation exactly as installed.
        setattr(target, _MARKER, True)
        return False
    if rgb.kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible signature {signature}"
        )

    @functools.wraps(original)
    def preprocess(self, *args, **kwargs):
        bound = signature.bind_partial(self, *args, **kwargs)
        if "do_convert_rgb" not in bound.arguments:
            # Transformers already converted RGB in _prepare_input_videos and
            # popped this keyword before calling _preprocess. The legacy M3
            # implementation accepts but does not use it; do not convert twice.
            kwargs["do_convert_rgb"] = self.do_convert_rgb
        return original(self, *args, **kwargs)

    processor._preprocess = preprocess
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=target,
        marker=_MARKER,
        callback=lambda: apply_to_module(target),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
