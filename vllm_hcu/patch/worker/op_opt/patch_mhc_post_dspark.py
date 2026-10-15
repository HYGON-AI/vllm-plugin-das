# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route the DSV4.1 draft model's trailing mHC Post through AITER."""

from __future__ import annotations

import functools
import os
from types import ModuleType

import torch

from ..core_fix import patch_mhc_backend as backend
from ..core_fix._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v41.amd.dspark"
PATCH_ID = "worker.op_opt.dspark_mhc_post"
_MARKER = "_vllm_hcu_dspark_mhc_post_applied"
_ORIGINAL = "_vllm_hcu_original_mhc_post_torch"


def apply_to_module(module: ModuleType) -> bool:
    draft = load_exact_module(TARGET_MODULE, module)
    if getattr(draft, _MARKER, False):
        if not callable(getattr(draft, _ORIGINAL, None)):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_MODULE} is stale"
            )
        return False
    original = require_callable(
        draft, "mhc_post_torch", f"{TARGET_MODULE}.mhc_post_torch"
    )
    require_exact_signature(
        original,
        f"{TARGET_MODULE}.mhc_post_torch",
        positional=("x", "residual", "post_layer_mix", "comb_res_mix"),
    )
    post_available = backend._aiter_mhc_post_runtime_available()

    @functools.wraps(original)
    def mhc_post(
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        if (
            post_available
            and backend._aiter_mhc_post_enabled()
            and backend._aiter_mhc_post_supported(
                x, residual, post_layer_mix, comb_res_mix
            )
        ):
            return backend._call_aiter_mhc_post(
                x, residual, post_layer_mix, comb_res_mix
            )
        return original(x, residual, post_layer_mix, comb_res_mix)

    if post_available and backend._aiter_mhc_post_enabled():
        os.environ.setdefault("AITER_MHC_POST_KERNEL", "v3")
    setattr(draft, _ORIGINAL, original)
    draft.mhc_post_torch = mhc_post
    setattr(draft, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
