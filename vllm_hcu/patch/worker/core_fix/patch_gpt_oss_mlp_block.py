# SPDX-License-Identifier: Apache-2.0
"""Adapt GPT-OSS router GEMM to HCU's optional NN weight layout."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.models.gpt_oss"
PATCH_ID = "worker.core_fix.gpt_oss.rocm_unquantized_gemm_layout"
TARGET_SYMBOL = f"{TARGET_MODULE}.rocm_unquantized_gemm"
_MODULE_MARKER = "_vllm_hcu_gpt_oss_gemm_adapter_applied"
_WRAPPER_MARKER = "_vllm_hcu_gpt_oss_gemm_adapter"


def _use_nn_layout() -> bool:
    try:
        from vllm_hcu.platforms import envs as henvs

        return bool(henvs.VLLM_USE_NN)
    except (AttributeError, ImportError) as exc:
        raise PatchCompatibilityError(
            "required HCU feature flag VLLM_USE_NN is unavailable"
        ) from exc


def apply_to_module(module: ModuleType) -> bool:
    """Replace only the GPT-OSS module binding, not ``MLPBlock.forward``."""

    gpt_oss = load_exact_module(TARGET_MODULE, module)
    current = require_callable(gpt_oss, "rocm_unquantized_gemm", TARGET_SYMBOL)
    if getattr(gpt_oss, _MODULE_MARKER, False):
        if not getattr(current, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return False

    require_exact_signature(
        current,
        TARGET_SYMBOL,
        positional=("layer", "x", "weight", "bias"),
        defaults={"bias": None},
    )
    try:
        linear = gpt_oss.torch.nn.functional.linear
    except AttributeError as exc:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.torch.nn.functional.linear "
            "is missing"
        ) from exc
    if not callable(linear):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.torch.nn.functional.linear "
            "is missing"
        )
    original = current

    @functools.wraps(original)
    def hcu_rocm_unquantized_gemm(layer, x, weight, bias=None):
        if _use_nn_layout():
            transpose = getattr(weight, "t", None)
            if not callable(transpose):
                raise PatchCompatibilityError(
                    f"{TARGET_SYMBOL} received an NN-layout weight without t()"
                )
            # NN layout stores unquantized weights as [in_features,
            # out_features].  The upstream ROCm custom op and its tuned GEMM
            # branches expect a contiguous [out_features, in_features]
            # parameter, so do not feed the transposed view back into that op.
            return linear(x, transpose(), bias)
        return original(layer, x, weight, bias)

    setattr(hcu_rocm_unquantized_gemm, _WRAPPER_MARKER, True)
    setattr(gpt_oss, "_vllm_hcu_original_rocm_unquantized_gemm", original)
    setattr(gpt_oss, "rocm_unquantized_gemm", hcu_rocm_unquantized_gemm)
    setattr(gpt_oss, _MODULE_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    """Explicit direct entry point for tests and preloaded modules."""

    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
