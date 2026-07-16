# SPDX-License-Identifier: Apache-2.0
"""Provide the audited TF32 MHC pre-norm fallback without source mutation."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.utils.deep_gemm"
PATCH_ID = "worker.op_opt.deep_gemm.tf32_hc_prenorm_fallback"
TARGETS = (f"{TARGET_MODULE}.tf32_hc_prenorm_gemm",)
_MODULE_MARKER = "_vllm_hcu_tf32_prenorm_fallback_applied"
_WRAPPER_MARKER = "_vllm_hcu_tf32_prenorm_fallback_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    deep_gemm = load_exact_module(TARGET_MODULE, module)
    if already_applied(
        deep_gemm,
        _MODULE_MARKER,
        ((deep_gemm, "tf32_hc_prenorm_gemm", TARGETS[0], _WRAPPER_MARKER),),
    ):
        return False

    original = require_callable(deep_gemm, "tf32_hc_prenorm_gemm", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("x", "fn", "out", "sqrsum", "num_split"),
    )
    lazy_init = require_callable(deep_gemm, "_lazy_init", f"{TARGET_MODULE}._lazy_init")
    require_exact_signature(lazy_init, f"{TARGET_MODULE}._lazy_init")

    @functools.wraps(original)
    def hcu_tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_split):
        lazy_init()
        if getattr(deep_gemm, "_tf32_hc_prenorm_gemm_impl", None) is not None:
            return original(x, fn, out, sqrsum, num_split)

        if (
            x.ndim != 2
            or fn.ndim != 2
            or out.ndim != 3
            or sqrsum.ndim != 2
            or x.shape[1] != fn.shape[1]
            or out.shape != (num_split, x.shape[0], fn.shape[0])
            or sqrsum.shape != (num_split, x.shape[0])
            or num_split < 1
        ):
            raise ValueError(
                "HCU tf32_hc_prenorm_gemm fallback received incompatible shapes: "
                f"x={tuple(x.shape)}, fn={tuple(fn.shape)}, out={tuple(out.shape)}, "
                f"sqrsum={tuple(sqrsum.shape)}, num_split={num_split}"
            )
        out.zero_()
        sqrsum.zero_()
        out[0].copy_(deep_gemm.torch.matmul(x.float(), fn.t().float()))
        sqrsum[0].copy_(x.float().square().sum(dim=-1))
        return out

    setattr(hcu_tf32_hc_prenorm_gemm, _WRAPPER_MARKER, True)
    setattr(deep_gemm, "_vllm_hcu_original_tf32_hc_prenorm_gemm", original)
    setattr(deep_gemm, "tf32_hc_prenorm_gemm", hcu_tf32_hc_prenorm_gemm)
    setattr(deep_gemm, _MODULE_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
