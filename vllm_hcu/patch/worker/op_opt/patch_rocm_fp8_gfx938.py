# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Allow PerTensor torch FP8 scaled-mm on gfx938 without mutating vLLM sources.

The shared ``_rocm_torch_fp8_scaled_mm_supported`` predicate also gates
RowWise. Leaving it unchanged keeps gfx938 channel-FP8 on the existing HCU
ChannelWise kernel.
"""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.kernels.linear.scaled_mm.pytorch"
PATCH_ID = "worker.op_opt.scaled_mm.rocm_gfx938_torch_fp8"
_KERNEL_CLASS = "PerTensorTorchFP8ScaledMMLinearKernel"
TARGETS = (f"{TARGET_MODULE}.{_KERNEL_CLASS}.is_supported",)
_MODULE_MARKER = "_vllm_hcu_rocm_gfx938_torch_fp8_applied"
_WRAPPER_MARKER = "_vllm_hcu_rocm_gfx938_torch_fp8_wrapper"


def _is_gfx938() -> bool:
    try:
        from vllm.platforms.rocm import _GCN_ARCH
    except (AttributeError, ImportError):
        return False
    return "gfx938" in _GCN_ARCH


def apply_to_module(module: ModuleType) -> bool:
    pytorch_scaled_mm = load_exact_module(TARGET_MODULE, module)
    kernel_cls = require_class(pytorch_scaled_mm, _KERNEL_CLASS, TARGETS[0])
    if already_applied(
        kernel_cls,
        _MODULE_MARKER,
        ((kernel_cls, "is_supported", TARGETS[0], _WRAPPER_MARKER),),
    ):
        return False

    original = require_callable(kernel_cls, "is_supported", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("compute_capability",),
        defaults={"compute_capability": None},
    )

    @functools.wraps(original)
    def hcu_per_tensor_is_supported(cls, compute_capability=None):
        del cls
        supported, reason = original(compute_capability)
        if supported or not _is_gfx938():
            return supported, reason
        return True, None

    setattr(hcu_per_tensor_is_supported, _WRAPPER_MARKER, True)
    setattr(kernel_cls, "_vllm_hcu_original_is_supported", original)
    setattr(kernel_cls, "is_supported", classmethod(hcu_per_tensor_is_supported))
    setattr(kernel_cls, _MODULE_MARKER, True)
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
