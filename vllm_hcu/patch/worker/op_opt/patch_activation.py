# SPDX-License-Identifier: Apache-2.0
"""Enable the HCU clamp SwiGLU device operator on ROCm."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.layers.activation"
PATCH_ID = "worker.op_opt.activation.silu_and_mul_with_clamp"
TARGETS = (f"{TARGET_MODULE}.SiluAndMulWithClamp.__init__",)
_CLASS_MARKER = "_vllm_hcu_clamp_swiglu_applied"
_WRAPPER_MARKER = "_vllm_hcu_clamp_swiglu_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    activation = load_exact_module(TARGET_MODULE, module)
    op_class = require_class(
        activation,
        "SiluAndMulWithClamp",
        f"{TARGET_MODULE}.SiluAndMulWithClamp",
    )
    if already_applied(
        op_class,
        _CLASS_MARKER,
        ((op_class, "__init__", TARGETS[0], _WRAPPER_MARKER),),
    ):
        return False

    original = require_callable(op_class, "__init__", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("self", "swiglu_limit", "alpha", "beta"),
        keyword_only=("compile_native",),
        defaults={"alpha": 1.0, "beta": 0.0, "compile_native": True},
    )
    if len(op_class.__mro__) < 2:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has no base"
        )
    base_class = op_class.__mro__[1]
    if base_class.__name__ != "CustomOp":
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has unexpected base "
            f"{base_class.__name__}"
        )
    base_init = require_callable(
        base_class, "__init__", f"{base_class.__name__}.__init__"
    )
    require_exact_signature(
        base_init,
        f"{base_class.__name__}.__init__",
        positional=("self",),
        keyword_only=("enforce_enable", "compile_native"),
        defaults={"enforce_enable": False, "compile_native": False},
    )

    @functools.wraps(original)
    def hcu_init(
        self,
        swiglu_limit: float,
        alpha: float = 1.0,
        beta: float = 0.0,
        *,
        compile_native: bool = True,
    ):
        base_init(self, enforce_enable=True, compile_native=compile_native)
        self.swiglu_limit = float(swiglu_limit)
        self.alpha = float(alpha)
        self.beta = float(beta)
        platform = activation.current_platform
        if platform.is_rocm() or platform.is_cuda_alike() or platform.is_xpu():
            try:
                self.op = activation.torch.ops._C.silu_and_mul_with_clamp
            except AttributeError as exc:
                raise RuntimeError(
                    "HCU clamp SwiGLU is required, but "
                    "torch.ops._C.silu_and_mul_with_clamp is unavailable"
                ) from exc
        elif platform.is_cpu():
            self._forward_method = self.forward_native

    setattr(hcu_init, _WRAPPER_MARKER, True)
    setattr(op_class, "_vllm_hcu_original_init", original)
    setattr(op_class, "__init__", hcu_init)
    setattr(op_class, _CLASS_MARKER, True)
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
