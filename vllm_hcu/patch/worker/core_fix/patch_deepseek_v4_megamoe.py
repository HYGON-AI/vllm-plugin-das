# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Select the opt-in MegaMoE method before fallback kernel construction.

On HCU, ``--moe-backend deep_gemm_mega_moe`` selects the standalone megamoe
package instead of vLLM's NVIDIA DeepGEMM MegaMoE path. Other backends
retain their existing selection logic.
"""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_class,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.routed_experts"
PATCH_ID = "worker.core_fix.deepseek_v4.hcu_megamoe"
TARGET_SYMBOL = f"{TARGET_MODULE}.RoutedExperts._get_quant_method"
_MARKER = "_vllm_hcu_megamoe_wrapper"
MEGA_MOE_BACKEND = "deep_gemm_mega_moe"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    cls = require_class(target, "RoutedExperts", TARGET_SYMBOL)
    original = require_callable(cls, "_get_quant_method", TARGET_SYMBOL)
    if getattr(cls, _MARKER, False):
        if not getattr(original, _MARKER, False):
            raise PatchCompatibilityError(f"stale HCU MegaMoE marker on {TARGET_SYMBOL}")
        return False
    require_exact_signature(
        original, TARGET_SYMBOL,
        positional=("self", "prefix", "quant_config", "moe_config"),
    )

    @functools.wraps(original)
    def hcu_get_quant_method(self, prefix, quant_config, moe_config):
        if getattr(moe_config, "moe_backend", None) == MEGA_MOE_BACKEND:
            from vllm_hcu.model_executor.layers.fused_moe.megamoe_runtime import (
                select_method,
            )

            method = select_method(self, quant_config, moe_config)
            if method is not None:
                return method
            # vLLM's own deep_gemm_mega_moe path is NVIDIA-only.
            raise ValueError(
                f"moe_backend={MEGA_MOE_BACKEND!r} on HCU supports "
                "DeepSeek-V4 target/DSpark models only"
            )
        return original(self, prefix, quant_config, moe_config)

    setattr(hcu_get_quant_method, _MARKER, True)
    cls._get_quant_method = hcu_get_quant_method
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
