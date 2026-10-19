# SPDX-License-Identifier: Apache-2.0
"""Split only K3 SlimQuant mixed KDA projections at construction."""
from types import ModuleType
from ._common import load_exact_module, require_callable

TARGET_MODULE = "vllm.models.kimi_k3.amd.kda"
PATCH_ID = "worker.op_opt.kimi_k3.mixed_kda"
TARGETS = (f"{TARGET_MODULE}._KimiGDNMergedColumnParallelLinear",)
_MARKER = "_hcu_kimi_mixed_kda"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    original = require_callable(target, "_KimiGDNMergedColumnParallelLinear", TARGETS[0])
    from vllm_hcu.model_executor.layers.kimi_k3_mixed_precision import (
        KimiK3MixedKDAProjection, is_mixed_slimquant,
    )

    def projection(*args, **kwargs):
        if not is_mixed_slimquant(kwargs.get("quant_config")):
            return original(*args, **kwargs)
        return KimiK3MixedKDAProjection(
            *args, column_linear=target.ColumnParallelLinear,
            merged_linear=original, **kwargs)

    target._KimiGDNMergedColumnParallelLinear = projection
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
