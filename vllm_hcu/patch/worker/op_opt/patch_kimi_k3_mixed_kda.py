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
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )

    def projection(*args, **kwargs):
        config = kwargs.get("quant_config")
        if not is_mixed_slimquant(config):
            return original(*args, **kwargs)
        parent = kwargs["prefix"].removesuffix(".in_proj_qkvgfab")

        def uses_w8a8(branch):
            prefix = f"{parent}.{branch}"
            matching = dict(fused_mapping=config.packed_modules_mapping, use_fnmatch=True)
            return (not (config.ignore and should_ignore_layer(
                        prefix, ignore=config.ignore, **matching))
                    and should_ignore_layer(prefix, ignore=config.w8a8_include, **matching))

        # A mixed-precision declaration can select only MLPs or other layers.
        # Preserve the native merged layout when all three Q/K/V stay floating.
        # Partial Q/K/V quantization still reaches the split constructor's
        # explicit validation rather than silently loading INT8 into BF16.
        if not any(uses_w8a8(branch) for branch in ("q_proj", "k_proj", "v_proj")):
            return original(*args, **kwargs)
        return KimiK3MixedKDAProjection(
            *args, column_linear=target.ColumnParallelLinear,
            merged_linear=original, **kwargs)

    target._KimiGDNMergedColumnParallelLinear = projection
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
