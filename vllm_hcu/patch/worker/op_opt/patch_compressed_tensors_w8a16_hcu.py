# SPDX-License-Identifier: Apache-2.0
# Ported from vllm-hcu-main 70f5373: route compressed-tensors pack-quantized
# int8 (channelwise, symmetric, weight-only) layers on HCU to the vllm-hcu
# W8A16 GEMM path instead of the marlin/exllama MP-linear kernel selection
# (exllama crashes with negative tensor dims on this config).

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors"
)
PATCH_ID = "worker.op_opt.compressed_tensors.w8a16_hcu"
TARGETS = (f"{TARGET_MODULE}.CompressedTensorsConfig._get_scheme_from_parts",)
_MARKER = "_vllm_hcu_w8a16_patched"
_WRAPPER = "_vllm_hcu_w8a16_scheme_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    ct = load_exact_module(TARGET_MODULE, module)
    config_cls = getattr(ct, "CompressedTensorsConfig", None)
    if config_cls is None:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.CompressedTensorsConfig "
            "is missing"
        )
    if getattr(ct, _MARKER, False):
        if not getattr(
            config_cls._get_scheme_from_parts, _WRAPPER, False
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale; "
                "restart the process"
            )
        return False
    original = require_callable(
        config_cls, "_get_scheme_from_parts", TARGETS[0]
    )
    require_exact_signature(
        original,
        TARGETS[0],
        positional=(
            "self",
            "weight_quant",
            "input_quant",
            "output_quant",
            "format",
            "layer_name",
        ),
        defaults={
            "output_quant": None,
            "format": None,
            "layer_name": None,
        },
    )
    original_signature = inspect.signature(original)

    from vllm_hcu.model_executor.layers.quantization.compressed_tensors_w8a16_hcu import (
        CompressedTensorsW8A16Hcu,
        is_hcu_w8a16_int8_channel,
    )

    @functools.wraps(original)
    def _get_scheme_from_parts(self, weight_quant, input_quant,
                               output_quant=None, format=None,
                               layer_name=None):
        fmt = format if format is not None else self.quant_format
        if is_hcu_w8a16_int8_channel(weight_quant, input_quant, fmt):
            return CompressedTensorsW8A16Hcu(layer_name=layer_name)
        return original(self, weight_quant, input_quant,
                        output_quant, format, layer_name)

    setattr(_get_scheme_from_parts, _WRAPPER, True)
    config_cls._get_scheme_from_parts = _get_scheme_from_parts
    setattr(ct, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
