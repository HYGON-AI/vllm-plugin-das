# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Expose Channel-FP8 scale aliases to the DSV4.1 DSpark draft loader."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v41.amd.dspark"
PATCH_ID = "worker.core_fix.deepseek_v41_amd.dspark_load_weights_scale_remap"
TARGET_SYMBOL = f"{TARGET_MODULE}.DSparkDeepseekV4ForCausalLM.load_weights"
_CLASS_MARKER = "_vllm_hcu_dsv41_dspark_load_weights_applied"
_WRAPPER_MARKER = "_vllm_hcu_dsv41_dspark_load_weights_wrapper"


def _scale_alias(name: str) -> str | None:
    if name.endswith((".weight_scale", "_weight_scale")):
        return f"{name}_inv"
    if name.endswith((".weight_scale_inv", "_weight_scale_inv")):
        return name.removesuffix("_inv")
    return None


def apply_to_module(module: ModuleType) -> bool:
    dspark = load_exact_module(TARGET_MODULE, module)
    model_class = require_class(
        dspark,
        "DSparkDeepseekV4ForCausalLM",
        TARGET_SYMBOL.rsplit(".", 1)[0],
    )
    original = require_callable(model_class, "load_weights", TARGET_SYMBOL)
    if getattr(model_class, _CLASS_MARKER, False):
        current = vars(model_class).get("load_weights")
        if not getattr(current, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return False
    require_exact_signature(
        original,
        TARGET_SYMBOL,
        positional=("self", "weights"),
    )

    @functools.wraps(original)
    def hcu_load_weights(self, weights):
        instance_attributes = vars(self)
        had_instance_method = "named_parameters" in instance_attributes
        previous_instance_method = instance_attributes.get("named_parameters")
        original_named_parameters = self.named_parameters

        def named_parameters_with_scale_aliases(*args, **kwargs):
            for name, parameter in original_named_parameters(*args, **kwargs):
                yield name, parameter
                alias = _scale_alias(name)
                if alias is not None:
                    yield alias, parameter

        self.named_parameters = named_parameters_with_scale_aliases
        try:
            return original(self, weights)
        finally:
            if had_instance_method:
                self.named_parameters = previous_instance_method
            else:
                del self.named_parameters

    setattr(hcu_load_weights, _WRAPPER_MARKER, True)
    setattr(
        model_class,
        "_vllm_hcu_original_dsv41_dspark_load_weights",
        original,
    )
    setattr(model_class, "load_weights", hcu_load_weights)
    setattr(model_class, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGET_SYMBOL",
    "apply",
    "apply_to_module",
]
