# SPDX-License-Identifier: Apache-2.0
"""Register SlimQuant through vLLM's public out-of-tree quant registry."""

from __future__ import annotations

from types import ModuleType

from ._common import PatchCompatibilityError, apply_once, load_exact_module

TARGET_MODULE = "vllm.model_executor.layers.quantization"
PATCH_ID = "platform.core_fix.hcu_config.slimquant_registry"
_SLIMQUANT_NAMES = (
    "slimquant_marlin",
    "slimquant_compressed_tensors_marlin",
    "slimquant_w4a8",
)
TARGETS = tuple(
    f"{TARGET_MODULE}.register_quantization_config[{name}]"
    for name in _SLIMQUANT_NAMES
)
_MARKER = "_vllm_hcu_slimquant_registry_patch_applied"


def apply_to_module(module: ModuleType) -> bool:
    quantization = load_exact_module(TARGET_MODULE, module)
    if getattr(quantization, _MARKER, False):
        return False

    # Importing the facades imports vLLM's QuantizationConfig base.  Keep it
    # inside the exact post-import callback so merely arming platform patches
    # cannot import the target package ahead of the coordinator.
    from vllm_hcu.model_executor.layers.quantization.slimquant_facade import (
        SLIMQUANT_FACADES,
    )

    if tuple(SLIMQUANT_FACADES) != _SLIMQUANT_NAMES:
        raise PatchCompatibilityError(
            "HCU SlimQuant facade registry no longer matches the audited names"
        )

    register = getattr(quantization, "register_quantization_config", None)
    methods = getattr(quantization, "QUANTIZATION_METHODS", None)
    if not callable(register) or not isinstance(methods, list):
        raise PatchCompatibilityError(
            "vLLM public quantization registry API is missing or incompatible"
        )

    customized = getattr(quantization, "_CUSTOMIZED_METHOD_TO_QUANT_CONFIG", None)
    if customized is not None and not isinstance(customized, dict):
        raise PatchCompatibilityError(
            "vLLM customized quantization registry has incompatible type"
        )

    for name, facade in SLIMQUANT_FACADES.items():
        if name in methods:
            existing = customized.get(name) if isinstance(customized, dict) else None
            if existing is facade:
                continue
            raise PatchCompatibilityError(
                f"quantization method {name!r} is already registered by "
                f"{existing or 'an unknown provider'}"
            )
        decorator = register(name)
        if not callable(decorator):
            raise PatchCompatibilityError(
                "register_quantization_config did not return a decorator"
            )
        registered = decorator(facade)
        if registered is not facade or name not in methods:
            raise PatchCompatibilityError(
                f"vLLM failed to register SlimQuant facade {name!r}"
            )

    setattr(quantization, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    quantization = load_exact_module(TARGET_MODULE, module)
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=quantization,
        marker=_MARKER,
        callback=lambda: apply_to_module(quantization),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
