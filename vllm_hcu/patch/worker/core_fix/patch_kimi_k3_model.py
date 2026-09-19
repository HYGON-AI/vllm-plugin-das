"""Validate the Kimi-K3 model registry boundary in Worker processes."""

from __future__ import annotations

from types import ModuleType

from ._common import PatchCompatibilityError, load_exact_module

TARGET_MODULE = "vllm.model_executor.models.registry"
PATCH_ID = "worker.core_fix.kimi_k3.model_registry"
TARGETS = (f"{TARGET_MODULE}.ModelRegistry.register_model",)
_MARKER = "_vllm_hcu_kimi_k3_model_registry_audited"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    registry = getattr(target, "ModelRegistry", None)
    register = getattr(registry, "register_model", None)
    if not callable(register):
        raise PatchCompatibilityError("vLLM ModelRegistry.register_model is unavailable")
    if getattr(target, _MARKER, False):
        return False
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    return apply_to_module(target)


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
