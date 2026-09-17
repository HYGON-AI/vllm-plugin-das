"""Register the HCU-owned Kimi-K3 Transformers config lazily."""

from __future__ import annotations

from types import ModuleType

from ._common import PatchCompatibilityError, apply_once, load_exact_module

TARGET_MODULE = "vllm.transformers_utils.config"
PATCH_ID = "platform.core_fix.kimi_k3.config_registry"
TARGETS = (f"{TARGET_MODULE}._CONFIG_REGISTRY[kimi_k3]",)
_MARKER = "_vllm_hcu_kimi_k3_config_registry_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    registry = getattr(target, "_CONFIG_REGISTRY", None)
    if not hasattr(registry, "__setitem__"):
        raise PatchCompatibilityError("vLLM config registry is not mutable")
    from vllm_hcu.transformers_utils.configs.kimi_k3 import KimiK3Config

    existing = registry.get("kimi_k3")
    if existing is not None and existing is not KimiK3Config:
        raise PatchCompatibilityError("kimi_k3 config is already owned by another provider")
    registry["kimi_k3"] = KimiK3Config
    setattr(target, _MARKER, True)
    return existing is None


def apply(module: ModuleType | None = None) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=target,
        marker=_MARKER,
        callback=lambda: apply_to_module(target),
    )


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
