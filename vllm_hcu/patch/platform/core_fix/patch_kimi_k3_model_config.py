"""Register Kimi-K3 architecture config adapters in vLLM."""

from __future__ import annotations

from types import ModuleType

from ._common import PatchCompatibilityError, apply_once, load_exact_module

TARGET_MODULE = "vllm.model_executor.models.config"
PATCH_ID = "platform.core_fix.kimi_k3.model_config"
TARGETS = (f"{TARGET_MODULE}.MODELS_CONFIG_MAP",)
_MARKER = "_vllm_hcu_kimi_k3_model_config_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    mapping = getattr(target, "MODELS_CONFIG_MAP", None)
    if not isinstance(mapping, dict):
        raise PatchCompatibilityError("vLLM MODELS_CONFIG_MAP is not a dict")
    from vllm_hcu.models.kimi_k3.config import KimiK3ConfigAdapter

    changed = False
    for architecture in ("KimiK3ForConditionalGeneration", "KimiK3MTPModel"):
        existing = mapping.get(architecture)
        if existing is not None and existing is not KimiK3ConfigAdapter:
            raise PatchCompatibilityError(
                f"{architecture} config adapter is already owned by another provider"
            )
        if existing is None:
            mapping[architecture] = KimiK3ConfigAdapter
            changed = True
    setattr(target, _MARKER, True)
    return changed


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
