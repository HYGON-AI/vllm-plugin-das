"""Environment policy for optional DAS optimizations."""

from __future__ import annotations

import os


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_MASTER_SWITCH = "VLLM_HCU_USE_CUSTOM_OPS"
_RMS_NORM_SWITCH = "VLLM_HCU_USE_CUSTOM_RMS_NORM"
_RMS_NORM_COMPILE_FACTOR = "VLLM_HCU_DAS_RMS_NORM_ENABLED"


def _flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be a boolean value, got {raw!r}"
    )


def custom_ops_enabled() -> bool:
    """Return the process-wide rollback state for DAS custom operators."""

    return _flag(_MASTER_SWITCH, default=False)


def rms_norm_enabled() -> bool:
    """Return whether the migrated LightOp RMSNorm paths are enabled."""

    master_enabled = custom_ops_enabled()
    child_enabled = _flag(_RMS_NORM_SWITCH, default=False)
    return master_enabled and child_enabled


def preflight_compile_factors(vllm_envs: object) -> None:
    """Check registry ownership and policy without publishing a getter."""
    registry = getattr(vllm_envs, "environment_variables", None)
    if not isinstance(registry, dict) or not callable(getattr(vllm_envs, "compile_factors", None)):
        raise RuntimeError("vLLM environment compile-factor registry is unavailable")
    if getattr(vllm_envs, "_is_envs_cache_enabled", lambda: False)():
        raise RuntimeError("register RMS factors before vLLM caches env values")
    existing = registry.get(_RMS_NORM_COMPILE_FACTOR)
    if existing is not None and existing is not rms_norm_enabled:
        raise RuntimeError("HCU RMSNorm compile factor already has another owner")
    rms_norm_enabled()


def register_compile_factors(vllm_envs: object) -> None:
    """Include effective RMS policy in vLLM's normal and AOT cache keys.

    vLLM 0.32 hashes the getters in its environment registry for both caches.
    A derived getter keeps equivalent disabled switch combinations identical;
    the two raw switches alone would unnecessarily split those cache entries.
    This registration must happen before model construction or AOT loading.
    """
    preflight_compile_factors(vllm_envs)
    vllm_envs.environment_variables[_RMS_NORM_COMPILE_FACTOR] = rms_norm_enabled


__all__ = ["custom_ops_enabled", "rms_norm_enabled", "preflight_compile_factors", "register_compile_factors"]
