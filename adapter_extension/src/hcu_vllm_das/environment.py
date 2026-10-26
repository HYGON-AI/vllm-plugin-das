"""Environment policy for optional DAS optimizations."""

from __future__ import annotations

import os


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_MASTER_SWITCH = "VLLM_HCU_USE_CUSTOM_OPS"
_SILU_SWITCH = "VLLM_HCU_USE_CUSTOM_SILU_AND_MUL"
_RMS_NORM_SWITCH = "VLLM_HCU_USE_CUSTOM_RMS_NORM"


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

    return _flag(_MASTER_SWITCH, default=True)


def silu_and_mul_enabled() -> bool:
    """Return whether the migrated LightOp SwiGLU path is enabled."""

    return custom_ops_enabled() and _flag(_SILU_SWITCH, default=True)


def rms_norm_enabled() -> bool:
    """Return whether the migrated LightOp RMSNorm paths are enabled."""

    return custom_ops_enabled() and _flag(_RMS_NORM_SWITCH, default=True)


__all__ = ["custom_ops_enabled", "rms_norm_enabled", "silu_and_mul_enabled"]
