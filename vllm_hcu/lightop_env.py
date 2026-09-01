# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Configure LightOp environment aliases without importing LightOp itself."""

from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping


logger = logging.getLogger(__name__)

_MAPPINGS = (
    (
        "VLLM_HCU_FUSED_MOE_CHUNK_SIZE",
        "VLLM_FUSED_MOE_CHUNK_SIZE",
        "LMSLIM_FUSED_MOE_CHUNK_SIZE",
        "int",
    ),
    (
        "VLLM_HCU_USE_GLOBAL_MOE_CACHE",
        "VLLM_USE_GLOBAL_CACHE13",
        "LMSLIM_USE_GLOBAL_MOE_CACHE",
        "bool",
    ),
    (
        "VLLM_HCU_USE_FUSED_RMS_QUANT",
        "USE_FUSED_RMS_QUANT",
        "LMSLIM_USE_FUSED_RMS_QUANT",
        "bool",
    ),
    (
        "VLLM_HCU_USE_FUSE_SILU_AND_MUL",
        "VLLM_USE_FUSE_SILU_AND_MUL",
        "LMSLIM_USE_FUSE_SILU_AND_MUL",
        "bool",
    ),
)
_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
_FALSE_VALUES = frozenset(("0", "false", "no", "off"))
_WARNED_LEGACY_NAMES: set[str] = set()


class LightOpEnvironmentError(ValueError):
    """Raised when LightOp environment settings disagree or are invalid."""


def _normalize(name: str, value: str, kind: str) -> str:
    if kind == "int":
        try:
            return str(int(value))
        except ValueError as exc:
            raise LightOpEnvironmentError(
                f"{name} must be an integer, got {value!r}."
            ) from exc

    normalized = value.lower()
    if normalized in _TRUE_VALUES:
        return "1"
    if normalized in _FALSE_VALUES:
        return "0"
    raise LightOpEnvironmentError(
        f"{name} must be a boolean, got {value!r}."
    )


def _conflict_message(configured: dict[str, str]) -> str:
    values = ", ".join(
        f"{name}={value!r}" for name, value in configured.items()
    )
    return f"conflicting LightOp environment settings: {values}."


def _warn_legacy_once(legacy_name: str, hcu_name: str) -> None:
    if legacy_name in _WARNED_LEGACY_NAMES:
        return
    _WARNED_LEGACY_NAMES.add(legacy_name)
    logger.warning("%s is deprecated; use %s instead.", legacy_name, hcu_name)


def configure_lightop_environment(
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Bridge configured HCU names to aliases recognized by LightOp 0.6."""

    environ = os.environ if environ is None else environ
    for hcu_name, alias_name, legacy_name, kind in _MAPPINGS:
        configured = {
            name: _normalize(name, environ[name], kind)
            for name in (hcu_name, alias_name, legacy_name)
            if environ.get(name, "") != ""
        }
        if len(set(configured.values())) > 1:
            raise LightOpEnvironmentError(_conflict_message(configured))
        if legacy_name in configured:
            _warn_legacy_once(legacy_name, hcu_name)
        if configured:
            canonical = next(iter(configured.values()))
            if not environ.get(hcu_name):
                environ[hcu_name] = canonical
            if not environ.get(alias_name):
                environ[alias_name] = canonical
