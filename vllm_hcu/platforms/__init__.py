# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""Dependency-light exports for the HCU platform package.

Patch dispatchers import :mod:`vllm_hcu.platforms.envs` during plugin
discovery.  Importing the platform implementation here would load torch and
vLLM before exact module replacements are armed, so the compatibility exports
below are resolved only when callers explicitly request them.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name not in {"HCUPlatform", "current_platform"}:
        raise AttributeError(name)

    from .hcu import HCUPlatform

    globals()["HCUPlatform"] = HCUPlatform
    if name == "HCUPlatform":
        return HCUPlatform

    platform = HCUPlatform()
    globals()["current_platform"] = platform
    return platform


__all__ = ["HCUPlatform", "current_platform"]
