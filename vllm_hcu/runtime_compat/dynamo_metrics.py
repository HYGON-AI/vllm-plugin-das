# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Compatibility for PyTorch Dynamo compilation-metric serialization."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


_PATCH_MARKER = "_hcu_dynamo_metrics_logging_patch_applied"
_MISSING = object()


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe value, or ``_MISSING`` for unsupported objects."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            safe_item = _json_safe(item)
            if safe_item is not _MISSING:
                result[str(key)] = safe_item
        return result
    if isinstance(value, (set, frozenset)):
        safe_items = [_json_safe(item) for item in value]
        if any(item is _MISSING for item in safe_items):
            return _MISSING
        return sorted(safe_items, key=repr)
    if isinstance(value, (list, tuple)):
        safe_items = [_json_safe(item) for item in value]
        if any(item is _MISSING for item in safe_items):
            return _MISSING
        return safe_items
    try:
        json.dumps(value)
    except (TypeError, ValueError, OverflowError):
        return _MISSING
    return value


def install_dynamo_metrics_compat() -> bool:
    """Make Dynamo config logging tolerate newer callable config fields.

    PyTorch 2.11 adds callable values such as ``ignore_logging_functions`` to
    the Dynamo config, while its config logger still serializes every field
    directly.  The failure only affects diagnostic logging, but it pollutes
    otherwise successful inference requests.  Filter unsupported fields in a
    process-local wrapper and leave compilation and metric collection enabled.
    """

    try:
        from torch._dynamo import config, utils
    except (ImportError, ModuleNotFoundError):
        return False

    if getattr(utils, _PATCH_MARKER, False):
        return True

    original = getattr(utils, "_get_dynamo_config_for_logging", None)
    if not callable(original):
        return False

    def safe_get_dynamo_config_for_logging() -> str | None:
        try:
            return original()
        except (TypeError, ValueError, OverflowError):
            safe_config = _json_safe(config.get_config_copy())
            if safe_config is _MISSING:
                return None
            return json.dumps(safe_config, sort_keys=True)

    setattr(utils, "_get_dynamo_config_for_logging", safe_get_dynamo_config_for_logging)
    setattr(utils, _PATCH_MARKER, True)
    return True


__all__ = ["install_dynamo_metrics_compat"]
