# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep HCU runtime attributes out of upstream dataclass replacement."""

from __future__ import annotations

import functools
import inspect
from dataclasses import fields
from types import ModuleType

from ._common import PatchCompatibilityError, load_exact_module, require_callable

TARGET_MODULE = "vllm.config.utils"
PATCH_ID = "platform.core_fix.config_utils.replace_runtime_attributes"
TARGET_SYMBOL = f"{TARGET_MODULE}.replace"
TARGETS = (TARGET_SYMBOL,)
_MARKER = "_vllm_hcu_config_utils_replace_applied"
_WRAPPER_MARKER = "_vllm_hcu_config_utils_replace_wrapper"


def _require_replace_signature(replace) -> None:
    parameters = tuple(inspect.signature(replace).parameters.values())
    if (
        len(parameters) != 2
        or parameters[0].name != "dataclass_instance"
        or parameters[0].kind is not inspect.Parameter.POSITIONAL_ONLY
        or parameters[1].name != "kwargs"
        or parameters[1].kind is not inspect.Parameter.VAR_KEYWORD
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_SYMBOL} has incompatible "
            f"signature {inspect.signature(replace)}"
        )


def apply_to_module(module: ModuleType) -> bool:
    config_utils = load_exact_module(TARGET_MODULE, module)
    original = require_callable(config_utils, "replace", TARGET_SYMBOL)
    is_init_field = require_callable(
        config_utils,
        "is_init_field",
        f"{TARGET_MODULE}.is_init_field",
    )
    _require_replace_signature(original)
    if getattr(config_utils, _MARKER, False):
        current = vars(config_utils).get("replace")
        if not getattr(current, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return False

    @functools.wraps(original)
    def hcu_replace(dataclass_instance, /, **kwargs):
        cls = type(dataclass_instance)
        field_names = {field.name for field in fields(cls)}
        dataclass_dict = {
            name: value
            for name, value in dataclass_instance.__dict__.items()
            if name in field_names and is_init_field(cls, name)
        }
        dataclass_dict.update(kwargs)
        return cls(**dataclass_dict)

    setattr(hcu_replace, _WRAPPER_MARKER, True)
    setattr(config_utils, "_vllm_hcu_original_replace", original)
    setattr(config_utils, "replace", hcu_replace)
    setattr(config_utils, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
