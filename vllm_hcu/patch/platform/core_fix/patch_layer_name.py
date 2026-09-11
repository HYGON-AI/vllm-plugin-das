# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Adapt vLLM's LayerName registration to the HCU torch opaque API."""

from __future__ import annotations

import functools
import sys
from types import ModuleType


PATCH_ID = "platform.core.layer_name_value_member"
TARGET_MODULE = "vllm.utils.torch_utils"
TARGETS = ("vllm.utils.torch_utils.LayerName.value",)
_BRIDGE_MARKER = "_vllm_hcu_layer_name_registration_bridge"


def apply_to_module(module: ModuleType) -> None:
    if not module._USE_LAYERNAME:
        return

    from torch._library.opaque_object import MemberType, get_opaque_obj_info

    type_info = get_opaque_obj_info(module.LayerName)
    if type_info is None:
        raise RuntimeError("vLLM LayerName is not registered as an opaque type")
    type_info.members.setdefault("value", MemberType.USE_REAL)


def arm_partial_import_bridge(module: ModuleType | None = None) -> bool:
    """Add the member contract while LayerName is first registered."""
    if module is None:
        candidate = sys.modules.get(TARGET_MODULE)
        if not isinstance(candidate, ModuleType):
            return False
        module = candidate

    if hasattr(module, "LayerName"):
        return False
    spec = getattr(module, "__spec__", None)
    if not bool(getattr(spec, "_initializing", False)):
        return False

    import torch._library.opaque_object as opaque_object

    current_register = opaque_object.register_opaque_type
    if getattr(current_register, _BRIDGE_MARKER, None) is not None:
        return False

    @functools.wraps(current_register)
    def register_opaque_type(cls, **kwargs):
        is_layer_name = (
            cls.__module__ == TARGET_MODULE and cls.__qualname__ == "LayerName"
        )
        if not is_layer_name:
            return current_register(cls, **kwargs)

        members = dict(kwargs.get("members") or {})
        members.setdefault("value", opaque_object.MemberType.USE_REAL)
        kwargs["members"] = members
        try:
            return current_register(cls, **kwargs)
        finally:
            opaque_object.register_opaque_type = current_register

    setattr(register_opaque_type, _BRIDGE_MARKER, current_register)
    opaque_object.register_opaque_type = register_opaque_type
    return True


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply_to_module",
    "arm_partial_import_bridge",
]
