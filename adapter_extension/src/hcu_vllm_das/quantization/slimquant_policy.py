# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-name precision policy, independent of vLLM and device state.

A future loader must map its exact checkpoint names through this policy before
fusing or sharding. This module does not register a vLLM quantization method.
"""
from __future__ import annotations
from .checkpoint_audit import _excluded_module


def weight_precision(module_name, weight_map, modules_to_not_convert):
    """Return checkpoint storage policy for an unfused, canonical module name.

    Only modules present in the index are accepted. Expert w1/w2/w3 weights are
    interpreted as packed W4 under the caller's validated SlimQuant contract;
    a non-expert int8-looking weight with a scale is W8. Without a scale this
    function returns `unquantized`, which still requires dtype/header checking.
    """
    if not isinstance(module_name, str) or not module_name or module_name.endswith(".weight"):
        raise ValueError("module_name must be a canonical module path without .weight")
    name = module_name + ".weight"
    if name not in weight_map:
        raise KeyError(f"checkpoint does not contain {name}")
    scale = name + "_scale"
    excluded = _excluded_module(name, modules_to_not_convert)
    if scale in weight_map:
        if excluded:
            raise ValueError(f"quantized weight is explicitly excluded: {name}")
        parts = module_name.split(".")
        if "experts" in parts and parts[-1] in ("w1", "w2", "w3"):
            i = parts.index("experts")
            if len(parts) != i + 3 or not parts[i + 1].isdecimal():
                raise ValueError(f"unsupported expert path: {name}")
            return "w4a8"
        return "w8a8"
    return "unquantized"


def fused_precision(module_names, weight_map, modules_to_not_convert):
    """Return a common precision or fail when a native fused layer is mixed."""
    if not module_names:
        raise ValueError("at least one module is required")
    modes = [weight_precision(name, weight_map, modules_to_not_convert)
             for name in module_names]
    if len(set(modes)) != 1:
        raise ValueError(f"mixed precision fused projection requires split loading: {modes}")
    return modes[0]
