# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Register the HCU AITER INT8 MoE backend in vLLM's v0.25 oracle."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_parameter_names,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.oracle.int8"
PATCH_ID = "worker.op_opt.moe.oracle.int8_aiter"
TARGETS = (
    f"{TARGET_MODULE}.Int8MoeBackend",
    f"{TARGET_MODULE}.backend_to_kernel_cls",
    f"{TARGET_MODULE}.map_int8_backend",
    f"{TARGET_MODULE}.convert_to_int8_moe_kernel_format",
)
_MARKER = "_vllm_hcu_int8_aiter_oracle_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False

    old_enum = require_class(target, "Int8MoeBackend", TARGETS[0])
    backend_to_cls = require_callable(target, "backend_to_kernel_cls", TARGETS[1])
    map_backend = require_callable(target, "map_int8_backend", TARGETS[2])
    convert = require_callable(
        target,
        "convert_to_int8_moe_kernel_format",
        TARGETS[3],
    )
    require_parameter_names(backend_to_cls, TARGETS[1], ("backend",))
    require_parameter_names(map_backend, TARGETS[2], ("runner_backend",))
    require_parameter_names(
        convert,
        TARGETS[3],
        ("int8_backend", "w13", "w2", "layer", "w13_scale"),
    )

    values = {member.name: member.value for member in old_enum}
    if "AITER" in values:
        raise PatchCompatibilityError(
            "AITER INT8 MoE backend is already present outside the HCU adapter"
        )
    values["AITER"] = "AITER"
    hcu_enum = target.Enum("Int8MoeBackend", values, module=target.__name__)
    target._vllm_hcu_original_int8_moe_backend = old_enum
    target.Int8MoeBackend = hcu_enum

    @functools.wraps(backend_to_cls)
    def hcu_backend_to_kernel_cls(backend):
        if backend == hcu_enum.AITER:
            from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
                AiterExperts,
            )

            return [AiterExperts]
        return backend_to_cls(backend)

    @functools.wraps(map_backend)
    def hcu_map_int8_backend(runner_backend):
        if runner_backend == "aiter":
            return hcu_enum.AITER
        return map_backend(runner_backend)

    @functools.wraps(convert)
    def hcu_convert_to_int8_moe_kernel_format(
        int8_backend,
        w13,
        w2,
        layer=None,
        w13_scale=None,
    ):
        if int8_backend == hcu_enum.AITER:
            return w13, w2
        return convert(int8_backend, w13, w2, layer, w13_scale)

    target._vllm_hcu_original_backend_to_kernel_cls = backend_to_cls
    target.backend_to_kernel_cls = hcu_backend_to_kernel_cls
    target._vllm_hcu_original_map_int8_backend = map_backend
    target.map_int8_backend = hcu_map_int8_backend
    target._vllm_hcu_original_convert_to_int8_moe_kernel_format = convert
    target.convert_to_int8_moe_kernel_format = hcu_convert_to_int8_moe_kernel_format
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
