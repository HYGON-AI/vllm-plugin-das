# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Add an opt-in LightOp W16A16 backend to vLLM's unquantized MoE oracle."""

from __future__ import annotations

import functools
from types import ModuleType

from vllm_hcu.patch.worker.op_opt.moe._common import (
    PatchCompatibilityError,
    check_module_marker,
    load_exact_module,
    require_callable,
    require_class,
    require_parameter_names,
)

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.oracle.unquantized"
PATCH_ID = "worker.op_opt.moe.oracle.unquantized_lightop_w16a16"
TARGETS = (
    f"{TARGET_MODULE}.UnquantizedMoeBackend",
    f"{TARGET_MODULE}.backend_to_kernel_cls",
    f"{TARGET_MODULE}.map_unquantized_backend",
    f"{TARGET_MODULE}.select_unquantized_moe_backend",
    f"{TARGET_MODULE}.convert_to_unquantized_kernel_format",
    f"{TARGET_MODULE}.make_unquantized_moe_kernel",
)
_MARKER = "_vllm_hcu_lightop_w16a16_oracle_applied"
_BINDING_MARKER = "_vllm_hcu_lightop_w16a16_oracle_binding"


def _normalize_backend(backend, old_enum, new_enum):
    if isinstance(backend, old_enum):
        return new_enum[backend.name]
    return backend


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    bindings = tuple(
        (target, name, _BINDING_MARKER)
        for name in (
            "UnquantizedMoeBackend",
            "backend_to_kernel_cls",
            "map_unquantized_backend",
            "select_unquantized_moe_backend",
            "convert_to_unquantized_kernel_format",
            "make_unquantized_moe_kernel",
        )
    )
    if check_module_marker(target, _MARKER, bindings):
        return False

    old_enum = require_class(target, "UnquantizedMoeBackend", TARGETS[0])
    backend_to_cls = require_callable(target, "backend_to_kernel_cls", TARGETS[1])
    map_backend = require_callable(target, "map_unquantized_backend", TARGETS[2])
    select_backend = require_callable(
        target, "select_unquantized_moe_backend", TARGETS[3]
    )
    convert = require_callable(
        target, "convert_to_unquantized_kernel_format", TARGETS[4]
    )
    make_kernel = require_callable(target, "make_unquantized_moe_kernel", TARGETS[5])
    require_parameter_names(backend_to_cls, TARGETS[1], ("backend",))
    require_parameter_names(map_backend, TARGETS[2], ("runner_backend",))
    require_parameter_names(select_backend, TARGETS[3], ("moe_config",))
    require_parameter_names(
        convert,
        TARGETS[4],
        ("unquantized_backend", "moe_config", "w13_weight", "w2_weight"),
    )
    require_parameter_names(
        make_kernel,
        TARGETS[5],
        ("quant_config", "moe_config", "backend", "experts_cls", "routing_tables"),
    )

    values = {member.name: member.value for member in old_enum}
    if "HCU_LIGHTOP_W16A16" in values:
        raise PatchCompatibilityError(
            "HCU LightOp W16A16 backend is already present outside the HCU adapter"
        )
    values["HCU_LIGHTOP_W16A16"] = "HCU LightOp W16A16"
    hcu_enum = target.Enum("UnquantizedMoeBackend", values, module=target.__name__)
    target._vllm_hcu_original_unquantized_moe_backend = old_enum
    target.UnquantizedMoeBackend = hcu_enum

    @functools.wraps(backend_to_cls)
    def hcu_backend_to_kernel_cls(backend):
        if backend == hcu_enum.HCU_LIGHTOP_W16A16:
            from vllm_hcu.model_executor.layers.fused_moe.experts import (
                lightop_w16a16_moe,
            )

            return [lightop_w16a16_moe.LightopW16A16Experts]
        return backend_to_cls(_normalize_backend(backend, old_enum, hcu_enum))

    @functools.wraps(map_backend)
    def hcu_map_unquantized_backend(runner_backend):
        mapped = map_backend(runner_backend)
        return _normalize_backend(mapped, old_enum, hcu_enum)

    @functools.wraps(select_backend)
    def hcu_select_unquantized_moe_backend(moe_config):
        from vllm_hcu.model_executor.layers.fused_moe import (
            lightop_w16a16_runtime,
        )
        from vllm_hcu.model_executor.layers.fused_moe.experts import (
            lightop_w16a16_moe,
        )
        from vllm_hcu.platforms import envs as henvs

        if (
            getattr(moe_config, "moe_backend", "auto") == "auto"
            and henvs.VLLM_HCU_USE_CUSTOM_OPS
            and henvs.VLLM_HCU_USE_LIGHTOP_W16A16_MOE
        ):
            activation_format = (
                target.mk.FusedMoEActivationFormat.BatchedExperts
                if moe_config.moe_parallel_config.use_batched_activation_format
                else target.mk.FusedMoEActivationFormat.Standard
            )
            experts_cls = lightop_w16a16_moe.LightopW16A16Experts
            supported, _ = experts_cls.is_supported_config(
                experts_cls,
                moe_config,
                None,
                None,
                activation_format,
            )
            if supported:
                # Backend selection precedes layout mutation.  A missing M=1
                # decode config therefore delegates while weights are still
                # canonical and safe for official Triton/AITER paths.
                max_tokens = int(getattr(moe_config, "max_num_tokens", 0))
                configs_available = all(
                    lightop_w16a16_runtime.select_lightop_w16a16_config(
                        moe_config, expected_m=tokens, device="cuda"
                    )
                    is not None
                    for tokens in range(1, max_tokens + 1)
                )
                if configs_available:
                    return hcu_enum.HCU_LIGHTOP_W16A16, experts_cls

        backend, experts = select_backend(moe_config)
        return _normalize_backend(backend, old_enum, hcu_enum), experts

    @functools.wraps(convert)
    def hcu_convert_to_unquantized_kernel_format(
        unquantized_backend,
        moe_config,
        w13_weight,
        w2_weight,
    ):
        if unquantized_backend == hcu_enum.HCU_LIGHTOP_W16A16:
            from vllm_hcu.model_executor.layers.fused_moe import (
                lightop_w16a16_runtime,
            )

            packed13, packed2, _ = (
                lightop_w16a16_runtime.pack_lightop_w16a16_weights(
                    w13_weight, w2_weight
                )
            )
            return packed13, packed2
        return convert(
            _normalize_backend(unquantized_backend, old_enum, hcu_enum),
            moe_config,
            w13_weight,
            w2_weight,
        )

    @functools.wraps(make_kernel)
    def hcu_make_unquantized_moe_kernel(
        quant_config,
        moe_config,
        backend,
        experts_cls,
        routing_tables=None,
    ):
        return make_kernel(
            quant_config,
            moe_config,
            _normalize_backend(backend, old_enum, hcu_enum),
            experts_cls,
            routing_tables,
        )

    for replacement in (
        hcu_enum,
        hcu_backend_to_kernel_cls,
        hcu_map_unquantized_backend,
        hcu_select_unquantized_moe_backend,
        hcu_convert_to_unquantized_kernel_format,
        hcu_make_unquantized_moe_kernel,
    ):
        setattr(replacement, _BINDING_MARKER, True)
    target.backend_to_kernel_cls = hcu_backend_to_kernel_cls
    target.map_unquantized_backend = hcu_map_unquantized_backend
    target.select_unquantized_moe_backend = hcu_select_unquantized_moe_backend
    target.convert_to_unquantized_kernel_format = (
        hcu_convert_to_unquantized_kernel_format
    )
    target.make_unquantized_moe_kernel = hcu_make_unquantized_moe_kernel
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
