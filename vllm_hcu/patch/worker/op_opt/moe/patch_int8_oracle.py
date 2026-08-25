# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Register the HCU AITER INT8 MoE backend in vLLM's v0.25 oracle."""

from __future__ import annotations

import functools
from types import ModuleType

from vllm_hcu.patch.config import get_hcu_config

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
    f"{TARGET_MODULE}.select_int8_moe_backend",
    f"{TARGET_MODULE}.convert_to_int8_moe_kernel_format",
    f"{TARGET_MODULE}.make_int8_moe_quant_config",
    f"{TARGET_MODULE}.int8_w8a8_moe_quant_config",
)
_MARKER = "_vllm_hcu_int8_aiter_oracle_applied"


def _sidecar_config(config):
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        vllm_config = getattr(config, "_hcu_vllm_config", None)
    return get_hcu_config(vllm_config)


def _sidecar_backend(config) -> str:
    return _sidecar_config(config).moe_backend


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False

    old_enum = require_class(target, "Int8MoeBackend", TARGETS[0])
    backend_to_cls = require_callable(target, "backend_to_kernel_cls", TARGETS[1])
    map_backend = require_callable(target, "map_int8_backend", TARGETS[2])
    select_backend = require_callable(target, "select_int8_moe_backend", TARGETS[3])
    convert = require_callable(
        target,
        "convert_to_int8_moe_kernel_format",
        TARGETS[4],
    )
    make_quant_config = require_callable(
        target,
        "make_int8_moe_quant_config",
        TARGETS[5],
    )
    make_w8a8_quant_config = require_callable(
        target,
        "int8_w8a8_moe_quant_config",
        TARGETS[6],
    )
    require_parameter_names(backend_to_cls, TARGETS[1], ("backend",))
    require_parameter_names(map_backend, TARGETS[2], ("runner_backend",))
    require_parameter_names(
        select_backend,
        TARGETS[3],
        ("config", "weight_key", "activation_key"),
    )
    require_parameter_names(
        convert,
        TARGETS[4],
        ("int8_backend", "w13", "w2", "layer", "w13_scale"),
    )
    require_parameter_names(
        make_quant_config,
        TARGETS[4],
        (
            "int8_backend",
            "w1_scale",
            "w2_scale",
            "a1_scale",
            "a2_scale",
            "w1_bias",
            "w2_bias",
            "per_act_token_quant",
            "layer",
        ),
    )

    values = {member.name: member.value for member in old_enum}
    if "AITER" in values or "DPSK_DEEPGEMM" in values:
        raise PatchCompatibilityError(
            "HCU INT8 MoE backend is already present outside the HCU adapter"
        )
    values["AITER"] = "AITER"
    values["DPSK_DEEPGEMM"] = "DPSK_DEEPGEMM"
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
        if backend == hcu_enum.DPSK_DEEPGEMM:
            from vllm_hcu.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
                BatchedDeepGemmExperts,
            )
            from vllm_hcu.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
                DeepGemmExperts,
            )

            return [DeepGemmExperts, BatchedDeepGemmExperts]
        return backend_to_cls(backend)

    @functools.wraps(map_backend)
    def hcu_map_int8_backend(runner_backend):
        if runner_backend == "aiter":
            return hcu_enum.AITER
        if runner_backend == "dpsk_deep_gemm":
            return hcu_enum.DPSK_DEEPGEMM
        return map_backend(runner_backend)

    @functools.wraps(select_backend)
    def hcu_select_int8_moe_backend(config, weight_key, activation_key):
        if _sidecar_backend(config) != "dpsk_deep_gemm":
            return select_backend(config, weight_key, activation_key)
        if getattr(config, "moe_backend", "auto") != "auto":
            raise ValueError(
                "HCU sidecar selects dpsk_deep_gemm but official FusedMoEConfig "
                f"selects {config.moe_backend!r}; official backend must remain 'auto'"
            )
        activation_format = (
            target.mk.FusedMoEActivationFormat.BatchedExperts
            if config.moe_parallel_config.use_batched_activation_format
            else target.mk.FusedMoEActivationFormat.Standard
        )
        reasons = []
        for kernel_cls in hcu_backend_to_kernel_cls(hcu_enum.DPSK_DEEPGEMM):
            supported, reason = kernel_cls.is_supported_config(
                kernel_cls,
                config,
                weight_key,
                activation_key,
                activation_format,
            )
            if supported:
                return hcu_enum.DPSK_DEEPGEMM, kernel_cls
            reasons.append(f"{kernel_cls.__name__}: {reason or 'unsupported'}")
        raise ValueError(
            "dpsk_deep_gemm is required by HCU sidecar but does not support "
            "this INT8 MoE configuration: " + "; ".join(reasons)
        )

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
        if int8_backend == hcu_enum.DPSK_DEEPGEMM:
            from vllm_hcu.model_executor.layers.quantization.int8_runtime import (
                weight8bit_nt_kpack2_marlin2,
            )

            return (
                weight8bit_nt_kpack2_marlin2(w13),
                weight8bit_nt_kpack2_marlin2(w2),
            )
        return convert(int8_backend, w13, w2, layer, w13_scale)

    @functools.wraps(make_quant_config)
    def hcu_make_int8_moe_quant_config(
        int8_backend,
        w1_scale,
        w2_scale,
        a1_scale=None,
        a2_scale=None,
        w1_bias=None,
        w2_bias=None,
        per_act_token_quant=False,
        layer=None,
    ):
        if int8_backend in (
            hcu_enum.AITER,
            hcu_enum.DPSK_DEEPGEMM,
        ) and per_act_token_quant:
            return make_w8a8_quant_config(
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                w1_bias=w1_bias,
                w2_bias=w2_bias,
                per_act_token_quant=True,
            )
        return make_quant_config(
            int8_backend=int8_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            per_act_token_quant=per_act_token_quant,
            layer=layer,
        )

    target._vllm_hcu_original_backend_to_kernel_cls = backend_to_cls
    target.backend_to_kernel_cls = hcu_backend_to_kernel_cls
    target._vllm_hcu_original_map_int8_backend = map_backend
    target.map_int8_backend = hcu_map_int8_backend
    target._vllm_hcu_original_select_int8_moe_backend = select_backend
    target.select_int8_moe_backend = hcu_select_int8_moe_backend
    target._vllm_hcu_original_convert_to_int8_moe_kernel_format = convert
    target.convert_to_int8_moe_kernel_format = hcu_convert_to_int8_moe_kernel_format
    target._vllm_hcu_original_make_int8_moe_quant_config = make_quant_config
    target.make_int8_moe_quant_config = hcu_make_int8_moe_quant_config
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
