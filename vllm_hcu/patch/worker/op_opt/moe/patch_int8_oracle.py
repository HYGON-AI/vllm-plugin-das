# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Extend vLLM's v0.25 INT8 oracle for HCU AITER and DeepGEMM kernels."""

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
    f"{TARGET_MODULE}.make_int8_moe_kernel",
)
_MARKER = "_vllm_hcu_int8_aiter_oracle_applied"


def _sidecar_config(config):
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        vllm_config = getattr(config, "_hcu_vllm_config", None)
    return get_hcu_config(vllm_config)


def _model_architectures(config) -> tuple[str, ...]:
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        vllm_config = getattr(config, "_hcu_vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    architectures = getattr(model_config, "architectures", None)
    if architectures is None:
        hf_config = getattr(model_config, "hf_config", None)
        architectures = getattr(hf_config, "architectures", ())
    return tuple(architectures or ())


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
    make_kernel = require_callable(target, "make_int8_moe_kernel", TARGETS[7])
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt8DynamicTokenSym,
        kInt8StaticChannelSym,
    )

    channel_int8_scheme = (kInt8StaticChannelSym, kInt8DynamicTokenSym)
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
        TARGETS[5],
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
    require_parameter_names(
        make_kernel,
        TARGETS[7],
        (
            "int8_backend",
            "moe_quant_config",
            "moe_config",
            "experts_cls",
            "routing_tables",
            "layer",
        ),
    )

    values = {member.name: member.value for member in old_enum}
    if "AITER" in values or "HCU_DEEPGEMM" in values:
        raise PatchCompatibilityError(
            "HCU INT8 MoE backend is already present outside the HCU adapter"
        )
    values["AITER"] = "AITER"
    values["HCU_DEEPGEMM"] = "HCU_DEEPGEMM"
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
        if backend == hcu_enum.HCU_DEEPGEMM:
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
        if runner_backend == "deep_gemm":
            return hcu_enum.HCU_DEEPGEMM
        return map_backend(runner_backend)

    @functools.wraps(select_backend)
    def hcu_select_int8_moe_backend(config, weight_key, activation_key):
        sidecar = _sidecar_config(config)
        if sidecar.deepep_auto:
            architectures = _model_architectures(config)
            if "DeepseekV4ForCausalLM" not in architectures:
                raise ValueError(
                    "HCU Channel-INT8 deepep_auto is validated only for "
                    "DeepSeek-V4; use an explicit DeepEP high-throughput or "
                    "low-latency backend for other models. "
                    f"Got architectures={architectures!r}."
                )
            if sidecar.moe_backend not in ("auto", "deep_gemm"):
                raise ValueError(
                    "deepep_auto requires moe_backend='auto' or 'deep_gemm'"
                )
            if getattr(config, "moe_backend", "auto") != sidecar.moe_backend:
                raise ValueError(
                    "deepep_auto requires the official FusedMoEConfig "
                    f"moe_backend to match {sidecar.moe_backend!r}"
                )
            if (weight_key, activation_key) != channel_int8_scheme:
                raise ValueError(
                    "deepep_auto HCU DeepGEMM supports only channel-wise "
                    "INT8 weights with dynamic per-token INT8 activations"
                )
            from vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe import (
                DeepEPDeepGemmContiguousExperts,
            )

            return hcu_enum.HCU_DEEPGEMM, DeepEPDeepGemmContiguousExperts
        if sidecar.moe_backend != "deep_gemm":
            return select_backend(config, weight_key, activation_key)
        if getattr(config, "moe_backend", "auto") != "deep_gemm":
            raise ValueError(
                "HCU sidecar selects deep_gemm but official FusedMoEConfig "
                f"selects {config.moe_backend!r}; official backend must match "
                "'deep_gemm'"
            )
        activation_format = (
            target.mk.FusedMoEActivationFormat.BatchedExperts
            if config.moe_parallel_config.use_batched_activation_format
            else target.mk.FusedMoEActivationFormat.Standard
        )
        reasons = []
        for kernel_cls in hcu_backend_to_kernel_cls(hcu_enum.HCU_DEEPGEMM):
            supported, reason = kernel_cls.is_supported_config(
                kernel_cls,
                config,
                weight_key,
                activation_key,
                activation_format,
            )
            if supported:
                return hcu_enum.HCU_DEEPGEMM, kernel_cls
            reasons.append(f"{kernel_cls.__name__}: {reason or 'unsupported'}")
        raise ValueError(
            "deep_gemm is required by HCU sidecar but does not support "
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
        if int8_backend == hcu_enum.HCU_DEEPGEMM:
            if layer is None or not hasattr(layer, "moe_config"):
                raise ValueError(
                    "HCU DeepGEMM INT8 weight conversion requires the MoE layer "
                    "to select its masked or contiguous weight layout"
                )
            parallel_config = layer.moe_config.moe_parallel_config
            if getattr(parallel_config, "use_deepep_auto_kernels", False):
                return w13, w2
            from deepgemm import (
                marlin_i8_contiguous_weight,
                marlin_i8_masked_weight,
            )

            use_batched = bool(parallel_config.use_batched_activation_format)
            pack_weight = (
                marlin_i8_masked_weight
                if use_batched
                else marlin_i8_contiguous_weight
            )
            return pack_weight(w13), pack_weight(w2)
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
            hcu_enum.HCU_DEEPGEMM,
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

    @functools.wraps(make_kernel)
    def hcu_make_int8_moe_kernel(
        int8_backend,
        moe_quant_config,
        moe_config,
        experts_cls,
        routing_tables=None,
        layer=None,
    ):
        if getattr(
            moe_config.moe_parallel_config,
            "use_deepep_auto_kernels",
            False,
        ):
            if int8_backend != hcu_enum.HCU_DEEPGEMM:
                raise ValueError(
                    "deepep_auto currently supports only the "
                    f"HCU_DEEPGEMM INT8 backend, got {int8_backend.value}"
                )
            from vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe import (
                make_deepep_auto_deepgemm_int8_moe_kernel,
            )

            moe_kernel = make_deepep_auto_deepgemm_int8_moe_kernel(
                moe_quant_config=moe_quant_config,
                moe_config=moe_config,
                routing_tables=routing_tables,
            )
            fused_experts = getattr(moe_kernel, "fused_experts", None)
            experts = getattr(fused_experts, "experts", fused_experts)
            process = getattr(experts, "process_weights_after_loading", None)
            if layer is None or not callable(process):
                raise RuntimeError(
                    "deepep_auto HCU DeepGEMM INT8 kernel did not construct "
                    "modular experts before weight postprocessing"
                )
            process(layer)
            return moe_kernel
        return make_kernel(
            int8_backend,
            moe_quant_config,
            moe_config,
            experts_cls,
            routing_tables,
            layer,
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
    target._vllm_hcu_original_make_int8_moe_kernel = make_kernel
    target.make_int8_moe_kernel = hcu_make_int8_moe_kernel
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
