# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Register/select the HCU DPSK DeepGEMM FP8 MoE backend via sidecar config."""

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

TARGET_MODULE = "vllm.model_executor.layers.fused_moe.oracle.fp8"
PATCH_ID = "worker.op_opt.moe.oracle.fp8_dpsk"
TARGETS = (
    f"{TARGET_MODULE}.Fp8MoeBackend",
    f"{TARGET_MODULE}.backend_to_kernel_cls",
    f"{TARGET_MODULE}.map_fp8_backend",
    f"{TARGET_MODULE}.select_fp8_moe_backend",
    f"{TARGET_MODULE}.convert_to_fp8_moe_kernel_format",
    f"{TARGET_MODULE}.make_fp8_moe_kernel",
)
_MARKER = "_vllm_hcu_fp8_dpsk_oracle_applied"


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
    old_enum = require_class(target, "Fp8MoeBackend", TARGETS[0])
    backend_to_cls = require_callable(target, "backend_to_kernel_cls", TARGETS[1])
    map_backend = require_callable(target, "map_fp8_backend", TARGETS[2])
    select_backend = require_callable(target, "select_fp8_moe_backend", TARGETS[3])
    convert = require_callable(target, "convert_to_fp8_moe_kernel_format", TARGETS[4])
    make_kernel = require_callable(target, "make_fp8_moe_kernel", TARGETS[5])
    require_parameter_names(backend_to_cls, TARGETS[1], ("backend",))
    require_parameter_names(map_backend, TARGETS[2], ("runner_backend",))
    require_parameter_names(
        select_backend,
        TARGETS[3],
        ("config", "weight_key", "activation_key", "allow_vllm_cutlass"),
    )
    require_parameter_names(
        convert,
        TARGETS[4],
        (
            "fp8_backend",
            "layer",
            "w13",
            "w2",
            "w13_scale",
            "w2_scale",
            "w13_input_scale",
            "w2_input_scale",
        ),
    )
    require_parameter_names(
        make_kernel,
        TARGETS[5],
        (
            "moe_quant_config",
            "moe_config",
            "experts_cls",
            "fp8_backend",
            "routing_tables",
        ),
    )
    values = {member.name: member.value for member in old_enum}
    if "DPSK_DEEPGEMM" in values:
        raise PatchCompatibilityError("DPSK backend is already present outside HCU adapter")
    values["DPSK_DEEPGEMM"] = "DPSK_DEEPGEMM"
    hcu_enum = target.Enum("Fp8MoeBackend", values, module=target.__name__)
    target._vllm_hcu_original_fp8_moe_backend = old_enum
    target.Fp8MoeBackend = hcu_enum

    @functools.wraps(backend_to_cls)
    def hcu_backend_to_kernel_cls(backend):
        if backend == hcu_enum.DPSK_DEEPGEMM:
            try:
                from vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe import (
                    DeepEPDeepGemmContiguousExperts,
                    DeepEPDeepGemmMaskedExperts,
                )
            except (ImportError, AttributeError) as exc:
                raise RuntimeError(
                    "dpsk_deep_gemm was selected, but HCU DeepGEMM/LightOP "
                    "expert dependencies are unavailable"
                ) from exc
            return [DeepEPDeepGemmContiguousExperts, DeepEPDeepGemmMaskedExperts]
        return backend_to_cls(backend)

    @functools.wraps(map_backend)
    def hcu_map_fp8_backend(runner_backend):
        if runner_backend == "dpsk_deep_gemm":
            return hcu_enum.DPSK_DEEPGEMM
        return map_backend(runner_backend)

    @functools.wraps(select_backend)
    def hcu_select_fp8_moe_backend(
        config,
        weight_key,
        activation_key,
        allow_vllm_cutlass=False,
    ):
        sidecar = _sidecar_config(config)
        if sidecar.deepep_auto:
            if sidecar.moe_backend not in ("auto", "dpsk_deep_gemm"):
                raise ValueError(
                    "deepep_auto requires moe_backend='auto' or "
                    "'dpsk_deep_gemm'"
                )
            if getattr(config, "moe_backend", "auto") != "auto":
                raise ValueError(
                    "deepep_auto requires the official FusedMoEConfig "
                    "moe_backend to remain 'auto'"
                )
            from vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe import (
                DeepEPDeepGemmContiguousExperts,
            )

            return hcu_enum.DPSK_DEEPGEMM, DeepEPDeepGemmContiguousExperts
        if sidecar.moe_backend != "dpsk_deep_gemm":
            return select_backend(config, weight_key, activation_key, allow_vllm_cutlass)
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
            "this MoE configuration: " + "; ".join(reasons)
        )

    @functools.wraps(convert)
    def hcu_convert_to_fp8_moe_kernel_format(
        fp8_backend,
        layer,
        w13,
        w2,
        w13_scale,
        w2_scale,
        w13_input_scale,
        w2_input_scale,
    ):
        if fp8_backend == hcu_enum.DPSK_DEEPGEMM:
            return w13, w2, w13_scale, w2_scale
        if fp8_backend in (hcu_enum.DEEPGEMM, hcu_enum.BATCHED_DEEPGEMM) and getattr(
            layer, "weight_block_size", None
        ) is None:
            return w13, w2, w13_scale, w2_scale
        return convert(
            fp8_backend,
            layer,
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_input_scale,
            w2_input_scale,
        )

    @functools.wraps(make_kernel)
    def hcu_make_fp8_moe_kernel(
        moe_quant_config,
        moe_config,
        experts_cls,
        fp8_backend,
        routing_tables=None,
    ):
        if getattr(
            moe_config.moe_parallel_config,
            "use_deepep_auto_kernels",
            False,
        ):
            if fp8_backend != hcu_enum.DPSK_DEEPGEMM:
                raise ValueError(
                    "deepep_auto currently supports only the HCU "
                    f"DPSK_DEEPGEMM FP8 backend, got {fp8_backend.value}"
                )
            from vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe import (
                make_deepep_auto_deepgemm_fp8_moe_kernel,
            )

            return make_deepep_auto_deepgemm_fp8_moe_kernel(
                moe_quant_config=moe_quant_config,
                moe_config=moe_config,
                routing_tables=routing_tables,
            )
        return make_kernel(
            moe_quant_config,
            moe_config,
            experts_cls,
            fp8_backend,
            routing_tables,
        )

    target._vllm_hcu_original_backend_to_kernel_cls = backend_to_cls
    target.backend_to_kernel_cls = hcu_backend_to_kernel_cls
    target._vllm_hcu_original_map_fp8_backend = map_backend
    target.map_fp8_backend = hcu_map_fp8_backend
    target._vllm_hcu_original_select_fp8_moe_backend = select_backend
    target.select_fp8_moe_backend = hcu_select_fp8_moe_backend
    target._vllm_hcu_original_convert_to_fp8_moe_kernel_format = convert
    target.convert_to_fp8_moe_kernel_format = hcu_convert_to_fp8_moe_kernel_format
    target._vllm_hcu_original_make_fp8_moe_kernel = make_kernel
    target.make_fp8_moe_kernel = hcu_make_fp8_moe_kernel
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
