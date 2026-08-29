# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Select HCU FP8 kernels for vLLM's official ``deep_gemm`` backend."""

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
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
        kFp8StaticChannelSym,
    )

    channel_fp8_scheme = (kFp8StaticChannelSym, kFp8DynamicTokenSym)
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
    if "HCU_DEEPGEMM" in values:
        raise PatchCompatibilityError(
            "HCU DeepGEMM backend is already present outside the HCU adapter"
        )
    values["HCU_DEEPGEMM"] = "HCU_DEEPGEMM"
    hcu_enum = target.Enum("Fp8MoeBackend", values, module=target.__name__)
    target._vllm_hcu_original_fp8_moe_backend = old_enum
    target.Fp8MoeBackend = hcu_enum

    @functools.wraps(backend_to_cls)
    def hcu_backend_to_kernel_cls(backend):
        if backend == hcu_enum.HCU_DEEPGEMM:
            try:
                from vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe import (
                    DeepEPDeepGemmContiguousExperts,
                    DeepEPDeepGemmMaskedExperts,
                )
            except (ImportError, AttributeError) as exc:
                raise RuntimeError(
                    "deep_gemm was selected, but HCU DeepGEMM/LightOP "
                    "expert dependencies are unavailable"
                ) from exc
            return [DeepEPDeepGemmContiguousExperts, DeepEPDeepGemmMaskedExperts]
        return backend_to_cls(backend)

    @functools.wraps(map_backend)
    def hcu_map_fp8_backend(runner_backend):
        # Keep the public backend mapped to vLLM's official enum.  The HCU
        # channel-FP8 specialization needs quantization keys, which are only
        # available in select_fp8_moe_backend().  Replacing the global mapping
        # here would also redirect block-FP8 to the channel-only HCU experts.
        mapped = map_backend(runner_backend)
        if isinstance(mapped, old_enum):
            return hcu_enum[mapped.name]
        return mapped

    @functools.wraps(select_backend)
    def hcu_select_fp8_moe_backend(
        config,
        weight_key,
        activation_key,
        allow_vllm_cutlass=False,
    ):
        sidecar = _sidecar_config(config)
        is_channel_fp8 = (weight_key, activation_key) == channel_fp8_scheme
        if sidecar.deepep_auto:
            if sidecar.moe_backend not in ("auto", "deep_gemm"):
                raise ValueError(
                    "deepep_auto requires moe_backend='auto' or "
                    "'deep_gemm'"
                )
            expected_backend = sidecar.moe_backend
            if getattr(config, "moe_backend", "auto") != expected_backend:
                raise ValueError(
                    "deepep_auto requires the official FusedMoEConfig "
                    f"moe_backend to match {expected_backend!r}"
                )
            if not is_channel_fp8:
                raise ValueError(
                    "deepep_auto HCU DeepGEMM supports only channel-wise "
                    "FP8 weights with dynamic per-token FP8 activations"
                )
            from vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe import (
                DeepEPDeepGemmContiguousExperts,
            )

            return hcu_enum.HCU_DEEPGEMM, DeepEPDeepGemmContiguousExperts
        if sidecar.moe_backend != "deep_gemm":
            return select_backend(config, weight_key, activation_key, allow_vllm_cutlass)
        if getattr(config, "moe_backend", "auto") != "deep_gemm":
            raise ValueError(
                "HCU sidecar selects deep_gemm but official FusedMoEConfig "
                f"selects {config.moe_backend!r}; official backend must match "
                "'deep_gemm'"
            )
        if not is_channel_fp8:
            # Preserve the official DEEPGEMM/BATCHED_DEEPGEMM selection for
            # block-FP8 and every other scheme not owned by the HCU adapter.
            return select_backend(
                config,
                weight_key,
                activation_key,
                allow_vllm_cutlass,
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
        if fp8_backend == hcu_enum.HCU_DEEPGEMM:
            return w13, w2, w13_scale, w2_scale
        if fp8_backend == hcu_enum.AITER:
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
            if fp8_backend != hcu_enum.HCU_DEEPGEMM:
                raise ValueError(
                    "deepep_auto currently supports only the "
                    f"HCU_DEEPGEMM FP8 backend, got {fp8_backend.value}"
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
