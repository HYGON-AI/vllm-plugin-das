# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Select the optional public-LightOp backend for eligible AutoAWQ layers."""

from __future__ import annotations

import functools
from types import ModuleType

from vllm_hcu.platforms import envs as henvs

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.layers.quantization.auto_awq"
PATCH_ID = "worker.op_opt.quantization.lightop_autoawq"
TARGETS = (f"{TARGET_MODULE}.AutoAWQConfig.get_quant_method",)
_CLASS_MARKER = "_vllm_hcu_lightop_autoawq_applied"
_WRAPPER_MARKER = "_vllm_hcu_lightop_autoawq_wrapper"


def _public_lightop_awq_available() -> bool:
    try:
        from lightop.gemm_ops import (
            awq_gemm_marlin_weight_repack,
            gemm_awq_w4a16_marlin,
        )
    except (ImportError, AttributeError):
        return False
    return callable(awq_gemm_marlin_weight_repack) and callable(
        gemm_awq_w4a16_marlin
    )


def _config_is_supported(config: object) -> bool:
    return (
        getattr(config, "weight_bits", None) == 4
        and getattr(config, "group_size", None) == 128
        and getattr(config, "zero_point", None) is True
    )


def apply_to_module(module: ModuleType) -> bool:
    auto_awq = load_exact_module(TARGET_MODULE, module)
    config_class = require_class(
        auto_awq, "AutoAWQConfig", f"{TARGET_MODULE}.AutoAWQConfig"
    )
    if already_applied(
        config_class,
        _CLASS_MARKER,
        (
            (
                config_class,
                "get_quant_method",
                TARGETS[0],
                _WRAPPER_MARKER,
            ),
        ),
    ):
        return False

    original = require_callable(config_class, "get_quant_method", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("self", "layer", "prefix"),
    )
    linear_method = require_class(
        auto_awq,
        "AutoAWQLinearMethod",
        f"{TARGET_MODULE}.AutoAWQLinearMethod",
    )
    marlin_method = require_class(
        auto_awq,
        "AutoAWQMarlinLinearMethod",
        f"{TARGET_MODULE}.AutoAWQMarlinLinearMethod",
    )
    feature_enabled = henvs.lightop_awq_enabled()

    @functools.wraps(original)
    def hcu_get_quant_method(self, layer, prefix):
        selected = original(self, layer, prefix)
        if (
            not feature_enabled
            or not _config_is_supported(self)
            or not isinstance(selected, (linear_method, marlin_method))
            or not _public_lightop_awq_available()
        ):
            return selected

        from vllm_hcu.model_executor.layers.quantization.lightop_autoawq import (
            LightOpAutoAWQLinearMethod,
        )

        return LightOpAutoAWQLinearMethod(selected, self)

    setattr(hcu_get_quant_method, _WRAPPER_MARKER, True)
    setattr(config_class, "_vllm_hcu_original_get_quant_method", original)
    setattr(config_class, "get_quant_method", hcu_get_quant_method)
    setattr(config_class, _CLASS_MARKER, True)
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
