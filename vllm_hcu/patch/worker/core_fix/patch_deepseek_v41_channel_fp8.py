# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route DeepSeek V4.1 Channel-FP8 layers to compressed-tensors on HCU.

The V4.1 Flash Channel-FP8 checkpoint declares ``quant_method=fp8`` with
``is_per_channel=true`` and stores one FP32 scale per output channel
(``[out, 1]``).  The upstream native ``Fp8Config`` path only understands
tensor and block weight scales, so on HCU every scaled-mm candidate kernel
rejects it before any weight is loaded.

The HCU plugin already carries the channel-wise FP8 machinery
(``CompressedTensorsW8A8Fp8``, ``ChannelWiseTorchFP8ScaledMMLinearKernel``,
``runtime_compat.scaled_mm`` and the Channel-FP8 MoE patches).  This adapter
only supplies the missing routing: when the current model is a Channel-FP8
V4.1 checkpoint, ordinary Linear layers and routed experts are served through
the compressed-tensors channel scheme.  Everything else keeps the official
``DeepseekV4FP8Config`` behaviour.
"""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v41.quant_config"
PATCH_ID = "worker.core_fix.deepseek_v41.channel_fp8_quant_method"
TARGETS = (
    f"{TARGET_MODULE}.DeepseekV4FP8Config.get_quant_method",
)
_CLASS_MARKER = "_vllm_hcu_dsv41_channel_fp8_applied"
_GET_MARKER = "_vllm_hcu_dsv41_channel_fp8_get_quant_method_wrapper"
_CHANNEL_CONFIG_ATTR = "_vllm_hcu_dsv41_channel_config"
_IS_CHANNEL_ATTR = "_vllm_hcu_dsv41_channel_fp8"

# ``wo_a`` ships as BF16 without a scale in the Channel-FP8 checkpoint and the
# official ROCm path consumes it as a plain BF16 linear.
_WO_A_SUFFIX = ".wo_a"

_CHANNEL_GROUP = {
    "targets": ["Linear"],
    "weights": {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "dynamic": False,
        "strategy": "channel",
        "group_size": None,
    },
    "input_activations": {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "dynamic": True,
        "strategy": "token",
        "group_size": None,
    },
}


def _channel_fp8_config_dict() -> dict:
    """Build the compressed-tensors config equivalent to the checkpoint.

    ``ignore`` keeps BF16 ``wo_a`` and the MoE router/gate out of the scheme;
    routed experts are matched through the ``Linear`` target expansion that
    ``CompressedTensorsMoEMethod.get_moe_method`` performs.
    """

    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "config_groups": {"group_0": dict(_CHANNEL_GROUP)},
        "ignore": ["re:.*attn\\.wo_a.*"],
    }


def _is_channel_fp8_quant_config(config: object) -> bool:
    """Detect the V4.1 Channel-FP8 checkpoint schema.

    The predicate mirrors the sglang translation rule: ``fp8`` weights with
    ``is_per_channel`` and dynamic activations, without a block size.  Any
    other FP8 checkpoint (block, tensor, MXFP8, Quark) keeps the official
    behaviour.
    """

    return (
        isinstance(config, dict)
        and config.get("quant_method") in ("fp8", "deepseek_v4_fp8")
        and bool(config.get("is_per_channel"))
        and config.get("weight_block_size") is None
        and config.get("activation_scheme", "dynamic") == "dynamic"
    )


def _hf_quantization_config(self) -> object:
    """Read ``quantization_config`` from the active HF config if available."""

    try:
        from vllm.config import get_current_vllm_config

        hf_config = get_current_vllm_config().model_config.hf_config
    except Exception:
        return None
    return getattr(hf_config, "quantization_config", None)


def _resolve_channel_flag(self, quant_config: object) -> bool:
    """Latch the checkpoint schema decision on the config instance.

    The decision is read once from the HF config so a single model cannot
    switch routes between layers.
    """

    resolved = getattr(self, _IS_CHANNEL_ATTR, None)
    if resolved is not None:
        return resolved
    if quant_config is None:
        # vllm_config is not active yet; do not latch an incorrect decision.
        return False
    resolved = _is_channel_fp8_quant_config(quant_config)
    setattr(self, _IS_CHANNEL_ATTR, resolved)
    return resolved


def _channel_quant_config(self):
    """Create (once) the compressed-tensors config for this model."""

    config = getattr(self, _CHANNEL_CONFIG_ATTR, None)
    if config is not None:
        return config
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    config = CompressedTensorsConfig.from_config(_channel_fp8_config_dict())
    setattr(self, _CHANNEL_CONFIG_ATTR, config)
    return config


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    config_class = require_class(
        target,
        "DeepseekV4FP8Config",
        f"{TARGET_MODULE}.DeepseekV4FP8Config",
    )
    original = require_callable(
        config_class, "get_quant_method", TARGETS[0]
    )
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("self", "layer", "prefix"),
    )

    if getattr(config_class, _CLASS_MARKER, False):
        if not getattr(original, _GET_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale; "
                "restart the process"
            )
        return False

    @functools.wraps(original)
    def hcu_get_quant_method(self, layer, prefix):
        if not _resolve_channel_flag(self, _hf_quantization_config(self)):
            return original(self, layer, prefix)

        from vllm.model_executor.layers.fused_moe import RoutedExperts
        from vllm.model_executor.layers.linear import (
            LinearBase,
            UnquantizedLinearMethod,
        )

        if isinstance(layer, LinearBase):
            if prefix.endswith(_WO_A_SUFFIX):
                return UnquantizedLinearMethod()
            return _channel_quant_config(self).get_quant_method(layer, prefix)

        if isinstance(layer, RoutedExperts):
            return _channel_quant_config(self).get_quant_method(layer, prefix)

        return original(self, layer, prefix)

    setattr(hcu_get_quant_method, _GET_MARKER, True)
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
