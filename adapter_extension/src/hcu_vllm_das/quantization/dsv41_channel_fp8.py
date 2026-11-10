# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 Channel-FP8 schema routing without changing official config owners.

The native V4.1 configuration is subclassed, rather than reconstructing its
constructor parameter list, so native FP8 policy and future optional arguments
remain owned by vLLM. Only the reviewed channel schema changes linear/expert
routing. CT continues to own its quantization methods and kernel selection.
"""
from __future__ import annotations

from collections.abc import Mapping

from ..environment import _flag

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config
from vllm.platforms import current_platform

from hcu_vllm_das.dsv41_activation import QUANTIZATION_METHOD


def _field(config, name, default=None):
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _is_v41(hf_config) -> bool:
    for config in (hf_config, _field(hf_config, "text_config")):
        if config is None:
            continue
        if _field(config, "model_type") in ("deepseek_v41", "deepseek_v41_text"):
            return True
        if "DeepseekV41ForCausalLM" in (_field(config, "architectures") or ()):
            return True
    return False


def _is_channel_schema(config) -> bool:
    return (
        isinstance(config, dict)
        and config.get("quant_method") in ("fp8", "deepseek_v4_fp8")
        and config.get("is_per_channel") is True
        and config.get("weight_block_size") is None
        and config.get("activation_scheme", "dynamic") == "dynamic"
    )


def _is_hcu() -> bool:
    return current_platform.is_rocm() and current_platform.device_name == "hcu"


def _ct_description() -> dict:
    # Allocate each nested dictionary per configuration. The checkpoint and
    # another model's parsed scheme must not be mutated by CT or a mapper.
    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8, "type": "float", "symmetric": True,
                    "dynamic": False, "strategy": "channel", "group_size": None,
                },
                "input_activations": {
                    "num_bits": 8, "type": "float", "symmetric": True,
                    "dynamic": True, "strategy": "token", "group_size": None,
                },
            },
        },
        "ignore": [r"re:.*attn\.wo_a.*"],
    }


class HcuDSV41ChannelFP8Config(DeepseekV4FP8Config):
    """A distinct public configuration; native FP8 and CT remain unmodified."""

    @classmethod
    def get_name(cls):
        return QUANTIZATION_METHOD

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if (
            (not _flag("VLLM_HCU_DSV41_PCP_ATTENTION", default=False)
             and user_quant != QUANTIZATION_METHOD)
            or user_quant not in (None, "fp8", "deepseek_v4_fp8", QUANTIZATION_METHOD)
            or not _is_hcu()
            or not _is_v41(hf_config)
            or not _is_channel_schema(hf_quant_cfg)
        ):
            return None
        return QUANTIZATION_METHOD

    @classmethod
    def from_config(cls, config):
        if not _is_channel_schema(config):
            raise ValueError(
                "DSV4.1 Channel-FP8 requires channel weights and dynamic activations"
            )
        if not _is_hcu():
            raise ValueError("DSV4.1 Channel-FP8 routing requires the HCU platform")
        values = dict(config)
        values.setdefault("activation_scheme", "dynamic")
        return super().from_config(values)

    def _channel_config(self):
        config = getattr(self, "_hcu_channel_config", None)
        if config is None:
            config = CompressedTensorsConfig.from_config(_ct_description())
            self._hcu_channel_config = config
        # vLLM updates this mapping as models are constructed. Keep that
        # native state current without translating FP8 exclusions into CT's
        # different ignore syntax.
        config.packed_modules_mapping = self.packed_modules_mapping
        return config

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, RoutedExperts) and is_layer_skipped(
            prefix=prefix,
            ignored_layers=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            return super().get_quant_method(layer, prefix)
        if isinstance(layer, (LinearBase, RoutedExperts)) and is_layer_skipped(
            prefix=prefix,
            ignored_layers=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
            match_mode=self.ignored_layers_match_mode,
        ):
            return super().get_quant_method(layer, prefix)
        if isinstance(layer, LinearBase):
            if prefix.endswith(".wo_a"):
                return UnquantizedLinearMethod()
            channel_config = self._channel_config()
            method = channel_config.get_quant_method(layer, prefix)
            from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
                CompressedTensorsLinearMethod,
            )
            from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
                CompressedTensorsW8A8Fp8,
            )

            if isinstance(method, CompressedTensorsLinearMethod) and isinstance(
                layer.scheme, CompressedTensorsW8A8Fp8
            ):
                from .dsv41_channel_fp8_linear import HcuDSV41ChannelFP8LinearMethod

                return HcuDSV41ChannelFP8LinearMethod(channel_config)
            return method
        if isinstance(layer, RoutedExperts):
            if layer.moe_config.moe_backend != "deep_gemm_mega_moe":
                raise ValueError("HCU V4.1 PCP requires deep_gemm_mega_moe")
            from ..moe.v41_megamoe import HcuV41MegaMoEMethod
            scheme = self._channel_config().target_scheme_map["Linear"]
            return HcuV41MegaMoEMethod(scheme["weights"], scheme["input_activations"],
                                     layer.moe_config, prefix)
        return super().get_quant_method(layer, prefix)
