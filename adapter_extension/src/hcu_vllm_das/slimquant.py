# SPDX-License-Identifier: Apache-2.0
"""SlimQuant checkpoint decoding with official CT linear and expert execution."""

from __future__ import annotations

import torch
from compressed_tensors.quantization import QuantizationArgs
from torch import nn

from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 import (
    CompressedTensorsW8A8Int8MoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)


def unpack_signed_int4(packed: torch.Tensor) -> torch.Tensor:
    """Decode high nibble first, keeping canonical signed [-8, 7] values."""
    if packed.dtype != torch.int8 or not packed.ndim:
        raise ValueError("SlimQuant packed weights must be non-scalar INT8 tensors")
    high = packed >> 4
    low = ((packed & 15) ^ 8) - 8
    return torch.stack((high, low), dim=-1).flatten(-2)


class SlimQuantInt8Experts(CompressedTensorsW8A8Int8MoEMethod):
    """Convert packed bytes and high-nibble scales for the official INT8 loader."""

    def create_weights(self, layer, *args, **kwargs):
        super().create_weights(layer, *args, **kwargs)
        for name in ("w13_weight", "w2_weight"):
            parameter = getattr(layer, name)
            original_loader = parameter.weight_loader

            def load(
                param,
                loaded_weight,
                *loader_args,
                _loader=original_loader,
                **loader_kwargs,
            ):
                return _loader(
                    param,
                    unpack_signed_int4(loaded_weight),
                    *loader_args,
                    **loader_kwargs,
                )

            parameter.weight_loader = load

        for name in ("w13_weight_scale", "w2_weight_scale"):
            parameter = getattr(layer, name)
            original_loader = parameter.weight_loader

            def load_scale(
                param,
                loaded_weight,
                *loader_args,
                _loader=original_loader,
                **loader_kwargs,
            ):
                # Checkpoint scales describe signed nibbles shifted left by4.
                # Canonical INT4 weights need the corresponding scale times16.
                return _loader(
                    param,
                    loaded_weight.float() * 16.0,
                    *loader_args,
                    **loader_kwargs,
                )

            parameter.weight_loader = load_scale


class SlimQuantConfig(CompressedTensorsConfig):
    """Distinct format; official CT quantization and MoE kernel selection remain owners."""

    def __init__(self, config):
        self.weight_quant = QuantizationArgs(
            num_bits=8, type="int", strategy="channel", symmetric=True, dynamic=False
        )
        self.input_quant = QuantizationArgs(
            num_bits=8, type="int", strategy="token", symmetric=True, dynamic=True
        )
        super().__init__(
            target_scheme_map={
                "Linear": {
                    "weights": self.weight_quant,
                    "input_activations": self.input_quant,
                    "format": "int-quantized",
                }
            },
            ignore=list(config.get("ignore") or []),
            quant_format="int-quantized",
        )
        mixed = config.get("mixed_precision") or {}
        experts = mixed.get("experts") or {}
        if (
            experts.get("format"),
            experts.get("weight_bits"),
            experts.get("scope"),
        ) != ("pack-int4-nibble", 4, "per_channel"):
            raise ValueError(
                "SlimQuant requires declared per-channel packed INT4 experts"
            )
        self.expert_modules = list(experts.get("modules") or [])
        dense = mixed.get("w8a8_include") or {}
        if (
            dense.get("format"),
            dense.get("weight_bits"),
            dense.get("activation_bits"),
            dense.get("activation_scheme"),
        ) != ("int8-per-channel", 8, 8, "dynamic_per_token"):
            raise ValueError(
                "SlimQuant requires declared dynamic-token INT8 dense modules"
            )
        self.dense_modules = list(dense.get("modules") or [])
        ngram = mixed.get("ngram_embedding") or {}
        if (ngram.get("format"), ngram.get("weight_bits"), ngram.get("scope")) != (
            "int8",
            8,
            "per_channel",
        ):
            raise ValueError("SlimQuant requires declared per-row INT8 n-gram tables")
        self.ngram_modules = list(ngram.get("modules") or [])
        if not all((self.expert_modules, self.dense_modules, self.ngram_modules)):
            raise ValueError("SlimQuant mixed-precision module lists must be non-empty")

    @classmethod
    def from_config(cls, config):
        return cls(config)

    @classmethod
    def get_name(cls):
        return "slimquant_w4a8"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_config_filenames(cls):
        return []

    def apply_vllm_mapper(self, mapper):
        super().apply_vllm_mapper(mapper)
        self.expert_modules = mapper.apply_list(self.expert_modules)
        self.dense_modules = mapper.apply_list(self.dense_modules)
        self.ngram_modules = mapper.apply_list(self.ngram_modules)

    def _matches(self, prefix, patterns):
        return should_ignore_layer(
            prefix,
            ignore=patterns,
            fused_mapping=self.packed_modules_mapping,
            use_fnmatch=True,
        )

    def get_embedding_method(self, prefix, embedding_method_base):
        from vllm.models.qwen4_exp.common.ngram_embedding import (
            Qwen4ExpPLEPerRowEmbeddingMethod,
            Qwen4ExpPLEUnquantizedEmbeddingMethod,
        )

        if self._matches(prefix, self.ignore):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        context = get_current_vllm_config_or_none()
        model_config = getattr(context, "model_config", None)
        text_config = getattr(model_config, "hf_text_config", None)
        parts = getattr(text_config, "split_ngram_parts", None)
        if isinstance(parts, bool) or not isinstance(parts, int) or parts <= 0:
            raise ValueError(
                "SlimQuant PLE requires positive split_ngram_parts from "
                "the current model's text configuration"
            )
        # One runtime PLE parameter stores every serialized shard. Evaluate the
        # actual names, preserving Core's glob/regex semantics and each model's
        # own shard count, rather than treating shard_0 as a format proxy.
        shards = [f"{prefix}.shard_{index}" for index in range(parts)]
        ignored = [self._matches(name, self.ignore) for name in shards]
        if all(ignored):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if any(ignored):
            raise ValueError(f"SlimQuant does not support partially ignored PLE {prefix!r}")
        declared = [self._matches(name, self.ngram_modules) for name in shards]
        if not any(declared):
            return Qwen4ExpPLEUnquantizedEmbeddingMethod()
        if not all(declared):
            raise ValueError(f"SlimQuant does not support partially declared PLE {prefix!r}")
        return Qwen4ExpPLEPerRowEmbeddingMethod(torch.int8)

    def get_quant_method(self, layer: nn.Module, prefix: str):
        ignored = self._matches(prefix, self.ignore)
        if isinstance(layer, RoutedExperts):
            if ignored:
                return UnquantizedFusedMoEMethod(layer.moe_config)
            projections = [f"{prefix}.{name}" for name in ("gate_up_proj", "down_proj")]
            ignored_projections = [self._matches(name, self.ignore) for name in projections]
            if all(ignored_projections):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if any(ignored_projections):
                raise ValueError(
                    f"SlimQuant does not support partially ignored expert group {prefix!r}"
                )
            # RoutedExperts groups the serialized gate_up_proj/down_proj modules.
            if not all(
                self._matches(name, self.expert_modules) for name in projections
            ):
                raise ValueError(f"Undeclared SlimQuant expert group {prefix!r}")
            return SlimQuantInt8Experts(
                self.weight_quant, self.input_quant, layer.moe_config
            )
        if isinstance(layer, LinearBase):
            if ignored or not self._matches(prefix, self.dense_modules):
                return UnquantizedLinearMethod()
        return super().get_quant_method(layer, prefix)
