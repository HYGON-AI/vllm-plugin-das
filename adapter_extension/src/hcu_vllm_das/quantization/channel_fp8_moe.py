# SPDX-License-Identifier: Apache-2.0
"""Owned Channel-FP8 RoutedExperts method with native CT parameter loaders."""
from __future__ import annotations

from compressed_tensors.quantization import QuantizationStrategy
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import CompressedTensorsMoEMethod
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8 import CompressedTensorsW8A8Fp8MoEMethod
from vllm.model_executor.utils import replace_parameter
from hcu_vllm_das.moe import channel_fp8_runtime as runtime
from hcu_vllm_das.moe.channel_fp8_prepare import (
    HcuChannelFP8HTPrepareAndFinalize, HcuChannelFP8LLPrepareAndFinalize,
    make_channel_prepare,
)
import torch


class HcuDSV41ChannelFP8MoEMethod(CompressedTensorsW8A8Fp8MoEMethod):
    """Bypass the block-only FP8 oracle while inheriting native create_weights."""

    def __init__(self, weight_quant, input_quant, moe, layer_name=None):
        if not (weight_quant.strategy == QuantizationStrategy.CHANNEL
                and input_quant.strategy == QuantizationStrategy.TOKEN
                and weight_quant.num_bits == input_quant.num_bits == 8
                and weight_quant.type == input_quant.type == "float"
                and not weight_quant.dynamic and input_quant.dynamic
                and weight_quant.block_structure is None):
            raise ValueError("HCU Channel-FP8 MoE requires static channel weights and dynamic token activations")
        if moe.moe_parallel_config.enable_eplb:
            raise ValueError("Channel-FP8 DeepGEMM EPLB is unsupported")
        if moe.activation.value != "silu" or not moe.is_act_and_mul:
            raise ValueError("Channel-FP8 DeepGEMM requires gated SiLU")
        if getattr(moe, "swiglu_alpha", None) is not None or getattr(moe, "swiglu_beta", None) is not None:
            raise ValueError("Channel-FP8 DeepGEMM does not support custom SiLU alpha/beta")
        CompressedTensorsMoEMethod.__init__(self, moe)
        self.weight_quant, self.input_quant = weight_quant, input_quant
        self.weight_block_size = None
        self.block_quant = self.static_input_scales = False
        self._layer_name = layer_name
        # Capture construction-time VllmConfig so AG-RS draft process_weights
        # can validate ownership after set_current_vllm_config restores target.
        from vllm.config import get_current_vllm_config

        self._config = get_current_vllm_config()
        self._backend_mode = runtime.select_channel_backend(moe.moe_parallel_config, requested=moe.moe_backend)
        self.experts_cls = runtime.ChannelFP8MaskedExperts if self._backend_mode == "ll" else runtime.ChannelFP8ContiguousExperts
        self._packed = None

    @property
    def supports_eplb(self):
        return False

    def get_fused_moe_quant_config(self, layer):
        if getattr(layer, "swiglu_alpha", None) is not None or getattr(layer, "swiglu_beta", None) is not None:
            raise ValueError("Channel-FP8 DeepGEMM does not support custom SiLU alpha/beta")
        return FusedMoEQuantConfig.make(
            torch.float8_e4m3fn, w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale, per_act_token_quant=True,
            per_out_ch_quant=True, block_shape=None, is_scale_swizzled=False,
            gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
        )

    def process_weights_after_loading(self, layer):
        layout = "masked" if self._backend_mode == "ll" else "contiguous"
        if self._packed is not None:
            runtime.pack_channel_weights(layer.w13_weight, layer.w2_weight,
                                         layer.w13_weight_scale, layer.w2_weight_scale,
                                         layout=layout, prior=self._packed)
            return
        # Prepare construction validates the route before destructive parameter replacement.
        quant_config = self.get_fused_moe_quant_config(layer)
        prepare = make_channel_prepare(
            self.moe, quant_config, layer._expert_routing_tables(), config=self._config
        )
        if self._backend_mode == "ll":
            expected = HcuChannelFP8LLPrepareAndFinalize
        elif self._backend_mode == "ht":
            expected = HcuChannelFP8HTPrepareAndFinalize
        else:
            from hcu_vllm_das.dspark_agrs import HcuChannelFP8AgRsPrepareAndFinalize

            expected = HcuChannelFP8AgRsPrepareAndFinalize
        if not isinstance(prepare, expected):
            raise runtime.ChannelFP8RuntimeError("Channel-FP8 prepare owner/backend mismatch")
        limits = dict(max_num_tokens=prepare.max_num_tokens_per_rank(), num_dispatchers=prepare.num_dispatchers()) if self._backend_mode == "ll" else {}
        packed = runtime.pack_channel_weights(layer.w13_weight, layer.w2_weight,
                                              layer.w13_weight_scale, layer.w2_weight_scale,
                                              layout=layout, **({"max_tokens": limits["max_num_tokens"], "num_dispatchers": limits["num_dispatchers"]} if limits else {}))
        for name, tensor in (("w13_weight", packed.w1), ("w2_weight", packed.w2),
                             ("w13_weight_scale", packed.s1), ("w2_weight_scale", packed.s2)):
            replace_parameter(layer, name, tensor)
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        experts = self.experts_cls(self.moe, self.moe_quant_config, **limits)
        experts.packed = packed
        packed.validate(layer.w13_weight, layer.w2_weight, layer.w13_weight_scale, layer.w2_weight_scale)
        self.moe_kernel = mk.FusedMoEKernel(prepare, experts)
        self._packed = packed
