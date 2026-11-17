# SPDX-License-Identifier: Apache-2.0
"""Unregistered load-only SlimQuant parameter factories for vLLM 0.33.

No quantization config, model binding or executable inference path is
published here. These factories retain raw checkpoint representation; they
must not be selected in production before loader/IPC lifecycle validation.
"""
from __future__ import annotations
import torch
from vllm.model_executor.layers.linear import LinearMethodBase, LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
from vllm.model_executor.parameter import ModelWeightParameter, ChannelQuantScaleParameter
from vllm.model_executor.utils import set_weight_attrs


class SlimQuantW8A8LoadOnlyLinear(LinearMethodBase):
    """Create per-channel INT8 parameters with official shard loaders."""
    supports_pre_processed_weights = True
    requires_device_loading = False
    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **attrs):
        del input_size, output_size, params_dtype
        loader = attrs.get("weight_loader")
        if not callable(loader):
            raise ValueError("an official vLLM weight_loader is required")
        if input_size_per_partition <= 0 or not output_partition_sizes or any(n <= 0 for n in output_partition_sizes):
            raise ValueError("invalid linear shard dimensions")
        layer.logical_widths = list(output_partition_sizes)
        layer.register_parameter("weight", ModelWeightParameter(
            data=torch.empty(sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8),
            input_dim=1, output_dim=0, weight_loader=loader))
        layer.register_parameter("weight_scale", ChannelQuantScaleParameter(
            data=torch.empty(sum(output_partition_sizes), 1, dtype=torch.float32),
            output_dim=0, weight_loader=loader))
        layer.register_parameter("input_scale", None)
        layer.register_parameter("input_zero_point", None)

    def apply(self, layer, x, bias=None):
        raise RuntimeError("SlimQuant preload candidate does not support forward")


class SlimQuantW4A8LoadOnlyExperts(FusedMoEMethodBase):
    """Create raw signed high-nibble-first W4 expert parameters and scales."""
    supports_pre_processed_weights = True
    requires_device_loading = False
    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **attrs):
        del params_dtype
        if min(num_experts, hidden_size, intermediate_size_per_partition) <= 0:
            raise ValueError("expert dimensions must be positive")
        if hidden_size % 2 or intermediate_size_per_partition % 2:
            raise ValueError("packed INT4 dimensions must be even")
        if not callable(attrs.get("weight_loader")):
            raise ValueError("an official vLLM expert weight_loader is required")
        i, h, e = intermediate_size_per_partition, hidden_size, num_experts
        for name, shape in (("w13_weight", (e,2*i,h//2)), ("w2_weight", (e,h,i//2))):
            param = torch.nn.Parameter(torch.zeros(shape, dtype=torch.int8), requires_grad=False)
            layer.register_parameter(name, param)
            set_weight_attrs(param, dict(attrs))
        for name, shape in (("w13_weight_scale", (e,2*i,1)), ("w2_weight_scale", (e,h,1))):
            param = torch.nn.Parameter(torch.ones(shape, dtype=torch.float32), requires_grad=False)
            layer.register_parameter(name, param)
            set_weight_attrs(param, {**attrs, "quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        layer.register_parameter("w13_input_scale", None)
        layer.register_parameter("w2_input_scale", None)

    def get_fused_moe_quant_config(self, layer):
        raise RuntimeError("SlimQuant preload candidate has no inference backend")

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        raise RuntimeError("SlimQuant preload candidate does not support forward")


class Kimi31SlimQuantLoadOnlyConfig(QuantizationConfig):
    """Strict target precision selection, not a registered vLLM config.

    This candidate resolves only observed checkpoint-backed linear paths.
    Fused mixed KDA requires a split projection and is deliberately rejected.
    """
    _EXPECTED_IGNORES = frozenset({
        "lm_head", "embed_tokens", "vision_tower", "mm_projector",
        "block_sparse_moe.gate", "routed_expert_norm", "mlp.gate_proj",
        "mlp.up_proj", "mlp.down_proj", "self_attention_res_proj", "mlp_res_proj",
        "output_attn_res_proj", "self_attn.o_proj", "self_attn.kv_b_proj",
        "self_attn.f_a_proj", "self_attn.f_b_proj", "self_attn.g_proj",
        "self_attn.b_proj", "self_attn.q_conv1d", "self_attn.k_conv1d",
        "self_attn.v_conv1d", "self_attn.indexer", "self_attn.o_norm",
        "self_attn.q_a_layernorm", "self_attn.kv_a_layernorm",
    })
    def __init__(self, config):
        super().__init__()
        if config.get("quant_method") != "slimquant_w4a8" or config.get("activation_scheme") != "dynamic":
            raise ValueError("requires dynamic slimquant_w4a8")
        ignores = config.get("modules_to_not_convert")
        if not isinstance(ignores, list) or any(not isinstance(x, str) for x in ignores):
            raise ValueError("invalid modules_to_not_convert")
        if len(ignores) != len(self._EXPECTED_IGNORES) or frozenset(ignores) != self._EXPECTED_IGNORES:
            raise ValueError("unsupported SlimQuant ignore set; checkpoint must be audited")
        self.ignores = tuple(ignores)

    @classmethod
    def get_name(cls):
        return "slimquant_w4a8"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, RoutedExperts):
            if not prefix.endswith(".block_sparse_moe.experts"):
                raise ValueError(f"unsupported expert path: {prefix}")
            return SlimQuantW4A8LoadOnlyExperts(layer.moe_config)
        if isinstance(layer, LinearBase):
            if self.linear_precision(prefix) == "w8a8":
                return SlimQuantW8A8LoadOnlyLinear()
            return UnquantizedLinearMethod()
        return None

    def linear_precision(self, prefix):
        """Resolve canonical vLLM module names; reject unknown/fused layouts."""
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("missing linear prefix")
        if prefix.endswith(".self_attn.in_proj_qkvgfab"):
            raise ValueError("mixed KDA projection requires split INT8/BF16 loader")
        # K3.1's official module mapper renames indexer -> dsa_indexer.
        if any(part in ("." + prefix + ".") for part in (".dsa_indexer.", ".vision_tower.", ".mm_projector.")):
            return "unquantized"
        from .checkpoint_audit import _excluded_module
        if _excluded_module(prefix + ".weight", self.ignores):
            return "unquantized"
        if prefix.endswith(".mlp.gate_up_proj") and ".shared_experts." not in prefix:
            return "unquantized"
        known = (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj",
                 ".self_attn.q_a_proj", ".self_attn.q_b_proj",
                 ".self_attn.kv_a_proj_with_mqa", ".self_attn.fused_qkv_a_proj",
                 ".block_sparse_moe.routed_expert_down_proj",
                 ".block_sparse_moe.routed_expert_up_proj",
                 ".shared_experts.gate_up_proj", ".shared_experts.down_proj")
        if prefix.endswith(known):
            return "w8a8"
        raise ValueError(f"unsupported checkpoint linear module: {prefix}")
