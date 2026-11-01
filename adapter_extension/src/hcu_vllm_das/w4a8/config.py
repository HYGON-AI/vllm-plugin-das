"""SlimQuant serialization over the official compressed-tensors protocols.

Dense W8 and ignored layers use real CT schemes. Packed W4 is explicitly its
own format, never CT's int32 packed format. The official loader, model, PLE,
runner, and fallback modular INT8 experts retain ownership.
"""

from __future__ import annotations

import copy
import re
from fnmatch import translate

import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

from . import runtime


def args(bits, strategy, dynamic=False):
    return QuantizationArgs(
        num_bits=bits, type="int", strategy=strategy, symmetric=True, dynamic=dynamic
    )


def scheme(bits, format):
    return dict(
        weights=args(bits, "channel"),
        input_activations=args(8, "token", True),
        output_activations=None,
        format=format,
    )


class SlimQuantW4A8Config(CompressedTensorsConfig):
    @classmethod
    def get_name(cls):
        return "slimquant_w4a8"

    @classmethod
    def get_min_capability(cls):
        # HIP architecture/provider checks belong to the selected route.
        return 0

    @classmethod
    def from_config(cls, config):
        if config.get("quant_method") != cls.get_name():
            raise ValueError("Expected SlimQuant W4A8 checkpoint metadata")
        if config.get("activation_scheme", "dynamic") != "dynamic":
            raise ValueError("SlimQuant requires dynamic per-token INT8 activation")
        mixed = config.get("mixed_precision", {})
        experts = mixed.get("experts", {})
        if (
            experts.get("format") != "pack-int4-nibble"
            or experts.get("weight_bits") != 4
            or experts.get("scope") != "per_channel"
        ):
            raise ValueError("Only SlimQuant signed nibble/channel W4 is supported")
        if mixed.get("ngram_embedding") is not None:
            raise NotImplementedError(
                "INT8 PLE requires a separate embedding protocol gate"
            )
        dense = mixed.get("w8a8_include", {})
        if dense and any(
            dense.get(key) != value
            for key, value in (
                ("format", "int8-per-channel"),
                ("weight_bits", 8),
                ("activation_bits", 8),
                ("activation_scheme", "dynamic_per_token"),
                ("scope", "per_channel"),
            )
        ):
            raise ValueError("Unsupported mixed dense quantization metadata")
        obj = cls({}, list(config.get("ignore", [])), "pack-int4-nibble", config=config)
        obj._ignore_modules = list(config.get("ignore", []))
        obj._modules = {
            **{
                name: scheme(4, "pack-int4-nibble")
                for name in experts.get("modules", [])
            },
            **{name: scheme(8, "int-quantized") for name in dense.get("modules", [])},
        }
        if not experts.get("modules"):
            raise ValueError("Explicit packed expert targets are required")
        obj._refresh_targets()
        return obj

    def _refresh_targets(self):
        modules = dict(self._modules)
        # Describe logical expert projections too, for official mixed-precision
        # and shared-expert compatibility queries. Serialized bytes remain W4.
        for name, value in self._modules.items():
            if value["weights"].num_bits != 4:
                continue
            if name.endswith(".experts.gate_up_proj"):
                for suffix in ("gate_proj", "up_proj"):
                    modules[name.removesuffix("gate_up_proj") + "*." + suffix] = value
            elif name.endswith(".experts.down_proj"):
                modules[name.removesuffix("down_proj") + "*.down_proj"] = value
        self.target_scheme_map = {
            name if name.startswith("re:") else "re:" + translate(name): value
            for name, value in modules.items()
        }
        self.ignore = [
            name if name.startswith("re:") else "re:" + translate(name)
            for name in self._ignore_modules
        ]

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        # Map checkpoint paths before translating globs into CT regex targets.
        self._modules = hf_to_vllm_mapper.apply_dict(self._modules)
        self._ignore_modules = hf_to_vllm_mapper.apply_list(self._ignore_modules)
        self._refresh_targets()

    @staticmethod
    def get_checkpoint_weight_mapper():
        from vllm.model_executor.models.utils import WeightsMapper

        return CompressedTensorsConfig.get_checkpoint_weight_mapper() | WeightsMapper(
            orig_to_new_regex={
                re.compile(
                    r"(\.experts\.(?:gate_up_proj|down_proj))\.weight_scale$"
                ): r"\1_scale",
            },
        )

    def get_quant_method(self, layer, prefix):
        from . import lifecycle

        lifecycle.require_selection()
        if not isinstance(layer, RoutedExperts):
            return super().get_quant_method(layer, prefix)
        if should_ignore_layer(prefix, self.ignore, self.packed_modules_mapping):
            return UnquantizedFusedMoEMethod(layer.moe_config)
        projection_schemes = [
            self.get_scheme_dict(layer, f"{prefix}.{name}")
            for name in ("gate_up_proj", "down_proj")
        ]
        projection_schemes += [
            self.get_scheme_dict(layer, f"{prefix}.0.{name}")
            for name in ("gate_proj", "up_proj", "down_proj")
        ]
        if any(value != projection_schemes[0] for value in projection_schemes[1:]):
            raise ValueError("All routed expert projections require the same scheme")
        if projection_schemes[0] is None:
            return UnquantizedFusedMoEMethod(layer.moe_config)
        if projection_schemes[0]["weights"].num_bits != 4:
            raise ValueError("SlimQuant routed experts must use packed W4")
        native = (
            runtime.prepare_native(layer.moe_config)
            if (lifecycle.native_requested() and not layer.apply_router_weight_on_input)
            else None
        )
        method = SlimQuantW4A8MoEMethod(layer.moe_config, native)
        lifecycle.record(prefix, method)
        return method

    def get_linear_method(self):
        return SlimQuantW4A8LinearMethod(self)


class SlimQuantW4A8LinearMethod(CompressedTensorsLinearMethod):
    """Keep the official tensor path; consume legacy quant tuples explicitly.

    The a198 INT8 kernel does not consume QuantizedActivation yet. Its public
    scaled GEMM is reused for the same dynamic symmetric token/channel math.
    Never discard a supplied scale or silently quantize INT8 a second time.
    """

    def supports_quanted_inputs(self):
        return True

    def apply(self, layer, x, bias=None, x_and_scale_quanted=None):
        from vllm.model_executor.layers.fusion.quant_activation import (
            QuantizedActivation,
            as_quantized_activation,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kInt8DynamicTokenSym,
        )

        if isinstance(x, QuantizedActivation):
            if x_and_scale_quanted is not None:
                raise runtime.W4A8Error("Provide only one prequantized activation")
            qa = as_quantized_activation(x, kInt8DynamicTokenSym)
            quantized, scale, dtype, shape = (
                qa.data,
                qa.scale,
                qa.orig_dtype,
                qa.orig_shape,
            )
        elif x_and_scale_quanted is not None:
            quantized, scale = x_and_scale_quanted
            dtype, shape = x.dtype, x.shape
        else:
            return super().apply(layer, x, bias)
        from vllm.model_executor.kernels.linear.scaled_mm.triton import (
            TritonInt8ScaledMMLinearKernel,
        )
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
            triton_scaled_mm,
        )

        if not isinstance(layer.scheme.kernel, TritonInt8ScaledMMLinearKernel):
            raise runtime.W4A8Error(
                "Prequantized W8 input requires the official Triton INT8 kernel"
            )
        if (
            quantized.dtype != torch.int8
            or scale.dtype != torch.float32
            or tuple(quantized.shape) != tuple(shape)
            or scale.numel() != quantized.numel() // shape[-1]
            or dtype not in (torch.bfloat16, torch.float16)
        ):
            raise runtime.W4A8Error("Expected INT8 activation and FP32 per-token scale")
        return triton_scaled_mm(
            quantized.reshape(-1, shape[-1]),
            layer.weight,
            scale_a=scale.reshape(-1, 1),
            scale_b=layer.weight_scale,
            out_dtype=dtype,
            bias=bias,
        ).reshape(*shape[:-1], layer.weight.shape[-1])


class SlimQuantW4A8MoEMethod(FusedMoEMethodBase):
    def __init__(self, moe, native):
        super().__init__(moe)
        self.native = native
        self._processed = False
        self.delegate = None
        if native is None:
            if moe.moe_backend not in ("auto", "triton"):
                raise runtime.W4A8Error("Explicit backend cannot be silently replaced")
            # Spawn unpickles this config before EngineCore loads plugins.
            # Only the selected fallback needs the INT8 oracle's consumer.
            from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 import (  # noqa: E501
                CompressedTensorsW8A8Int8MoEMethod,
            )

            fallback = copy.copy(moe)
            fallback.moe_backend = "triton"
            self.delegate = CompressedTensorsW8A8Int8MoEMethod(
                args(8, "channel"),
                args(8, "token", True),
                fallback,
            )

    @property
    def topk_indices_dtype(self):
        if self.native is not None:
            return torch.int32
        return super().topk_indices_dtype

    @property
    def skip_forward_padding(self):
        # MoERunner owns the original-width router/shared expert. Physical
        # native weight padding must start only after those consumers run.
        return self.native is not None

    @property
    def has_unpadded_output(self):
        return self.native is not None

    def maybe_roundup_sizes(
        self,
        hidden_size,
        intermediate_size_per_partition,
        act_dtype,
        moe_parallel_config,
    ):
        if hidden_size % 2 or intermediate_size_per_partition % 2:
            raise runtime.W4A8Error("TP must preserve whole packed nibble pairs")
        if self.native is not None:
            return self.native.shape.hidden, self.native.shape.intermediate
        return hidden_size, intermediate_size_per_partition

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if not self.moe.is_act_and_mul or self.moe.has_bias:
            raise runtime.W4A8Error(
                "This checkpoint ABI requires gated bias-free experts"
            )
        e, h, i = num_experts, hidden_size, intermediate_size_per_partition
        if h % 2 or i % 2:
            raise runtime.W4A8Error("Odd packed K is unsupported")
        for name, shape in (
            ("w13_weight", (e, 2 * i, h // 2)),
            ("w2_weight", (e, h, i // 2)),
        ):
            # Official TP loading narrows padded destinations to checkpoint
            # width. Initialize the untouched tail to zero for both GEMMs.
            p = torch.nn.Parameter(
                torch.zeros(shape, dtype=torch.int8), requires_grad=False
            )
            layer.register_parameter(name, p)
            set_weight_attrs(p, extra_weight_attrs)
        attrs = dict(
            extra_weight_attrs, quant_method=FusedMoeWeightScaleSupported.CHANNEL.value
        )
        for name, shape in (
            ("w13_weight_scale", (e, 2 * i, 1)),
            ("w2_weight_scale", (e, h, 1)),
        ):
            p = torch.nn.Parameter(
                torch.ones(shape, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter(name, p)
            set_weight_attrs(p, attrs)
        layer.w13_input_scale = layer.w2_input_scale = None

    def get_fused_moe_quant_config(self, layer):
        if self.delegate is not None:
            return self.delegate.get_fused_moe_quant_config(layer)
        from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

        return FusedMoEQuantConfig.make(
            torch.int8,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            per_act_token_quant=True,
            weight_dtype="int4",
        )

    def process_weights_after_loading(self, layer):
        from . import lifecycle

        lifecycle.require_selection()
        if self._processed:
            raise runtime.W4A8Error("Packed W4 parameters have already been processed")
        e, h, i = (
            self.moe.num_local_experts,
            self.moe.hidden_dim,
            self.moe.intermediate_size_per_partition,
        )
        weights = (layer.w13_weight, layer.w2_weight)
        scales = (layer.w13_weight_scale, layer.w2_weight_scale)
        shapes = ((e, 2 * i, h // 2), (e, h, i // 2))
        for w, s, shape in zip(weights, scales, shapes):
            if (
                w.dtype != torch.int8
                or tuple(w.shape) != shape
                or not w.is_contiguous()
            ):
                raise runtime.W4A8Error("Expected canonical packed SlimQuant weights")
            if s.dtype != torch.float32 or tuple(s.shape) != shape[:2] + (1,):
                raise runtime.W4A8Error("Expected FP32 per-output-channel scales")
        if layer.w13_input_scale is not None or layer.w2_input_scale is not None:
            raise runtime.W4A8Error("Static activation scales are unsupported")
        if self.native is None:
            converted, converted_scales = runtime.int8_weights_and_scales(
                weights, scales
            )
        else:
            converted, converted_scales = self.native.pack(weights), scales
        prepared = copy.copy(layer)
        prepared._parameters = layer._parameters.copy()
        for name, value in zip(("w13_weight", "w2_weight"), converted):
            replace_parameter(prepared, name, value)
        for name, value in zip(
            ("w13_weight_scale", "w2_weight_scale"), converted_scales
        ):
            replace_parameter(prepared, name, value)
        if self.native is None:
            self.delegate.process_weights_after_loading(prepared)
        quant_config = self.get_fused_moe_quant_config(prepared)
        # Kernel imports, allocation, packing, and configuration all succeeded
        # against the private module before replacing any canonical parameter.
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            replace_parameter(layer, name, getattr(prepared, name))
        if self.native is None:
            self.moe_kernel = self.delegate.moe_kernel
        self.moe_quant_config = quant_config
        self._processed = True

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if not self._processed:
            raise runtime.W4A8Error("W4A8 post-load processing has not completed")
        if self.delegate is not None:
            return self.delegate.apply(
                layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
            )
        if layer.expert_map is not None or layer.apply_router_weight_on_input:
            raise runtime.W4A8Error("Native W4A8 EP/router-weight-on-input not enabled")
        logical_hidden = self.moe.hidden_dim_unpadded
        if x.ndim != 2 or x.shape[-1] != logical_hidden:
            raise runtime.W4A8Error("Native W4A8 requires the logical hidden width")
        if not x.is_contiguous():
            raise runtime.W4A8Error("Native W4A8 requires contiguous logical input")
        if self.native.shape.hidden < logical_hidden:
            raise runtime.W4A8Error("Native W4A8 physical width is smaller than logical")
        padded = (
            torch.nn.functional.pad(x, (0, self.native.shape.hidden - logical_hidden))
            if self.native.shape.hidden != logical_hidden
            else x
        )
        output = self.native.run(
            padded,
            layer.w13_weight,
            layer.w2_weight,
            (layer.w13_weight_scale, layer.w2_weight_scale),
            topk_weights,
            topk_ids,
            global_experts=layer.global_num_experts,
        )
        if x.shape[0] == 0:
            return torch.empty_like(x)
        # Match the official custom-op fake's logical shape and strides before
        # shared expert combination, all-reduce, and the next model layer.
        if self.native.shape.hidden != logical_hidden:
            # contiguous() may keep the physical row stride when M=1.
            return output[..., :logical_hidden].clone(
                memory_format=torch.contiguous_format
            )
        return output
