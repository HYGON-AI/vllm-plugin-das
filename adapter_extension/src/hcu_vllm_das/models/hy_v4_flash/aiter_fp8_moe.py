# SPDX-License-Identifier: Apache-2.0
"""Model-local channel-FP8 AITER experts; Core retains loading and TP ownership."""

from copy import deepcopy

import torch
from hcu_vllm_das.moe import aiter_fp8 as common
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    AiterFp8ChannelExperts,
)
from vllm.model_executor.layers.fused_moe.layer import RoutedExperts
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    make_fp8_moe_kernel,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8 import (
    CompressedTensorsW8A8Fp8MoEMethod,
)

from .quantization import validate_flash_quant_parallelism

logger = init_logger(__name__)


def _problem(tokens, intermediate):
    return common.Problem(
        tokens,
        512,
        2 * intermediate,
        1536,
        1536,
        16,
        torch.bfloat16,
        activation="silu",
        gated=True,
        gemm1_limit=10.0,
        architecture="gfx938",
    )


def _profiles(problem, use_shuffle):
    return (
        common.QualifiedProfile(
            problem.weight_key,
            use_shuffle,
            (("SOL_ID1", 11007), ("SOL_ID2", 21000), ("BLOCK_SIZE_M", 32)),
            evidence="HYV4 Flash gfx938 TP4/TP8 FP8 channel clamp10 raw/shuffle native qualification",
            max_tokens=8192,
        ),
    )


def _is_gfx938():
    from .runtime import is_hcu_platform

    return bool(
        is_hcu_platform()
        and torch.version.hip
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).gcnArchName.split(":")[0]
        == "gfx938"
    )


def _validate_scheme(weight, activation):
    if not (
        activation is not None
        and weight.num_bits == activation.num_bits == 8
        and weight.type == activation.type == "float"
        and weight.strategy == "channel"
        and activation.strategy == "token"
        and weight.symmetric
        and activation.symmetric
        and not weight.dynamic
        and activation.dynamic
        and all(
            getattr(q, "block_structure", None) is None for q in (weight, activation)
        )
        and all(
            getattr(q, "group_size", None) in (None, -1) for q in (weight, activation)
        )
    ):
        raise ValueError(
            "Flash AITER requires symmetric channel FP8 / dynamic-token FP8"
        )


class HYV4FlashAiterFP8Config(CompressedTensorsConfig):
    """A private CT copy for routed experts; all other methods remain Core's."""

    @classmethod
    def for_moe(cls, source, *, use_shuffle=True):
        if type(use_shuffle) is not bool:
            raise TypeError("Flash AITER shuffle policy must be a boolean")
        if isinstance(source, cls):
            if source._use_shuffle != use_shuffle:
                raise ValueError("Flash AITER CT layout policy is already frozen")
            return source
        if not isinstance(source, CompressedTensorsConfig):
            raise TypeError("Flash AITER requires Core CompressedTensorsConfig")
        if (
            source.kv_cache_scheme is not None
            or source.transform_config is not None
            or source.online_quantization_config is not None
        ):
            raise ValueError(
                "Flash AITER does not admit CT transforms/KV/online quantization"
            )
        result = cls(
            target_scheme_map=deepcopy(source.target_scheme_map),
            ignore=deepcopy(source.ignore),
            quant_format=source.quant_format,
            config=deepcopy(source.config),
            total_num_heads=source.total_num_heads,
            total_num_kv_heads=source.total_num_kv_heads,
        )
        result.packed_modules_mapping = deepcopy(source.packed_modules_mapping)
        result._add_fused_moe_to_target_scheme_map()
        result._use_shuffle = use_shuffle
        return result

    def get_quant_method(self, layer, prefix):
        if not isinstance(layer, RoutedExperts):
            return super().get_quant_method(layer, prefix)
        current = get_current_vllm_config()
        validate_flash_quant_parallelism(current)
        if current.kernel_config.moe_backend != "aiter":
            raise ValueError("Flash AITER config requires explicit --moe-backend aiter")
        schemes = [
            self.get_scheme_dict(layer, f"{prefix}.{expert}.{projection}")
            for expert in range(layer.moe_config.num_logical_experts)
            for projection in (
                layer.ckpt_gate_proj_name,
                layer.ckpt_up_proj_name,
                layer.ckpt_down_proj_name,
            )
        ]
        if (
            not schemes
            or schemes[0] is None
            or any(s != schemes[0] for s in schemes[1:])
        ):
            raise ValueError(
                "Flash AITER requires identical non-ignored expert schemes"
            )
        scheme = schemes[0]
        if (
            scheme["format"] != "float-quantized"
            or scheme.get("output_activations") is not None
        ):
            raise ValueError("Flash AITER requires serialized float-quantized experts")
        weight, activation = scheme["weights"], scheme["input_activations"]
        _validate_scheme(weight, activation)
        return HYV4FlashAiterFP8Method(
            weight, activation, layer.moe_config, use_shuffle=self._use_shuffle
        )


class HYV4FlashAiterFP8Experts(AiterFp8ChannelExperts):
    """Raw FN FP8 ASM stages and clamped SiLU, inside the official modular runner."""

    use_shuffle = False

    @staticmethod
    def _supports_current_device():
        return _is_gfx938()

    @staticmethod
    def _supports_activation(activation):
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(parallel):
        return bool(
            parallel.tp_size in (4, 8)
            and parallel.dp_size == 1
            and not parallel.use_ep
            and not parallel.use_deepep_ht_kernels
            and not parallel.use_deepep_ll_kernels
            and not getattr(parallel, "use_mori_kernels", False)
        )

    def __init__(self, moe_config, quant_config):
        super().__init__(moe_config, quant_config)
        if not _is_gfx938():
            raise NotImplementedError("Flash AITER FP8 requires genuine gfx938 HCU")
        if (
            moe_config.moe_backend != "aiter"
            or moe_config.activation != MoEActivation.SILU
        ):
            raise ValueError("Flash AITER FP8 requires explicit AITER and gated SiLU")
        if self.activation_config.clamp_limit != 10.0:
            raise ValueError(
                "Flash AITER FP8 requires the qualified clamp10 activation"
            )
        tp_size = moe_config.moe_parallel_config.tp_size
        if tp_size not in (4, 8):
            raise ValueError("Flash AITER FP8 is qualified for ordinary TP4/TP8")
        self._intermediate = 1536 // tp_size
        self._prepared = None
        self._configs = {}

    def bind_prepared_weights(self, prepared, config):
        if bool(config.vendor_config.need_shuffle) != self.use_shuffle:
            raise ValueError("Flash AITER expert/weight layout mismatch")
        self._prepared = prepared
        self._configs = {config.problem.tokens: config}

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        if (
            activation != MoEActivation.SILU
            or global_num_experts != 512
            or expert_map is not None
            or expert_tokens_meta is not None
            or apply_router_weight_on_input
            or a1q_scale is not None
            or a2_scale is not None
        ):
            raise ValueError(
                "Flash AITER FP8 requires ordinary TP and unquantized BF16 inputs"
            )
        tokens = hidden_states.shape[0]
        if (
            hidden_states.ndim != 2
            or hidden_states.shape[1] != 1536
            or not 0 <= tokens <= 8192
            or hidden_states.dtype != torch.bfloat16
            or not hidden_states.is_contiguous()
            or tuple(w1.shape) != (512, 2 * self._intermediate, 1536)
            or tuple(w2.shape) != (512, 1536, self._intermediate)
            or tuple(topk_ids.shape) != (tokens, 16)
            or tuple(topk_weights.shape) != (tokens, 16)
        ):
            raise ValueError(
                "Flash AITER FP8 shape is outside the qualified TP4/TP8 top16 profile"
            )
        if tokens == 0:
            return
        if self._prepared is None:
            raise RuntimeError("Flash AITER weights were not prepared at postload")
        self._prepared.validate_bindings(
            (w1, w2, self.quant_config.w1_scale, self.quant_config.w2_scale)
        )
        config = self._configs.get(tokens)
        if config is None:
            problem = _problem(tokens, self._intermediate)
            config = common.get_config(
                problem,
                use_shuffle=self.use_shuffle,
                profiles=_profiles(problem, self.use_shuffle),
            )
            self._configs[tokens] = config
        common.run(
            hidden_states,
            topk_weights,
            topk_ids,
            self._prepared,
            config,
            output=output,
            routed_scaling_factor=1.0,
        )


class HYV4FlashAiterFP8ShuffledExperts(HYV4FlashAiterFP8Experts):
    use_shuffle = True


class HYV4FlashAiterFP8Method(CompressedTensorsW8A8Fp8MoEMethod):
    """Retain CT create_weights/apply; choose the model-owned modular expert."""

    def __init__(
        self, weight_quant, input_quant, moe, layer_name=None, *, use_shuffle=True
    ):
        if type(use_shuffle) is not bool:
            raise TypeError("Flash AITER shuffle policy must be a boolean")
        _validate_scheme(weight_quant, input_quant)
        CompressedTensorsMoEMethod.__init__(self, moe)
        self.weight_quant, self.input_quant = weight_quant, input_quant
        self.weight_block_size = None
        self.block_quant = self.static_input_scales = False
        self.fp8_backend = Fp8MoeBackend.AITER
        self._use_shuffle = use_shuffle
        self._prepared = None
        self.experts_cls = (
            HYV4FlashAiterFP8ShuffledExperts
            if use_shuffle
            else HYV4FlashAiterFP8Experts
        )

    @property
    def supports_eplb(self):
        return False

    def process_weights_after_loading(self, layer):
        current = get_current_vllm_config()
        validate_flash_quant_parallelism(current)
        tp_size = current.parallel_config.tensor_parallel_size
        if tp_size not in (4, 8) or not _is_gfx938():
            raise NotImplementedError(
                "Flash AITER channel FP8 is qualified for gfx938 ordinary TP4/TP8"
            )
        tensors = (
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        )
        if self._prepared is not None:
            if self._prepared.is_current(tensors):
                return
            raise ValueError(
                "AITER weights changed; construct a new method from canonical weights before capture"
            )
        if self.moe.in_dtype != torch.bfloat16:
            raise NotImplementedError(
                "Flash AITER channel FP8 currently requires BF16 inputs"
            )
        if (
            layer.activation != MoEActivation.SILU
            or getattr(layer, "swiglu_alpha", None) not in (None, 1.0)
            or getattr(layer, "swiglu_beta", None) not in (None, 0.0)
            or getattr(layer, "swiglu_limit", None) != 10.0
            or layer.apply_router_weight_on_input
            or layer.expert_map is not None
        ):
            raise ValueError(
                "Flash AITER requires gated clamp10 and output-side router weights"
            )
        weights = (layer.w13_weight, layer.w2_weight)
        scales = (layer.w13_weight_scale, layer.w2_weight_scale)
        intermediate = 1536 // tp_size
        expected = ((512, 2 * intermediate, 1536), (512, 1536, intermediate))
        if any(
            tuple(w.shape) != shape
            or w.dtype != torch.float8_e4m3fn
            or not w.is_contiguous()
            for w, shape in zip(weights, expected)
        ):
            raise ValueError("Flash AITER requires canonical raw FN FP8 TP4 weights")
        if any(
            tuple(s.shape) != (w.shape[0], w.shape[1], 1)
            or s.dtype != torch.float32
            or not s.is_contiguous()
            or s.device != w.device
            for s, w in zip(scales, weights)
        ):
            raise ValueError("Flash AITER requires canonical per-channel FP32 scales")
        problem = _problem(1, intermediate)
        selected = common.get_config(
            problem,
            use_shuffle=self._use_shuffle,
            profiles=_profiles(problem, self._use_shuffle),
        )
        self.experts_cls = (
            HYV4FlashAiterFP8ShuffledExperts
            if selected.vendor_config.need_shuffle
            else HYV4FlashAiterFP8Experts
        )
        prepared = common.prepare_weights(*tensors, selected)
        self._prepared = prepared.copy_into(tensors)
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = make_fp8_moe_kernel(
            self.moe_quant_config,
            self.moe,
            self.experts_cls,
            self.fp8_backend,
            routing_tables=layer._expert_routing_tables(),
        )
        self.moe_kernel.fused_experts.bind_prepared_weights(self._prepared, selected)
        logger.info_once(
            "HYV4 Flash MoE backend=aiter provider=%s config_source=%s shuffle=%s "
            "raw_fp8=E4M3FN w13=%s w2=%s top16 clamp10 router_factor=Core-owned",
            selected.vendor_config.solution_type,
            selected.provenance,
            bool(selected.vendor_config.need_shuffle),
            tuple(weights[0].shape),
            tuple(weights[1].shape),
        )
