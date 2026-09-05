# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""HCU compressed-tensors methods owned by the LightOp Marlin backend."""
import enum
import torch
from enum import Enum
from typing import Optional
from compressed_tensors.quantization import (QuantizationStrategy)
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from torch.nn.parameter import Parameter
from vllm.distributed import get_ep_group, get_dp_group
from vllm.model_executor.layers.fused_moe import (
    FusedMoEActivationFormat, FusedMoEMethodBase,
    FusedMoeWeightScaleSupported, FusedMoEConfig, RoutedExperts,
    SharedExperts)
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.fused_moe import config as fused_moe_config
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalizeModular,
    FusedMoeWeightScaleSupported,
)
from vllm_hcu.model_executor.layers.quantization.int8_runtime import (
    weight8bit_nt_kpack2_marlin2,
)
from vllm_hcu.model_executor.layers.quantization.lightop_marlin_moe_compat import (
    ensure_safe_marlin_moe_alignment,
)
logger = init_logger(__name__)

__all__ = [
    "CompressedTensorsW8A8Int8MarlinMoEMethod",
    "CompressedTensorsW8A8FP8MarlinMoEMethod",
]
# ── Weight layout helpers (Marlin interleave) ───────────────────────

def get_w8a8_int8_marlin_weights(
         weight,
         k_tile=64):
    if weight.dim() == 2:
        # [N, K] -> [K // k_tile, N * k_tile]
        weight = weight.T
        size_k, size_n = weight.shape
        assert size_k % k_tile == 0, (
            "Marlin W8A8 MoE weight K dimension must be divisible by "
            f"{k_tile}, got {size_k}."
        )
        weight = weight.reshape(size_k // k_tile, k_tile, size_n)
        weight = weight.transpose(1, 2)
        return weight.reshape(size_k // k_tile, size_n * k_tile)

    if weight.dim() == 3:
        # [E, N, K] -> [E, K // k_tile, N * k_tile]
        num_experts, size_n, size_k = weight.shape
        assert size_k % k_tile == 0, (
            "Marlin W8A8 MoE weight K dimension must be divisible by "
            f"{k_tile}, got {size_k}."
        )
        weight = weight.transpose(1, 2)
        weight = weight.reshape(
            num_experts, size_k // k_tile, k_tile, size_n
        )
        weight = weight.transpose(2, 3)
        return weight.reshape(
            num_experts, size_k // k_tile, size_n * k_tile
        )

    raise ValueError(f"Expected 2D or 3D weight, got {weight.dim()}D")


def w8a8_nt_kpack2_marlin_weight(w8a8_w, # [size_n, size_k// 2 ]
                                k_tile=16,
                                n_tile=16, ):
    assert w8a8_w.dtype == torch.int8, "w8a8_w 必须是 int8 类型"
    size_n, size_k = w8a8_w.shape
    assert size_n % k_tile == 0 and size_k % n_tile == 0, "k_tile / n_tile 必须能整除对应维度"

    w8a8_w = w8a8_w.reshape((size_n // n_tile,  n_tile, size_k // k_tile, k_tile))
    w8a8_w = w8a8_w.permute((0, 2, 1, 3)).contiguous()
    w8a8_w = w8a8_w.reshape((size_n // k_tile, size_k * k_tile))
    return w8a8_w

def fp32_to_fp8_e4m3fn(t: torch.Tensor) -> torch.Tensor:
    """更合理的FP32到Float8_e4m3fn转换，使用最近值而不是简单舍弃尾数"""
    # torch.float8_e4m3fn的数值范围约[-448, 448]
    fp8_min, fp8_max = -448.0, 448.0
    t_clamped = t.clamp(min=fp8_min, max=fp8_max)
    # 保证不会下溢到0
    # 转换前到float16再转fp8可能提升精度（float8实现本身通常通过float16做rounding）
    t_fp16 = t_clamped.to(torch.float16)
    return t_fp16.to(torch.float8_e4m3fn)


def w8a8_fp8_nt_kpack2_marlin_weight(w8a8_w,  # [size_n, size_k// 2 ]
                                     k_tile=16,
                                     n_tile=16, ):
    size_n, size_k = w8a8_w.shape
    assert size_n % k_tile == 0 and size_k % n_tile == 0, "k_tile / n_tile 必须能整除对应维度"

    w8a8_w = w8a8_w.reshape((size_n // n_tile, n_tile, size_k // k_tile, k_tile))
    w8a8_w = w8a8_w.permute((0, 2, 1, 3)).contiguous()
    w8a8_w = w8a8_w.reshape((size_n // k_tile, size_k * k_tile))
    return w8a8_w

class CompressedTensorsMarlinMoEMethod(FusedMoEMethodBase):
    def __init_(self, moe: FusedMoEConfig):
        super().__init__(moe)

    @property
    def supports_inplace_output(self) -> bool:
        return not self.use_deepep

    @staticmethod
    def _allows_inplace_output(
        x: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> bool:
        return (
            shared_experts is None
            or shared_experts_input is None
            or shared_experts.allows_inplace_routed_output(
                x,
                shared_experts_input,
            )
        )

    @staticmethod
    def get_moe_method(
        quant_config: "SlimQuantCompressedTensorsMarlinConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
    ) -> "CompressedTensorsMarlinMoEMethod":

        # are supported + check if the layer is being ignored.
        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get(
            "input_activations")
        if quant_config._is_fp8_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8FP8MarlinMoEMethod(quant_config, layer.moe_config)
        elif quant_config._is_dynamic_token_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8Int8MarlinMoEMethod(quant_config, layer.moe_config)
        else:
            raise RuntimeError(
                f"Slimquant_marlin does not support the FusedMoe scheme: {weight_quant}, {input_quant}")


class CompressedTensorsW8A8FP8MarlinMoEMethod(CompressedTensorsMarlinMoEMethod):
    def __init__(
            self,
            quant_config: "CompressedTensorsMarlinConfig",  # type: ignore # noqa E501
            moe: FusedMoEConfig
    ):
        self.quant_config = quant_config
        super().__init__(moe)
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get(
            "weights")
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get(
            "input_activations")

        per_channel = (
                self.weight_quant.strategy == QuantizationStrategy.CHANNEL
                and self.input_quant.strategy == QuantizationStrategy.TOKEN)
        if not per_channel:
            raise ValueError(
                "For FP8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}")

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For FP8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales.")
        self.fused_experts = self.fused_moe_forward
        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        self.dp_size = get_dp_group().world_size
        self.use_deepep = moe.moe_parallel_config.use_ep and \
            parallel_config.all2all_backend in (
                "deepep_high_throughput",
                "deepep_low_latency",
            )

        if self.use_deepep:
            all2all_manager = get_ep_group().device_communicator.all2all_manager
            assert all2all_manager is not None
            self.num_dispatchers = all2all_manager.world_size


    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> Optional[FusedMoEQuantConfig]:
        return fp8_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            per_act_token_quant=True,
            per_out_ch_quant=False,
            block_shape=None,
        )


    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        if self.use_deepep:
            self.N = 2 * intermediate_size_per_partition
            self.K = hidden_size

        params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
            requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
            requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        w13_weight_scale = torch.nn.Parameter(torch.ones(
            num_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=torch.float32),
            requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                        hidden_size,
                                                        1,
                                                        dtype=torch.float32),
                                             requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.use_deepep:
            # FP8 DeepEP (HT + LL): pack full [E, N, K] -> 6D layout.
            try:
                from deepgemm.m_group_gemm import (
                    pack_int8_weight_enk_to_w6_low_latency,
                )
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "HCU FP8 DeepEP weight packing requires the HCU-specific "
                    "deepgemm package; install the matching proprietary wheel"
                ) from error
            w1_marlin = pack_int8_weight_enk_to_w6_low_latency(layer.w13_weight)
            w2_marlin = pack_int8_weight_enk_to_w6_low_latency(layer.w2_weight)
            layer.w13_weight = Parameter(w1_marlin, requires_grad=False)
            layer.w2_weight = Parameter(w2_marlin, requires_grad=False)
            return

        # Repack the full [E, N, K] FP8 tensors in one view/transpose path.
        # This avoids both FP32 widening and per-expert stack temporaries.
        w1_marlin = get_w8a8_int8_marlin_weights(layer.w13_weight)
        w2_marlin = get_w8a8_int8_marlin_weights(layer.w2_weight)
        layer.w13_weight = Parameter(w1_marlin, requires_grad=False)
        layer.w2_weight = Parameter(w2_marlin, requires_grad=False)

    def fused_moe_forward(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            global_num_experts: int = -1,
            expert_map: Optional[torch.Tensor] = None,
            apply_router_weight_on_input: bool = False,
            activation: str = "silu",
            routed_scaling_factor: Optional[float] = None,
            shared_output: Optional[torch.Tensor] = None,
            i_q: torch.Tensor | None = None,
            i_s: torch.Tensor | None = None,
            inplace: bool = True,
    ):
        from lightop.moe import fused_experts_impl_fp8_marlin
        ensure_safe_marlin_moe_alignment(fused_experts_impl_fp8_marlin)
        return fused_experts_impl_fp8_marlin(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=inplace,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_fp8_w8a8=True,
            per_channel_quant=True,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            i_q=i_q,
            i_s=i_s,
            shared_output=shared_output,
            routed_scaling_factor=routed_scaling_factor)

    def apply(
            self,
            layer: RoutedExperts,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts: SharedExperts | None,
            shared_experts_input: torch.Tensor | None,
            i_q: torch.Tensor | None = None,
            i_s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inplace = self._allows_inplace_output(
            x,
            shared_experts,
            shared_experts_input,
        )
        return self.fused_experts(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            activation=layer.activation.value,
            routed_scaling_factor=1.0,
            shared_output=None,
            i_q=i_q,
            i_s=i_s,
            inplace=inplace,
        )

    @property
    def supports_eplb(self) -> bool:
        return True

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> FusedMoEExpertsModular:
        from vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
            BatchedDeepGemmExperts,
        )
        from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
            DeepGemmExperts,
        )

        if (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            max_num_tokens_per_rank = prepare_finalize.max_num_tokens_per_rank()
            assert max_num_tokens_per_rank is not None

            logger.debug("BatchedDeepGemmExperts(%s)", self.__class__.__name__)
            return BatchedDeepGemmExperts(
                moe_config=self.moe,
                max_num_tokens=max_num_tokens_per_rank,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                quant_config=self.moe_quant_config,
            )

        else:
            logger.debug("DeepGemmExperts(%s)", self.__class__.__name__)
            return DeepGemmExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
            )


class CompressedTensorsW8A8Int8MarlinMoEMethod(CompressedTensorsMarlinMoEMethod):
    def __init__(
            self,
            quant_config: "CompressedTensorsMarlinConfig",  # type: ignore # noqa E501
            moe: FusedMoEConfig
    ):
        self.quant_config = quant_config
        super().__init__(moe)
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get(
            "weights")
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get(
            "input_activations")

        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN)
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}")

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales.")

        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        self.dp_size = get_dp_group().world_size
        self.use_deepep = self.dp_size > 1 and parallel_config.enable_expert_parallel and \
            parallel_config.all2all_backend in (
                "deepep_high_throughput",
                "deepep_low_latency",
            )
        if self.use_deepep:
            all2all_manager = get_ep_group().device_communicator.all2all_manager
            assert all2all_manager is not None
            self.num_dispatchers = all2all_manager.world_size

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return fused_moe_config.int8_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            per_act_token_quant=True,
            block_shape=None,
        )

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        if self.use_deepep:
            self.N = 2 * intermediate_size_per_partition
            self.K = hidden_size

        params_dtype = torch.int8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        w13_weight_scale = torch.nn.Parameter(torch.ones(
            num_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=torch.float32),
                                              requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                        hidden_size,
                                                        1,
                                                        dtype=torch.float32),
                                             requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # LightOp Marlin weight interleave path.
        #if not self.use_deepep:
        w1_marlin_list = []
        for ii in range(layer.w13_weight.shape[0]):
            if not self.use_deepep:
                w1_marlin_in = get_w8a8_int8_marlin_weights(layer.w13_weight[ii])
            else:
                w1_marlin_in = weight8bit_nt_kpack2_marlin2(layer.w13_weight[ii])
            w1_marlin_list.append(w1_marlin_in)
        w1_marlin = torch.stack(w1_marlin_list, dim=0)

        del w1_marlin_list
        w2_marlin_list = []
        for ii in range(layer.w2_weight.shape[0]):
            if not self.use_deepep:
                w2_marlin_in = get_w8a8_int8_marlin_weights(layer.w2_weight[ii])
            else:
                w2_marlin_in = weight8bit_nt_kpack2_marlin2(layer.w2_weight[ii])
            w2_marlin_list.append(w2_marlin_in)
        w2_marlin = torch.stack(w2_marlin_list, dim=0)

        layer.w13_weight = Parameter(w1_marlin, requires_grad=False)
        layer.w2_weight = Parameter(w2_marlin, requires_grad=False)

    # ── apply ───────────────────────────────────────────────────────
    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
        i_q: torch.Tensor | None = None,
        i_s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # LightOp Marlin INT8 path.
        from lightop.moe import fused_experts_impl_int8_marlin
        ensure_safe_marlin_moe_alignment(fused_experts_impl_int8_marlin)
        inplace = self._allows_inplace_output(
            x,
            shared_experts,
            shared_experts_input,
        )
        return fused_experts_impl_int8_marlin(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=inplace,
            activation=layer.activation.value,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            use_int8_w8a8=True,
            per_channel_quant=True,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            i_q=i_q,
            i_s=i_s,
            shared_output=None,
            routed_scaling_factor=1.0,
        )

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> FusedMoEExpertsModular:
        from vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
            BatchedDeepGemmExperts,
        )
        from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
            DeepGemmExperts,
        )

        if (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            max_num_tokens_per_rank = prepare_finalize.max_num_tokens_per_rank()
            assert max_num_tokens_per_rank is not None

            logger.debug("BatchedDeepGemmExperts(%s)", self.__class__.__name__)
            return BatchedDeepGemmExperts(
                moe_config=self.moe,
                max_num_tokens=max_num_tokens_per_rank,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                quant_config=self.moe_quant_config,
            )

        else:
            logger.debug("DeepGemmExperts(%s)", self.__class__.__name__)
            return DeepGemmExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
            )
