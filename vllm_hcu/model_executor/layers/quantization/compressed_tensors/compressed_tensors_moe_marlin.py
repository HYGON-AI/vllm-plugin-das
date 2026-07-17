# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AITER W8A8 MoE修改说明:
Patch for vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe_marlin
-- AITER W8A8 MoE 双后端方案

=== 替换方式 ===

本模块是 HCU 定制的 W8A8 MoE 实现，采用双后端设计，通过环境变量切换：

  - **AITER W8A8 MoE**（VLLM_ROCM_USE_AITER=1 + VLLM_ROCM_USE_AITER_MOE=1）
    通过 aiter.moe 的 hipBLASLt 底层算子执行 W8A8 INT8 MoE
  - **Marlin W8A8 MoE**（默认）
    通过 lmslim 的 Marlin 内核执行 INT8/FP8 MoE

=== 替换内容 ===

CompressedTensorsW8A8Int8MarlinMoEMethod 类实现：

    重写的方法                                  HCU 新增行为
    ───────────────────────────────────────    ──────────────────────────────────
    process_weights_after_loading(layer)       如果 AITER MoE 启用：
                                              → 预加载 aiter 模块，设置 MoE_C 缓存属性
                                              → 跳过 Marlin 权重重排
                                              否则 → Marlin interleave / kpack2 权重重排

    apply(layer, x, topk_weights,             如果 AITER MoE 启用：
         topk_ids, ...)                       → _get_aiter_moe_runtime_config() 获取配置
                                              → _get_aiter_weights_for_solution() 准备权重
                                              → 调用 aiter.moe.aiter_moe()
                                              否则 → 调用 lmslim fused_experts_impl_int8_marlin

    新增的辅助方法：
      _get_aiter_moe_runtime_config()     —— 获取 AITER MoE 运行时配置（带缓存）
      _get_aiter_weights_for_solution()   —— 按 solution_type 准备 MoE_C 重排权重

    新增的模块级辅助函数：
      _is_hcu_aiter_w8a8_moe_requested()  —— 环境变量检测

CompressedTensorsW8A8FP8MarlinMoEMethod 类：
  使用 Marlin FP8 路径（lmslim fused_experts_impl_fp8_marlin）

=== 环境变量 ===

    VLLM_ROCM_USE_AITER=1      启用 AITER 加速
    VLLM_ROCM_USE_AITER_MOE=1  启用 AITER W8A8 MoE 路径

=== 相比旧方案的优势 ===

  旧方案（v0.18.1）：直接内联在 upstream vllm 代码中，无法独立维护。
  新方案（canako.py 风格）：独立的 HCU 模块，清晰的 AITER / Marlin 双分支，
  通过环境变量一键切换，代码组织清晰、易于维护。
"""
import enum
import torch
from enum import Enum
from typing import Callable, Optional
from compressed_tensors.quantization import (QuantizationStrategy)
import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from torch.nn.parameter import Parameter
from vllm.distributed import get_ep_group, get_dp_group
from vllm.model_executor.layers.fused_moe import (
    FusedMoE, FusedMoEActivationFormat, FusedMoEMethodBase,
    FusedMoeWeightScaleSupported, FusedMoEConfig)
from vllm.model_executor.utils import set_weight_attrs
from vllm_hcu.model_executor.layers.quantization.int8_runtime import (
    weight8bit_nt_kpack2_marlin2,
)
from vllm.model_executor.layers.fused_moe import config as fused_moe_config
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalizeModular,
    FusedMoeWeightScaleSupported,
)
from deepgemm.m_group_gemm import pack_int8_weight_enk_to_w6_low_latency

logger = init_logger(__name__)

__all__ = [
    "CompressedTensorsW8A8Int8MarlinMoEMethod",
    "CompressedTensorsW8A8FP8MarlinMoEMethod",
]
# ── AITER W8A8 MoE env guard ────────────────────────────────────────

def _is_hcu_aiter_w8a8_moe_requested() -> bool:
    return envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_MOE


# ── Weight layout helpers (Marlin interleave) ───────────────────────

def get_w8a8_int8_marlin_weights(
         weight,
         k_tile=64):
    # 7168, 512
    weight = weight.T
    size_k, size_n = weight.shape
    assert size_k // k_tile
    weight = weight.reshape(size_k // k_tile, k_tile, size_n)
    weight = weight.transpose(1, 2)
    weight = weight.reshape(size_k // k_tile, size_n * k_tile)

    return weight


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
            w1_marlin = pack_int8_weight_enk_to_w6_low_latency(layer.w13_weight)
            w2_marlin = pack_int8_weight_enk_to_w6_low_latency(layer.w2_weight)
            layer.w13_weight = Parameter(w1_marlin, requires_grad=False)
            layer.w2_weight = Parameter(w2_marlin, requires_grad=False)
            return

        w1_marlin_list = []
        for ii in range(layer.w13_weight.shape[0]):
            w1_marlin_in = get_w8a8_int8_marlin_weights(layer.w13_weight[ii])
            w1_marlin_list.append(w1_marlin_in.float() if w1_marlin_in.dtype == torch.float8_e4m3fn else w1_marlin_in)
        w1_marlin = torch.stack(w1_marlin_list, dim=0)
        w1_marlin = fp32_to_fp8_e4m3fn(w1_marlin)

        del w1_marlin_list
        w2_marlin_list = []
        for ii in range(layer.w2_weight.shape[0]):
            w2_marlin_in = get_w8a8_int8_marlin_weights(layer.w2_weight[ii])
            w2_marlin_list.append(w2_marlin_in.float() if w2_marlin_in.dtype == torch.float8_e4m3fn else w2_marlin_in)
        w2_marlin = torch.stack(w2_marlin_list, dim=0)
        w2_marlin = fp32_to_fp8_e4m3fn(w2_marlin)
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
    ):
        from lmslim.layers.fused_moe.fuse_moe_fp8_marlin import fused_experts_impl_fp8_marlin
        return fused_experts_impl_fp8_marlin(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
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
            use_nn_moe=False,
            i_q=i_q,
            i_s=i_s,
            shared_output=shared_output,
            routed_scaling_factor=routed_scaling_factor)

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            global_num_experts: int = -1,
            expert_map: Optional[torch.Tensor] = None,
            custom_routing_function: Optional[Callable] = None,
            scoring_func: str = "softmax",
            e_score_correction_bias: Optional[torch.Tensor] = None,
            apply_router_weight_on_input: bool = False,
            activation: str = "silu",
            enable_eplb: bool = False,
            shared_experts_input: torch.Tensor | None = None,
            use_nn_moe: Optional[bool] = False,
            routed_scaling_factor: Optional[float] = None,
            use_fused_gate: Optional[bool] = False,
            expert_load_view: Optional[torch.Tensor] = None,
            logical_to_physical_map: Optional[torch.Tensor] = None,
            logical_replica_count: Optional[torch.Tensor] = None,
            shared_output: Optional[torch.Tensor] = None,
            i_q: torch.Tensor | None = None,
            i_s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.fused_experts(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            shared_output=shared_output,
            i_q=i_q,
            i_s=i_s, )

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
                N=self.N,
                K=self.K
            )

        else:
            logger.debug("DeepGemmExperts(%s)", self.__class__.__name__)
            return DeepGemmExperts(moe_config=self.moe,
                                   quant_config=self.moe_quant_config,
                                   N=self.N,
                                   K=self.K)


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

        # ── AITER W8A8 config cache ───────────────────────────
        self._aiter_moe_config_cache: dict[
            tuple[int, int, torch.dtype, str], object
        ] = {}

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
        # AITER W8A8 MoE fast-path: skip Marlin interleave, defer to AITER
        if _is_hcu_aiter_w8a8_moe_requested():
            if not rocm_aiter_ops.is_fused_moe_enabled():
                raise RuntimeError(
                    "VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_USE_AITER_MOE=1 "
                    "requested AITER W8A8 MoE, but rocm_aiter_ops fused MoE "
                    "support is unavailable."
                )
            if layer.apply_router_weight_on_input:
                raise RuntimeError(
                    "AITER W8A8 MoE does not support "
                    "apply_router_weight_on_input=True."
                )

            try:
                from aiter.ops.shuffle import (  # noqa: F401
                    moe_layout_shuffle_gemm1,
                    moe_layout_shuffle_gemm2,
                )
                from aiter.moe import (  # noqa: F401
                    MoeQuantType,
                    MoeSolutionType,
                    aiter_moe,
                    get_aiter_moe_config,
                )
            except Exception as exc:
                raise RuntimeError(
                    "AITER W8A8 MoE is enabled but required aiter modules "
                    "are unavailable."
                ) from exc

            setattr(layer, "_hcu_aiter_moe_c_w13_weight", None)
            setattr(layer, "_hcu_aiter_moe_c_w2_weight", None)
            return
        # Default Marlin weight interleave path
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

    # ── AITER W8A8 MoE runtime helpers ───────────────────────────────
    def _get_aiter_moe_runtime_config(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        """Get or cache the AITER MoE runtime configuration for this layer."""
        from aiter.moe import MoeQuantType, get_aiter_moe_config

        activation = getattr(layer.activation, "value", layer.activation)
        activation = str(activation)
        cache_key = (x.shape[0], topk_ids.shape[1], x.dtype, activation)
        moe_config = self._aiter_moe_config_cache.get(cache_key)
        if moe_config is not None:
            return moe_config

        status, moe_config = get_aiter_moe_config(
            M=x.shape[0],
            E=layer.w13_weight.shape[0],
            N1=layer.w13_weight.shape[1],
            N2=layer.w2_weight.shape[1],
            K=layer.w13_weight.shape[2],
            top_k=topk_ids.shape[1],
            block_size=0,
            dtype=x.dtype,
            quant_type=MoeQuantType.W8A8,
            activation=activation,
        )
        if not status:
            raise RuntimeError(
                "AITER W8A8 MoE did not find a valid backend config for "
                f"layer '{getattr(layer, 'layer_name', 'unknown')}' with "
                f"M={x.shape[0]}, top_k={topk_ids.shape[1]}, "
                f"dtype={x.dtype}."
            )

        self._aiter_moe_config_cache[cache_key] = moe_config
        return moe_config

    def _get_aiter_weights_for_solution(
        self,
        layer: FusedMoE,
        solution_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return weights, optionally shuffled for MoE_C solution type."""
        from aiter.moe import MoeSolutionType
        from aiter.ops.shuffle import (
            moe_layout_shuffle_gemm1,
            moe_layout_shuffle_gemm2,
        )

        if solution_type != MoeSolutionType.MOE_C:
            return layer.w13_weight, layer.w2_weight

        w1_moe_c = getattr(layer, "_hcu_aiter_moe_c_w13_weight", None)
        w2_moe_c = getattr(layer, "_hcu_aiter_moe_c_w2_weight", None)
        if w1_moe_c is not None and w2_moe_c is not None:
            return w1_moe_c, w2_moe_c

        with torch.no_grad():
            w1_moe_c = moe_layout_shuffle_gemm1(layer.w13_weight).view(
                *layer.w13_weight.shape
            )
            w2_moe_c = moe_layout_shuffle_gemm2(layer.w2_weight).view(
                *layer.w2_weight.shape
            )

        setattr(layer, "_hcu_aiter_moe_c_w13_weight", w1_moe_c)
        setattr(layer, "_hcu_aiter_moe_c_w2_weight", w2_moe_c)
        return w1_moe_c, w2_moe_c

    # ── apply ───────────────────────────────────────────────────────
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None = None,
        use_nn_moe: bool | None = False,
        i_q: torch.Tensor | None = None,
        i_s: torch.Tensor | None = None,
        shared_output: Optional[torch.Tensor] = None,
        routed_scaling_factor: Optional[float] = 1.0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # AITER W8A8 MoE fast-path
        if _is_hcu_aiter_w8a8_moe_requested():
            from aiter.moe import aiter_moe

            if not rocm_aiter_ops.is_fused_moe_enabled():
                raise RuntimeError(
                    "VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_USE_AITER_MOE=1 "
                    "requested AITER W8A8 MoE, but rocm_aiter_ops fused MoE "
                    "support is unavailable."
                )
            if layer.apply_router_weight_on_input:
                raise RuntimeError(
                    "AITER W8A8 MoE does not support "
                    "apply_router_weight_on_input=True."
                )

            moe_config = self._get_aiter_moe_runtime_config(
                layer, x, topk_ids
            )
            w1, w2 = self._get_aiter_weights_for_solution(
                layer, moe_config.solution_type
            )
            output = aiter_moe(
                hidden_states=x,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights.to(torch.float32),
                topk_ids=topk_ids.to(torch.int32),
                moe_config=moe_config,
                inplace=not self.moe.disable_inplace,
                activation=layer.activation.value,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                w1_zp=None,
                w2_zp=None,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                block_shape=None,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                routed_scaling_factor=(
                    routed_scaling_factor
                    if routed_scaling_factor is not None
                    else 1.0
                ),
            )
            if shared_output is not None:
                output = output + shared_output
            return output

        # Default Marlin INT8 path

        from lmslim.layers.fused_moe.fuse_moe_int8_marlin import fused_experts_impl_int8_marlin
        return fused_experts_impl_int8_marlin(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
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
            use_nn_moe=False,
            i_q=i_q,
            i_s=i_s,
            shared_output=shared_output,
            routed_scaling_factor=routed_scaling_factor,
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
                N=self.N,
                K=self.K
            )

        else:
            logger.debug("DeepGemmExperts(%s)", self.__class__.__name__)
            return DeepGemmExperts(moe_config=self.moe,
                                   quant_config=self.moe_quant_config,
                                   N=self.N,
                                   K=self.K)
