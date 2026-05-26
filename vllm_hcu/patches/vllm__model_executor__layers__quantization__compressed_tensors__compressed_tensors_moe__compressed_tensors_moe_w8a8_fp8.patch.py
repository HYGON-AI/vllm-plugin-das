# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe_w8a8_fp8
"""

PATCHES = [
(
"""
logger = init_logger(__name__)
""",
"""
import vllm_hcu.platforms.envs as henvs
from vllm._aiter_ops import rocm_aiter_ops

logger = init_logger(__name__)
""",
),

(
"""
        self.input_quant = input_quant
""",
"""
        self.input_quant = input_quant

        self._aiter_moe_config_cache: dict[
            tuple[int, int, torch.dtype, str], object
        ] = {}
""",
),

(
"""
    def apply(
""",
"""
    def _get_aiter_moe_runtime_config(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        # Get or cache the AITER MoE runtime configuration for this layer.
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
            quant_type=MoeQuantType.FP8_W8A8,
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
        # Return weights, optionally shuffled for MoE_C solution type.
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

    def apply(
""",
),

(
"""
        assert not self.is_monolithic
""",
"""
        # AITER W8A8 MoE fast-path
        if henvs.VLLM_HCU_USE_AITER_W8A8_FP8_MOE and henvs.VLLM_HCU_USE_CUSTOM_OPS:
            from aiter.moe import aiter_moe

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
            routed_scaling_factor = None
            shared_output = None
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

        assert not self.is_monolithic
""",
),
]
