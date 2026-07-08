# SPDX-License-Identifier: Apache-2.0
"""
Patch for vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe
Enable AITER W4A16 MoE zero-point tensors for CompressedTensorsWNA16MoEMethod.
"""



PATCHES = [
(
"""
logger = init_logger(__name__)
""",
"""
logger = init_logger(__name__)
import vllm_hcu.platforms.envs as henvs
""",
),

(
"""
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Reconfigure packed weights and scales to match moe_wna16 format
""",
"""
        if henvs.VLLM_HCU_USE_CUSTOM_OPS and henvs.VLLM_HCU_USE_AITER_W4A16_MOE:
            logger.info_once("Using aiter moe")
            w1_zp = torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // self.group_size // 2 if self.num_bits == 4 else hidden_size // self.group_size,
                    dtype=torch.uint8,
                )
            if self.num_bits == 4: w1_zp[:] = 136
            w13_qzeros = torch.nn.Parameter(
                w1_zp,
                requires_grad=False,
            )
            layer.register_parameter("w13_qzeros", w13_qzeros)
            set_weight_attrs(w13_qzeros, extra_weight_attrs)

            w2_zp = torch.empty(
                    num_experts,
                    intermediate_size_per_partition,
                    hidden_size // self.group_size // 2 if self.num_bits == 4 else hidden_size // self.group_size,
                    dtype=torch.uint8,
                )
            if self.num_bits == 4: w2_zp[:] = 136
            w2_qzeros = torch.nn.Parameter(
                w2_zp,
                requires_grad=False,
            )
            layer.register_parameter("w2_qzeros", w2_qzeros)
            set_weight_attrs(w2_qzeros, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Reconfigure packed weights and scales to match moe_wna16 format
""",
),

(
"""
        return config_builder(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_zp=None,
            w2_zp=None,
            block_shape=[0, self.group_size],
        )
""",
"""        
        return config_builder(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_zp=layer.w13_qzeros if henvs.VLLM_HCU_USE_CUSTOM_OPS and henvs.VLLM_HCU_USE_AITER_W4A16_MOE else None,
            w2_zp=layer.w2_qzeros if henvs.VLLM_HCU_USE_CUSTOM_OPS and henvs.VLLM_HCU_USE_AITER_W4A16_MOE else None,
            block_shape=[0, self.group_size],
        )
""",
    ),
]

