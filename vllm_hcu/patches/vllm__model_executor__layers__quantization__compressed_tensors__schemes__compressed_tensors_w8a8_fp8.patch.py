# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8
"""

PATCHES = [
(
"""
logger = init_logger(__name__)
""",
"""
import vllm_hcu.platforms.envs as henvs

logger = init_logger(__name__)
""",
),

(
"""
            weight, weight_scale, input_scale = process_fp8_weight_channel_strategy(
                layer.weight, layer.weight_scale, getattr(layer, "input_scale", None)
            )
            weight = weight.t()
""",
"""        
            weight, weight_scale, input_scale = process_fp8_weight_channel_strategy(
                layer.weight, layer.weight_scale, getattr(layer, "input_scale", None)
            )
            if henvs.VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM:
                weight = weight.contiguous()
            else:
                weight = weight.t()
""",
    ),
]
