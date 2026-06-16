# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht:
- per-act-token + DeepEP prepare paths
- expert_alignment for INT8/FP8 Marlin HT (align with SGLang groupgemm path)
"""

PATCHES = [
(
"""
        if not quant_config.is_block_quantized and not defer_input_quant:
""",
"""
        if not quant_config.is_block_quantized and not defer_input_quant and not quant_config.is_per_act_token:
""",
),
(
"""
        if quant_config.is_block_quantized and not defer_input_quant:
""",
"""
        if (quant_config.is_block_quantized or quant_config.is_per_act_token) and not defer_input_quant:
""",
),
(
"""
            expert_alignment=1,
""",
"""
            expert_alignment=256 if ((quant_config.use_int8_w8a8 or quant_config.use_fp8_w8a8) and quant_config.is_per_act_token and not quant_config.is_block_quantized) else 1,
""",
),
]
