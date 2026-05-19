# SPDX-License-Identifier: Apache-2.0

"""
vllm.model_executor.layers.fused_moe.deepep_ht_prepare_finalize: per-act-token + DeepEP prepare paths
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
]
