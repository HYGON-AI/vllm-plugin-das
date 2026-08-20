# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

from vllm import ModelRegistry

def register_model():
    
    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM", "vllm_hcu.models.deepseek_v2:DeepseekV3ForCausalLM"
    )

    ModelRegistry.register_model(
        "DeepseekV32ForCausalLM", "vllm_hcu.models.deepseek_v2:DeepseekV3ForCausalLM"
    )

    ModelRegistry.register_model(
        "DeepSeekMTPModel", "vllm_hcu.models.deepseek_mtp:DeepSeekMTP"
    )

    ModelRegistry.register_model(
        "GlmMoeDsaForCausalLM", "vllm_hcu.models.deepseek_v2:GlmMoeDsaForCausalLM"
    )

    ModelRegistry.register_model(
        "Glm4MoeForCausalLM", "vllm_hcu.models.glm4_moe:Glm4MoeForCausalLM"
    )

    ModelRegistry.register_model(
        "Glm4MoeMTPModel", "vllm_hcu.models.glm4_moe_mtp:Glm4MoeMTP"
    )
    
    ModelRegistry.register_model(
        "HYV3ForCausalLM", "vllm_hcu.models.hy_v3:HYV3ForCausalLM"
    )
    
    ModelRegistry.register_model(
        "HYV3MTPModel", "vllm_hcu.models.hy_v3_mtp:HYV3MTP"
    )

    ModelRegistry.register_model(
        "DSparkDraftModel",
        "vllm_hcu.models.deepseek_v4_dspark:DSparkDeepseekV4ForCausalLM",
    )


def register_quant_method():
    """to do"""
