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
        "HYV3ForCausalLM", "vllm_hcu.models.hy_v3:HYV3ForCausalLM"
    )
    
    ModelRegistry.register_model(
        "HYV3MTPModel", "vllm_hcu.models.hy_v3_mtp:HYV3MTP"
    )


def register_quant_method():
    """to do"""
