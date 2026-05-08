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


def register_quant_method():
    """to do"""
