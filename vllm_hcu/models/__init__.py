# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

from vllm import ModelRegistry


def register_model():
    # This only arms a post-import patch on
    # vllm.model_executor.models.gemma4, so it stays inert until a Gemma4
    # model is actually loaded.
    from vllm_hcu.utils.gemma4_switches import (
        gemma4_bw1000_enabled,
        warn_retired_switches,
    )

    warn_retired_switches()
    if gemma4_bw1000_enabled():
        from vllm_hcu.models.gemma4 import register_attention

        register_attention()

    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM", "vllm_hcu.models.deepseek_v2:DeepseekV3ForCausalLM"
    )

    ModelRegistry.register_model(
        "DeepseekV32ForCausalLM", "vllm_hcu.models.deepseek_v2:DeepseekV3ForCausalLM"
    )

    ModelRegistry.register_model(
        "DeepSeekMTPModel", "vllm_hcu.models.deepseek_mtp:DeepSeekMTP"
    )

    # Official main selects this architecture for DeepSeek-V3.2 and
    # GLM-MoE-DSA MTP drafts.  Keep the official architecture contract while
    # routing execution through the HCU implementation used by the paired
    # v0.25.1 stack.
    ModelRegistry.register_model(
        "DeepseekV32MTPModel", "vllm_hcu.models.deepseek_mtp:DeepSeekMTP"
    )

    # W8A8 MTP checkpoints can use .scale for parameters named weight_scale.
    ModelRegistry.register_model(
        "DeepSeekV4MTPModel",
        "vllm_hcu.models.deepseek_v4_mtp_compat:DeepSeekV4MTP",
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

    ModelRegistry.register_model("HYV3MTPModel", "vllm_hcu.models.hy_v3_mtp:HYV3MTP")

    ModelRegistry.register_model(
        "HYV4ForCausalLM", "vllm_hcu.models.hy_v4:HYV4ForCausalLM"
    )

    ModelRegistry.register_model("HYV4MTPModel", "vllm_hcu.models.hy_v4:HYV4MTP")

    ModelRegistry.register_model(
        "DSparkDraftModel",
        "vllm_hcu.models.deepseek_v4_dspark:DSparkDeepseekV4ForCausalLM",
    )

    ModelRegistry.register_model(
        "Step5ForCausalLM", "vllm_hcu.models.step5_text:Step5ForCausalLM"
    )
    ModelRegistry.register_model(
        "Step5ForConditionalGeneration",
        "vllm_hcu.models.step5_vl:Step5ForConditionalGeneration",
    )
    ModelRegistry.register_model("Step5MTP", "vllm_hcu.models.step5_mtp:Step5MTP")


def register_quant_method():
    """to do"""
