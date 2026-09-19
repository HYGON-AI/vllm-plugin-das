"""Kimi-K3 model-config validation owned by the HCU plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.model_executor.models.config import VerifyAndUpdateConfig

from vllm_hcu.model_executor.layers.quantization.kimi_k3_w4a8 import (
    KIMI_K3_W4A8_QUANT_METHOD,
    SLIMQUANT_W4A8_QUANT_METHOD,
    validate_kimi_k3_w4a8_metadata,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig


class KimiK3ConfigAdapter(VerifyAndUpdateConfig):
    """Normalize SlimQuant metadata before vLLM resolves quantization."""

    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        configs = (
            model_config.hf_config,
            getattr(model_config, "hf_text_config", None),
            getattr(model_config, "model_arch_config", None),
        )
        for config in configs:
            quant = getattr(config, "quantization_config", None)
            if not isinstance(quant, dict):
                continue
            if quant.get("quant_method") not in (
                SLIMQUANT_W4A8_QUANT_METHOD,
                KIMI_K3_W4A8_QUANT_METHOD,
            ):
                continue
            metadata = validate_kimi_k3_w4a8_metadata(
                quant, getattr(model_config, "hf_text_config", None)
            )
            quant.update(
                {
                    "quant_method": KIMI_K3_W4A8_QUANT_METHOD,
                    "format": SLIMQUANT_W4A8_QUANT_METHOD,
                    "model_version": metadata.model_version,
                    "num_experts": metadata.num_experts,
                    "top_k": metadata.top_k,
                    "hidden_size": metadata.hidden_size,
                    "intermediate_size": metadata.intermediate_size,
                }
            )


__all__ = ["KimiK3ConfigAdapter"]
