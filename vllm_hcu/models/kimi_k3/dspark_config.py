# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Transformers/vLLM config registration for the K3 DSpark draft model."""

from __future__ import annotations

from transformers import AutoConfig, DeepseekV2Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


class K3DSparkConfig(DeepseekV2Config):
    """Configuration for the dense MLA draft checkpoint used by K3 DSpark."""

    model_type = "k3_dspark"
    has_no_defaults_at_init = True

    def __init__(
        self,
        mla_use_nope: bool = False,
        mla_use_output_gate: bool = False,
        mla_use_qk_norm: bool = False,
        rope_theta: float = 50000.0,
        **kwargs,
    ) -> None:
        # DeepseekV2Config defaults to a MoE topology. The K3 draft is dense.
        kwargs.setdefault("n_routed_experts", 0)
        kwargs.setdefault("n_shared_experts", 0)
        kwargs.setdefault("num_experts_per_tok", 0)

        rope_parameters = kwargs.get("rope_parameters")
        if rope_parameters is None:
            kwargs["rope_parameters"] = {
                "rope_type": "default",
                "rope_theta": rope_theta,
            }
        else:
            rope_parameters = dict(rope_parameters)
            rope_parameters.setdefault("rope_type", "default")
            rope_parameters.setdefault("rope_theta", rope_theta)
            kwargs["rope_parameters"] = rope_parameters

        super().__init__(**kwargs)
        self.mla_use_nope = mla_use_nope
        self.mla_use_output_gate = mla_use_output_gate
        self.mla_use_qk_norm = mla_use_qk_norm

        unsupported = [
            name
            for name in (
                "mla_use_nope",
                "mla_use_output_gate",
                "mla_use_qk_norm",
                "dspark_bonus_anchor",
            )
            if getattr(self, name, False)
        ]
        if self.q_lora_rank is None:
            unsupported.append("q_lora_rank=None")
        if unsupported:
            raise ValueError("MLA DSpark does not support " + ", ".join(unsupported))

        self.draft_vocab_size = (
            getattr(self, "draft_vocab_size", None) or self.vocab_size
        )
        if self.draft_vocab_size != self.vocab_size:
            raise ValueError(
                "MLA DSpark requires draft_vocab_size to equal vocab_size when "
                "sharing the target embedding and LM head."
            )

        target_layer_ids = getattr(self, "target_layer_ids", None)
        if not target_layer_ids or getattr(self, "num_target_layers", None) != len(
            target_layer_ids
        ):
            raise ValueError(
                "MLA DSpark requires non-empty target_layer_ids and a matching "
                "num_target_layers."
            )


def register_k3_dspark_config() -> None:
    """Register the checkpoint config without changing the vLLM checkout."""
    from vllm.transformers_utils import config as vllm_config

    model_type = K3DSparkConfig.model_type
    existing_vllm = vllm_config._CONFIG_REGISTRY.get(model_type)
    if existing_vllm is not None and existing_vllm is not K3DSparkConfig:
        raise RuntimeError(
            f"vLLM config {model_type!r} has a different owner: {existing_vllm!r}"
        )
    existing_transformers = (
        CONFIG_MAPPING[model_type] if model_type in CONFIG_MAPPING else None
    )
    if (
        existing_transformers is not None
        and existing_transformers is not K3DSparkConfig
    ):
        raise RuntimeError(
            "Transformers config "
            f"{model_type!r} has a different owner: {existing_transformers!r}"
        )

    vllm_config._CONFIG_REGISTRY[model_type] = K3DSparkConfig
    AutoConfig.register(model_type, K3DSparkConfig, exist_ok=True)


__all__ = ["K3DSparkConfig", "register_k3_dspark_config"]
