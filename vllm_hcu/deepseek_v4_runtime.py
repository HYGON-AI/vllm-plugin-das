# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Shared DeepSeek-V4 runtime configuration predicates."""

from __future__ import annotations


def model_architectures(vllm_config: object | None) -> tuple[str, ...]:
    model_config = getattr(vllm_config, "model_config", None)
    architectures = getattr(model_config, "architectures", None)
    if architectures is None:
        hf_config = getattr(model_config, "hf_config", None)
        architectures = getattr(hf_config, "architectures", ())
    return tuple(architectures or ())


def is_deepseek_v4(vllm_config: object | None) -> bool:
    return "DeepseekV4ForCausalLM" in model_architectures(vllm_config)


def is_dspark_enabled(vllm_config: object | None) -> bool:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    use_dspark = getattr(speculative_config, "use_dspark", None)
    if callable(use_dspark):
        return bool(use_dspark())
    return getattr(speculative_config, "method", None) == "dspark"


def dspark_mooncake_pd_use_low_latency(
    vllm_config: object | None,
) -> bool | None:
    """Return the fixed DeepEP layout for a supported Mooncake P/D role."""

    if not is_dspark_enabled(vllm_config):
        return None
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if getattr(kv_transfer_config, "kv_connector", None) != "MooncakeConnector":
        return None
    if not is_deepseek_v4(vllm_config):
        return None

    role = getattr(kv_transfer_config, "kv_role", None)
    if role == "kv_producer":
        return False
    if role == "kv_consumer":
        return True
    return None


__all__ = [
    "dspark_mooncake_pd_use_low_latency",
    "is_deepseek_v4",
    "is_dspark_enabled",
    "model_architectures",
]
