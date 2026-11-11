# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Shared DeepSeek-V4 runtime configuration predicates."""

from __future__ import annotations

from collections.abc import Mapping


def config_field(owner: object | None, name: str, default: object = None) -> object:
    """Read a field from either an object config or its serialized mapping."""

    if isinstance(owner, Mapping):
        return owner.get(name, default)
    return getattr(owner, name, default)


def model_architectures(vllm_config: object | None) -> tuple[str, ...]:
    model_config = config_field(vllm_config, "model_config")
    architectures = config_field(model_config, "architectures")
    if architectures is None:
        hf_config = config_field(model_config, "hf_config")
        architectures = config_field(hf_config, "architectures", ())
    return tuple(architectures or ())


def is_deepseek_v4(vllm_config: object | None) -> bool:
    return "DeepseekV4ForCausalLM" in model_architectures(vllm_config)


def is_dspark_enabled(vllm_config: object | None) -> bool:
    speculative_config = config_field(vllm_config, "speculative_config")
    use_dspark = config_field(speculative_config, "use_dspark")
    if callable(use_dspark):
        return bool(use_dspark())
    return config_field(speculative_config, "method") == "dspark"


def dspark_mooncake_pd_use_low_latency(
    vllm_config: object | None,
) -> bool | None:
    """Return the fixed DeepEP layout for a supported Mooncake P/D role."""

    if not is_dspark_enabled(vllm_config):
        return None
    kv_transfer_config = config_field(vllm_config, "kv_transfer_config")
    if config_field(kv_transfer_config, "kv_connector") != "MooncakeConnector":
        return None
    if not is_deepseek_v4(vllm_config):
        return None

    role = config_field(kv_transfer_config, "kv_role")
    if role == "kv_producer":
        return False
    if role == "kv_consumer":
        return True
    return None


__all__ = [
    "config_field",
    "dspark_mooncake_pd_use_low_latency",
    "is_deepseek_v4",
    "is_dspark_enabled",
    "model_architectures",
]
