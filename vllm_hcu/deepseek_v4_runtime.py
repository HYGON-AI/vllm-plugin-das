# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Shared DeepSeek-V4 runtime configuration predicates."""

from __future__ import annotations


# V4 and V4.1 share the MLA/MoE runtime contracts the callers below gate on,
# so both names belong to the family predicate.
_DSV4_ARCHITECTURES = frozenset(
    {"DeepseekV4ForCausalLM", "DeepseekV41ForCausalLM"}
)


def model_architectures(vllm_config: object | None) -> tuple[str, ...]:
    model_config = getattr(vllm_config, "model_config", None)
    architectures = getattr(model_config, "architectures", None)
    if architectures is None:
        hf_config = getattr(model_config, "hf_config", None)
        architectures = getattr(hf_config, "architectures", ())
    return tuple(architectures or ())


def is_deepseek_v4(vllm_config: object | None) -> bool:
    """Whether the model belongs to the DeepSeek-V4/V4.1 family."""
    return bool(_DSV4_ARCHITECTURES.intersection(model_architectures(vllm_config)))


def is_dspark_enabled(vllm_config: object | None) -> bool:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    use_dspark = getattr(speculative_config, "use_dspark", None)
    if callable(use_dspark):
        return bool(use_dspark())
    return getattr(speculative_config, "method", None) == "dspark"


__all__ = ["is_deepseek_v4", "is_dspark_enabled", "model_architectures"]
