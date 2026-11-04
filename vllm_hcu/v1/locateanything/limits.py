# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Shared physical/probe/logical context contract for LocateAnything Fast."""

PROBE_TOKENS = 6


def reject_speculative_decoding(speculative_config) -> None:
    if speculative_config is not None:
        raise ValueError("LocateAnything Fast uses six-position block generation and does not support vLLM speculative decoding")


def logical_context_limit(max_model_len: int) -> int:
    """Reserve probes and leave room for a nonempty prompt and output."""
    limit = max_model_len - PROBE_TOKENS
    if limit < 2:
        raise ValueError("LocateAnything Fast max-model-len must be at least 8 (including six probe slots)")
    return limit


def context_error(max_model_len: int, prompt_tokens: int, max_new_tokens: int | None) -> str | None:
    limit = logical_context_limit(max_model_len)
    if prompt_tokens >= limit or (max_new_tokens is not None and prompt_tokens + max_new_tokens > limit):
        return (f"Fast logical context is {limit} tokens for max-model-len={max_model_len}; "
                "six physical probe slots are reserved separately")
    return None
