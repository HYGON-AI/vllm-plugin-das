# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only synthetic request geometry shared by startup and its unit tests."""


def warmup_prompts(max_num_reqs, max_num_tokens):
    """Keep the synthetic prefill and all decode steps within runner buffers."""
    minimum_prompt = 13
    physical_tail = 6
    if max_num_reqs < 1 or max_num_tokens < minimum_prompt + physical_tail:
        raise ValueError("Fast warmup requires a request and at least 19 tokens")
    n = min(max_num_reqs, max_num_tokens // (minimum_prompt + physical_tail))
    maximum_prompt = min(minimum_prompt + n - 1,
                         max_num_tokens // n - physical_tail)
    period = maximum_prompt - minimum_prompt + 1
    return [list(range(minimum_prompt + i % period)) for i in range(n)]
