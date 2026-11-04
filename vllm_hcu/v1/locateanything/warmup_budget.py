# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only synthetic request geometry shared by startup and its unit tests."""


def warmup_prompts(max_num_reqs, max_num_tokens, logical_limit=None):
    """Keep the synthetic prefill and all decode steps within runner buffers."""
    if logical_limit is not None and logical_limit < 2:
        raise ValueError("Fast warmup needs a logical context of at least two tokens")
    minimum_prompt = 13 if logical_limit is None else min(13, logical_limit - 1)
    physical_tail = 6
    max_decode_query = 12 if logical_limit is None else min(12, logical_limit + 5)
    minimum_budget = max(minimum_prompt + physical_tail, max_decode_query)
    if max_num_reqs < 1 or max_num_tokens < minimum_budget:
        raise ValueError(f"Fast warmup requires a request and at least {minimum_budget} tokens")
    n = min(max_num_reqs, max_num_tokens // minimum_budget)
    maximum_prompt = min(minimum_prompt + n - 1,
                         max_num_tokens // n - physical_tail)
    if logical_limit is not None:
        maximum_prompt = min(maximum_prompt, logical_limit - 1)
    period = maximum_prompt - minimum_prompt + 1
    return [list(range(minimum_prompt + i % period)) for i in range(n)]


def warmup_request_plan(prompts, logical_limit, num_blocks, specs, block_count):
    """Fit whole requests in the real per-group KV reservation budget."""
    available = num_blocks - 1  # Block 0 is the shared null block.
    plan = []
    for prompt in prompts:
        budget = min(128, logical_limit - len(prompt))
        blocks = tuple(block_count(len(prompt) + budget + 6, spec) for spec in specs)
        cost = sum(blocks)
        if cost > available:
            break
        plan.append((prompt, budget, blocks))
        available -= cost
    if not plan:
        raise ValueError("LocateAnything Fast warmup has insufficient KV blocks for one synthetic request (block 0 is reserved)")
    return plan
