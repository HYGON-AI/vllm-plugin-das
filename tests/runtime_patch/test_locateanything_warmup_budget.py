# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU regression cases for Fast synthetic requests exceeding runner buffers."""
import pytest
from vllm_hcu.v1.locateanything.warmup_budget import warmup_prompts


@pytest.mark.parametrize("capacity,budget", [
    (1,32768),(16,32768),(128,32768),(192,32768),(256,32768),
    (512,32768),(1024,32768),(2048,32768),(256,64),(4096,32768),
])
def test_real_prefill_and_worst_decode_fit_token_budget(capacity, budget):
    prompts = warmup_prompts(capacity, budget)
    assert 0 < len(prompts) <= capacity
    assert all(len(p) >= 13 and p == list(range(len(p))) for p in prompts)
    assert sum(len(p) + 6 for p in prompts) <= budget
    # Each cached step can carry six uncommitted outputs plus six physical slots.
    assert len(prompts) * 12 <= budget


def test_256_request_regression_and_128_request_compatibility():
    make = warmup_prompts
    assert sum(13 + i + 6 for i in range(256)) > 32768
    assert sum(len(p) + 6 for p in make(256,32768)) <= 32768
    assert make(128,32768) == [list(range(13+i)) for i in range(128)]


@pytest.mark.parametrize("capacity,budget", [(0,32768),(16,18)])
def test_unusable_warmup_budget_is_rejected(capacity, budget):
    with pytest.raises(ValueError, match="at least 19 tokens"):
        warmup_prompts(capacity, budget)
