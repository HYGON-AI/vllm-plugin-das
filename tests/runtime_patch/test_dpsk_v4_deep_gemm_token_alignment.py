# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Regression tests for the fix that routes deepgemm_moe_permute and
compute_aligned_M through the plugin's deep_gemm_utils, which enforces a
256-token-per-expert alignment on ROCm instead of the upstream default (128).
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


# ---------------------------------------------------------------------------
# Runtime alignment arithmetic
# ---------------------------------------------------------------------------

def test_expert_num_tokens_round_up_and_sum_pads_each_expert() -> None:
    """Every expert's token count is independently rounded up to 256.

    Experts: 1 → 256, 255 → 256, 256 → 256, 257 → 512, 0 → 0  =  1280 total.
    """
    from vllm_hcu.model_executor.layers.fused_moe.deep_gemm_utils import (
        expert_num_tokens_round_up_and_sum,
    )

    tokens = torch.tensor([1, 255, 256, 257, 0], dtype=torch.int32)
    result = expert_num_tokens_round_up_and_sum(tokens, alignment=256)
    assert result == 256 + 256 + 256 + 512 + 0


def test_compute_aligned_M_rocm_ignores_sub_256_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On ROCm, alignment is floored to 256 even when caller passes 128.

    This is the core contract broken before the fix: the upstream call site
    passed alignment=128 (get_mk_alignment_for_contiguous_layout()), so each
    expert was padded to only 128 rows — violating DeepGEMM's requirement.
    """
    import vllm_hcu.model_executor.layers.fused_moe.deep_gemm_utils as utils

    monkeypatch.setattr(utils.current_platform, "is_rocm", lambda: True)

    M_sum, align_used = utils.compute_aligned_M_and_alignment(
        M=10,
        num_topk=2,
        local_num_experts=4,
        alignment=128,  # what the upstream path passes
        expert_tokens_meta=None,
    )

    assert align_used == 256, f"expected align_used=256 on ROCm, got {align_used}"
    assert M_sum % 256 == 0, f"M_sum={M_sum} is not a multiple of 256"


def test_compute_aligned_M_rocm_with_expert_meta_pads_per_expert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-expert rounding via expert_tokens_meta also uses 256 on ROCm.

    Expert 0: 100 tokens → 256, Expert 1: 200 tokens → 256.  Sum ≥ 512.
    """
    import vllm_hcu.model_executor.layers.fused_moe.deep_gemm_utils as utils

    monkeypatch.setattr(utils.current_platform, "is_rocm", lambda: True)

    meta = SimpleNamespace(
        expert_num_tokens_cpu=torch.tensor([100, 200], dtype=torch.int32),
        expert_num_tokens=None,
    )

    M_sum, align_used = utils.compute_aligned_M_and_alignment(
        M=150,
        num_topk=2,
        local_num_experts=2,
        alignment=128,
        expert_tokens_meta=meta,
    )

    assert align_used == 256
    assert M_sum >= 512, f"expected at least 512 (2×256), got {M_sum}"
    assert M_sum % 256 == 0, f"M_sum={M_sum} is not a multiple of 256"
