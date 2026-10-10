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
# Helpers for building fake packed weight tensors
# ---------------------------------------------------------------------------

def _make_6dim_weights(
    num_experts: int,
    n: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fake 6-dim packed (w13, w2) matching [E, K/64, N/16, 4, 16, 16].

    w13 has output size 2*n (gate+up fused); w2 maps intermediate→hidden (n→k).
    The w2_n check requires: w2.size(2)*w2.size(4) == k.
    """
    assert k % 64 == 0 and (2 * n) % 16 == 0, "n/k must be 16/64-aligned"
    assert k % 16 == 0 and n % 64 == 0, "w2 dims must be 16/64-aligned"
    w13 = torch.empty(num_experts, k // 64, (2 * n) // 16, 4, 16, 16)
    w2 = torch.empty(num_experts, n // 64, k // 16, 4, 16, 16)
    return w13, w2


def _make_7dim_weights(
    num_experts: int,
    n: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fake 7-dim packed (w13, w2) matching [E, K/64, N/32, 2, 4, 16, 16].

    The w2_n check requires: w2.size(2)*w2.size(3)*w2.size(5) == k.
    """
    assert k % 64 == 0 and (2 * n) % 32 == 0, "n/k must be 32/64-aligned"
    assert k % 32 == 0 and n % 64 == 0, "w2 dims must be 32/64-aligned"
    w13 = torch.empty(num_experts, k // 64, (2 * n) // 32, 2, 4, 16, 16)
    w2 = torch.empty(num_experts, n // 64, k // 32, 2, 4, 16, 16)
    return w13, w2


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


# ---------------------------------------------------------------------------
# _packed_deepgemm_problem_shape: 6-dim and 7-dim layout reconstruction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "num_experts,n,k",
    [
        (4, 512, 1024),
        (8, 1024, 2048),
        (1, 512, 512),
    ],
)
def test_packed_shape_6dim_reconstructs_n_and_k(
    num_experts: int, n: int, k: int
) -> None:
    """6-dim packer layout [E, K/64, N/16, 4, 16, 16]: N and K are recovered correctly."""
    from vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe import (
        DeepEPDeepGemmContiguousExperts,
    )

    w13, w2 = _make_6dim_weights(num_experts, n, k)
    got_e, got_n, got_k = DeepEPDeepGemmContiguousExperts._packed_deepgemm_problem_shape(w13, w2)

    assert got_e == num_experts
    assert got_n == 2 * n, f"expected n={2 * n}, got {got_n}"
    assert got_k == k, f"expected k={k}, got {got_k}"


@pytest.mark.parametrize(
    "num_experts,n,k",
    [
        (4, 512, 1024),
        (8, 1024, 2048),
        (1, 512, 512),
    ],
)
def test_packed_shape_7dim_reconstructs_n_and_k(
    num_experts: int, n: int, k: int
) -> None:
    """7-dim packer layout [E, K/64, N/32, 2, 4, 16, 16]: N and K are recovered correctly."""
    from vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe import (
        DeepEPDeepGemmContiguousExperts,
    )

    w13, w2 = _make_7dim_weights(num_experts, n, k)
    got_e, got_n, got_k = DeepEPDeepGemmContiguousExperts._packed_deepgemm_problem_shape(w13, w2)

    assert got_e == num_experts
    assert got_n == 2 * n, f"expected n={2 * n}, got {got_n}"
    assert got_k == k, f"expected k={k}, got {got_k}"


def test_packed_shape_mixed_dims_rejected() -> None:
    """w13 and w2 must have the same number of dimensions; mixing 6 and 7 raises."""
    from vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe import (
        DeepEPDeepGemmContiguousExperts,
    )

    w13_6, _ = _make_6dim_weights(4, 512, 1024)
    _, w2_7 = _make_7dim_weights(4, 512, 1024)

    with pytest.raises(RuntimeError, match="packed before apply"):
        DeepEPDeepGemmContiguousExperts._packed_deepgemm_problem_shape(w13_6, w2_7)


def test_packed_shape_mismatched_experts_rejected() -> None:
    """w13 and w2 with different expert counts raise RuntimeError."""
    from vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe import (
        DeepEPDeepGemmContiguousExperts,
    )

    w13, _ = _make_7dim_weights(4, 512, 1024)
    _, w2_wrong_e = _make_7dim_weights(8, 512, 1024)

    with pytest.raises(RuntimeError, match="mismatched experts"):
        DeepEPDeepGemmContiguousExperts._packed_deepgemm_problem_shape(w13, w2_wrong_e)


def test_packed_shape_hidden_size_mismatch_rejected() -> None:
    """w2 whose reconstructed N ≠ K from w13 raises RuntimeError (down-proj size mismatch)."""
    from vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe import (
        DeepEPDeepGemmContiguousExperts,
    )

    w13, _ = _make_7dim_weights(4, 512, 1024)
    # Build a w2 that is valid on its own but whose w2_n != k of w13 (k=1024).
    # Use k=512 for w2 so hidden-size reconstructed from w2 is 512 ≠ 1024.
    _, w2_bad = _make_7dim_weights(4, 256, 512)

    with pytest.raises(RuntimeError, match="down weight output size does not match"):
        DeepEPDeepGemmContiguousExperts._packed_deepgemm_problem_shape(w13, w2_bad)


def test_packed_shape_wrong_ndim_rejected() -> None:
    """Weights with fewer than 6 dimensions are rejected before any reconstruction."""
    from vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe import (
        DeepEPDeepGemmContiguousExperts,
    )

    w_bad = torch.empty(4, 16, 64)  # 3-dim — not a packed layout
    with pytest.raises(RuntimeError, match="packed before apply"):
        DeepEPDeepGemmContiguousExperts._packed_deepgemm_problem_shape(w_bad, w_bad)


# ---------------------------------------------------------------------------
# moe_problem_size: 2D and 3D activations with 6-dim and 7-dim weights
# ---------------------------------------------------------------------------

def _bare_contiguous_expert():
    """Create a DeepEPDeepGemmContiguousExperts instance without calling __init__.

    moe_problem_size() only calls _packed_deepgemm_problem_shape() (a staticmethod)
    and reads tensor shapes, so no instance state is needed.
    """
    from vllm_hcu.model_executor.layers.fused_moe.experts.dpsk_v4_deep_gemm_moe import (
        DeepEPDeepGemmContiguousExperts,
    )

    return object.__new__(DeepEPDeepGemmContiguousExperts)


@pytest.mark.parametrize("weight_factory,label", [
    (_make_6dim_weights, "6dim"),
    (_make_7dim_weights, "7dim"),
])
def test_moe_problem_size_2d_activation(weight_factory, label) -> None:
    """moe_problem_size with [M, K] activations returns (E, M, N, K, topk)."""
    obj = _bare_contiguous_expert()
    num_experts, n, k, M, topk = 4, 512, 1024, 16, 2

    w13, w2 = weight_factory(num_experts, n, k)
    a1 = torch.empty(M, k)
    topk_ids = torch.zeros(M, topk, dtype=torch.int64)

    got = obj.moe_problem_size(a1, w13, w2, topk_ids)

    assert got == (num_experts, M, 2 * n, k, topk), (
        f"[{label}] expected ({num_experts}, {M}, {2*n}, {k}, {topk}), got {got}"
    )


@pytest.mark.parametrize("weight_factory,label", [
    (_make_6dim_weights, "6dim"),
    (_make_7dim_weights, "7dim"),
])
def test_moe_problem_size_3d_activation(weight_factory, label) -> None:
    """moe_problem_size with [E, max_tokens, K] activations returns (E, max_tokens, N, K, topk)."""
    obj = _bare_contiguous_expert()
    num_experts, n, k, max_tokens, topk = 4, 512, 1024, 32, 2

    w13, w2 = weight_factory(num_experts, n, k)
    a1 = torch.empty(num_experts, max_tokens, k)
    # topk_ids shape for 3D path: [M_flat, topk]; M_flat can be any value
    topk_ids = torch.zeros(max_tokens, topk, dtype=torch.int64)

    got = obj.moe_problem_size(a1, w13, w2, topk_ids)

    assert got == (num_experts, max_tokens, 2 * n, k, topk), (
        f"[{label}] expected ({num_experts}, {max_tokens}, {2*n}, {k}, {topk}), got {got}"
    )


def test_moe_problem_size_3d_expert_count_mismatch_raises() -> None:
    """3D activation whose dim-0 != local_num_experts raises AssertionError."""
    obj = _bare_contiguous_expert()
    num_experts, n, k = 4, 512, 1024

    w13, w2 = _make_7dim_weights(num_experts, n, k)
    a1 = torch.empty(num_experts + 1, 32, k)  # wrong expert count
    topk_ids = torch.zeros(32, 2, dtype=torch.int64)

    with pytest.raises(AssertionError):
        obj.moe_problem_size(a1, w13, w2, topk_ids)
