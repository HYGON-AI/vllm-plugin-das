# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Live-HCU accuracy for the screened DeepSeek-V4 WO_A batched GEMM."""

from __future__ import annotations

import pytest
import torch


pytestmark = pytest.mark.hcu

_BW200B_CONFIG = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 1,
    "num_warps": 8,
    "num_stages": 2,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16,
}


def _load_operator():
    try:
        from aiter.ops.triton.batched_gemm_bf16 import batched_gemm_bf16
    except ImportError as exc:
        pytest.skip(f"AITER batched_gemm_bf16 is unavailable: {exc}")
    return batched_gemm_bf16


@pytest.mark.parametrize(
    ("tokens", "groups"),
    ((1, 8), (16, 8), (64, 8), (8, 4), (8, 2), (8, 1)),
)
def test_aiter_batched_gemm_matches_deepseek_v4_fp32_reference(
    tokens: int,
    groups: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("a CUDA/ROCm device is required")
    properties = torch.cuda.get_device_properties(0)
    if not hasattr(properties, "gcnArchName"):
        pytest.skip("the active device is not an HCU/ROCm device")

    operator = _load_operator()
    generator = torch.Generator(device="cuda").manual_seed(
        20260906 + tokens + groups
    )
    source = torch.randn(
        (tokens, groups, 4096),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).contiguous()
    weight = (
        torch.randn(
            (groups, 1024, 4096),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.02
    ).contiguous()
    source_before = source.clone()
    weight_before = weight.clone()
    reference = torch.einsum("tgd,grd->tgr", source.float(), weight.float())

    batch_major = source.transpose(0, 1).contiguous()
    actual = operator(
        batch_major,
        weight,
        dtype=torch.bfloat16,
        config=_BW200B_CONFIG,
    ).transpose(0, 1).contiguous()
    torch.cuda.synchronize()

    assert actual.shape == (tokens, groups, 1024)
    assert actual.dtype is torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), reference, rtol=0.02, atol=0.05)
    torch.testing.assert_close(source, source_before, rtol=0, atol=0)
    torch.testing.assert_close(weight, weight_before, rtol=0, atol=0)
