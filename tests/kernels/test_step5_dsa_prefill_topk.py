# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the exact-count resident prefill selector."""

import pytest
import torch

pytest.importorskip("triton")

kernels = pytest.importorskip("vllm_hcu.model_executor.layers.step4_dsa_kernels")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton device required")
def test_exact_emit_preserves_packed_metadata(monkeypatch):
    width = 16384
    topk = 512
    generator = torch.Generator(device="cuda").manual_seed(1602)
    scores = torch.randn((6, width), device="cuda", generator=generator)
    scores[4].mul_(4).round_()
    scores[5].zero_()
    lengths = torch.tensor(
        [0, 36, 512, width, width - 18, width], device="cuda", dtype=torch.int32
    )
    positions = lengths * 8 + 2
    block_table = torch.arange(width // 8, device="cuda", dtype=torch.int32)[None]
    block_table[0, 5:11] = -1
    monkeypatch.setattr(kernels.henvs, "optional_custom_op_enabled", lambda: True)

    def pack(exact):
        monkeypatch.setattr(kernels, "_PREFILL_TOPK_EXACT_EMIT", exact)
        logical = torch.empty((scores.shape[0], topk), device="cuda", dtype=torch.int32)
        packed = kernels.region_topk_pack(
            scores, lengths, positions, block_table,
            topk=topk, regions_per_page=8, _logical_out=logical,
        )
        return packed, logical

    reference_packed, reference_logical = pack(False)
    actual_packed, actual_logical = pack(True)
    torch.testing.assert_close(actual_packed, reference_packed, rtol=0, atol=0)
    torch.testing.assert_close(actual_logical, reference_logical, rtol=0, atol=0)
