# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU numerical parity for the optional BoltOPs FLA chunk pipeline."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.parametrize("lengths", [(64,), (32, 64)])
def test_boltops_chunk_pipeline_matches_official(
    monkeypatch: pytest.MonkeyPatch,
    lengths: tuple[int, ...],
):
    if not torch.cuda.is_available():
        pytest.skip("HCU is required")
    pytest.importorskip("boltops.fla.gdn")

    from vllm.third_party.flash_linear_attention.ops import chunk
    from vllm_hcu.patch.worker.op_opt import (
        patch_fla_chunk_delta_h,
        patch_fla_chunk_o,
        patch_fla_recompute_w_u,
    )
    from vllm_hcu.platforms import envs as henvs

    patch_fla_chunk_delta_h.apply_to_module(chunk)
    patch_fla_chunk_o.apply_to_module(chunk)
    patch_fla_recompute_w_u.apply_to_module(chunk)

    total_tokens = sum(lengths)
    torch.manual_seed(7)
    q = torch.randn(
        1, total_tokens, 2, 64, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn_like(q) * 0.1
    g = torch.nn.functional.logsigmoid(
        torch.randn(1, total_tokens, 2, device="cuda", dtype=torch.float32)
    )
    beta = torch.sigmoid(
        torch.randn(1, total_tokens, 2, device="cuda", dtype=torch.float32)
    )
    if len(lengths) == 1:
        cu_seqlens = None
        initial_state = None
    else:
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.long)
        initial_state = torch.randn(
            len(lengths), 2, 64, 64, device="cuda", dtype=torch.float32
        ) * 0.01

    def run(enabled: bool):
        monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
        monkeypatch.setattr(
            henvs, "VLLM_HCU_USE_CUSTOM_AITER_FLA", enabled
        )
        output_buffer = torch.empty(
            v.numel() + 16, device="cuda", dtype=v.dtype
        )
        output, final_state = chunk.chunk_gated_delta_rule(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            initial_state=(
                None if initial_state is None else initial_state.clone()
            ),
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            core_attn_out=output_buffer,
        )
        torch.cuda.synchronize()
        assert output.data_ptr() == output_buffer.data_ptr()
        return output.clone(), final_state.clone()

    expected_output, expected_state = run(False)
    actual_output, actual_state = run(True)
    torch.testing.assert_close(
        actual_output, expected_output, rtol=5e-2, atol=5e-3
    )
    torch.testing.assert_close(
        actual_state, expected_state, rtol=5e-2, atol=5e-3
    )
