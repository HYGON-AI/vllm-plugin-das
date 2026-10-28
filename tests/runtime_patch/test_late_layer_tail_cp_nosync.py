# SPDX-License-Identifier: Apache-2.0
"""SGLang-style nonzero_static helpers for late-layer / DualChunk tail."""

from __future__ import annotations

import torch

from vllm_hcu.patch.worker.core_fix.late_layer_tail_cp import (
    index_copy_masked,
    late_layer_tail_cp_layout,
    late_layer_tail_layout,
    nonzero_static_flat,
)


def test_nonzero_static_flat_pads():
    mask = torch.tensor([False, True, False, True, False])
    idx = nonzero_static_flat(mask, size=5, fill_value=-1)
    assert idx.tolist() == [1, 3, -1, -1, -1]


def test_index_copy_masked_ignores_fill():
    dest = torch.zeros(4, 2)
    index = torch.tensor([1, 3, -1, -1])
    source = torch.tensor([[1.0, 1.0], [3.0, 3.0], [9.0, 9.0], [9.0, 9.0]])
    out = index_copy_masked(dest, index, source)
    assert out[1].tolist() == [1.0, 1.0]
    assert out[3].tolist() == [3.0, 3.0]
    assert out[0].tolist() == [0.0, 0.0]
    assert out[2].tolist() == [0.0, 0.0]


def test_late_layer_tail_cp_layout_matches_legacy_nonzero():
    device = torch.device("cpu")
    # nonzero_static is CUDA-oriented in some builds; layout with pcp_size>1 uses
    # it on the token device — keep CPU path covered via pcp_size==1 + formula.
    token_indices = torch.arange(17, device=device, dtype=torch.long)
    positions = token_indices.to(torch.int32)
    floors = torch.zeros_like(positions)
    plan = late_layer_tail_cp_layout(
        token_indices=token_indices,
        tail_lens=torch.tensor([5, 6, 6], dtype=torch.int32),
        positions=positions,
        window_floors=floors,
        pcp_rank=0,
        pcp_size=1,
        batch_size=3,
    )
    assert plan.local_tail_rows.tolist() == list(range(17))
    assert plan.max_local == 17


def test_late_layer_tail_layout_per_request_keeps_full_extend():
    """prompt_logprobs rows can keep the full extend while others take window."""
    device = torch.device("cpu")
    token_indices, tail_lens, floors = late_layer_tail_layout(
        extend_lens_cpu=[8, 8],
        seq_lens_cpu=[100, 200],
        tail_len=4,
        device=device,
        per_request_tail_lens=[8, 4],
    )
    assert tail_lens == [8, 4]
    assert token_indices.tolist() == list(range(0, 8)) + list(range(12, 16))
    # Floors are seq_len - tail_len per kept row.
    assert floors.tolist() == [92] * 8 + [196] * 4
