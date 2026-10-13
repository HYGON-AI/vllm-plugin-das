# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for DSA pending-region scheduling and buffer ownership."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("route", ["general", "decode"])
def test_pending_heads_survive_earlier_row_allocation(route):
    from vllm_hcu.model_executor.layers.step4_dsa import (
        Step4DSAGeometry,
        Step4DSAState,
    )

    state = Step4DSAState(1, Step4DSAGeometry(), 2, torch.bfloat16, spec_query_len=4)
    state.regions_per_page = 1
    original_row = state.pending.acquire(10)
    state.pending.acquire(11)
    state.pending.retire(10)
    # Row 0 starts a new region; row 1 rolls back into a retired region.
    # Its old tokens must be protected before row 0 claims any buffer.
    table = torch.tensor([[12], [10]], dtype=torch.int32)
    seq = torch.tensor([1, 2], dtype=torch.int32)
    md = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens_cpu=seq,
        seq_lens=seq,
        block_table=table,
        block_table_np=table.numpy(),
        num_decodes=2 if route == "decode" else 0,
        num_prefills=0 if route == "decode" else 2,
    )
    if route == "general":
        schedule = state._general_schedule(md)
        assert schedule.requests[1].pending_row_in == original_row
    else:
        schedule = state._decode_schedule(md)
        assert schedule.pending_in[1].item() == original_row
    assert state.pending.get(10) == original_row
    assert state.pending.get(12) != original_row
