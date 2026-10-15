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


@pytest.mark.parametrize("next_batch", ["smaller", "dummy"])
@pytest.mark.parametrize("device_kind", ["cpu", "cuda_graph"])
def test_spec_schedule_clears_padding_seen_by_captured_views(
    monkeypatch, next_batch, device_kind
):
    from vllm_hcu.model_executor.layers.step4_dsa import (
        Step4DSAGeometry,
        Step4DSAState,
    )

    if device_kind == "cpu":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            pytest.skip("CUDA Graph replay requires an accelerator")
        device = torch.device("cuda", torch.cuda.current_device())
    state = Step4DSAState(
        1,
        Step4DSAGeometry(),
        16,
        torch.bfloat16,
        max_num_reqs=4,
        max_model_len=64,
        spec_query_len=4,
    )
    state.regions_per_page = 1
    state.num_kv_groups = 1
    state.graph_regions = 8
    state._decode_logits = torch.zeros((16, 8), device=device)

    table_cpu = torch.arange(1, 33, dtype=torch.int32).reshape(4, 8)
    table = table_cpu.to(device)
    first = SimpleNamespace(
        num_decodes=4,
        num_decode_tokens=16,
        query_start_loc_cpu=torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([10, 18, 26, 34], dtype=torch.int32),
        block_table=table,
        block_table_np=table_cpu.numpy(),
    )
    for request, past in enumerate([6, 14, 22, 30]):
        state.pending.acquire(int(table_cpu[request, past // 8]))
    state._spec_decode_schedule(first, persistent=True)
    captured_pending = state._spec_raw[state._spec_layout["pending_out"][0] :][:4]
    captured_keep = state._spec_raw[state._spec_layout["keep_rows"][0] :][:8]
    captured_buf = state._spec_raw[state._spec_layout["tok_buf"][0] :][:16]
    captured_seq = state._spec_i32["tok_seq"][:16]
    captured_table = state._spec_table[:16]
    seq_pointer = captured_seq.data_ptr()
    table_pointer = captured_table.data_ptr()
    assert torch.count_nonzero(captured_seq[4:]) > 0
    assert torch.count_nonzero(captured_table[4:]) > 0

    captured = {
        "seq": captured_seq,
        "table": captured_table,
        "pending": captured_pending,
        "keep": captured_keep,
        "buf": captured_buf,
        "nscore": state._spec_i32["tok_nscore"],
        "counts": state._spec_i32["counts"],
    }
    observed = captured
    graph = None
    if device_kind == "cuda_graph":
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            observed = {name: value.clone() for name, value in captured.items()}

    dummy = next_batch == "dummy"
    tokens = 1 if dummy else 4
    small_table_cpu = torch.zeros((1, 8), dtype=torch.int32) if dummy else table_cpu[:1]
    small_table = small_table_cpu.to(device)
    second = SimpleNamespace(
        num_decodes=1,
        num_decode_tokens=tokens,
        query_start_loc_cpu=torch.tensor([0, 0 if dummy else 4], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 0 if dummy else 4], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([0 if dummy else 10], dtype=torch.int32),
        block_table=small_table,
        block_table_np=small_table_cpu.numpy(),
    )
    schedule = state._spec_decode_schedule(second, persistent=True)
    if graph is not None:
        graph.replay()
        torch.cuda.synchronize()

    # A graph retains the larger views and must not see lengths/pages of an
    # earlier request in rows beyond this step's logical batch.
    assert captured_seq.data_ptr() == seq_pointer
    assert captured_table.data_ptr() == table_pointer
    assert torch.all(observed["pending"][1:] == state.pending_trash_row)
    assert torch.all(observed["keep"][2:] == state.pending_trash_row)
    assert torch.all(observed["buf"][tokens:] >= 16)
    assert torch.count_nonzero(observed["seq"][tokens:]) == 0
    assert torch.count_nonzero(observed["nscore"][tokens:]) == 0
    assert torch.count_nonzero(observed["counts"][2:]) == 0
    assert torch.count_nonzero(observed["table"][tokens:]) == 0
    if not dummy:
        assert torch.equal(
            schedule.tok_seq,
            torch.tensor([7, 8, 9, 10], dtype=torch.int32, device=device),
        )
        assert torch.equal(schedule.tok_table, table[:1].expand(4, -1))


@pytest.mark.parametrize("next_batch", ["smaller", "dummy"])
@pytest.mark.parametrize("device_kind", ["cpu", "cuda_graph"])
def test_decode_schedule_clears_inactive_captured_rows(
    monkeypatch, next_batch, device_kind
):
    from vllm_hcu.model_executor.layers import step4_dsa as dsa

    if device_kind == "cpu":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            pytest.skip("CUDA Graph replay requires an accelerator")
        device = torch.device("cuda", torch.cuda.current_device())

    state = dsa.Step4DSAState(
        1,
        dsa.Step4DSAGeometry(),
        16,
        torch.bfloat16,
        max_num_reqs=4,
        max_model_len=64,
        spec_query_len=4,
    )
    state.regions_per_page = 1
    state.graph_regions = 8
    state._buf = {
        name: torch.zeros(4, dtype=dtype, device=device)
        for name, dtype in {
            "head": torch.int64,
            "tail": torch.int64,
            "slots": torch.int64,
            "pending_in": torch.int64,
            "pending_out": torch.int64,
            "nscore": torch.int32,
            "counts": torch.int32,
            "live": torch.bool,
        }.items()
    }
    state._buf["starts"] = torch.arange(4, dtype=torch.int32, device=device) * 8
    pin = device.type == "cuda"
    state._host = {
        name: [
            torch.zeros(4, dtype=torch.int64, pin_memory=pin)
            for _ in range(dsa._STAGING_DEPTH)
        ]
        for name in ["pending_in", "pending_out"]
    }
    state._stage = 0
    state._stage_ev = (
        [torch.cuda.Event() for _ in range(dsa._STAGING_DEPTH)] if pin else []
    )
    for event in state._stage_ev:
        event.record()

    table_cpu = torch.arange(1, 33, dtype=torch.int32).reshape(4, 8)
    table = table_cpu.to(device)
    seq_cpu = torch.tensor([7, 16, 23, 32], dtype=torch.int32)
    for row, seq in enumerate(seq_cpu.tolist()):
        state.pending.acquire(int(table_cpu[row, (seq - 1) // 8]))
    first = SimpleNamespace(
        num_decodes=4,
        seq_lens=seq_cpu.to(device),
        seq_lens_cpu=seq_cpu,
        block_table=table,
        block_table_np=table_cpu.numpy(),
    )
    state._decode_schedule(first, persistent=True)
    captured = dict(state._buf)
    pointers = {name: tensor.data_ptr() for name, tensor in captured.items()}
    assert torch.count_nonzero(captured["counts"][1:]) > 0
    assert torch.count_nonzero(captured["nscore"][1:]) > 0
    observed = captured
    graph = None
    if device_kind == "cuda_graph":
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            observed = {name: tensor.clone() for name, tensor in captured.items()}

    dummy = next_batch == "dummy"
    small_cpu = torch.zeros((1, 8), dtype=torch.int32) if dummy else table_cpu[:1]
    next_seq = torch.tensor([0 if dummy else 7], dtype=torch.int32)
    second = SimpleNamespace(
        num_decodes=1,
        seq_lens=next_seq.to(device),
        seq_lens_cpu=next_seq,
        block_table=small_cpu.to(device),
        block_table_np=small_cpu.numpy(),
    )
    schedule = state._decode_schedule(second, persistent=True)
    if graph is not None:
        graph.replay()
        torch.cuda.synchronize()

    for name in ["head", "tail", "nscore", "counts", "live"]:
        assert torch.count_nonzero(observed[name][1:]) == 0, name
    assert torch.all(observed["slots"][1:] == -1)
    for name in ["pending_in", "pending_out"]:
        assert torch.all(observed[name][1:] == state.pending_trash_row), name
    assert all(
        state._buf[name].data_ptr() == pointer for name, pointer in pointers.items()
    )
    if not dummy:
        assert schedule.head.item() == 6
        assert schedule.tail.item() == 7
        assert schedule.nscore.item() == 0
