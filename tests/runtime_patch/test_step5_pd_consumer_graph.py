# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Imported auxiliary data and builder schedules remain live across replay."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_graph_replay_reads_new_imports_and_reused_pending_rows():
    if not torch.cuda.is_available():
        pytest.skip('GPU is required')
    from vllm_hcu.model_executor.layers.step4_dsa import Step4DSAGeometry, Step4DSAState
    path = Path(__file__).resolve().parents[2] / 'vllm_hcu/distributed/kv_transfer/kv_connector/v1/mooncake/step5_aux.py'
    spec = importlib.util.spec_from_file_location('step5_aux_graph_tested', path)
    aux = importlib.util.module_from_spec(spec);spec.loader.exec_module(aux)
    state = Step4DSAState(1, Step4DSAGeometry(proxy_dim=32), 4,
                         torch.bfloat16, max_num_reqs=1,
                         max_model_len=64, spec_query_len=4)
    state.num_kv_groups = state.heads_per_group = 1
    cache = torch.zeros((13, 1, 16, 64), dtype=torch.uint8, device='cuda')
    worker = SimpleNamespace(is_kv_producer=False, seen_base_addresses=[],
        vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=1),
                                   model_config=SimpleNamespace(enforce_eager=False)),
        _layer_group_indices={'layer': 0},
        _logical_to_kernel_block_ids=lambda groups: groups,
        engine=SimpleNamespace(batch_register_memory=lambda ptrs, sizes: 0))
    manager = aux.Step5AuxManager(worker, {'layer': SimpleNamespace(state=state, slot=0)}, {'layer': cache})
    pointers = tuple(t.data_ptr() for t in (state.summary, state.pending_key, state.pending_z))
    summary_ids = torch.zeros(1, dtype=torch.int64, device='cuda')
    graph = None;rows_seen = set();schedule_pointer = None
    # Revisit physical blocks after different-length prompts, including both
    # sides of region/page boundaries. No Python indexer runs during replay.
    for i, (tokens, first_block) in enumerate([(5,2),(8,5),(9,2),(15,7),(16,2),(17,5),(31,2),(5,2)]):
        blocks = [first_block, first_block + 1, first_block + 2, first_block + 3]
        slot = blocks[(tokens - 1)//16]*2 + ((tokens - 1)%16)//8
        state._schedule = SimpleNamespace(completed_slots=[slot])
        request_id = f'case-{i}'
        pull = SimpleNamespace(d_req_id=request_id, local_block_ids=[blocks[:(tokens + 15)//16]], num_prompt_tokens=tokens, dsa_aux={})
        manager.prepare_pull(pull)
        row = state.pending._row_of_slot[slot];rows_seen.add(row)
        assert pull.dsa_aux['layer'][4] == state.pending_key[0, row].data_ptr()
        state.pending_key[:, row].zero_();state.pending_z[:, row].zero_()
        length = (tokens - 1)%8 + 1
        state.pending_key[:, row, :length].fill_(i + 1)
        state.pending_z[:, row, :length].fill_(i + 11)
        state.summary[0, slot].copy_(torch.full((1,32), i + 1, device='cuda').to(torch.float8_e4m3fn))
        manager.finish_pulls({request_id}, set())
        summary_ids.fill_(slot)
        table_cpu = torch.tensor([blocks], dtype=torch.int32)
        starts = torch.tensor([0,4], dtype=torch.int32)
        md = SimpleNamespace(num_decodes=1, num_decode_tokens=4,
            query_start_loc_cpu=starts, query_start_loc=starts.to('cuda'),
            seq_lens_cpu=torch.tensor([tokens + 4], dtype=torch.int32),
            block_table=table_cpu.to('cuda'), block_table_np=table_cpu.numpy())
        schedule = state._spec_decode_schedule(md, persistent=True)
        state._schedule = schedule
        if graph is None:
            schedule_pointer = state._spec_raw.data_ptr()
            # Warm the same operations before recording their fixed pointers.
            state.pending_key.index_select(1, schedule.pending_in)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                seen_k = state.pending_key.index_select(1, schedule.pending_in)
                seen_z = state.pending_z.index_select(1, schedule.pending_in)
                seen_summary = state.summary[0].view(torch.uint8).index_select(0, summary_ids)
        graph.replay();torch.cuda.synchronize()
        head = tokens % 8
        expected_k = torch.zeros_like(seen_k);expected_z = torch.zeros_like(seen_z)
        expected_k[:, :, :head].fill_(i + 1);expected_z[:, :, :head].fill_(i + 11)
        assert torch.equal(seen_k, expected_k) and torch.equal(seen_z, expected_z)
        assert torch.equal(seen_summary, state.summary[0,slot:slot+1].view(torch.uint8))
        assert state._spec_raw.data_ptr() == schedule_pointer
        assert tuple(t.data_ptr() for t in (state.summary,state.pending_key,state.pending_z)) == pointers
        manager._validate_allocations()
        manager.finish_pulls(set(), {request_id})
        assert not manager._imports
    assert len(rows_seen) > 1
