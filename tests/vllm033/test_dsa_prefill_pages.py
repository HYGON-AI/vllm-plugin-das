# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from hcu_vllm_das.dsa_prefill_pages import (
    PageGroup,
    PrefillPages,
    active_prefill_pages,
    fresh_prefill_page_ids,
    fresh_swa_prefill_page_ids,
    prefill_pages_scope,
)


SOURCES = (2, 8, 14, 20)
INDEX_SOURCES = (*SOURCES, 24, 28, 32, 36)


def main(layer):
    return f'model.layers.{layer}.attn'


def index(layer):
    return main(layer) + '.indexer.k_cache'


def output(*requests):
    return NS(
        num_scheduled_tokens={req.req_id: req.scheduled for req in requests},
        scheduled_cached_reqs=NS(req_ids=[]),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        scheduled_new_reqs=list(requests),
    )


def request(rid='a', scheduled=65, blocks=(8, 2, 6)):
    return NS(req_id=rid, scheduled=scheduled, prompt_token_ids=[1] * 96,
              block_ids=([10], [9], list(blocks)), num_computed_tokens=0,
              mm_features=[], replay_start=0, sampling_params=None)


def test_fresh_prefill_unions_request_pages_at_the_scheduler_group_index():
    batch = output(request(), request('b', 64, (3, 7)), request('c', 33, (8, 9)))
    groups = (PageGroup(2, 64, 16, (main(2), index(2))),)
    assert fresh_prefill_page_ids(batch, groups) == {2: (2, 3, 8)}


@pytest.mark.parametrize('attr,value', [
    ('num_computed_tokens', 1), ('replay_start', 1), ('mm_features', [object()]),
    ('scheduled', 97), ('block_ids', ([1], [1], [5, 0])),
    ('block_ids', ([1], [1], [5, 16])), ('block_ids', ([1], [1], [5])),
])
def test_inexact_or_incompatible_page_plan_uses_the_native_path(attr, value):
    req = request()
    setattr(req, attr, value)
    assert fresh_prefill_page_ids(output(req), (PageGroup(2, 64, 16, (main(2),)),)) is None


@pytest.mark.parametrize('attr,value', [
    ('scheduled_cached_reqs', NS(req_ids=['cached'])),
    ('scheduled_spec_decode_tokens', {'a': [1]}),
    ('scheduled_encoder_inputs', {'a': [0]}), ('has_sync_kv_loads', True),
    ('kv_connector_metadata', object()),
])
def test_mixed_decode_prefix_transfer_and_draft_are_not_fresh_prefills(attr, value):
    batch = output(request())
    setattr(batch, attr, value)
    assert fresh_prefill_page_ids(batch, (PageGroup(2, 64, 16, (main(2),)),)) is None


def test_scope_cleans_up_after_a_failed_forward_and_supports_nested_calls():
    plan = PrefillPages({}, {})
    with pytest.raises(RuntimeError, match='forward failed'), prefill_pages_scope(plan):
        assert active_prefill_pages() is plan
        with prefill_pages_scope(None):
            assert active_prefill_pages() is None
        assert active_prefill_pages() is plan
        raise RuntimeError('forward failed')
    assert active_prefill_pages() is None


@pytest.mark.parametrize('alias_index,expected', [
    (True, [2, 8, 14, 20, 24, 28, 32, 36]), (False, [2, 8, 14, 20]),
])
def test_dsv41_reindex_layers_must_rebroadcast_only_if_they_clobber_main(alias_index, expected):
    groups = {main(i): 2 for i in SOURCES}
    groups.update({index(i): 2 if alias_index else 3 for i in SOURCES})
    plan = PrefillPages({}, groups)
    transfers = []
    for layer in range(2, 40):
        source = max(i for i in SOURCES if i <= layer)
        if layer == source:
            plan.source_written(main(source))
        if layer in INDEX_SOURCES:
            plan.finish_transfer(index(source))
        if not plan.ready(main(source)):
            plan.finish_transfer(main(source))
            transfers.append(layer)
        assert plan.ready(main(source))
    assert transfers == expected


def test_planned_transfer_reuses_receiver_until_indexer_overwrites_or_source_rewrites(monkeypatch):
    import torch
    from hcu_vllm_das import dsa_cache_layer_split as runtime

    source, indexer, swa = main(14), index(14), main(16) + '.swa_cache'
    owners = torch.zeros(256, dtype=torch.uint8)
    receivers = torch.zeros_like(owners)
    owner = owners[:24].view(6, 4)
    reverse = receivers[:24].view(6, 4)
    device_ids = torch.tensor([1, 4], dtype=torch.int64)
    plan_owner = PrefillPages({source: device_ids, indexer: device_ids, swa: device_ids},
                              {source: 2, indexer: 2})
    plan_receiver = PrefillPages(dict(plan_owner.by_prefix), dict(plan_owner.group_by_prefix))
    state = NS(rank_in_group=0, packets=[], cursor=0)

    def broadcast(wire, src):
        assert src == 0
        if state.rank_in_group == 0:
            state.packets.append(wire.clone())
        else:
            wire.copy_(state.packets[state.cursor])
            state.cursor += 1

    group = NS(device_group=object(), broadcast=broadcast)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(runtime, '_transport_regions', {prefix: (32, 224, 6)
                                                        for prefix in (source, indexer, swa)})
    monkeypatch.setattr(runtime, 'get_forward_context',
                        lambda: NS(attn_metadata={source: object(), indexer: object(), swa: object()}))
    monkeypatch.setattr(runtime, 'get_pcp_group', lambda: group)
    monkeypatch.setattr(runtime, '_live_block_ids', Mock(side_effect=AssertionError('read back')))
    monkeypatch.setattr(runtime.dist, 'all_reduce', Mock(side_effect=AssertionError('bitmap')))

    def run(prefix, value, num_after, rewrite=False):
        owner.fill_(value)
        for rank, cache, plan in [(0, owner, plan_owner), (1, reverse, plan_receiver)]:
            group.rank_in_group = state.rank_in_group = rank
            with prefill_pages_scope(plan):
                if rewrite:
                    plan.source_written(prefix)
                runtime.broadcast_kv_source_cache(prefix, 0, cache)
        assert len(state.packets) == state.cursor == num_after

    run(source, 12, 1)
    assert reverse[:, 0].tolist() == [0, 12, 0, 0, 12, 0]
    # Switching to a replay view doesn't rebind the globally planned pages.
    plan_owner.finish_transfer(swa)
    plan_receiver.finish_transfer(swa)
    run(source, 12, 1)
    run(indexer, 45, 2)  # The indexer uses the same scratch bytes on receivers.
    assert reverse[:, 0].tolist() == [0, 45, 0, 0, 45, 0]
    run(source, 12, 3)
    run(source, 67, 4, rewrite=True)
    assert reverse[:, 0].tolist() == [0, 67, 0, 0, 67, 0]
    run(swa, 21, 5)
    run(swa, 22, 6)
    assert reverse[:, 0].tolist() == [0, 22, 0, 0, 22, 0]


def test_schedule_plan_equals_legacy_union_with_zero_query_ranks(monkeypatch):
    import torch
    from hcu_vllm_das import dsa_cache_layer_split as runtime

    prefix, k_prefix = main(14), index(14)
    monkeypatch.setattr(runtime, '_source_block_sizes', {prefix: 64, k_prefix: 64})
    a, b = request(), request('b', 64, (3, 7, 9))
    # PCP locally compacts rows. Ranks can see different subsets (including
    # none), while the scheduler's page coordinates are global.
    selected = {name: set() for name in (prefix, k_prefix)}
    for rows in [(a,), (), (b,), (a, b), (), (b,), (), (a,)]:
        table = torch.tensor([r.block_ids[2] for r in rows]) if rows else None
        main_meta = NS(num_reqs=len(rows), block_table=table)
        index_meta = NS(block_table=table, num_decodes=0, num_prefills=len(rows),
                        num_decode_tokens=0, num_prefill_tokens=sum(r.scheduled for r in rows),
                        seq_lens=torch.tensor([r.scheduled for r in rows]))
        metadata = {prefix: main_meta, k_prefix: index_meta}
        for name, page_ids in selected.items():
            page_ids.update(runtime._live_block_ids(metadata, name, 16))
    groups = (PageGroup(2, 64, 16, (prefix, k_prefix)),)
    planned = fresh_prefill_page_ids(output(a, b), groups)
    assert selected[prefix] == selected[k_prefix] == set(planned[2])


def test_runner_scopes_plan_to_real_prefill_and_preserves_worker_arguments(monkeypatch):
    import torch
    from hcu_vllm_das import dsa_prefill_runner as runner

    marker = object()
    groups = (PageGroup(2, 64, 16, (main(2), index(2))),)
    monkeypatch.setattr(runner, '_page_groups', lambda *args: groups)

    class Core:
        device = torch.device('cpu')
        pcp_manager = object()
        kv_cache_config = object()
        block_tables = NS(block_sizes=[32, 32, 64], redirect_writes_to_null_block=False)

        def execute_model(self, batch, intermediate_tensors, **kwargs):
            assert intermediate_tensors is marker
            return active_prefill_pages()

    value = runner.layer_split_prefill_runner_cls(Core)()
    plan = value.execute_model(output(request()), marker)
    assert plan.by_prefix[main(2)] is plan.by_prefix[index(2)]
    assert plan.by_prefix[main(2)].tolist() == [2, 8]
    assert active_prefill_pages() is None
    assert value.execute_model(output(request()), marker, dummy_run=True) is None
    assert value.execute_model(output(request()), marker).received == {}


def swa_request(rid, length, offset, *, prompt_logprobs=False, block_size=64):
    num_pages = (length + block_size - 1) // block_size
    blocks = list(range(offset + num_pages, offset, -1))
    req = request(rid, length, blocks)
    req.prompt_token_ids = [1] * length
    if prompt_logprobs:
        req.sampling_params = NS(prompt_logprobs=0, prompt_logprob_token_ids=None)
    return req


@pytest.mark.parametrize('lengths,keep', [
    ((32768,), False), ((257,), False), ((129,), False), ((128,), False),
    ((17,), False), ((3,), False), ((257, 513, 17), False),
    ((129, 256, 9), True),
])
@pytest.mark.parametrize('size', [32, 64])
def test_swa_plans_match_old_reads_across_native_pcp_partitions(monkeypatch, lengths, keep, size):
    import numpy as np
    import torch
    from vllm.v1.attention.backends.utils import split_decodes_and_prefills
    from vllm.v1.worker.gpu.pcp_manager import PCPManager
    from hcu_vllm_das import dsa_cache_layer_split as runtime

    window, num_blocks = 128, 4096
    name = main(2) + '.swa_cache'
    group = PageGroup(2, size, num_blocks, (name,), window)
    lengths = np.array(lengths)
    reqs = [swa_request(str(i), int(n), 600 * i, prompt_logprobs=keep and i == 0, block_size=size)
            for i, n in enumerate(lengths)]
    plans = fresh_swa_prefill_page_ids(output(*reqs), (group,), replay_window=window)[2]
    monkeypatch.setattr(runtime, '_source_block_sizes', {name: size})
    pcp = PCPManager.__new__(PCPManager)
    pcp.pcp_world_size, pcp.dcp_world_size, pcp.shard_decode_requests = 8, 1, True
    for mode, planned in zip(('full', 'first', 'later'), plans):
        selected = set()
        for rank in range(pcp.pcp_world_size):
            segments = sorted(pcp._iter_rank_chunks(rank, lengths, np.ones(len(reqs), dtype=bool)))
            if not segments:
                continue
            ends, queries, floors, tables = [], [], [], []
            for req_idx, start, amount in segments:
                req, end = reqs[req_idx], start + amount
                cut = (max(0, req.scheduled - window)
                       if mode != 'full' and req.sampling_params is None else 0)
                queries.append(max(0, end - max(start, cut)))
                ends.append(end)
                floors.append(cut if mode == 'later' else 0)
                row = req.block_ids[2]
                tables.append(row + [0] * (max(len(r.block_ids[2]) for r in reqs) - len(row)))
            if not sum(queries):
                continue
            common = NS(max_query_len=max(n for _, _, n in segments), num_reqs=len(ends),
                        num_actual_tokens=sum(queries),
                        query_start_loc_cpu=torch.tensor([0, *np.cumsum(queries)]))
            ndec, npref, d_tokens, p_tokens = split_decodes_and_prefills(common, decode_threshold=6)
            # CPU equivalent of the native causal decode slots and Triton gather
            # lengths. Use the legacy extractor to determine actual physical IDs.
            rows = []
            for table, end, q, floor in zip(tables[:ndec], ends[:ndec], queries[:ndec], floors):
                for pos in range(end - q, end):
                    rows.append([table[t // size] * size + t % size
                                 for t in range(max(floor, pos - window + 1), pos + 1)])
            decode = torch.tensor([r + [-1] * (window - len(r)) for r in rows], dtype=torch.int64)
            gathers = [q + min(max(0, end - q - floor), window - 1)
                       for end, q, floor in zip(ends[ndec:], queries[ndec:], floors[ndec:])]
            meta = NS(block_size=size, block_table=torch.tensor(tables), num_prefills=npref,
                      num_decodes=ndec, num_prefill_tokens=p_tokens, num_decode_tokens=d_tokens,
                      decode_swa_indices=decode, decode_swa_lens=torch.tensor(list(map(len, rows))),
                      prefill_seq_lens=torch.tensor(ends[ndec:]), prefill_gather_lens=torch.tensor(gathers))
            selected.update(runtime._live_block_ids({name: meta}, name, num_blocks))
        if planned is None:
            assert mode == 'first' and len(reqs) > 1
        else:
            assert set(planned) == selected


def test_swa_plan_selects_actual_replay_context():
    import torch
    from hcu_vllm_das.dsa_prefill_runner import _add_swa_pages

    cut, later = (main(i) + '.swa_cache' for i in (20, 21))
    group = PageGroup(2, 64, 64, (cut, later), 128)
    batch = output(swa_request('a', 513, 0))
    replay_meta = {}
    layers = NS(window=128, first_swa_prefix=cut, replay_batch=None)
    state = NS(decoder_replay_layers=layers,
               vllm_config=NS(aux_output_config=NS(enable_return_routed_experts=False)))
    main_ids = torch.tensor([8])
    plan = PrefillPages({main(2): main_ids}, {main(2): 0})
    _add_swa_pages(plan, (group,), batch, state, torch.device('cpu'))
    assert plan.ids(cut, replay_meta) is plan.ids(later, {})
    assert plan.ids(cut, replay_meta).numel() == 9
    layers.replay_batch = NS(forward_context=NS(attn_metadata=replay_meta))
    assert plan.ids(cut, replay_meta).tolist() == [1, 2, 3, 4, 5]
    assert plan.ids(later, replay_meta).tolist() == [1, 2, 3]
    assert plan.ids(main(2), replay_meta) is main_ids
    assert plan.ids(cut, {}).numel() == 9

    mixed = PrefillPages({}, {})
    _add_swa_pages(mixed, (group,), output(*batch.scheduled_new_reqs, swa_request('b', 16, 20)),
                   state, torch.device('cpu'))
    assert mixed.ids(cut, replay_meta) is None
    assert mixed.ids(later, replay_meta).tolist() == [1, 2, 3, 21]
