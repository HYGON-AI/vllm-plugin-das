# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Byte-exact replicated DSA state over nonidentity P/D block mappings."""
from __future__ import annotations

import ast
import asyncio
import ctypes
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def aux():
    path = Path(__file__).resolve().parents[2] / "vllm_hcu/distributed/kv_transfer/kv_connector/v1/mooncake/step5_aux.py"
    spec = importlib.util.spec_from_file_location("step5_aux_tested", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Pending:
    def __init__(self):
        self.rows = {}

    def acquire(self, slot):
        return self.rows.setdefault(slot, len(self.rows))


def make_state():
    state = SimpleNamespace(
        num_layers=2, dtype=torch.bfloat16, block_size=16, regions_per_page=2,
        geometry=SimpleNamespace(region_size=8, proxy_dim=4),
        pending=Pending(), step_cache={},
        summary=torch.zeros((2, 26, 1, 4), dtype=torch.float8_e4m3fn),
        pending_key=torch.zeros((2, 32, 8, 4), dtype=torch.bfloat16),
        pending_z=torch.zeros((2, 32, 8, 4), dtype=torch.bfloat16),
    )
    state.ensure_allocated = lambda cache, device: None
    return state


def make_manager(aux, producer, rank=0):
    state = make_state()
    names = [f"model.layers.{i}.self_attn" for i in range(2)]
    indexers = {name: SimpleNamespace(state=state, slot=i) for i, name in enumerate(names)}
    caches = {name: torch.empty((13, 4 if producer else 1, 16, 8), dtype=torch.uint8)
              for name in names}
    engine = SimpleNamespace(registrations=[])
    engine.batch_register_memory = lambda ptrs, sizes: (engine.registrations.extend(zip(ptrs, sizes)) or 0)
    worker = SimpleNamespace(
        is_kv_producer=producer, engine=engine, seen_base_addresses=[], tp_rank=rank,
        vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=4)),
        _layer_group_indices={name: 0 for name in names},
        _logical_to_kernel_block_ids=lambda groups: [list(g) for g in groups],
        transfer_topo=SimpleNamespace(tp_ratio=lambda other: -4),
    )
    return aux.Step5AuxManager(worker, indexers, caches), state, indexers, worker


def connector_lifecycle_methods():
    source = Path(__file__).resolve().parents[2] / "vllm_hcu/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
    tree = ast.parse(source.read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    methods = {"MooncakeConnector": ["bind_connector_metadata", "get_finished"],
               "MooncakeConnectorWorker": ["start_load_kv", "_start_load_kv", "get_finished"]}
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for name, wanted in methods.items():
        node = classes[name]
        node.body = [item for item in node.body
                     if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and item.name in wanted]
        body.append(node)

    class Base:
        def bind_connector_metadata(self, metadata):
            self._connector_metadata = metadata

    namespace = {"KVConnectorBase_V1": Base, "MooncakeConnectorMetadata": SimpleNamespace,
                 "SupportsHMA": type("SupportsHMA", (), {}), "asyncio": asyncio,
                 "logger": logging.getLogger('step5-aux-tests')}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 str(source), "exec"), namespace)
    return namespace["MooncakeConnector"], namespace["MooncakeConnectorWorker"]


def test_model_runner_v2_binding_captures_before_deferred_start_load(aux):
    connector_type, worker_type = connector_lifecycle_methods()
    manager, state, indexers, worker = make_manager(aux, True)
    worker._step5_aux = manager
    worker.is_kv_consumer = False
    connector = connector_type()
    connector.connector_worker = worker
    metadata = SimpleNamespace(dsa_source_blocks={"x": [[7]]},
                               reqs_not_processed=set(), reqs_to_recv={}, reqs_to_send={})
    # Runner V2 binds before forward, but calls start_load_kv after forward.
    connector.bind_connector_metadata(metadata)
    assert connector._connector_metadata is metadata
    md = SimpleNamespace(query_start_loc=torch.tensor([0, 6]),
                         seq_lens=torch.tensor([6]), block_table=torch.tensor([[7]]))
    schedule = SimpleNamespace(decode=False, spec=False,
                               requests=[SimpleNamespace(row=0, pending_row_in=-1)])
    values = torch.arange(24).reshape(6, 1, 4).to(torch.bfloat16)
    manager.capture(next(iter(indexers.values())), md, schedule, values, values + 1)
    row = manager.pools[id(state)].get((0, 7))
    assert torch.equal(manager.pools[id(state)].tensor[0, row, 0, :6], values[:, 0])
    # A late start hook must not revive a transfer whose acknowledgements won
    # the race with post-forward bookkeeping.
    manager.release("x")
    worker_type.start_load_kv(worker, metadata)
    assert not manager._sources


def test_binding_releases_aborted_source_before_forward(aux):
    connector_type, _ = connector_lifecycle_methods()
    manager, state, _, worker = make_manager(aux, True)
    worker._step5_aux = manager
    manager.track_source("x", [[7]])
    manager.pools[id(state)].acquire((0, 7))
    connector = connector_type()
    connector.connector_worker = worker
    connector.bind_connector_metadata(SimpleNamespace(dsa_source_blocks={},
                                                      reqs_not_processed={"x"}))
    assert not manager._sources
    with pytest.raises(RuntimeError, match="no indexer tail"):
        manager.pools[id(state)].get((0, 7))


def test_binding_without_step5_preserves_base_contract():
    connector_type, _ = connector_lifecycle_methods()
    connector = connector_type()
    connector.connector_worker = SimpleNamespace()
    metadata = object()
    connector.bind_connector_metadata(metadata)
    assert connector._connector_metadata is metadata


@pytest.mark.parametrize("blocks", [[], [[]], [[], []]])
def test_notification_only_pull_skips_all_auxiliary_restoration(aux, blocks):
    manager = object.__new__(aux.Step5AuxManager)
    pull = SimpleNamespace(local_block_ids=blocks, num_prompt_tokens=0,
                           dsa_aux={"stale": object()})
    # No worker/mapper exists: notification handling must return before either
    # translating blocks or accessing any summary/pending-region storage.
    manager.prepare_pull(pull)
    assert pull.dsa_aux == {}


def test_allocated_pull_still_requires_prompt_length(aux):
    manager, state, _, _ = make_manager(aux, False)
    with pytest.raises(ValueError, match="prompt token count"):
        manager.prepare_pull(SimpleNamespace(local_block_ids=[[2]], num_prompt_tokens=0))
    assert not state.pending.rows


@pytest.mark.parametrize("cancelled_first", [True, False])
def test_preallocation_abort_notification_survives_mixed_worker_batch(aux, monkeypatch, cancelled_first):
    source = Path(__file__).resolve().parents[2] / "vllm_hcu/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
    tree = ast.parse(source.read_text())
    scheduler = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                     and n.name == "MooncakeConnectorScheduler")
    finish = next(n for n in scheduler.body if isinstance(n, ast.FunctionDef)
                  and n.name == "request_finished")
    namespace = {"logger": logging.getLogger(__name__)}
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), finish]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 str(source), "exec"), namespace)
    scheduler_state = SimpleNamespace(is_kv_producer=False, _reqs_need_recv={})
    request = SimpleNamespace(request_id="cancelled", status="aborted",
        kv_transfer_params={"transfer_id": "x", "do_remote_prefill": True})
    assert namespace['request_finished'](scheduler_state, request, ()) == (False, None)
    _, blocks = scheduler_state._reqs_need_recv['cancelled']
    assert blocks == []

    _, worker_type = connector_lifecycle_methods()
    manager, state, _, worker = make_manager(aux, False)
    worker._step5_aux = manager
    worker.is_kv_consumer = True
    worker._remote_agents = {"producer": object()}
    worker.receiver_loop = object()
    cancelled = SimpleNamespace(local_block_ids=blocks, num_prompt_tokens=0, dsa_aux={})
    active = SimpleNamespace(local_block_ids=[[2]], num_prompt_tokens=8, dsa_aux={})
    pulls = dict([("cancelled", cancelled), ("active", active)] if cancelled_first
                 else [("active", active), ("cancelled", cancelled)])
    dispatched = []
    worker.receive_kv = lambda engine, batch: dispatched.append((engine, batch))
    worker._start_load_kv = worker_type._start_load_kv.__get__(worker)
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", lambda coro, loop: asyncio.run(coro))
    worker_type.start_load_kv(worker, SimpleNamespace(reqs_to_recv={"producer": pulls},
        reqs_to_send={}, reqs_not_processed=set()))
    assert dispatched == [("producer", pulls)]
    assert cancelled.dsa_aux == {}
    assert active.dsa_aux
    assert len(state.pending.rows) == 1


def copy_registered(source_worker, destination_worker, pointers):
    sources, targets, sizes = pointers
    for source, target, size in zip(sources, targets, sizes):
        assert any(base <= source and source + size <= base + length
                   for base, length in source_worker.engine.registrations)
        assert any(base <= target and target + size <= base + length
                   for base, length in destination_worker.engine.registrations)
        ctypes.memmove(target, source, size)


@pytest.mark.parametrize("tokens", [1, 5, 8, 9, 15, 16, 17, 31, 32, 33])
def test_replicated_summaries_and_last_region_survive_tp1_to_tp4(aux, tokens):
    producer, pstate, indexers, pworker = make_manager(aux, True)
    count = (tokens + 15) // 16
    source_blocks = [7, 3, 11][:count]
    destination_blocks = [2, 5, 1][:count]
    producer.track_source("x", [[source_blocks[0]]])
    md = SimpleNamespace(
        query_start_loc=torch.tensor([0, tokens]), seq_lens=torch.tensor([tokens]),
        block_table=torch.tensor([source_blocks]),
    )
    schedule = SimpleNamespace(decode=False, spec=False,
                               requests=[SimpleNamespace(row=0, pending_row_in=-1)])
    expected = []
    for indexer in indexers.values():
        values = torch.arange(tokens * 4).reshape(tokens, 1, 4).to(torch.bfloat16)
        values += indexer.slot * 100
        producer.capture(indexer, md, schedule, values, values + 1)
        pstate.summary[indexer.slot].copy_((torch.arange(104).reshape(26, 1, 4) % 31).to(torch.float8_e4m3fn))
        length = (tokens - 1) % 8 + 1
        tail = torch.zeros((2, 8, 4), dtype=torch.bfloat16)
        tail[0, :length] = values[-length:, 0]
        tail[1, :length] = (values + 1)[-length:, 0]
        expected.append(tail)
    # Growing a chunked prefill's block list must retain its snapshot row.
    producer.track_source("x", [source_blocks])

    for rank in range(4):
        consumer, dstate, _, dworker = make_manager(aux, False, rank)
        pull = SimpleNamespace(local_block_ids=[destination_blocks], num_prompt_tokens=tokens, dsa_aux={})
        consumer.prepare_pull(pull)
        peer = SimpleNamespace(remote_tp_size=4, dsa_aux={"d": pull.dsa_aux})
        copy_registered(pworker, dworker, producer.transfer_parameters(
            "d", peer,
            [source_blocks], [destination_blocks],
        ))
        slot = destination_blocks[-1] * 2 + ((tokens - 1) % 16) // 8
        row = dstate.pending.rows[slot]
        for layer in range(2):
            for source_block, destination_block in zip(source_blocks, destination_blocks):
                assert torch.equal(
                    pstate.summary[layer, source_block * 2:(source_block + 1) * 2].view(torch.uint8),
                    dstate.summary[layer, destination_block * 2:(destination_block + 1) * 2].view(torch.uint8),
                )
            assert torch.equal(dstate.pending_key[layer, row], expected[layer][0])
            assert torch.equal(dstate.pending_z[layer, row], expected[layer][1])
    producer.release("x")
    with pytest.raises(RuntimeError, match="no indexer tail"):
        producer.pools[id(pstate)].get((0, source_blocks[0]))


@pytest.mark.parametrize("past,query,decode", [(3, 2, False), (7, 1, True),
                                               (15, 1, True), (29, 4, False)])
def test_tail_capture_precedes_pending_release_at_chunk_and_region_boundaries(aux, past, query, decode):
    manager, state, indexers, _ = make_manager(aux, True)
    tokens = past + query
    blocks = [7, 3, 11][:(tokens + 15) // 16]
    manager.track_source("x", [blocks])
    md = SimpleNamespace(query_start_loc=torch.tensor([0, query]),
                         seq_lens=torch.tensor([tokens]), block_table=torch.tensor([blocks]))
    schedule = SimpleNamespace(decode=decode, spec=False, pending_in=torch.tensor([2]),
                               requests=[SimpleNamespace(row=0, pending_row_in=2)])
    length = (tokens - 1) % 8 + 1
    prefix = max(length - query, 0)
    for indexer in indexers.values():
        state.pending_key[indexer.slot, 2].fill_(7)
        state.pending_z[indexer.slot, 2].fill_(9)
        keys = torch.full((query, 1, 4), 3, dtype=torch.bfloat16)
        manager.capture(indexer, md, schedule, keys, keys + 1)
        # The normal compression may release/reuse this row immediately.
        state.pending_key[indexer.slot, 2].zero_()
        snapshot = manager.pools[id(state)].tensor[indexer.slot, manager.pools[id(state)].get((0, blocks[0]))]
        assert (snapshot[0, :prefix] == 7).all()
        assert (snapshot[1, :prefix] == 9).all()
        assert (snapshot[0, prefix:length] == 3).all()
        assert (snapshot[1, prefix:length] == 4).all()


def test_nonleader_producer_does_not_race_replicated_auxiliary_writes(aux):
    manager, _, _, worker = make_manager(aux, True, rank=1)
    worker.transfer_topo.tp_ratio = lambda other: 2
    assert manager.transfer_parameters("d", SimpleNamespace(remote_tp_size=1, dsa_aux={}), [[1]], [[2]]) == ([], [], [])


def test_snapshot_pool_never_evicts_an_unacknowledged_transfer(aux):
    pool = aux._SnapshotPool(make_state(), 1, torch.device("cpu"))
    row = pool.acquire(7)
    assert pool.acquire(7) == row
    with pytest.raises(RuntimeError, match="pool is full"):
        pool.acquire(8)
    pool.release(7)
    assert pool.acquire(8) == row


def test_missing_peer_auxiliary_state_fails_instead_of_decoding_with_empty_summaries(aux):
    manager, _, _, _ = make_manager(aux, True)
    with pytest.raises(ValueError, match="did not provide compatible"):
        manager.transfer_parameters("d", SimpleNamespace(remote_tp_size=4, dsa_aux={}), [[1]], [[2]])


@pytest.mark.parametrize("field,value", [(0, 1), (2, 999), (3, 0), (4, 0), (6, 1), (7, 2)])
def test_incompatible_auxiliary_descriptors_fail_before_any_memory_copy(aux, field, value):
    producer, _, _, _ = make_manager(aux, True)
    consumer, _, _, _ = make_manager(aux, False)
    pull = SimpleNamespace(local_block_ids=[[2]], num_prompt_tokens=5, dsa_aux={})
    consumer.prepare_pull(pull)
    name = next(iter(pull.dsa_aux))
    descriptor = list(pull.dsa_aux[name])
    descriptor[field] = value
    pull.dsa_aux[name] = tuple(descriptor)
    peer = SimpleNamespace(remote_tp_size=4, dsa_aux={"d": pull.dsa_aux})
    with pytest.raises(ValueError, match="geometry|descriptor"):
        producer.transfer_parameters("d", peer, [[1]], [[2]])


@pytest.mark.parametrize("source,destination", [(100, 2), (1, 100)])
def test_auxiliary_block_bounds_are_checked(aux, source, destination):
    producer, _, _, _ = make_manager(aux, True)
    consumer, _, _, _ = make_manager(aux, False)
    pull = SimpleNamespace(local_block_ids=[[2]], num_prompt_tokens=5, dsa_aux={})
    consumer.prepare_pull(pull)
    peer = SimpleNamespace(remote_tp_size=4, dsa_aux={"d": pull.dsa_aux})
    with pytest.raises(ValueError, match="exceeds its registered pool"):
        producer.transfer_parameters("d", peer, [[source]], [[destination]])


def test_auxiliary_restore_requires_prompt_length(aux):
    consumer, _, _, _ = make_manager(aux, False)
    with pytest.raises(ValueError, match="prompt token count"):
        consumer.prepare_pull(SimpleNamespace(local_block_ids=[[2]], num_prompt_tokens=0))


def test_ordinary_non_pd_prefill_does_not_consume_a_snapshot_row(aux):
    manager, state, indexers, _ = make_manager(aux, True)
    manager.capture(next(iter(indexers.values())), None, None, torch.zeros((1, 1, 4)), torch.zeros((1, 1, 4)))
    assert manager.pools[id(state)]._rows == {}


def test_imported_tail_is_not_erased_by_deferred_cleanup_of_reused_block(aux):
    manager, state, _, _ = make_manager(aux, False)
    slot = 2 * 2
    state._schedule = SimpleNamespace(completed_slots=[slot, slot + 1])
    manager.prepare_pull(SimpleNamespace(local_block_ids=[[2]], num_prompt_tokens=5, dsa_aux={}))
    assert state._schedule.completed_slots == [slot + 1]
    assert slot in state.pending.rows


def test_snapshot_keys_preserve_kv_group_identity(aux):
    pool = aux._SnapshotPool(make_state(), 2, torch.device("cpu"))
    first = pool.acquire((0, 7))
    second = pool.acquire((1, 7))
    assert first != second
    pool.release((0, 7))
    assert pool.get((1, 7)) == second


@pytest.mark.parametrize("total,cut", [(8, 6), (9, 7), (27, 25),
                                       (31, 15), (33, 17), (35, 19)])
def test_interleaved_dbo_microbatches_keep_the_final_raw_region(aux, total, cut):
    manager, state, indexers, _ = make_manager(aux, True)
    blocks = [7, 3, 11][:(total + 15) // 16]
    manager.track_source("x", [blocks])
    metadata = [
        SimpleNamespace(query_start_loc=torch.tensor([0, cut]),
                        seq_lens=torch.tensor([cut]),
                        block_table=torch.tensor([blocks[:(cut + 15) // 16]])),
        SimpleNamespace(query_start_loc=torch.tensor([0, total - cut]),
                        seq_lens=torch.tensor([total]), block_table=torch.tensor([blocks])),
    ]
    schedules = [
        SimpleNamespace(decode=False, spec=False,
                        requests=[SimpleNamespace(row=0, pending_row_in=-1)]),
        SimpleNamespace(decode=False, spec=False,
                        requests=[SimpleNamespace(row=0, pending_row_in=2)]),
    ]
    caches = [{}, {}]
    expected = []
    # HCU DBO alternates microbatches at each layer, restoring each one's
    # step_cache; their attention compute is ordered on one compute stream.
    for indexer in indexers.values():
        values = (torch.arange(total * 4).reshape(total, 1, 4)
                  + indexer.slot * 100).to(torch.bfloat16)
        for uid, (start, end) in enumerate(((0, cut), (cut, total))):
            state.step_cache = caches[uid]
            manager.capture(indexer, metadata[uid], schedules[uid],
                            values[start:end], values[start:end] + 1)
            if uid == 0 and cut % 8:
                # The normal first microbatch compression leaves this prefix
                # for the straddling request in the second microbatch.
                tail = values[cut - cut % 8:cut, 0]
                state.pending_key[indexer.slot, 2, :cut % 8].copy_(tail)
                state.pending_z[indexer.slot, 2, :cut % 8].copy_(tail + 1)
        expected.append(values[-((total - 1) % 8 + 1):, 0])
    row = manager.pools[id(state)].get((0, 7))
    snapshot = manager.pools[id(state)].tensor[:, row]
    for layer, values in enumerate(expected):
        assert torch.equal(snapshot[layer, 0, :len(values)], values)
        assert torch.equal(snapshot[layer, 1, :len(values)], values + 1)
    assert len(manager.pools[id(state)]._rows) == 1


@pytest.mark.parametrize("producer,enabled,owned,eager,threshold,allowed", [
    (True, True, True, True, 4096, True),
    (True, True, False, True, 4096, False),
    (False, True, True, True, 4096, False),
    (True, True, True, False, 4096, False),
    (True, False, False, True, 0, True),
    (True, False, False, True, 4096, False),
])
def test_dbo_requires_owned_eager_producer_runtime(
    aux, monkeypatch, producer, enabled, owned, eager, threshold, allowed,
):
    module = ModuleType("vllm_hcu.model_executor.layers.step4_dsa")
    module._INDEXERS = {"layer": object()}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("HCU_DBO_MIN_TOKENS", str(threshold))
    sentinel = object()
    monkeypatch.setattr(aux, "Step5AuxManager", lambda *args: sentinel)
    model = SimpleNamespace(hf_config=SimpleNamespace(model_type="step5"),
                            enforce_eager=eager)
    parallel = SimpleNamespace(enable_dbo=enabled, _hcu_step5_dbo=owned)
    worker = SimpleNamespace(
        model_config=model, is_kv_producer=producer, is_kv_consumer=not producer,
        vllm_config=SimpleNamespace(
            model_config=model, parallel_config=parallel,
            kv_transfer_config=SimpleNamespace(kv_connector="MooncakeConnector",
                kv_role="kv_producer" if producer else "kv_consumer"),
            cache_config=SimpleNamespace(enable_prefix_caching=False)),
    )
    if allowed:
        assert aux.create_step5_aux_manager(worker, {"layer": object()}) is sentinel
    else:
        with pytest.raises(NotImplementedError, match="DBO|eager"):
            aux.create_step5_aux_manager(worker, {"layer": object()})


@pytest.mark.parametrize("model_type,transfer", [
    ("llama", True), ("deepseek_v3", True), ("step3p5", True), ("step5", False),
])
def test_other_models_and_ifb_never_import_indexers(aux, monkeypatch, model_type, transfer):
    monkeypatch.setitem(sys.modules, "vllm_hcu.model_executor.layers.step4_dsa", None)
    model = SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))
    config = SimpleNamespace(model_config=model,
        kv_transfer_config=SimpleNamespace(kv_connector="MooncakeConnector",
            kv_role="kv_producer") if transfer else None)
    assert aux.create_step5_aux_manager(SimpleNamespace(vllm_config=config), {}) is None


@pytest.mark.parametrize('heads,padding,split,allowed', [
    (1, 0, False, True), (1, 32, False, False),
    (4, 0, False, False), (1, 0, True, True),
])
def test_compiled_consumer_requires_affine_sparse_kv(aux, monkeypatch, heads, padding, split, allowed):
    module = ModuleType('vllm_hcu.model_executor.layers.step4_dsa')
    module._INDEXERS = {'layer': object()}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv('HCU_DBO_MIN_TOKENS', '0')
    sentinel = object()
    monkeypatch.setattr(aux, 'Step5AuxManager', lambda *args: sentinel)
    model = SimpleNamespace(hf_config=SimpleNamespace(model_type='step5'), enforce_eager=False)
    config = SimpleNamespace(model_config=model, parallel_config=SimpleNamespace(enable_dbo=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        kv_transfer_config=SimpleNamespace(kv_connector='MooncakeConnector', kv_role='kv_consumer'))
    worker = SimpleNamespace(vllm_config=config, is_kv_producer=False, is_kv_consumer=True)
    if split:
        cache = torch.empty((2, 3, 16, heads, 4), dtype=torch.uint8)
    else:
        stride = heads * 16 * 8 + padding
        cache = torch.as_strided(torch.empty(3 * stride, dtype=torch.uint8),
            (3, heads, 16, 8), (stride, 16 * 8, 8, 1))
    if allowed:
        assert aux.create_step5_aux_manager(worker, {'layer': cache}) is sentinel
        config.cache_config.enable_prefix_caching = True
        with pytest.raises(NotImplementedError, match='prefix'):
            aux.create_step5_aux_manager(worker, {'layer': cache})
    else:
        with pytest.raises(NotImplementedError, match='affine'):
            aux.create_step5_aux_manager(worker, {'layer': cache})


@pytest.mark.parametrize('field', ['summary', 'pending_key', 'pending_z'])
def test_rebound_auxiliary_allocations_reject_transfer_but_keep_cleanup_usable(aux, field):
    manager, state, _, _ = make_manager(aux, False)
    original = getattr(state, field)
    setattr(state, field, original.clone())
    with pytest.raises(RuntimeError, match='re-register'):
        manager.prepare_pull(SimpleNamespace(local_block_ids=[[2]], num_prompt_tokens=5))
    cleanup = SimpleNamespace(local_block_ids=[], num_prompt_tokens=0, dsa_aux={'old': ()})
    manager.prepare_pull(cleanup)
    assert cleanup.dsa_aux == {}
    assert any(original is t for _, tensors, _ in manager._allocations.values() for t in tensors)


def test_inplace_storage_rebinding_is_rejected(aux):
    manager, state, _, _ = make_manager(aux, False)
    registered_pointer = state.pending_key.data_ptr()
    state.pending_key.set_(torch.zeros_like(state.pending_key))
    with pytest.raises(RuntimeError, match='re-register'):
        manager.prepare_pull(SimpleNamespace(local_block_ids=[[2]], num_prompt_tokens=5))
    assert any(s.data_ptr() == registered_pointer for s in manager._registered_storages)


def real_pending(capacity):
    path = Path(__file__).resolve().parents[2] / 'vllm_hcu/model_executor/layers/step4_dsa.py'
    node = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(n, ast.ClassDef) and n.name == 'Step4PendingTable')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    scope={};exec(compile(ast.fix_missing_locations(module),str(path),'exec'),scope)
    return scope['Step4PendingTable'](capacity)


def test_cancelled_import_stays_reserved_until_dma_finishes(aux):
    manager,state,_,_=make_manager(aux,False)
    state.pending=real_pending(2);manager._compiled_consumer=True
    manager.prepare_pull(SimpleNamespace(d_req_id='request',local_block_ids=[[2]],num_prompt_tokens=5))
    owner=manager._imports['request'].owner;slot=4
    for i in range(8):
        state.pending.begin_step();state.pending.acquire(10+i)
        state.pending.retire(slot);state.pending.release(slot)
        assert state.pending.is_reserved(slot,owner)
    manager.finish_pulls(set(),{'request'})
    assert state.pending.is_reserved(slot,owner)
    manager.finish_pulls({'request'},set())
    assert slot not in state.pending._row_of_slot and not manager._imports


def test_ready_import_waits_for_consumption_and_stale_cleanup_cannot_release_reuse(aux):
    manager,state,_,_=make_manager(aux,False)
    state.pending=real_pending(2);manager._compiled_consumer=True
    manager.prepare_pull(SimpleNamespace(d_req_id='request',local_block_ids=[[2]],num_prompt_tokens=5))
    old=manager._imports['request'].owner;slot=4
    manager.finish_pulls({'request'},set())
    assert state.pending.is_reserved(slot,old)
    for i in range(4):state.pending.begin_step();state.pending.acquire(20+i)
    state.pending.get(slot)
    manager.finish_pulls(set(),set())
    assert not manager._imports
    new=object();state.pending.reserve(slot,new)
    state.pending.release_reservation(slot,old)
    assert state.pending.is_reserved(slot,new)
    with pytest.raises(RuntimeError,match='another import'):
        state.pending.reserve(slot,object())


def test_all_reserved_rows_fail_instead_of_overwriting_live_dma():
    pending=real_pending(1);owner=object();pending.reserve(7,owner)
    pending.begin_step()
    with pytest.raises(RuntimeError,match='in use'):
        pending.acquire(8)
    assert pending.is_reserved(7,owner)
    pending.release_reservation(7,owner)
    assert pending.acquire(8)==0


@pytest.mark.parametrize('block',[-1,13])
def test_invalid_import_destination_does_not_reserve_a_pending_row(aux,block):
    manager,state,_,_=make_manager(aux,False)
    state.pending=real_pending(2);manager._compiled_consumer=True
    with pytest.raises(ValueError,match='registered pool'):
        manager.prepare_pull(SimpleNamespace(d_req_id='request',local_block_ids=[[block]],num_prompt_tokens=5))
    assert state.pending._reservations=={}


def test_receiver_waits_for_prior_compute_before_submitting_rdma():
    _,worker_type=connector_lifecycle_methods()
    events=[]
    event=SimpleNamespace(synchronize=lambda:events.append('compute-complete'))
    worker=SimpleNamespace(_remote_agents={'producer':object()},
        receive_kv=lambda *args:events.append('receive'))
    asyncio.run(worker_type._start_load_kv(worker,{'producer':{}},event))
    assert events==['compute-complete','receive']


def test_connector_reports_transfer_and_request_completion_to_import_leases(monkeypatch):
    connector_type,worker_type=connector_lifecycle_methods()
    calls=[]
    async def received():return {'received'}
    monkeypatch.setattr(asyncio,'run_coroutine_threadsafe',
        lambda coro,loop:SimpleNamespace(result=lambda:asyncio.run(coro)))
    worker=SimpleNamespace(is_kv_producer=False,is_kv_consumer=True,tp_rank=0,receiver_loop=object(),
        fetch_finished_recving_reqs=received,
        _step5_aux=SimpleNamespace(finish_pulls=lambda *args:calls.append(args)))
    worker.get_finished=worker_type.get_finished.__get__(worker)
    connector=object.__new__(connector_type);connector.connector_worker=worker
    assert connector.get_finished({'cancelled'})==(None,{'received'})
    assert calls==[({'received'},{'cancelled'})]


@pytest.mark.parametrize("source_groups,target_groups", [([], [[2]]), ([[1]], [])])
def test_missing_kv_group_is_reported_before_pointer_lookup(aux, source_groups, target_groups):
    producer, _, _, _ = make_manager(aux, True)
    consumer, _, _, _ = make_manager(aux, False)
    pull = SimpleNamespace(local_block_ids=[[2]], num_prompt_tokens=8, dsa_aux={})
    consumer.prepare_pull(pull)
    peer = SimpleNamespace(remote_tp_size=4, dsa_aux={"d": pull.dsa_aux})
    with pytest.raises(ValueError, match="group mapping"):
        producer.transfer_parameters("d", peer, source_groups, target_groups)
