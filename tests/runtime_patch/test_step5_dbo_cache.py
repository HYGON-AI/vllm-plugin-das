# SPDX-License-Identifier: Apache-2.0
"""DBO must retain each microbatch's DSA plan inputs across layer handoffs.

Execute the real schedule, pending table, planner, decode-row derivation and
first-indexer-layer reset on CPU. Only the GPU region selector is replaced.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
DSA = "vllm_hcu/model_executor/layers/step4_dsa.py"


def _tree(path):
    return ast.parse((ROOT / path).read_text(encoding="utf-8"))


def _exec(nodes, namespace, filename):
    module = ast.Module(body=[
        ast.ImportFrom("__future__", [ast.alias("annotations")], 0), *nodes,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)


def _runtime(monkeypatch):
    tree = _tree(DSA)
    subject = ModuleType("step5_dbo_cache_subject")
    monkeypatch.setitem(sys.modules, subject.__name__, subject)
    namespace = subject.__dict__
    namespace.update(torch=torch, np=np, dataclass=dataclass, field=field,
                     REGION_BLOCK_SIZE=8)
    classes = {"Step4DSAMetadata", "Step4SparsePlan", "Step4PendingTable",
               "_RequestPlan", "Step4DSASchedule"}
    nodes = [node for node in tree.body
             if isinstance(node, ast.ClassDef) and node.name in classes]
    state = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "Step4DSAState")
    methods = {"schedule", "_general_schedule", "finish_layer", "end_step"}
    nodes.append(ast.ClassDef(
        name=state.name, bases=[], keywords=[], decorator_list=[],
        body=[node for node in state.body
              if isinstance(node, ast.FunctionDef) and node.name in methods],
    ))
    nodes.append(next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "build_plan"))
    _exec(nodes, namespace, DSA)
    kernels = "vllm_hcu/model_executor/layers/step4_dsa_kernels.py"
    _exec([next(node for node in _tree(kernels).body
                if isinstance(node, ast.FunctionDef) and node.name == "decode_rows_meta")],
          namespace, kernels)
    actual_rows = namespace["decode_rows_meta"]
    derived = []
    selected = []

    def rows(*args):
        value = actual_rows(*args)
        derived.append(value)
        return value

    def select(*args, rows_meta, **kwargs):
        selected.append(rows_meta)
        return torch.empty((1, 1), dtype=torch.int64), rows_meta[1]

    namespace.update(decode_rows_meta=rows, decode_sparse_meta=select)

    ctx = dict(enabled=True, uid=0)
    ubatching = ModuleType("vllm.v1.worker.ubatching")
    ubatching.dbo_enabled = lambda: ctx["enabled"]
    ubatching.dbo_current_ubatch_id = lambda: ctx["uid"]
    monkeypatch.setitem(sys.modules, ubatching.__name__, ubatching)
    monkeypatch.setitem(sys.modules, "vllm_hcu.model_executor.layers.step4_dsa", subject)
    dbo_path = "vllm_hcu/v1/hcu_dbo.py"
    dbo_tree = _tree(dbo_path)
    dbo = dict(sys=sys, np=np)
    _exec([node for node in dbo_tree.body if isinstance(node, ast.FunctionDef)
           and node.name in {"_noop", "_patch_dsa", "_has_prefill_in_both_ubatches"}],
          dbo, dbo_path)
    dbo["_patch_dsa"]()

    indexer = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "Step4Indexer")
    run = next(node for node in indexer.body if isinstance(node, ast.FunctionDef)
               and node.name == "run")
    # Use the real reset statement, including its order after schedule(). A
    # fixture that skips this assignment would miss the original handoff bug.
    reset = next(node for node in run.body if isinstance(node, ast.If)
                 and isinstance(node.test, ast.Compare)
                 and isinstance(node.test.left, ast.Attribute)
                 and node.test.left.attr == "slot")
    reset_code = compile(ast.fix_missing_locations(ast.Module(
        body=[reset], type_ignores=[])), DSA, "exec")

    state = namespace["Step4DSAState"]()
    state.geometry = SimpleNamespace(region_size=8)
    state.regions_per_page = 2
    state.spec_query_len = 1
    state.pending = namespace["Step4PendingTable"](16)
    state.step_cache = {}
    state._schedule_key = None
    state._schedule = None
    state._layers_done = 0
    state._registered = {0, 1}
    state.first_slot = 0
    return SimpleNamespace(state=state, ctx=ctx, namespace=namespace,
                           reset=reset_code, selected=selected, derived=derived,
                           gate=dbo["_has_prefill_in_both_ubatches"])


def _metadata(runtime, queries, lengths, request_rows):
    qsl = torch.tensor([0, *np.cumsum(queries).tolist()], dtype=torch.int32)
    rows = [torch.arange(1 + row * 4096, 1 + row * 4096 + 2048,
                         dtype=torch.int32) for row in request_rows]
    table = torch.stack(rows)
    seq = torch.tensor(lengths, dtype=torch.int32)
    decodes = sum(query == 1 for query in queries)
    return runtime.namespace["Step4DSAMetadata"](
        num_decodes=decodes, num_prefills=len(queries) - decodes,
        num_decode_tokens=decodes, num_prefill_tokens=sum(queries) - decodes,
        num_actual_tokens=sum(queries), block_table=table,
        slot_mapping=torch.empty(sum(queries), dtype=torch.int64), seq_lens=seq,
        query_start_loc=qsl, query_start_loc_cpu=qsl, seq_lens_cpu=seq,
        positions=None, max_seq_len=max(lengths), block_size=16,
        block_table_np=table.numpy(),
    )


def _split_metadata(runtime, decode_length=8193):
    # Actual split point 2048 for [1, 2048, 2048]. The middle request
    # contributes its first 2047 tokens to ub0 and its last token to ub1.
    return (
        _metadata(runtime, [1, 2047], [decode_length, 2047], [0, 1]),
        _metadata(runtime, [1, 2048], [2048, 2048], [1, 2]),
    )


def _layer(runtime, md, uid, slot):
    runtime.ctx["uid"] = uid
    state = runtime.state
    sched = state.schedule(md)
    exec(runtime.reset, {"self": SimpleNamespace(slot=slot), "state": state})
    runtime.namespace["build_plan"](
        md, None, torch.empty((1, 1024)), topk=512,
        num_kv_groups=1, regions_per_page=state.regions_per_page,
        step_cache=state.step_cache,
    )
    rows_meta = runtime.selected[-1]
    state.finish_layer()
    return int(rows_meta[1][0]), state.step_cache, sched


def _two_layers(runtime, metadata):
    return [_layer(runtime, metadata[uid], uid, slot)
            for slot in (0, 1) for uid in (0, 1)]


def test_dbo_mixed_prefill_decode_caches_follow_each_microbatch(monkeypatch):
    runtime = _runtime(monkeypatch)
    assert runtime.gate([1, 2048, 2048], 4097, 2048, 1)
    metadata = _split_metadata(runtime)
    results = _two_layers(runtime, metadata)
    assert [value[0] for value in results] == [1024, 255, 1024, 255]
    assert results[0][1] is results[2][1]
    assert results[1][1] is results[3][1]
    assert results[0][1] is not results[1][1]
    assert results[0][2] is results[2][2]
    assert results[1][2] is results[3][2]
    assert len(runtime.derived) == 2
    # Both schedules still belong to one pending-table step. The closed tail
    # from the straddling request is transferred to the main schedule and
    # released by the existing layer-completion lifecycle.
    assert runtime.state.pending.step == 1
    assert runtime.state._schedule.completed_slots == []
    assert results[1][2].completed_slots == []


def test_new_dbo_step_discards_both_prior_microbatch_caches(monkeypatch):
    runtime = _runtime(monkeypatch)
    old = _two_layers(runtime, _split_metadata(runtime))
    newer = _two_layers(runtime, _split_metadata(runtime, decode_length=16385))
    assert [value[0] for value in newer] == [2048, 255, 2048, 255]
    assert newer[0][1] is not old[0][1]
    assert newer[1][1] is not old[1][1]
    assert len(runtime.derived) == 4
    assert runtime.state.pending.step == 2


def test_unsplit_forward_clears_dbo_cache_lifetime(monkeypatch):
    runtime = _runtime(monkeypatch)
    old = _two_layers(runtime, _split_metadata(runtime))
    runtime.ctx["enabled"] = False
    md = _metadata(runtime, [1, 2048, 2048], [24577, 2048, 2048], [0, 1, 2])
    whole = [_layer(runtime, md, 0, slot) for slot in (0, 1)]
    assert [value[0] for value in whole] == [3072, 3072]
    assert whole[0][1] is whole[1][1]
    assert not hasattr(runtime.state, "_hcu_dbo_sched")
    assert not hasattr(runtime.state, "_hcu_dbo_cache_uid")
    runtime.ctx["enabled"] = True
    split = _two_layers(runtime, _split_metadata(runtime, decode_length=16385))
    assert [value[0] for value in split] == [2048, 255, 2048, 255]
    assert split[0][1] is not old[0][1] and split[0][1] is not whole[0][1]
    assert split[1][1] is not old[1][1] and split[1][1] is not whole[0][1]
    assert len(runtime.derived) == 5
    assert runtime.state.pending.step == 3
