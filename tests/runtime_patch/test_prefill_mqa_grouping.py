"""CPU contracts for opt-in prefill MQA request grouping."""
from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
import random
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[2] / "vllm_hcu/v1/attention/ops/prefill_mqa.py"


def _module():
    spec = importlib.util.spec_from_file_location("prefill_mqa_contract", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("qs,ks,start,end,expected", [
    ([5, 7], [9, 13], 3, 9, ((0, 2, 0, 9), (2, 6, 9, 22))),
    ([0, 2, 0], [4, 7, 5], 0, 2, ((0, 2, 4, 11),)),
    ([3], [9], 0, 3, ()),
    ([0, 0], [4, 7], 0, 0, ()),
])
def test_cpu_plan_preserves_partial_request_and_zero_q_k_offsets(qs, ks, start, end, expected):
    # Losing zero-Q requests' K spans or failing to clip query chunks breaks indices.
    actual = _module().plan_prefill_mqa_chunks(qs, ks, start, end, min_saved_cells=0)
    assert actual == expected


def test_logits_budget_splits_q_without_changing_request_k_ranges():
    actual = _module().plan_prefill_mqa_chunks(
        [5, 3], [10, 20], min_saved_cells=0, max_logits_bytes=80,
    )
    assert actual == ((0, 2, 0, 10), (2, 4, 0, 10), (4, 5, 0, 10),
                      (5, 6, 10, 30), (6, 7, 10, 30), (7, 8, 10, 30))


@pytest.mark.parametrize("qs,ks,kwargs", [
    ([-1], [2], {}), ([1], [-1], {}), ([1], [2, 3], {}),
    ([1], [2], {"min_saved_cells": -1}),
    ([1], [2], {"query_start": -1}),
    ([1], [2], {"query_end": 2}),
    ([1], [2], {"max_logits_bytes": -1}),
])
def test_invalid_cpu_metadata_is_rejected(qs, ks, kwargs):
    with pytest.raises(ValueError):
        _module().plan_prefill_mqa_chunks(qs, ks, **kwargs)


def test_contiguous_group_cost_matches_independent_exhaustive_partitions():
    planner = _module().plan_prefill_mqa_chunks
    rng = random.Random(0)
    for _ in range(100):
        count = rng.randint(2, 6)
        qs = [rng.randint(1, 20) for _ in range(count)]
        ks = [rng.randint(1, 100) for _ in range(count)]
        threshold = rng.randint(0, 1000)
        q_starts = [0, *itertools.accumulate(qs)]
        k_starts = [0, *itertools.accumulate(ks)]
        def cost(groups):
            waste = 0
            for a, b, c, d in groups:
                waste += sum(qs[i] * (k_starts[i] - c)
                             for i in range(count) if a <= q_starts[i] < b)
            return waste + threshold * (len(groups) - 1), -len(groups)
        possibilities = []
        for mask in range(1 << (count - 1)):
            boundaries = [0] + [i for i in range(1, count) if mask & (1 << (i - 1))] + [count]
            groups = [(q_starts[a], q_starts[b], k_starts[a], k_starts[b])
                      for a, b in zip(boundaries, boundaries[1:])]
            possibilities.append(cost(groups))
        groups = planner(qs, ks, min_saved_cells=threshold) or ((0, sum(qs), 0, sum(ks)),)
        assert cost(groups) == min(possibilities)


def test_metadata_adapter_builds_plan_from_cpu_partial_query_lengths(monkeypatch):
    import torch
    from vllm_hcu.platforms import envs as henvs
    from vllm_hcu.patch.worker.op_opt.patch_mla_indexer import _wrap_prefill_chunk_builder
    monkeypatch.setattr(henvs, "VLLM_HCU_ENABLE_PREFILL_MQA_GROUPING", True, raising=False)
    monkeypatch.setattr(henvs, "VLLM_HCU_PREFILL_MQA_GROUP_MIN_SAVED_CELLS", 0, raising=False)
    chunk = SimpleNamespace()
    builder = _wrap_prefill_chunk_builder(lambda *args: chunk)
    result = builder(0, 2, None, torch.tensor([0, 5, 12]), None, None,
                     torch.tensor([9, 13]), None, 1, slice(3, 9))
    assert result.hcu_mqa_groups == ((0, 2, 0, 9), (2, 6, 9, 22))


@pytest.mark.parametrize("enabled,dcp,compress", [(False, 1, 1), (True, 2, 1), (True, 1, 4)])
def test_metadata_adapter_leaves_disabled_dcp_and_compressed_paths_unchanged(monkeypatch, enabled, dcp, compress):
    from vllm_hcu.platforms import envs as henvs
    from vllm_hcu.patch.worker.op_opt.patch_mla_indexer import _wrap_prefill_chunk_builder
    monkeypatch.setattr(henvs, "VLLM_HCU_ENABLE_PREFILL_MQA_GROUPING", enabled, raising=False)
    chunk = SimpleNamespace()
    builder = _wrap_prefill_chunk_builder(lambda *args: chunk)
    # CPU metadata deliberately unavailable: a guarded path must not read it.
    assert builder(0, 2, None, None, None, None, None, None, compress,
                   dcp_world_size=dcp) is chunk
    assert not hasattr(chunk, "hcu_mqa_groups")


def test_runtime_rejects_incomplete_plan_before_mutating_output():
    import torch
    fn = _module().run_grouped_prefill_mqa
    out = torch.full((3, 2), -77, dtype=torch.int32)
    def forbidden(*args):
        pytest.fail("invalid plan dispatched a kernel")
    assert not fn(torch.zeros(3, 1, 1), torch.zeros(4, 1), torch.ones(4),
                  torch.ones(3, 1), torch.zeros(3), torch.ones(3), out,
                  ((0, 2, 0, 4),), mqa=forbidden, topk=forbidden)
    assert out.tolist() == [[-77, -77]] * 3


@pytest.mark.parametrize("qs,ns,start,end", [
    ([5, 7], [9, 13], 3, 9), ([0, 2, 0], [4, 7, 5], 0, 2),
    ([2, 3], [0, 7], 0, 5),
])
def test_grouped_execution_matches_independent_cpu_reference(qs, ns, start, end):
    import torch
    helper = _module()
    torch.manual_seed(17)
    rows = end-start
    q, keys = torch.randn(rows, 2, 4), torch.randn(sum(ns), 4)
    scales, weights = torch.arange(1, sum(ns)+1).float()/16, torch.rand(rows, 2)
    starts, ends = [], []
    qc = kc = 0
    for nq,nk in zip(qs,ns):
        for i in range(nq):
            if start <= qc+i < end:
                starts.append(kc); ends.append(kc+nk)
        qc += nq; kc += nk
    ks,ke = torch.tensor(starts),torch.tensor(ends)
    expected = torch.full((rows,4),-1,dtype=torch.int32)
    scores = (torch.einsum('mhd,nd->mhn',q,keys).mul(scales).relu()*weights[:,:,None]).sum(1)
    for i,(a,b) in enumerate(zip(starts,ends)):
        n=min(4,b-a)
        expected[i,:n]=scores[i,a:b].topk(n).indices.int()
    actual=torch.full_like(expected,-77)
    def mqa(q,kv,weights,ks,ke):
        k,s=kv
        return (torch.einsum('mhd,nd->mhn',q,k).mul(s).relu()*weights[:,:,None]).sum(1)
    def topk(logits,ks,ke,out,k):
        out.fill_(-1)
        for i in range(len(out)):
            a,b=int(ks[i]),int(ke[i]);n=min(k,b-a)
            out[i,:n]=logits[i,a:b].topk(n).indices.int()
    chunks=helper.plan_prefill_mqa_chunks(qs,ns,start,end,min_saved_cells=0)
    assert helper.run_grouped_prefill_mqa(q,keys,scales,weights,ks,ke,actual,chunks,mqa=mqa,topk=topk)
    torch.testing.assert_close(actual.sort().values,expected.sort().values)
