# SPDX-License-Identifier: Apache-2.0
"""Exercise the eager gate scheduling seam without an accelerator or vLLM."""

import ast
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load_eager_method(namespace):
    tree = ast.parse((ROOT / "vllm_hcu/models/hy_v4/attention.py").read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "HYV4MLAAttention")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_indexer_and_attn")
    method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "attention.py", "exec"), namespace)
    return namespace["_indexer_and_attn"]


@pytest.mark.parametrize("gather_at", ["indexer", "mla", None])
def test_gate_runs_once_at_first_gather_or_after_attention(gather_at):
    calls = []
    active = []

    @contextmanager
    def scope(fn):
        active.append(fn)
        try:
            yield
        finally:
            active.pop()

    score = SimpleNamespace(record_stream=lambda stream: calls.append("record"))

    def gate(hidden):
        calls.append("gate")
        return (score,)

    def gather():
        calls.append("gather")
        return "gathered"

    def parallel(fn0, fn1, start, done, stream):
        calls.append("start")
        result0, result1 = fn0(), fn1()
        calls.append("join")
        return result0, result1

    def indexer(*args):
        calls.append("indexer_projection")
        if gather_at == "indexer":
            assert active[-1](gather) == "gathered"
        calls.append("indexer_compute")

    def mla(*args, **kwargs):
        if gather_at is not None:
            assert active[-1](gather) == "gathered"
        calls.append("attention_compute")
        return "attention"

    namespace = dict(cache_gather_overlap_scope=scope,
                     maybe_execute_in_parallel=parallel,
                     torch=SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: "main")))
    forward = _load_eager_method(namespace)
    model = SimpleNamespace(
        gate_proj_stream="aux", gate_proj_start_event="start",
        gate_proj_done_event="done", gate_a2a_stream=None,
        gated_mla=True, linear_gate=gate, indexer=indexer,
        is_sparse=True, skip_topk=False, indexer_rope_emb=None,
        mla_attn=mla, _apply_gate_in_place=lambda out, gate: calls.append("apply"),
    )
    out = SimpleNamespace(shape=(2, 4), copy_=lambda value: calls.append("copy"))
    forward(model, "hidden", None, None, None, None, None, out)
    assert calls.count("gate") == 1
    assert calls[-1] == "apply"
    assert not active
    if gather_at is None:
        assert calls.index("gate") > calls.index("attention_compute")
        assert "start" not in calls
    else:
        assert calls.index("gate") > calls.index("indexer_projection")
        assert calls.index("join") < calls.index("attention_compute")
        assert calls.count("start") == 1
        if gather_at == "indexer":
            assert calls.index("join") < calls.index("indexer_compute")


def test_pcp_gather_scope_wraps_live_collectives_and_restores_on_error():
    tree = ast.parse((ROOT / "vllm_hcu/model_executor/layers/attention/pcp.py").read_text())
    names = {"cache_gather_overlap_scope", "_gather_prefill_cache_inputs",
             "_pcp_world_size", "_rank_slot_slice", "_decode_only_slot_mapping"}
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    calls = []

    class Tensor:
        def __init__(self, size):
            self.shape = (size,)
            self.ndim = 1

        def __getitem__(self, item):
            start, stop, step = item.indices(self.shape[0])
            return Tensor(len(range(start, stop, step)))

        def contiguous(self):
            return self

    def gather(tensor, dim):
        calls.append("collective")
        return Tensor(tensor.shape[0] * 2)

    state = ContextVar("test_gather_overlap", default=None)
    namespace = dict(contextmanager=contextmanager, _CACHE_GATHER_OVERLAP=state,
                     get_pcp_group=lambda: SimpleNamespace(
                         world_size=2, rank_in_group=0, all_gather=gather))
    exec(compile(ast.fix_missing_locations(module), "pcp.py", "exec"), namespace)
    scope = namespace["cache_gather_overlap_scope"]
    gather_inputs = namespace["_gather_prefill_cache_inputs"]

    def overlap(fn):
        calls.append("overlap")
        return fn()

    with pytest.raises(RuntimeError, match="failed"):
        with scope(overlap):
            tensors, slots = gather_inputs(
                (Tensor(2),), Tensor(4),
                SimpleNamespace(pcp_world_size=2, num_decode_tokens=0))
            assert tensors[0].shape == slots.shape == (4,)
            assert calls == ["overlap", "collective", "collective"]
            calls.clear()
            gather_inputs((Tensor(2),), Tensor(4),
                          SimpleNamespace(pcp_world_size=2, num_decode_tokens=2))
            assert not calls  # Decode-only batches do not trigger the gate.
            raise RuntimeError("failed")
    assert state.get() is None
