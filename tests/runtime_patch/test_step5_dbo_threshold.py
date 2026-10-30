# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for Step5's DP-unanimous DBO eligibility decision.

Execute the real gate and the Step5 execute_model prefix through AST so these
tests need neither vLLM imports nor a GPU runtime. The collective double uses
concurrent ranks and a barrier: a missing rank vote fails instead of hanging.
"""
from __future__ import annotations

import ast
import contextlib
import contextvars
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).parents[2]


def _load_gate():
    path = ROOT / "vllm_hcu/v1/hcu_dbo.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"dbo_batch_scope", "_has_prefill_in_both_ubatches",
             "_dsa_decode_threshold", "_patch_dp_threshold"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    namespace = {"np": np, "sys": sys, "DBO_MIN_TOKENS": 4096, "_MIN_TAIL": 64,
                 "contextlib": contextlib,
                 "_DBO_BATCH": contextvars.ContextVar("test_dbo_batch", default=None)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _step5_dispatch_prefix(dispatch, scope):
    """Keep the actual metadata gathering and dispatch call before input slicing."""
    path = ROOT / "vllm_hcu/v1/step5_execution.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "execute_model")
    stop = next(i for i, statement in enumerate(function.body)
                if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                       and n.func.id == "dispatch_cg_and_sync_dp"
                       for n in ast.walk(statement)))
    function.body = function.body[:stop + 1] + [ast.Return(ast.Name("batch_desc", ast.Load()))]
    function.decorator_list = []
    module = ast.Module(body=[
        ast.ImportFrom("__future__", [ast.alias("annotations")], 0), function
    ], type_ignores=[])
    namespace = {"dispatch_cg_and_sync_dp": dispatch,
                 "dbo_batch_scope": scope,
                 "needs_context_eager": lambda *args: False}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["execute_model"]


class _Votes:
    def __init__(self, size):
        self.barrier = threading.Barrier(size)
        self.lock = threading.Lock()
        self.votes = []
        self.group = object()

    def all_reduce(self, tensor, *, op, group):
        assert op == "MIN" and group is self.group
        with self.lock:
            self.votes.append(int(tensor.item()))
        self.barrier.wait(timeout=5)
        tensor[:] = min(self.votes)


def _install_gate(monkeypatch, votes):
    namespace = _load_gate()
    parent = None
    for name in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu",
                 "vllm.v1.worker.gpu.dp_utils"):
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        if parent is not None:
            setattr(parent, name.rsplit(".", 1)[1], module)
        parent = module
    dpu = parent
    baseline_calls = []

    def baseline(*args):
        baseline_calls.append(args)
        return "upstream"

    def tensor(value, *, dtype, device):
        assert dtype is np.int32 and device == "cpu"
        return np.array(value, dtype=dtype)

    dpu.check_ubatch_thresholds = baseline
    dpu.dist = SimpleNamespace(all_reduce=votes.all_reduce,
                               ReduceOp=SimpleNamespace(MIN="MIN"))
    dpu.get_dp_group = lambda: SimpleNamespace(cpu_group=votes.group)
    namespace["torch"] = SimpleNamespace(tensor=tensor, int32=np.int32)
    namespace["_patch_dp_threshold"]()
    return namespace, dpu, baseline_calls


def _dsa_builders(monkeypatch):
    module = ModuleType("vllm_hcu.model_executor.layers.step4_dsa")

    class Builder:
        def __init__(self, threshold):
            self.reorder_batch_threshold = threshold

    module.Step4DSAMetadataBuilder = Builder
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return Builder


def _run_ranks(monkeypatch, layouts, *, thresholds=None, totals=None, uniform_counts=None,
               runner_options=None, batch_options=None):
    votes = _Votes(len(layouts))
    namespace, dpu, _ = _install_gate(monkeypatch, votes)
    builder_cls = _dsa_builders(monkeypatch)
    if totals is None:
        totals = [sum(rows) if rows is not None else 4096 for rows in layouts]
    across = np.array(totals, dtype=np.int32)

    def sync_cudagraph_and_dp_padding(config, num_tokens, uniform_decode):
        num_tokens_across_dp = across
        return dpu.check_ubatch_thresholds(
            config, int(num_tokens_across_dp.min()), uniform_decode
        )

    def dispatch_cg_and_sync_dp(manager, num_reqs, num_tokens, uniform_count,
                                dp_size, dp_rank, **kwargs):
        assert kwargs["allow_ubatching"]
        return sync_cudagraph_and_dp_padding(
            kwargs["parallel_config"], num_tokens, kwargs["uniform_decode"]
        ), None

    execute = _step5_dispatch_prefix(dispatch_cg_and_sync_dp, namespace["dbo_batch_scope"])

    def run(rank):
        rows = layouts[rank]
        batch = None if rows is None else SimpleNamespace(
            num_scheduled_tokens=np.array(rows, dtype=np.int32), num_tokens=totals[rank],
            is_prefilling_np=np.ones(len(rows), dtype=bool),
        )
        if batch_options is not None:
            vars(batch).update(batch_options[rank])
        runner = SimpleNamespace(
            gather_batch_req_state=lambda *args: (
                batch, None if uniform_counts is None else uniform_counts[rank]
            ),
            pcp_manager=None, lora_config=None, is_encoder_decoder=False,
            cudagraph_manager=None, parallel_config=SimpleNamespace(_hcu_step5_dbo=True),
            dp_size=len(layouts), dp_rank=rank, ubatch_runner=object(),
            decode_query_len=4,
            attn_groups=[[SimpleNamespace(metadata_builders=[
                builder_cls(4 if thresholds is None else thresholds[rank])
            ])]],
        )
        if runner_options is not None:
            vars(runner).update(runner_options[rank])
        # dummy_run skips request updates; gather returns the actual local
        # layout so the real Step5 prefix exercises the pre-dispatch contract.
        scheduler = SimpleNamespace(num_scheduled_tokens={"r": totals[rank]},
                                    total_num_scheduled_tokens=totals[rank])
        return execute(runner, scheduler, dummy_run=True)

    with ThreadPoolExecutor(max_workers=len(layouts)) as pool:
        results = list(pool.map(run, range(len(layouts))))
    return results, sorted(votes.votes)


@pytest.mark.parametrize("rows,threshold,expected", [
    ([4] * 512 + [2048], 4, False),  # reported MTP3: ubatch 0 is decode-only
    ([2048] + [4] * 512, 4, False),  # ubatch 1 is decode-only
    ([4] * 511 + [8] + [4] * 511, 4, False),  # straddling prefill becomes 4 + 4
    ([4096], 4, True),
    ([2048, 2048], 4, True),
    ([4] * 256 + [3072], 4, True),  # safe mixed batch retains DBO
    ([4] * 1024, 1, True),  # use the actual configured decode threshold
    ([4] * 1024, 4, False),
])
def test_actual_sliced_queries_require_prefill_in_both_halves(rows, threshold, expected):
    eligible = _load_gate()["_has_prefill_in_both_ubatches"]
    assert eligible(rows, 4096, 2048, threshold) is expected


@pytest.mark.parametrize("rows,threshold,total", [
    (None, 4, 4096), ([], 4, 4096), ([4096], None, 4096),
    ([4096], 4, 8192), ([0, 4096], 4, 4096),
])
def test_missing_or_different_layout_falls_back(rows, threshold, total):
    assert not _load_gate()["_has_prefill_in_both_ubatches"](rows, total, 2048, threshold)


@pytest.mark.parametrize("layouts,expected,votes", [
    ([[4096], [4] * 512 + [2048]], False, [0, 1]),
    ([[4096], [2048] + [4] * 512], False, [0, 1]),
    ([[4096], [4] * 256 + [3072]], True, [1, 1]),
    ([[2048, 2048], [2048, 2048]], True, [1, 1]),
    ([[4096], None], False, [0, 1]),
    ([[4096], [4160], [6144]], True, [1, 1, 1]),
    ([[4096], [8192]], False, []),  # all ranks fail the shared tail guard
])
def test_dp_ranks_agree_before_creating_ubatches(monkeypatch, layouts, expected, votes):
    results, actual_votes = _run_ranks(monkeypatch, layouts)
    assert results == [expected] * len(layouts)
    assert actual_votes == votes


def test_smallest_rank_uses_global_padded_split_point(monkeypatch):
    # A local 2048 split would be safe. At the actual shared 3072 split,
    # rank 0's second half contains only four-token decode rows.
    rows = [3072] + [4] * 256
    results, votes = _run_ranks(monkeypatch, [rows, [6144]])
    assert results == [False, False]
    assert votes == [0, 1]


def test_other_configs_keep_upstream_threshold_and_patch_is_idempotent(monkeypatch):
    namespace, dpu, calls = _install_gate(monkeypatch, _Votes(1))
    wrapper = dpu.check_ubatch_thresholds
    namespace["_patch_dp_threshold"]()
    assert dpu.check_ubatch_thresholds is wrapper
    config = SimpleNamespace()
    assert wrapper(config, 4096, False) == "upstream"
    assert calls == [(config, 4096, False)]


def test_unknown_caller_without_dp_agreement_falls_back(monkeypatch):
    _, dpu, _ = _install_gate(monkeypatch, _Votes(1))
    assert not dpu.check_ubatch_thresholds(SimpleNamespace(_hcu_step5_dbo=True), 4096, False)


def test_initialized_dsa_thresholds_control_eligibility(monkeypatch):
    # Parallel drafting can classify length-7 queries as decode even when the
    # runner's ordinary MTP3 decode width is 4.
    results, votes = _run_ranks(
        monkeypatch, [[4096], [7] * 300 + [1996]], thresholds=[4, 7]
    )
    assert results == [False, False]
    assert votes == [0, 1]


def test_dsa_threshold_uses_all_initialized_builders(monkeypatch):
    builder_cls = _dsa_builders(monkeypatch)
    runner = SimpleNamespace(attn_groups=[[SimpleNamespace(
        metadata_builders=[builder_cls(4), builder_cls(7), object()]
    )]])
    threshold = _load_gate()["_dsa_decode_threshold"]
    assert threshold(runner) == 7
    runner.attn_groups[0][0].metadata_builders.append(builder_cls(None))
    assert threshold(runner) is None
    assert threshold(SimpleNamespace()) is None


def test_missing_builder_threshold_vetoes_every_rank(monkeypatch):
    results, votes = _run_ranks(monkeypatch, [[4096], [4096]], thresholds=[4, None])
    assert results == [False, False]
    assert votes == [0, 1]


def test_transformed_token_layout_vetoes_every_rank(monkeypatch):
    results, votes = _run_ranks(monkeypatch, [[4096], [4096]], totals=[4096, 6144])
    assert results == [False, False]
    assert votes == [0, 1]


def test_local_uniform_decode_flag_participates_in_unanimous_vote(monkeypatch):
    results, votes = _run_ranks(
        monkeypatch, [[4096], [4096]], uniform_counts=[4, None]
    )
    assert results == [False, False]
    assert votes == [0, 1]


@pytest.mark.parametrize("option", ["adaptive_verification", "pcp_manager"])
def test_post_dispatch_boundary_rewrites_veto_every_rank(monkeypatch, option):
    options = [{}, {option: object()}]
    if option == "pcp_manager":
        # Even unchanged totals do not prove PCP preserves each rank's queries.
        options[1][option] = SimpleNamespace(get_num_tokens_for_dispatch=lambda *args: 4096)
    results, votes = _run_ranks(monkeypatch, [[4096], [4096]], runner_options=options)
    assert results == [False, False]
    assert votes == [0, 1]


@pytest.mark.parametrize("rows", [
    np.array(["4096"], dtype=object), [4096.0], [[4096]], [[1], [2, 3]],
])
def test_malformed_query_metadata_is_a_veto(rows):
    assert not _load_gate()["_has_prefill_in_both_ubatches"](rows, 4096, 2048, 4)


@pytest.mark.parametrize("rows", [
    np.array(["4096"], dtype=object), [4096.0], [[4096]], [[1], [2, 3]],
])
def test_malformed_query_metadata_cannot_strand_another_rank(monkeypatch, rows):
    results, votes = _run_ranks(
        monkeypatch, [[4096], [4096]],
        batch_options=[{}, {"num_scheduled_tokens": rows}],
    )
    assert results == [False, False]
    assert votes == [0, 1]


@pytest.mark.parametrize("threshold", ["4", object(), -1, True, 4.0])
def test_malformed_builder_threshold_is_a_unanimous_veto(monkeypatch, threshold):
    results, votes = _run_ranks(monkeypatch, [[4096], [4096]], thresholds=[4, threshold])
    assert results == [False, False]
    assert votes == [0, 1]


def test_unavailable_builder_collection_is_a_unanimous_veto(monkeypatch):
    results, votes = _run_ranks(
        monkeypatch, [[4096], [4096]],
        runner_options=[{}, {"attn_groups": [[SimpleNamespace(metadata_builders=None)]]}],
    )
    assert results == [False, False]
    assert votes == [0, 1]


def test_dispatch_scope_restores_nested_context_and_exceptions():
    namespace = _load_gate()
    current = namespace["_DBO_BATCH"]
    scope = namespace["dbo_batch_scope"]
    first, second = object(), object()
    assert current.get() is None
    with scope(first, "first-batch"):
        assert current.get() == (first, "first-batch")
        with pytest.raises(RuntimeError), scope(second, "second-batch"):
            assert current.get() == (second, "second-batch")
            raise RuntimeError("dispatch failed")
        assert current.get() == (first, "first-batch")
    assert current.get() is None
