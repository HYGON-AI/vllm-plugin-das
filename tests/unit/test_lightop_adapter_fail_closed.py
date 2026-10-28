"""Regression tests for ABI fail-closed behaviour (review P2).

Every import/native-ABI precondition miss must return ``False`` so the caller
keeps its exact Torch fallback. Only a native call that already started may
propagate a real runtime error.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


REPO = Path(__file__).resolve().parents[2]
ADAPTER_PATH = REPO / "vllm_hcu/v1/attention/ops/lightop_indexer_topk.py"
SPARSE_PATH = REPO / "vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py"


def _load(names):
    nodes = [
        node
        for node in ast.parse(ADAPTER_PATH.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"torch": torch, "Callable": object}
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), str(ADAPTER_PATH), "exec"
        ),
        namespace,
    )
    return namespace


def test_load_returns_none_when_native_interface_missing(monkeypatch):
    """Stale Python wrapper with a missing native export must fail closed."""
    namespace = _load({"_load_topk_transform_ragged_v2"})
    pkg = ModuleType("lightop")
    pkg.topk_transform_ragged_v2 = lambda *a, **k: None
    pkg.op = ModuleType("lightop.op")  # no topk_transform_ragged_v2_interface
    monkeypatch.setitem(sys.modules, "lightop", pkg)
    monkeypatch.setitem(sys.modules, "lightop.op", pkg.op)

    assert namespace["_load_topk_transform_ragged_v2"]() is None


def test_load_returns_callable_when_both_layers_present(monkeypatch):
    namespace = _load({"_load_topk_transform_ragged_v2"})
    pkg = ModuleType("lightop")
    sentinel = object()
    pkg.topk_transform_ragged_v2 = lambda *a, **k: sentinel
    pkg.op = ModuleType("lightop.op")
    pkg.op.topk_transform_ragged_v2_interface = lambda *a, **k: sentinel
    monkeypatch.setitem(sys.modules, "lightop", pkg)
    monkeypatch.setitem(sys.modules, "lightop.op", pkg.op)

    assert namespace["_load_topk_transform_ragged_v2"]() is not None


def test_load_swallows_oserror_from_dynamic_link_failure(monkeypatch):
    """A broken shared object raises OSError on import; that must not escape."""
    namespace = _load({"_load_topk_transform_ragged_v2"})

    class Boom(ModuleType):
        def __getattr__(self, item):
            raise OSError("cannot open shared object file")

    monkeypatch.setitem(sys.modules, "lightop", Boom("lightop"))
    assert namespace["_load_topk_transform_ragged_v2"]() is None


def test_decode_topk_missing_symbol_returns_false():
    namespace = _load({"decode_topk"})
    logits = torch.zeros((2, 16), dtype=torch.float32)
    out = torch.full((2, 8), 5, dtype=torch.int32)
    assert (
        namespace["decode_topk"](
            None, logits, torch.tensor([16, 8], dtype=torch.int32), out, 8
        )
        is False
    )
    assert torch.equal(out, torch.full_like(out, 5)), "output must stay untouched"


def test_decode_topk_bad_shape_returns_false_without_touching_output():
    namespace = _load({"decode_topk"})
    calls = []
    logits = torch.zeros((3, 16), dtype=torch.float32)
    out = torch.full((2, 8), 5, dtype=torch.int32)
    assert (
        namespace["decode_topk"](
            lambda *a: calls.append(a),
            logits, torch.tensor([16, 8], dtype=torch.int32), out, 8,
        )
        is False
    )
    assert calls == []
    assert torch.equal(out, torch.full_like(out, 5))


def test_decode_topk_native_runtime_error_still_propagates():
    """A gate that passed and then failed in the kernel is a real error."""
    namespace = _load({"decode_topk"})
    logits = torch.zeros((1, 16), dtype=torch.float32)
    out = torch.empty((1, 8), dtype=torch.int32)

    def boom(*_a):
        raise RuntimeError("kernel died")

    with pytest.raises(RuntimeError, match="kernel died"):
        namespace["decode_topk"](
            boom, logits, torch.tensor([16], dtype=torch.int32), out, 8
        )


def test_decode_path_falls_back_when_lightop_resolution_fails():
    """The non-512 branch must not resolve LightOp before deciding fallback."""
    nodes = [
        node
        for node in ast.parse(SPARSE_PATH.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_lightop_topk_indices_decode"
    ]
    used = []
    namespace = {
        "torch": torch,
        "decode_topk": lambda *_a: False,
        "decode_topk_v2": lambda *_a: False,
        "_decode_row_ends_from_seq_lens": lambda seq, n, rows: torch.zeros(
            rows, dtype=torch.int32
        ),
        "_topk_indices_torch": lambda logits, k, row_starts=None, row_ends=None: (
            used.append(k) or torch.full((logits.shape[0], k), 7, dtype=torch.int32)
        ),
        "_get_lightop_attention": lambda: (_ for _ in ()).throw(
            ImportError("lightop missing")
        ),
        "henvs": type(
            "E",
            (),
            {
                "VLLM_HCU_USE_CUSTOM_OPS": True,
                "VLLM_HCU_USE_LIGHTOP_INDEXER_TOPK_ADAPTER": True,
            },
        )(),
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SPARSE_PATH), "exec"),
        namespace,
    )
    out = torch.full((1, 64), 3, dtype=torch.int32)
    # K != 512 exercises the legacy-adapter branch.
    namespace["_lightop_topk_indices_decode"](
        torch.zeros((1, 16), dtype=torch.float32),
        torch.tensor([16], dtype=torch.int32), 1, out, 64,
    )
    assert used == [64], "Torch fallback must run when LightOp cannot resolve"
    assert torch.equal(out, torch.full_like(out, 7))


def test_decode_topk_rejects_output_columns_that_do_not_match_k():
    namespace = _load({"decode_topk"})
    calls = []
    logits = torch.zeros((2, 16), dtype=torch.float32)
    output = torch.full((2, 7), 5, dtype=torch.int32)
    assert (
        namespace["decode_topk"](
            lambda *args: calls.append(args),
            logits,
            torch.tensor([16, 8], dtype=torch.int32),
            output,
            8,
        )
        is False
    )
    assert calls == []
    assert torch.equal(output, torch.full_like(output, 5))


def test_decode_topk_contiguous_metadata_preserves_strided_values():
    namespace = _load({"decode_topk"})
    observed = []
    logits = torch.zeros((2, 16), dtype=torch.float32)
    output = torch.empty((2, 8), dtype=torch.int32)
    metadata_storage = torch.tensor([16, 999, 8, 777], dtype=torch.int32)
    row_ends = metadata_storage[::2]
    assert not row_ends.is_contiguous()

    def native(_logits, _next_n, lengths, native_output, *_args):
        observed.append((lengths.clone(), lengths.is_contiguous()))
        native_output.fill_(3)

    assert namespace["decode_topk"](native, logits, row_ends, output, 8)
    assert len(observed) == 1
    assert torch.equal(observed[0][0], torch.tensor([16, 8], dtype=torch.int32))
    assert observed[0][1] is True


def test_flag_off_noncontiguous_output_does_not_resolve_lightop():
    nodes = [
        node
        for node in ast.parse(SPARSE_PATH.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_lightop_topk_indices_decode"
    ]
    namespace = {
        "torch": torch,
        "decode_topk": lambda *_args: pytest.fail("adapter must be disabled"),
        "decode_topk_v2": lambda *_args: False,
        "_decode_row_ends_from_seq_lens": lambda seq, n, rows: torch.full(
            (rows,), 16, dtype=torch.int32
        ),
        "_topk_indices_torch": lambda logits, k, row_starts=None, row_ends=None: (
            torch.full((logits.shape[0], k), 7, dtype=torch.int32)
        ),
        "_get_lightop_attention": lambda: pytest.fail(
            "flag-off strided fallback must not import LightOp"
        ),
        "henvs": type(
            "E",
            (),
            {
                "VLLM_HCU_USE_CUSTOM_OPS": True,
                "VLLM_HCU_USE_LIGHTOP_INDEXER_TOPK_ADAPTER": False,
            },
        )(),
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SPARSE_PATH), "exec"),
        namespace,
    )
    backing = torch.full((1, 16), 3, dtype=torch.int32)
    output = backing[:, ::2]
    namespace["_lightop_topk_indices_decode"](
        torch.zeros((1, 16), dtype=torch.float32),
        torch.tensor([16], dtype=torch.int32),
        1,
        output,
        8,
    )
    assert torch.equal(output, torch.full_like(output, 7))
    assert torch.equal(backing[:, 1::2], torch.full_like(backing[:, 1::2], 3))
