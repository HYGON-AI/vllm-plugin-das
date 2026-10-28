"""CPU tests for the LightOp v2 selector inside the HCU FP4 indexer path.

The FP4 reader returns early from ``rocm_aiter_sparse_attn_indexer_native``, so
the shared decode helper is never reached there.  These tests pin the two FP4
Top-K sites (prefill chunk and decode) that now try the fused ragged-v2
selector first and fall back to the exact Torch reference.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO = Path(__file__).resolve().parents[2]
SPARSE_PATH = REPO / "vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py"
ADAPTER_PATH = REPO / "vllm_hcu/v1/attention/ops/lightop_indexer_topk.py"


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _exec_nodes(nodes, namespace):
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SPARSE_PATH), "exec"),
        namespace,
    )
    return namespace


def _fp4_namespace(*, v2, torch_topk, adapter=True, custom_ops=True):
    """Minimal namespace holding just the two FP4 Top-K statements."""
    node = _function_node(SPARSE_PATH, "_hcu_fp4_sparse_attn_indexer_native")
    hits = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.If):
            continue
        src = ast.unparse(sub)
        # Only the direct `if not (... and decode_topk_v2(...))` guards; the
        # enclosing prefill/decode blocks also contain the call but are not
        # the guard under test.
        if src.startswith("if not (") and "decode_topk_v2" in src:
            hits.append(sub)
    assert len(hits) == 2, f"expected 2 guarded Top-K sites, found {len(hits)}"
    return {
        "torch": torch,
        "decode_topk_v2": v2,
        "_topk_indices_torch": torch_topk,
        "henvs": SimpleNamespace(
            VLLM_HCU_USE_CUSTOM_OPS=custom_ops,
            VLLM_HCU_USE_LIGHTOP_INDEXER_TOPK_ADAPTER=adapter,
        ),
        "_sites": hits,
    }


def _run_site(ns, site_index, *, logits, row_ends, out, topk_tokens):
    """Execute one guarded site with the FP4-local variables bound."""
    site = ns["_sites"][site_index]
    module = ast.Module(body=[site], type_ignores=[])
    local = dict(ns)
    local.update(
        logits=logits, row_ends=row_ends, row_lens=row_ends,
        out=out, topk_tokens=topk_tokens,
    )
    exec(compile(module, str(SPARSE_PATH), "exec"), local)
    return out


@pytest.mark.parametrize("site_index", [0, 1])
def test_fp4_topk_site_prefers_v2_when_abi_matches(site_index):
    logits = torch.zeros((3, 64), dtype=torch.float32)
    row_ends = torch.tensor([64, 32, 8], dtype=torch.int32)
    out = torch.full((3, 512), 91, dtype=torch.int32)
    seen = []

    def v2(logits_arg, ends_arg, out_arg, k):
        seen.append((logits_arg.shape, ends_arg.tolist(), out_arg.shape, k))
        out_arg.fill_(7)
        return True

    ns = _fp4_namespace(
        v2=v2,
        torch_topk=lambda *_a, **_kw: pytest.fail("torch fallback ran"),
    )
    _run_site(
        ns, site_index,
        logits=logits, row_ends=row_ends, out=out, topk_tokens=512,
    )
    assert len(seen) == 1
    assert seen[0][0] == (3, 64)
    assert seen[0][1] == [64, 32, 8]
    assert seen[0][2] == (3, 512)
    assert seen[0][3] == 512
    assert torch.equal(out, torch.full_like(out, 7))


@pytest.mark.parametrize("site_index", [0, 1])
def test_fp4_topk_site_falls_back_when_v2_rejects(site_index):
    logits = torch.zeros((2, 32), dtype=torch.float32)
    row_ends = torch.tensor([32, 16], dtype=torch.int32)
    out = torch.full((2, 512), -5, dtype=torch.int32)
    calls = []

    def torch_topk(logits_arg, k, row_starts=None, row_ends=None):
        calls.append((logits_arg.shape, k, row_ends.tolist()))
        return torch.full((logits_arg.shape[0], k), 3, dtype=torch.int32)

    ns = _fp4_namespace(
        v2=lambda *_a, **_kw: False,
        torch_topk=torch_topk,
    )
    _run_site(
        ns, site_index,
        logits=logits, row_ends=row_ends, out=out, topk_tokens=512,
    )
    assert len(calls) == 1
    assert calls[0][0] == (2, 32)
    assert calls[0][1] == 512
    assert calls[0][2] == [32, 16]
    assert torch.equal(out, torch.full_like(out, 3))


@pytest.mark.parametrize("site_index", [0, 1])
def test_fp4_topk_site_skips_v2_when_adapter_flag_off(site_index):
    logits = torch.zeros((2, 16), dtype=torch.float32)
    row_ends = torch.tensor([16, 8], dtype=torch.int32)
    out = torch.full((2, 512), 0, dtype=torch.int32)

    ns = _fp4_namespace(
        v2=lambda *_a, **_kw: pytest.fail("v2 must not run when flag is off"),
        torch_topk=lambda logits_arg, k, **kw: torch.full(
            (logits_arg.shape[0], k), 11, dtype=torch.int32
        ),
        adapter=False,
    )
    _run_site(
        ns, site_index,
        logits=logits, row_ends=row_ends, out=out, topk_tokens=512,
    )
    assert torch.equal(out, torch.full_like(out, 11))


@pytest.mark.parametrize("site_index", [0, 1])
def test_fp4_topk_site_skips_v2_when_custom_ops_off(site_index):
    logits = torch.zeros((2, 16), dtype=torch.float32)
    row_ends = torch.tensor([16, 8], dtype=torch.int32)
    out = torch.full((2, 512), 0, dtype=torch.int32)

    ns = _fp4_namespace(
        v2=lambda *_a, **_kw: pytest.fail("v2 must not run without custom ops"),
        torch_topk=lambda logits_arg, k, **kw: torch.full(
            (logits_arg.shape[0], k), 13, dtype=torch.int32
        ),
        custom_ops=False,
    )
    _run_site(
        ns, site_index,
        logits=logits, row_ends=row_ends, out=out, topk_tokens=512,
    )
    assert torch.equal(out, torch.full_like(out, 13))


@pytest.mark.parametrize("site_index", [0, 1])
def test_fp4_topk_site_short_circuits_before_v2_when_k_not_512(site_index):
    """decode_topk_v2 itself rejects k!=512; the flag gates must still hold."""
    logits = torch.zeros((2, 16), dtype=torch.float32)
    row_ends = torch.tensor([16, 8], dtype=torch.int32)
    out = torch.full((2, 1024), 0, dtype=torch.int32)
    seen = []

    def v2(logits_arg, ends_arg, out_arg, k):
        seen.append(k)
        return False

    ns = _fp4_namespace(
        v2=v2,
        torch_topk=lambda logits_arg, k, **kw: torch.full(
            (logits_arg.shape[0], k), 17, dtype=torch.int32
        ),
    )
    _run_site(
        ns, site_index,
        logits=logits, row_ends=row_ends, out=out, topk_tokens=1024,
    )
    assert seen == [1024]
    assert torch.equal(out, torch.full_like(out, 17))


def _load_real_v2():
    nodes = [
        node
        for node in ast.parse(ADAPTER_PATH.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"decode_topk_v2", "_load_topk_transform_ragged_v2"}
    ]
    namespace = {"torch": torch}
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), str(ADAPTER_PATH), "exec"
        ),
        namespace,
    )
    return namespace["decode_topk_v2"]


def test_real_v2_rejects_strided_output_row_mismatch():
    """A padded buffer whose row count differs must fall back, not corrupt."""
    v2 = _load_real_v2()
    logits = torch.zeros((2, 32), dtype=torch.float32)
    row_ends = torch.tensor([32, 16], dtype=torch.int32)
    out = torch.full((5, 512), 41, dtype=torch.int32)
    assert v2(logits, row_ends, out, 512) is False
    assert torch.equal(out, torch.full_like(out, 41))


def test_real_v2_rejects_cpu_logits():
    v2 = _load_real_v2()
    logits = torch.zeros((2, 32), dtype=torch.float32)
    out = torch.full((2, 512), 41, dtype=torch.int32)
    assert v2(logits, torch.tensor([32, 16], dtype=torch.int32), out, 512) is False
    assert torch.equal(out, torch.full_like(out, 41))


def test_real_v2_accepts_strided_column_view_only_when_contiguous_columns():
    """stride(1)!=1 is rejected by the ABI gate before any native call."""
    v2 = _load_real_v2()
    base = torch.zeros((64, 2), dtype=torch.float32)
    logits = base.t()  # shape (2,64), stride (1,2) -> stride(1) != 1
    assert logits.stride(1) == 2
    out = torch.full((2, 512), 41, dtype=torch.int32)
    assert v2(logits, torch.tensor([64, 32], dtype=torch.int32), out, 512) is False


def test_fp4_sites_are_both_guarded_by_the_adapter_flag():
    """Guard the production shape of both edits, not just their behaviour."""
    node = _function_node(SPARSE_PATH, "_hcu_fp4_sparse_attn_indexer_native")
    guarded = 0
    for sub in ast.walk(node):
        if not isinstance(sub, ast.If):
            continue
        src = ast.unparse(sub)
        if src.startswith("if not (") and "decode_topk_v2" in src:
            guarded += 1
    assert guarded == 2, f"expected 2 guarded Top-K sites, found {guarded}"
