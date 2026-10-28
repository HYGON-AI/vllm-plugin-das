from __future__ import annotations

import ast
import math
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


REPO = Path(__file__).resolve().parents[2]


def _load_helpers():
    sparse_path = REPO / "vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py"
    adapter_path = REPO / "vllm_hcu/v1/attention/ops/lightop_indexer_topk.py"
    sparse_nodes = [
        node
        for node in ast.parse(sparse_path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_topk_indices_torch",
            "_decode_row_ends_from_seq_lens",
            "_use_lightop_sparse_mla_topk",
        }
    ]
    adapter_nodes = [
        node
        for node in ast.parse(adapter_path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"decode_topk", "decode_topk_v2", "_load_topk_transform_ragged_v2"}
    ]
    namespace = {
        "torch": torch,
        "Callable": Callable,
        "SimpleNamespace": SimpleNamespace,
        "henvs": SimpleNamespace(
            VLLM_HCU_USE_CUSTOM_OPS=True,
            VLLM_HCU_USE_LIGHTOP_SPARSE_MLA_TOPK=True,
            VLLM_HCU_USE_LIGHTOP_INDEXER_TOPK_ADAPTER=True,
        ),
    }
    exec(compile(ast.Module(body=sparse_nodes, type_ignores=[]), str(sparse_path), "exec"), namespace)
    exec(compile(ast.Module(body=adapter_nodes, type_ignores=[]), str(adapter_path), "exec"), namespace)
    return namespace


def test_decode_adapter_calls_real_boundary_and_preserves_noncontiguous_output():
    ns = _load_helpers()
    logits = torch.tensor(
        [[0.0, 8.0, 2.0, 1.0], [7.0, 3.0, 9.0, 4.0]], dtype=torch.float32
    )
    backing = torch.full((2, 4), -77, dtype=torch.int32)
    output = backing[:, ::2]
    assert not output.is_contiguous()
    calls = []

    def native(logits_arg, next_n, lengths, native_output, rows, stride0, stride1, k):
        calls.append((logits_arg, next_n, lengths, native_output, rows, stride0, stride1, k))
        assert native_output.is_contiguous()
        native_output.copy_(torch.tensor([[1, 0], [2, 0]], dtype=torch.int32))

    row_ends = ns["_decode_row_ends_from_seq_lens"](
        torch.tensor([[2, 3]], dtype=torch.int32), 2, 2
    )
    ns["decode_topk"](native, logits, row_ends, output, 2)

    assert torch.equal(output, torch.tensor([[1, 0], [2, 0]], dtype=torch.int32))
    assert calls[0][1] == 1
    assert torch.equal(calls[0][2], torch.tensor([2, 3], dtype=torch.int32))
    assert (calls[0][5], calls[0][6]) == (4, 1)


def test_decode_v2_adapter_rejects_cpu_as_unsupported_domain(
    monkeypatch: pytest.MonkeyPatch,
):
    ns = _load_helpers()
    calls = []
    lightop = ModuleType("lightop")

    def v2(scores, lengths, *, out_offsets, out_indices, row_starts):
        calls.append((scores, lengths, out_offsets, out_indices, row_starts))
        out_indices.fill_(7)

    lightop.topk_transform_ragged_v2 = v2
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    logits = torch.zeros((2, 8), dtype=torch.float32)
    backing = torch.full((2, 1024), 55, dtype=torch.int32)
    output = backing[:, ::2]
    assert not ns["decode_topk_v2"](
        logits, torch.tensor([0, 8], dtype=torch.int32), output, 512
    )
    assert calls == []
    assert torch.equal(output, torch.full_like(output, 55))
    assert torch.equal(backing[:, 1::2], torch.full_like(backing[:, 1::2], 55))


def test_decode_v2_import_error_is_fail_closed(monkeypatch: pytest.MonkeyPatch):
    ns = _load_helpers()
    monkeypatch.setitem(sys.modules, "lightop", ModuleType("lightop"))
    assert ns["_load_topk_transform_ragged_v2"]() is None


@pytest.mark.parametrize("enabled", [False, True])
def test_production_k512_route_flag_off_keeps_torch_and_on_uses_v2(enabled: bool):
    sparse_path = REPO / "vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py"
    function = next(
        node for node in ast.parse(sparse_path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_lightop_topk_indices_decode"
    )
    calls = []
    def fake_v2(logits, ends, output, k):
        calls.append((logits, ends.clone(), output, k))
        output.fill_(13)
        return True
    helper_ns = _load_helpers()
    namespace = {
        "torch": torch,
        "decode_topk_v2": fake_v2,
        "decode_topk": lambda *_args: pytest.fail("legacy adapter called for K512"),
        "_decode_row_ends_from_seq_lens": helper_ns["_decode_row_ends_from_seq_lens"],
        "_topk_indices_torch": helper_ns["_topk_indices_torch"],
        "_get_lightop_attention": lambda: pytest.fail("legacy native called for K512"),
        "henvs": SimpleNamespace(
            VLLM_HCU_USE_CUSTOM_OPS=True,
            VLLM_HCU_USE_LIGHTOP_INDEXER_TOPK_ADAPTER=enabled,
        ),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(sparse_path), "exec"), namespace)
    output = torch.full((2, 512), 99, dtype=torch.int32)
    namespace["_lightop_topk_indices_decode"](
        torch.arange(16, dtype=torch.float32).reshape(2, 8),
        torch.tensor([8, 6], dtype=torch.int32), 1, output, 512
    )
    if enabled:
        assert len(calls) == 1
        assert torch.equal(calls[0][1], torch.tensor([8, 6], dtype=torch.int32))
        assert calls[0][0].dtype is torch.float32
        assert calls[0][2].dtype is torch.int32
        assert calls[0][2].shape == (2, 512)
        assert calls[0][3] == 512
        assert torch.equal(output, torch.full_like(output, 13))
    else:
        assert calls == []
        assert torch.equal(output[0, :8], torch.arange(8, dtype=torch.int32))
        assert torch.equal(output[1, :6], torch.arange(6, dtype=torch.int32))
        assert torch.all(output[:, 8:] == -1)


def test_production_k512_v2_import_failure_falls_back_to_torch(monkeypatch):
    sparse_path = REPO / "vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py"
    function = next(
        node for node in ast.parse(sparse_path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_lightop_topk_indices_decode"
    )
    helper_ns = _load_helpers()
    monkeypatch.setitem(sys.modules, "lightop", ModuleType("lightop"))
    namespace = {
        "torch": torch,
        "decode_topk_v2": helper_ns["decode_topk_v2"],
        "decode_topk": lambda *_args: pytest.fail("legacy adapter called"),
        "_decode_row_ends_from_seq_lens": helper_ns["_decode_row_ends_from_seq_lens"],
        "_topk_indices_torch": helper_ns["_topk_indices_torch"],
        "_get_lightop_attention": lambda: pytest.fail("legacy native called"),
        "henvs": SimpleNamespace(
            VLLM_HCU_USE_CUSTOM_OPS=True,
            VLLM_HCU_USE_LIGHTOP_INDEXER_TOPK_ADAPTER=True,
        ),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(sparse_path), "exec"), namespace)
    output = torch.full((1, 512), 99, dtype=torch.int32)
    namespace["_lightop_topk_indices_decode"](
        torch.arange(8, dtype=torch.float32).reshape(1, 8),
        torch.tensor([8], dtype=torch.int32), 1, output, 512
    )
    assert torch.equal(output[0, :8], torch.arange(8, dtype=torch.int32))
    assert torch.all(output[0, 8:] == -1)


def test_production_k512_v2_runtime_error_propagates():
    sparse_path = REPO / "vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py"
    function = next(
        node for node in ast.parse(sparse_path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_lightop_topk_indices_decode"
    )
    namespace = {
        "torch": torch,
        "decode_topk_v2": lambda *_args: (_ for _ in ()).throw(RuntimeError("v2 failed")),
        "decode_topk": lambda *_args: pytest.fail("legacy adapter called"),
        "_decode_row_ends_from_seq_lens": _load_helpers()["_decode_row_ends_from_seq_lens"],
        "_topk_indices_torch": lambda *_args: pytest.fail("torch fallback swallowed v2 error"),
        "_get_lightop_attention": lambda: pytest.fail("legacy native called"),
        "henvs": SimpleNamespace(
            VLLM_HCU_USE_CUSTOM_OPS=True,
            VLLM_HCU_USE_LIGHTOP_INDEXER_TOPK_ADAPTER=True,
        ),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(sparse_path), "exec"), namespace)
    with pytest.raises(RuntimeError, match="v2 failed"):
        namespace["_lightop_topk_indices_decode"](
            torch.zeros((1, 8), dtype=torch.float32),
            torch.tensor([8], dtype=torch.int32), 1,
            torch.empty((1, 512), dtype=torch.int32), 512,
        )


def test_production_decode_helper_reaches_adapter_and_preserves_prefill_rows():
    sparse_path = REPO / "vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py"
    adapter = _load_helpers()["decode_topk"]
    function = next(
        node
        for node in ast.parse(sparse_path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_lightop_topk_indices_decode"
    )
    calls = []

    def native(logits, next_n, lengths, output, rows, stride0, stride1, k):
        calls.append((next_n, lengths.clone(), output.is_contiguous()))
        output.fill_(3)

    namespace = {
        "torch": torch,
        "decode_topk": adapter,
        "_decode_row_ends_from_seq_lens": _load_helpers()[
            "_decode_row_ends_from_seq_lens"
        ],
        "_get_lightop_attention": lambda: SimpleNamespace(
            top_k_per_row_decode=native
        ),
        "_topk_indices_torch": lambda *_args, **_kwargs: pytest.fail(
            "K=512 fallback unexpectedly reached"
        ),
        "henvs": SimpleNamespace(VLLM_HCU_USE_LIGHTOP_INDEXER_TOPK_ADAPTER=True),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(sparse_path), "exec"),
        namespace,
    )
    shared = torch.full((3, 2), 91, dtype=torch.int32)
    namespace["_lightop_topk_indices_decode"](
        torch.zeros((1, 4), dtype=torch.float32),
        torch.tensor([[4]], dtype=torch.int32),
        1,
        shared[:1],
        2,
    )
    assert torch.equal(shared[0], torch.tensor([3, 3], dtype=torch.int32))
    assert torch.equal(shared[1:], torch.full((2, 2), 91, dtype=torch.int32))
    assert len(calls) == 1
    assert calls[0][0] == 1
    assert torch.equal(calls[0][1], torch.tensor([4], dtype=torch.int32))
    assert calls[0][2] is True


def test_production_gate_honors_master_and_leaf_off():
    ns = _load_helpers()
    assert ns["_use_lightop_sparse_mla_topk"]()
    ns["henvs"].VLLM_HCU_USE_LIGHTOP_SPARSE_MLA_TOPK = False
    assert not ns["_use_lightop_sparse_mla_topk"]()
    ns["henvs"].VLLM_HCU_USE_LIGHTOP_SPARSE_MLA_TOPK = True
    ns["henvs"].VLLM_HCU_USE_CUSTOM_OPS = False
    assert not ns["_use_lightop_sparse_mla_topk"]()


def test_production_decode_helper_keeps_k512_torch_fallback():
    sparse_path = REPO / "vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py"
    function = next(
        node
        for node in ast.parse(sparse_path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_lightop_topk_indices_decode"
    )
    ns = _load_helpers()
    ns["_get_lightop_attention"] = lambda: pytest.fail("K=512 called native")
    ns["henvs"].VLLM_HCU_USE_CUSTOM_OPS = False
    ns["henvs"].VLLM_HCU_USE_LIGHTOP_INDEXER_TOPK_ADAPTER = False
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(sparse_path), "exec"),
        ns,
    )
    output = torch.empty((1, 512), dtype=torch.int32)
    ns["_lightop_topk_indices_decode"](
        torch.tensor([[2.0, 1.0]], dtype=torch.float32),
        torch.tensor([2], dtype=torch.int32),
        1,
        output,
        512,
    )
    assert torch.equal(output[0, :2], torch.tensor([0, 1], dtype=torch.int32))
    assert torch.all(output[0, 2:] == -1)


def test_production_decode_helper_default_off_uses_torch_for_strided_output():
    sparse_path = REPO / "vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py"
    function = next(
        node
        for node in ast.parse(sparse_path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_lightop_topk_indices_decode"
    )
    ns = _load_helpers()
    ns["henvs"].VLLM_HCU_USE_LIGHTOP_INDEXER_TOPK_ADAPTER = False
    ns["_get_lightop_attention"] = lambda: SimpleNamespace(
        top_k_per_row_decode=lambda *_args: pytest.fail("default-off adapter called")
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(sparse_path), "exec"),
        ns,
    )
    backing = torch.full((1, 4), 77, dtype=torch.int32)
    ns["_lightop_topk_indices_decode"](
        torch.tensor([[1.0, 4.0, 2.0]], dtype=torch.float32),
        torch.tensor([3], dtype=torch.int32),
        1,
        backing[:, ::2],
        2,
    )
    assert torch.equal(backing[0, ::2], torch.tensor([1, 2], dtype=torch.int32))
    assert backing[0, 1] == 77 and backing[0, 3] == 77


@pytest.mark.parametrize("rows", [0, 1, 5])
@pytest.mark.parametrize("next_n", [1, 2, 4])
def test_decode_row_end_expansion_handles_empty_short_and_long_rows(rows, next_n):
    ns = _load_helpers()
    seq_lens = torch.arange(rows, dtype=torch.int32) + 1
    num_rows = rows * next_n
    ends = ns["_decode_row_ends_from_seq_lens"](seq_lens, next_n, num_rows)
    assert ends.shape == (num_rows,)
    if rows:
        if next_n == 1:
            expected = seq_lens[:next_n]
        else:
            expected = (
                seq_lens[0]
                - next_n
                + torch.arange(next_n, dtype=torch.int32)
                + 1
            ).clamp_min(0)
        assert torch.equal(ends[:next_n], expected)


def test_decode_adapter_rejects_missing_symbol_without_fallback():
    ns = _load_helpers()
    logits = torch.zeros((1, 3), dtype=torch.float32)
    output = torch.empty((1, 2), dtype=torch.int32)
    with pytest.raises(RuntimeError, match="unavailable"):
        ns["decode_topk"](None, logits, torch.tensor([3]), output, 2)


def test_decode_adapter_rejects_unsupported_dtypes():
    ns = _load_helpers()
    output = torch.empty((1, 2), dtype=torch.int64)
    with pytest.raises(RuntimeError, match="int32"):
        ns["decode_topk"](
            lambda *_args: None,
            torch.zeros((1, 3), dtype=torch.float32),
            torch.tensor([3]),
            output,
            2,
        )


def test_decode_adapter_covers_empty_short_and_long_rows():
    ns = _load_helpers()
    logits = torch.zeros((3, 5), dtype=torch.float32)
    output = torch.empty((3, 3), dtype=torch.int32)

    def native(_logits, _next_n, lengths, native_output, *_args):
        expected = torch.full_like(native_output, -1)
        for row, length in enumerate(lengths.tolist()):
            expected[row, : min(length, native_output.shape[1])] = torch.arange(
                min(length, native_output.shape[1]), dtype=torch.int32
            )
        native_output.copy_(expected)

    ns["decode_topk"](
        native,
        logits,
        torch.tensor([0, 1, 5], dtype=torch.int32),
        output,
        3,
    )
    assert torch.equal(
        output,
        torch.tensor([[-1, -1, -1], [0, -1, -1], [0, 1, 2]], dtype=torch.int32),
    )


def _load_torch_decode_entry():
    sparse_path = REPO / "vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py"
    nodes = [
        node
        for node in ast.parse(sparse_path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {"_topk_indices_torch", "_decode_row_ends_from_seq_lens", "_lightop_topk_indices_decode"}
    ]
    namespace = {
        "torch": torch,
        "_get_lightop_attention": lambda: pytest.fail("K512 must not call native"),
        "decode_topk": lambda *_args: pytest.fail("K512 must not call adapter"),
        "decode_topk_v2": lambda *_args: False,
        "henvs": SimpleNamespace(
            VLLM_HCU_USE_CUSTOM_OPS=False,
            VLLM_HCU_USE_LIGHTOP_INDEXER_TOPK_ADAPTER=False,
        ),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(sparse_path), "exec"), namespace)
    return namespace


def _legal_topk_ids(scores: torch.Tensor, n: int, k: int) -> set[int]:
    values = scores[:n]
    threshold = torch.topk(values, min(k, n)).values[-1]
    return {int(i) for i, value in enumerate(values) if value >= threshold}


@pytest.mark.parametrize("n", [0, 511, 512, 513, 700, 2049])
@pytest.mark.parametrize("next_n", [1, 2])
def test_k512_decode_entry_score_aware_lengths_and_next_n(n, next_n):
    ns = _load_torch_decode_entry()
    width = max(n, 1)
    scores = torch.linspace(-3.0, 3.0, width, dtype=torch.float32).unsqueeze(0)
    scores = scores.expand(max(next_n, 1), -1).clone()
    scores[0, :width:17] = 0.0  # deliberate ties across the cutoff for long rows
    output = torch.full((max(next_n, 1), 512), 91, dtype=torch.int32)
    seq_lens = torch.tensor([[n] * next_n], dtype=torch.int32)
    ns["_lightop_topk_indices_decode"](scores, seq_lens, next_n, output, 512)

    for row in range(max(next_n, 1)):
        ids = output[row][output[row] >= 0].tolist()
        assert len(ids) == min(n, 512)
        assert len(set(ids)) == len(ids)
        assert all(0 <= i < n for i in ids)
        if n <= 512:
            assert ids == list(range(n))
        else:
            selected_values = scores[row, torch.tensor(ids)].sort().values
            expected_values = torch.topk(scores[row, :n], 512).values.sort().values
            torch.testing.assert_close(selected_values, expected_values)


@pytest.mark.parametrize("value", [0.0, -0.0, float("inf"), float("-inf")])
def test_k512_decode_entry_special_values_and_threshold(value):
    ns = _load_torch_decode_entry()
    n = 513
    scores = torch.full((1, n), -2.0, dtype=torch.float32)
    scores[0, 0] = value
    scores[0, 512] = 4.0
    output = torch.empty((1, 512), dtype=torch.int32)
    ns["_lightop_topk_indices_decode"](
        scores, torch.tensor([n], dtype=torch.int32), 1, output, 512
    )
    ids = output[0][output[0] >= 0].tolist()
    assert len(ids) == len(set(ids))
    assert all(0 <= i < n for i in ids)
    if math.isfinite(value) or value == float("inf"):
        selected = scores[0, torch.tensor(ids)]
        assert float(selected.min()) >= float(torch.topk(scores, 512).values[0, -1])


def test_k512_decode_entry_strided_output_preserves_canaries():
    ns = _load_torch_decode_entry()
    backing = torch.full((2, 1024), 0x5A5A5A5A, dtype=torch.int32)
    output = backing[:, ::2]
    scores = torch.arange(513, dtype=torch.float32).repeat(2, 1)
    ns["_lightop_topk_indices_decode"](
        scores, torch.tensor([513, 511], dtype=torch.int32), 1, output, 512
    )
    assert torch.equal(backing[:, 1::2], torch.full_like(backing[:, 1::2], 0x5A5A5A5A))
    assert torch.equal(output[1, 511:], torch.full((1,), -1, dtype=torch.int32))


def test_adaptive_v2_semantic_model_exposes_long_all_negative_infinity_difference():
    """The CPU model describes v2 raw-ID semantics; it is not native evidence."""
    ns = _load_torch_decode_entry()
    n = 513
    scores = torch.full((1, n), float("-inf"), dtype=torch.float32)
    output = torch.empty((1, 512), dtype=torch.int32)
    ns["_lightop_topk_indices_decode"](
        scores, torch.tensor([n], dtype=torch.int32), 1, output, 512
    )
    torch_ids = output[0].tolist()
    adaptive_raw_ids = list(range(512))
    assert all(i == -1 for i in torch_ids)
    assert adaptive_raw_ids[:3] == [0, 1, 2]
    assert len(set(adaptive_raw_ids)) == 512
