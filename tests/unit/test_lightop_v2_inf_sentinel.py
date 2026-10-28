"""Regression tests for the ragged-v2 -inf sentinel semantics (review P1).

The LightOp adaptive kernel validates only ``0 <= id < length`` and therefore
publishes a masked-out (``-inf``) candidate as a valid index when a long row
holds fewer than ``topk_tokens`` finite scores.  The Torch reference converts a
selected ``-inf`` into ``-1`` on such rows.  ``_mask_inf_selections`` restores
that contract in the HCU adapter, without touching short rows (which both
paths emit in sequence order, -inf positions included).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[2]
ADAPTER_PATH = REPO / "vllm_hcu/v1/attention/ops/lightop_indexer_topk.py"


def _load_mask_helper():
    nodes = [
        node
        for node in ast.parse(ADAPTER_PATH.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_mask_inf_selections"
    ]
    assert len(nodes) == 1, "_mask_inf_selections must exist exactly once"
    namespace = {"torch": torch}
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), str(ADAPTER_PATH), "exec"
        ),
        namespace,
    )
    return namespace["_mask_inf_selections"]


def _torch_reference(logits, row_ends, topk_tokens):
    """Mirror _topk_indices_torch for the K == topk_tokens case."""
    num_rows = logits.shape[0]
    k = min(topk_tokens, logits.shape[-1])
    values, indices = torch.topk(logits, k=k, dim=-1)
    indices = indices.to(torch.int32)
    indices = torch.where(
        values == float("-inf"),
        torch.full_like(indices, -1, dtype=torch.int32),
        indices,
    )
    row_lens = row_ends.clamp_min(0)
    if k == topk_tokens:
        selected = indices
    else:
        selected = torch.full(
            (num_rows, topk_tokens), -1, dtype=torch.int32
        )
        selected[:, :k] = indices
    seq = torch.arange(topk_tokens, dtype=torch.int32).unsqueeze(0)
    seq = seq.expand(num_rows, -1)
    all_valid = torch.where(
        seq < row_lens.unsqueeze(1), seq, torch.full_like(seq, -1)
    )
    return torch.where(
        (row_lens <= topk_tokens).unsqueeze(1), all_valid, selected
    )


def test_zero_width_logits_publish_only_padding():
    mask = _load_mask_helper()
    selected = torch.full((2, 512), 17, dtype=torch.int32)
    mask(
        torch.empty((2, 0), dtype=torch.float32),
        torch.tensor([0, 0], dtype=torch.int32),
        selected,
        512,
    )
    assert torch.equal(selected, torch.full_like(selected, -1))


def test_long_row_selected_inf_becomes_minus_one():
    mask = _load_mask_helper()
    topk_tokens = 512
    row_ends = torch.tensor([600], dtype=torch.int32)
    logits = torch.full((1, 600), float("-inf"), dtype=torch.float32)
    # Only 100 finite candidates; the kernel-style selection also emits
    # positions 100..511 whose scores are -inf.
    logits[0, :100] = torch.linspace(0.0, 1.0, 100, dtype=torch.float32)
    selected = torch.arange(topk_tokens, dtype=torch.int32).unsqueeze(0)

    mask(logits, row_ends, selected, topk_tokens)

    finite_ids = selected[0][selected[0] >= 0].tolist()
    assert len(finite_ids) == 100, "every -inf selection must become -1"
    assert finite_ids == list(range(100))


def test_short_row_inf_positions_are_preserved():
    """Short rows publish sequence order in both paths; do not mask them."""
    mask = _load_mask_helper()
    topk_tokens = 512
    row_ends = torch.tensor([8], dtype=torch.int32)
    logits = torch.full((1, 8), float("-inf"), dtype=torch.float32)
    selected = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7] + [-1] * (topk_tokens - 8)], dtype=torch.int32
    )

    mask(logits, row_ends, selected, topk_tokens)

    assert selected[0, :8].tolist() == list(range(8))
    assert (selected[0, 8:] == -1).all()


def test_padding_slots_are_not_rewritten():
    mask = _load_mask_helper()
    topk_tokens = 512
    row_ends = torch.tensor([600], dtype=torch.int32)
    logits = torch.zeros((1, 600), dtype=torch.float32)
    selected = torch.full((1, topk_tokens), -1, dtype=torch.int32)
    selected[0, 0] = 0  # a real finite selection
    selected[0, 1] = 599

    mask(logits, row_ends, selected, topk_tokens)

    assert selected[0, 0].item() == 0
    assert selected[0, 1].item() == 599
    assert (selected[0, 2:] == -1).all()


def test_positive_inf_selection_is_retained():
    """LightOp keeps +inf; the -inf sentinel rule must not touch it."""
    mask = _load_mask_helper()
    topk_tokens = 512
    row_ends = torch.tensor([600], dtype=torch.int32)
    logits = torch.zeros((1, 600), dtype=torch.float32)
    logits[0, 42] = float("inf")
    selected = torch.full((1, topk_tokens), -1, dtype=torch.int32)
    selected[0, 0] = 42

    mask(logits, row_ends, selected, topk_tokens)

    assert selected[0, 0].item() == 42


def test_masked_output_matches_torch_reference_on_long_rows():
    """End-to-end parity against the Torch reference for a long -inf row."""
    mask = _load_mask_helper()
    topk_tokens = 512
    row_ends = torch.tensor([700], dtype=torch.int32)
    logits = torch.full((1, 700), float("-inf"), dtype=torch.float32)
    logits[0, :120] = torch.linspace(-1.0, 1.0, 120, dtype=torch.float32)
    # Kernel-style emit: dense 0..511. The real kernel would reroute these
    # through score selection; the masking under test only depends on scores.
    selected = torch.arange(topk_tokens, dtype=torch.int32).unsqueeze(0).clone()

    mask(logits, row_ends, selected, topk_tokens)

    got = sorted(selected[0][selected[0] >= 0].tolist())
    expected = sorted(
        _torch_reference(logits, row_ends, topk_tokens)[0][
            _torch_reference(logits, row_ends, topk_tokens)[0] >= 0
        ].tolist()
    )
    assert got == expected


@pytest.mark.parametrize("num_rows", [1, 3, 8])
def test_mask_handles_multiple_rows_independently(num_rows):
    mask = _load_mask_helper()
    topk_tokens = 512
    row_ends = torch.full((num_rows,), 600, dtype=torch.int32)
    logits = torch.full((num_rows, 600), float("-inf"), dtype=torch.float32)
    for row in range(num_rows):
        logits[row, : 50 + row] = 1.0
    selected = (
        torch.arange(topk_tokens, dtype=torch.int32)
        .unsqueeze(0)
        .expand(num_rows, -1)
        .clone()
    )

    mask(logits, row_ends, selected, topk_tokens)

    for row in range(num_rows):
        valid = selected[row][selected[row] >= 0]
        assert len(valid) == 50 + row
