# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Source contracts for HCU model-runner speculative-token dispatch."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm_hcu"
    / "v1"
    / "hcu_model_runner.py"
)


def _propose_draft_method(tree: ast.Module) -> ast.FunctionDef:
    runner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GPUModelRunner"
    )
    return next(
        node
        for node in runner_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "propose_draft_token_ids"
    )


def _runner_method(tree: ast.Module, name: str) -> ast.FunctionDef:
    runner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GPUModelRunner"
    )
    return next(
        node
        for node in runner_class.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_runner_forwards_scheduler_dynamic_speculative_token_count() -> None:
    """Every proposer must receive the scheduler-selected draft length."""

    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    method = _propose_draft_method(tree)

    assignments = [
        node
        for node in method.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "num_spec_tokens_to_schedule"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    assert ast.unparse(assignments[0].value) == (
        "scheduler_output.num_spec_tokens_to_schedule"
    )

    proposer_calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "propose"
        and ast.unparse(node.func.value) == "self.drafter"
    ]
    assert len(proposer_calls) == 6

    positional_calls = 0
    keyword_calls = 0
    for call in proposer_calls:
        keyword = next(
            (item for item in call.keywords if item.arg == "num_speculative_tokens"),
            None,
        )
        if keyword is None:
            positional_calls += 1
            value = call.args[0]
        else:
            keyword_calls += 1
            value = keyword.value
        assert ast.unparse(value) == "num_spec_tokens_to_schedule"

    assert positional_calls == 3
    assert keyword_calls == 3


def test_runner_clamps_speculative_placeholders_at_embedding_boundary() -> None:
    """Keep scheduler placeholders out of the embedding lookup."""

    source = SOURCE_PATH.read_text(encoding="utf-8")
    method_source = ast.get_source_segment(
        source,
        _runner_method(ast.parse(source), "_preprocess"),
    )

    assert method_source is not None
    assert "if self.speculative_config is not None:" in method_source
    assert "self.input_ids.gpu[:num_input_tokens].clamp_(min=0)" in method_source


def test_runner_uses_upstream_spec_metadata_h2d_contract() -> None:
    """Do not revive the dynamic fused staging-buffer implementation."""

    source = SOURCE_PATH.read_text(encoding="utf-8")
    method = _runner_method(ast.parse(source), "_calc_spec_decode_metadata")
    method_source = ast.get_source_segment(source, method)

    assert method_source is not None
    assert method_source.count("async_tensor_h2d(") == 5
    assert "fused_meta_data" not in method_source
    assert "draft_token_ids[target_logits_indices + 1]" in method_source


def test_invalid_draft_suffixes_remain_rejected_in_metadata(monkeypatch) -> None:
    """The embedding clamp must not erase -1 from rejection metadata."""

    import vllm.v1.attention.backend as target_attention_backend

    # Some target wheels predate this HCU-only metadata alias.
    monkeypatch.setattr(
        target_attention_backend,
        "CpCommonAttentionMetadata",
        object,
        raising=False,
    )
    runner_module = importlib.import_module("vllm_hcu.v1.hcu_model_runner")
    runner = object.__new__(runner_module.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.arange_np = np.arange(64, dtype=np.int64)
    runner._arange_scratch = np.empty(64, dtype=np.int64)
    runner.input_ids = SimpleNamespace(
        gpu=torch.tensor(
            [99, 10, -1, 99, 12, 99, 13, -1],
            dtype=torch.int32,
        )
    )

    metadata = runner_module.GPUModelRunner._calc_spec_decode_metadata(
        runner,
        np.array([2, 1, 2], dtype=np.int32),
        np.array([3, 5, 8], dtype=np.int32),
    )

    assert metadata.draft_token_ids.tolist() == [10, -1, 12, 13, -1]
