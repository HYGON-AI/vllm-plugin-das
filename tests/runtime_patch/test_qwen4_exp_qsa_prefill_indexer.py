# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Routing tests for the opt-in Qwen4Exp prefill indexer."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.op_opt import (
    patch_qwen4_exp_qsa_prefill_indexer as prefill_indexer_patch,
)


class _Indexer:
    output_width = 11

    def _select(self, q, metadata, out):
        del q, metadata
        self.original_calls = getattr(self, "original_calls", 0) + 1
        if out is None:
            out = torch.empty(4, self.output_width, dtype=torch.int32)
        out.fill_(3)
        return out


def _module() -> ModuleType:
    module = ModuleType(prefill_indexer_patch.TARGET_MODULE)
    module.QSAIndexer = type("QSAIndexer", (_Indexer,), {})
    return module


def _metadata(
    *,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    num_prefills: int = 1,
    num_prefill_tokens: int = 4,
    decode_query_len: int = 0,
    prefix_len: int = 0,
):
    num_tokens = num_decode_tokens + num_prefill_tokens
    query_len = max(1, num_prefill_tokens // max(1, num_prefills))
    return SimpleNamespace(
        num_actual_tokens=num_tokens,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        decode_query_len=decode_query_len,
        max_query_len=query_len,
        block_table=torch.zeros(
            max(1, num_decodes + num_prefills), 2, dtype=torch.int32
        ),
        query_start_loc=torch.arange(
            0,
            (max(1, num_decodes + num_prefills) + 1) * query_len,
            query_len,
            dtype=torch.int32,
        ),
        visible_blocks=torch.ones(num_tokens, dtype=torch.int32),
        logical_positions=torch.arange(
            prefix_len,
            prefix_len + num_tokens,
            dtype=torch.int64,
        ),
        token_to_req=torch.zeros(num_tokens, dtype=torch.int32),
        seq_lens=torch.full(
            (max(1, num_decodes + num_prefills),),
            prefix_len + query_len,
            dtype=torch.int32,
        ),
    )


@pytest.fixture(autouse=True)
def _enable_feature(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(prefill_indexer_patch, "_RUNTIME_ENABLED", True)


@pytest.mark.parametrize("prefix_len", [0, 128])
def test_prefill_only_routes_to_specialized_indexer_and_preserves_output(
    monkeypatch: pytest.MonkeyPatch,
    prefix_len: int,
):
    calls = []

    def specialized(indexer, q, metadata, out):
        calls.append((indexer, q, metadata, out))
        return out.fill_(9)

    monkeypatch.setattr(
        prefill_indexer_patch,
        "run_qsa_prefill_selection",
        specialized,
    )
    module = _module()
    assert prefill_indexer_patch.apply_to_module(module) is True
    indexer = module.QSAIndexer()
    metadata = _metadata(prefix_len=prefix_len)
    q = torch.empty(4, 4, 128)
    out = torch.empty(4, indexer.output_width, dtype=torch.int32)

    result = indexer._select(q, metadata, out)

    assert result is out
    assert torch.all(out == 9)
    assert len(calls) == 1
    assert calls[0][0] is indexer
    assert calls[0][1] is q
    assert calls[0][2] is metadata
    assert calls[0][3] is out
    assert not hasattr(indexer, "original_calls")


@pytest.mark.parametrize(
    "metadata",
    [
        _metadata(
            num_decodes=1,
            num_decode_tokens=4,
            num_prefills=0,
            num_prefill_tokens=0,
            decode_query_len=4,
        ),
        _metadata(
            num_decodes=2,
            num_decode_tokens=4,
            num_prefills=0,
            num_prefill_tokens=0,
            decode_query_len=2,
        ),
        _metadata(
            num_decodes=1,
            num_decode_tokens=1,
            num_prefills=1,
            num_prefill_tokens=3,
            decode_query_len=1,
        ),
    ],
    ids=("decode", "spec_decode", "mixed"),
)
def test_decode_speculative_and_mixed_batches_keep_original_selection(
    monkeypatch: pytest.MonkeyPatch,
    metadata,
):
    monkeypatch.setattr(
        prefill_indexer_patch,
        "run_qsa_prefill_selection",
        lambda *args: pytest.fail("specialized prefill path must not run"),
    )
    module = _module()
    prefill_indexer_patch.apply_to_module(module)
    indexer = module.QSAIndexer()
    out = torch.empty(4, indexer.output_width, dtype=torch.int32)

    result = indexer._select(torch.empty(4, 4, 128), metadata, out)

    assert result is out
    assert indexer.original_calls == 1
    assert torch.all(out == 3)


def test_disabled_feature_keeps_original_selection(
    monkeypatch: pytest.MonkeyPatch,
):
    prefill_indexer_patch.configure(False)
    monkeypatch.setattr(
        prefill_indexer_patch,
        "run_qsa_prefill_selection",
        lambda *args: pytest.fail("disabled prefill path must not run"),
    )
    module = _module()
    prefill_indexer_patch.apply_to_module(module)
    indexer = module.QSAIndexer()
    out = torch.empty(4, indexer.output_width, dtype=torch.int32)

    result = indexer._select(torch.empty(4, 4, 128), _metadata(), out)

    assert result is out
    assert indexer.original_calls == 1


def test_incomplete_prefill_metadata_falls_back_before_launch(
    monkeypatch: pytest.MonkeyPatch,
):
    metadata = _metadata()
    del metadata.visible_blocks
    monkeypatch.setattr(
        prefill_indexer_patch,
        "run_qsa_prefill_selection",
        lambda *args: pytest.fail("malformed metadata must not launch"),
    )
    module = _module()
    prefill_indexer_patch.apply_to_module(module)
    indexer = module.QSAIndexer()

    indexer._select(
        torch.empty(4, 4, 128),
        metadata,
        torch.empty(4, indexer.output_width, dtype=torch.int32),
    )

    assert indexer.original_calls == 1


def test_patch_is_idempotent_and_keeps_wrapper_marker():
    module = _module()

    assert prefill_indexer_patch.apply_to_module(module) is True
    assert prefill_indexer_patch.apply_to_module(module) is False
    assert getattr(
        module.QSAIndexer._select,
        "_vllm_hcu_qsa_prefill_indexer_select",
        False,
    )
