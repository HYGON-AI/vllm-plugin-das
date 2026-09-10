# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Sparse MLA splitting contracts for rebuilt multi-step draft metadata."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


TARGET_MODULE = "vllm.v1.attention.backends.mla.indexer"
ADAPTER_MODULE = "vllm_hcu.patch.worker.op_opt.patch_mla_indexer"


def _patched_split(monkeypatch: pytest.MonkeyPatch):
    calls: list[bool] = []
    target = ModuleType(TARGET_MODULE)

    def split_indexer_prefill_chunks(
        seq_lens_cpu,
        query_lens_cpu,
        workspace_size,
        max_logits_bytes,
        request_offset=0,
    ):
        return []

    def split_decodes_and_prefills(
        common_attn_metadata,
        decode_threshold=1,
        require_uniform=False,
        treat_short_extends_as_decodes=True,
    ):
        calls.append(treat_short_extends_as_decodes)
        if not treat_short_extends_as_decodes:
            assert common_attn_metadata.is_prefilling is not None
        return (1, 0, 1, 0)

    class DeepseekV32IndexerMetadataBuilder:
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            return SimpleNamespace()

    target.split_indexer_prefill_chunks = split_indexer_prefill_chunks
    target.split_decodes_and_prefills = split_decodes_and_prefills
    target.DeepseekV32IndexerMetadataBuilder = DeepseekV32IndexerMetadataBuilder
    monkeypatch.setitem(sys.modules, TARGET_MODULE, target)
    monkeypatch.delitem(sys.modules, ADAPTER_MODULE, raising=False)
    adapter = importlib.import_module(ADAPTER_MODULE)
    adapter.apply_to_module(target)
    return target.split_decodes_and_prefills, calls


def test_draft_metadata_without_prefill_mask_is_treated_as_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_batch, calls = _patched_split(monkeypatch)
    common = SimpleNamespace(is_prefilling=None)

    assert split_batch(common, 1, False, False) == (1, 0, 1, 0)
    assert calls == [True]


def test_target_metadata_preserves_explicit_prefill_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_batch, calls = _patched_split(monkeypatch)
    common = SimpleNamespace(is_prefilling=object())

    assert split_batch(common, 1, False, False) == (1, 0, 1, 0)
    assert calls == [False]
