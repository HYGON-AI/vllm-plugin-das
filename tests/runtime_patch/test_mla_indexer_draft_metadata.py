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


def test_metadata_scope_controls_pcp_width_during_indexer_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_world_sizes: list[int] = []
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
        return (1, 0, 1, 0)

    class DeepseekV32IndexerMetadataBuilder:
        def __init__(self) -> None:
            self.pcp_world_size = 2

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            observed_world_sizes.append(self.pcp_world_size)
            if getattr(common_attn_metadata, "fail", False):
                raise RuntimeError("metadata build failed")
            return SimpleNamespace()

    target.split_indexer_prefill_chunks = split_indexer_prefill_chunks
    target.split_decodes_and_prefills = split_decodes_and_prefills
    target.DeepseekV32IndexerMetadataBuilder = DeepseekV32IndexerMetadataBuilder
    monkeypatch.setitem(sys.modules, TARGET_MODULE, target)
    monkeypatch.delitem(sys.modules, ADAPTER_MODULE, raising=False)
    adapter = importlib.import_module(ADAPTER_MODULE)
    adapter.apply_to_module(target)

    common = SimpleNamespace(num_actual_tokens=4, pcp_world_size=2)
    builder = target.DeepseekV32IndexerMetadataBuilder()
    from vllm_hcu.model_executor.layers.attention.pcp import (
        logical_pcp_metadata_scope,
        replicated_mtp_batch_scope,
    )

    with logical_pcp_metadata_scope(1):
        result = builder.build(0, common)

    assert builder.pcp_world_size == 2
    assert result.pcp_world_size == 1

    with pytest.raises(RuntimeError, match="metadata build failed"):
        with logical_pcp_metadata_scope(1):
            builder.build(0, SimpleNamespace(num_actual_tokens=4, fail=True))
    assert builder.pcp_world_size == 2

    with logical_pcp_metadata_scope(2):
        prefill_result = builder.build(0, common)
    assert prefill_result.pcp_world_size == 2

    with replicated_mtp_batch_scope(), logical_pcp_metadata_scope(2):
        draft_result = builder.build(0, common)
    assert observed_world_sizes == [1, 1, 2, 1]
    assert draft_result.pcp_world_size == 1
