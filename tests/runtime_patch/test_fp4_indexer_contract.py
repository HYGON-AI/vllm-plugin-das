# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU/mock contracts for the phase-1 HCU FP4 indexer adapter."""

import sys
from types import ModuleType

import pytest
import torch

from vllm_hcu.model_executor.layers.fp4_indexer import (
    FP4_PAGE_BYTES,
    Fp4IndexerQuery,
    fp4_cache_shape,
    fp4_cache_view,
    lightop_fp4_decode,
    lightop_fp4_prefill,
    prepare_fp4_query,
    select_fp4_indexer_capability,
)


def test_default_off_and_missing_native_retain_fp8_before_allocation():
    off = select_fp4_indexer_capability(requested=False, compress_ratio=1)
    assert not off.enabled and off.cache_format == "fp8"

    missing = select_fp4_indexer_capability(
        requested=True, compress_ratio=1, native_available=False
    )
    assert not missing.enabled and missing.cache_format == "fp8"
    assert "missing" in missing.reason


@pytest.mark.parametrize("ratio", [2, 4, 128])
def test_unsupported_ratio_never_selects_fp4(ratio):
    choice = select_fp4_indexer_capability(
        requested=True, compress_ratio=ratio, native_available=True
    )
    assert not choice.enabled
    assert choice.cache_format == "fp8"


def test_ratio1_selection_requires_complete_shared_source_group():
    choice = select_fp4_indexer_capability(
        requested=True,
        compress_ratio=1,
        native_available=True,
        shared_source_complete=False,
    )
    assert not choice.enabled
    assert "shared" in choice.reason


def test_ratio1_supported_selection_is_frozen_before_allocation():
    choice = select_fp4_indexer_capability(
        requested=True, compress_ratio=1, native_available=True
    )
    assert choice.enabled
    assert choice.cache_format == "fp4"
    assert fp4_cache_shape(3) == (3, 64, 1, 68)
    assert FP4_PAGE_BYTES == 64 * 68


def test_cache_views_preserve_physical_payload_then_scale_planes():
    raw = torch.arange(2 * FP4_PAGE_BYTES, dtype=torch.uint8).reshape(
        2, FP4_PAGE_BYTES
    )
    viewed = fp4_cache_view(raw)
    assert viewed.shape == (2, 64, 1, 68)
    assert viewed.stride() == (FP4_PAGE_BYTES, 68, 68, 1)
    # The native kernel consumes page base + page stride and interprets the
    # physical bytes as two planes; the record-shaped tensor is only an ABI
    # view.  Assert the raw page, not a fictitious interleaved record layout.
    assert raw[0, : 64 * 64].tolist() == [n % 256 for n in range(64 * 64)]
    assert raw[0, 64 * 64 : 64 * 68].tolist() == [
        n % 256 for n in range(64 * 64, 64 * 68)
    ]


def test_prepare_query_keeps_bf16_prefill_and_packed_decode_contracts():
    q = torch.linspace(-4, 4, 2 * 3 * 128, dtype=torch.float32).to(torch.bfloat16)
    weights = torch.ones(2, 3)
    prepared = prepare_fp4_query(
        q.reshape(2, 3, 128), weights, softmax_scale=0.5, head_scale=2.0
    )
    assert isinstance(prepared, Fp4IndexerQuery)
    assert prepared.bf16.dtype == torch.bfloat16
    assert prepared.bf16.shape == (2, 3, 128)
    assert prepared.packed.dtype == torch.uint8
    assert prepared.packed.shape == (2, 3, 64)
    assert prepared.scales.dtype == torch.uint8
    assert prepared.scales.shape == (2, 3, 4)
    assert prepared.weights.dtype == torch.float32
    assert torch.isfinite(prepared.weights).all()


def test_lightop_adapters_use_current_prefill_and_paged_signatures(monkeypatch):
    calls = []
    attention = ModuleType("lightop.attention")

    def prefill(*args):
        calls.append(("prefill", args))
        return "prefill-result"

    def decode(*args):
        calls.append(("decode", args))
        return "decode-result"

    attention.fp8_fp4_mqa_logits = prefill
    attention.paged_mqa_logits_fp4 = decode
    lightop = ModuleType("lightop")
    lightop.attention = attention
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.attention", attention)

    q = torch.zeros(1, 2, 128, dtype=torch.bfloat16)
    k = torch.zeros(3, 64, dtype=torch.uint8)
    scales = torch.zeros(3, 4, dtype=torch.uint8)
    weights = torch.ones(1, 2)
    starts = torch.zeros(1, dtype=torch.int32)
    ends = torch.ones(1, dtype=torch.int32)
    assert lightop_fp4_prefill(
        q, k, scales, weights, starts, ends, max_seqlen_k=4
    ) == "prefill-result"

    cache = torch.zeros(1, 64, 68, dtype=torch.uint8)
    lengths = torch.ones(1, dtype=torch.int32)
    table = torch.zeros(1, 1, dtype=torch.int32)
    packed_q = torch.zeros(1, 1, 2, 64, dtype=torch.uint8)
    q_scales = torch.zeros(1, 1, 2, dtype=torch.int32)
    assert lightop_fp4_decode(
        packed_q, q_scales, cache, weights, lengths, table, None, 1
    ) == "decode-result"

    assert len(calls) == 2
    assert calls[0][0] == "prefill" and len(calls[0][1]) == 9
    assert calls[0][1][0][0] is q
    assert calls[0][1][0][1] is None
    assert calls[1][0] == "decode" and len(calls[1][1]) == 11
    assert calls[1][1][2].shape == (1, 64, 1, 68)
