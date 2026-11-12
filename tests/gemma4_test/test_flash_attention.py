# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Routing and packed-cache regression tests for Gemma4 prefill."""

from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm import ModelRegistry

from vllm_hcu.models import gemma4, register_model


HEAD_SIZE = 512
NUM_HEADS = 4
NUM_KV_HEADS = 2
BLOCK_SIZE = 32


@pytest.fixture
def attention(monkeypatch):
    monkeypatch.setenv("HCU_DEBUG_ATTN_Q_LEN", "0")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    return gemma4.Gemma4FAAttentionImpl(
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        scale=1.0,
        num_kv_heads=NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )


def _metadata(query_lengths, kv_lengths, block_table):
    query_starts = [0]
    for length in query_lengths:
        query_starts.append(query_starts[-1] + length)
    return SimpleNamespace(
        num_actual_tokens=query_starts[-1],
        max_query_len=max(query_lengths),
        max_seq_len=max(kv_lengths),
        query_start_loc=torch.tensor(
            query_starts, dtype=torch.int32, device=block_table.device
        ),
        seq_lens=torch.tensor(
            kv_lengths, dtype=torch.int32, device=block_table.device
        ),
        block_table=block_table,
        use_cascade=False,
        causal=True,
        mm_prefix_range=None,
        mm_prefix_range_tensor=None,
    )


@pytest.mark.parametrize(
    ("custom_ops", "gemma4_fa", "expected"),
    [
        (None, None, False),
        ("1", "0", False),
        (None, "1", True),
        ("1", "1", True),
        ("0", "1", False),
        ("0", "invalid", False),
    ],
    ids=[
        "default-off",
        "explicit-off",
        "default-master-on",
        "both-on",
        "master-off",
        "master-off-ignores-feature-value",
    ],
)
def test_gemma4_fa_registration_respects_custom_ops(
    monkeypatch, custom_ops, gemma4_fa, expected
):
    for name, value in (
        ("VLLM_HCU_USE_CUSTOM_OPS", custom_ops),
        ("VLLM_HCU_GEMMA4_USE_FA", gemma4_fa),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    register_attention = Mock()
    register_other_models = Mock()
    monkeypatch.setattr(gemma4, "register_attention", register_attention)
    monkeypatch.setattr(ModelRegistry, "register_model", register_other_models)

    register_model()

    assert register_attention.call_count == int(expected)
    assert register_other_models.called


class _CustomBackend(gemma4.TritonAttentionBackend):
    pass


@pytest.mark.parametrize(
    ("dcp_size", "requested_backend", "global_backend"),
    [
        (1, None, "TRITON_ATTN"),
        (1, gemma4.TritonAttentionBackend, "TRITON_ATTN"),
        (1, _CustomBackend, "TRITON_ATTN"),
        (1, gemma4.TritonAttentionBackend, "FLASH_ATTN"),
        (2, None, "TRITON_ATTN"),
    ],
    ids=[
        "implicit-triton",
        "explicit-triton",
        "explicit-custom",
        "nontriton-config",
        "unsupported-dcp",
    ],
)
def test_gemma4_fa_backend_selection(
    monkeypatch, dcp_size, requested_backend, global_backend
):
    from vllm import config

    monkeypatch.setattr(
        config,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_size),
            attention_config=SimpleNamespace(backend=global_backend),
        ),
    )

    class Attention:
        def __init__(self, num_heads, head_size, scale, **kwargs):
            self.backend = kwargs.get("attn_backend")

    model = ModuleType("vllm.model_executor.models.gemma4")
    model.Attention = Attention
    gemma4._use_gemma4_fa(model)

    if dcp_size == 2:
        with pytest.raises(ValueError, match="decode_context_parallel_size > 1"):
            model.Attention(NUM_HEADS, HEAD_SIZE, 1.0)
    else:
        full = model.Attention(
            NUM_HEADS, HEAD_SIZE, 1.0, attn_backend=requested_backend
        )
        assert full.backend is gemma4.Gemma4FAAttentionBackend
        sliding = model.Attention(
            NUM_HEADS,
            HEAD_SIZE,
            1.0,
            per_layer_sliding_window=1024,
            attn_backend=requested_backend,
        )
        assert sliding.backend is requested_backend


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_gemma4_fa_passes_packed_views_and_writes_only_active_output(
    monkeypatch, attention, layout
):
    query = torch.empty(19, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device="cpu")
    output = torch.full_like(query, -1)
    packed = torch.empty(
        2, NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD_SIZE, dtype=query.dtype, device="cpu"
    )
    if layout == "NHD":
        packed = packed.transpose(1, 2).contiguous().transpose(1, 2)
    block_table = torch.tensor([[1, 0]], dtype=torch.int32, device="cpu")
    metadata = _metadata([17], [41], block_table)
    metadata.mm_prefix_range = {0: []}

    def write_output(q, k, v, *, out, **kwargs):
        out.fill_(1)
        return out

    flash = ModuleType("flash_attn")
    flash.flash_attn_varlen_func = Mock(side_effect=write_output)
    monkeypatch.setitem(sys.modules, "flash_attn", flash)

    result = attention.forward(None, query, None, None, packed, metadata, output)
    assert result is output

    flash.flash_attn_varlen_func.assert_called_once()
    (q, k, v), kwargs = flash.flash_attn_varlen_func.call_args
    assert q.shape == (17, NUM_HEADS, HEAD_SIZE)
    assert q.data_ptr() == query.data_ptr()
    assert k.shape == v.shape == (2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    assert k.stride() == v.stride() == packed.transpose(1, 2).stride()
    assert k.data_ptr() == packed.data_ptr()
    assert v.data_ptr() - k.data_ptr() == HEAD_SIZE * packed.element_size()
    assert kwargs["out"].data_ptr() == output.data_ptr()
    assert kwargs["block_table"] is metadata.block_table
    assert kwargs["cu_seqlens_q"] is metadata.query_start_loc
    assert kwargs["seqused_k"] is metadata.seq_lens
    assert kwargs["max_seqlen_q"] == 17
    assert kwargs["max_seqlen_k"] == 41
    assert kwargs["causal"] is True
    assert kwargs["window_size"] == (-1, -1)
    assert kwargs["softmax_scale"] == 1.0
    torch.testing.assert_close(output[:17], torch.ones_like(output[:17]))
    torch.testing.assert_close(output[17:], torch.full_like(output[17:], -1))


@pytest.mark.parametrize(
    ("query_length", "causal", "mm_prefix_range"),
    [(5, True, None), (16, True, None), (17, False, None), (17, True, {0: [(0, 3)]})],
    ids=["mtp", "threshold", "noncausal", "visible-mm-prefix"],
)
def test_gemma4_fa_ineligible_calls_fall_back_to_triton(
    monkeypatch, attention, query_length, causal, mm_prefix_range
):
    query = torch.empty(
        query_length, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device="cpu"
    )
    output = torch.empty_like(query)
    packed = torch.empty(
        2, NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD_SIZE, dtype=query.dtype, device="cpu"
    )
    block_table = torch.tensor([[1, 0]], dtype=torch.int32, device="cpu")
    metadata = _metadata([query_length], [41], block_table)
    metadata.causal = causal
    metadata.mm_prefix_range = mm_prefix_range
    triton = Mock(return_value=output)
    monkeypatch.setattr(gemma4.TritonAttentionImpl, "forward", triton)
    flash = Mock(side_effect=AssertionError("unexpected FlashAttention dispatch"))
    monkeypatch.setattr(attention, "_flash_attention", flash)

    result = attention.forward(None, query, None, None, packed, metadata, output)
    assert result is output

    flash.assert_not_called()
    triton.assert_called_once_with(
        None, query, None, None, packed, metadata, output, None, None
    )


@pytest.mark.hcu
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_gemma4_native_packed_prefill_matches_fp32(attention, layout):
    if not torch.cuda.is_available():
        pytest.skip("requires a live gfx938 DCU")
    if not getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").startswith(
        "gfx938"
    ):
        pytest.skip("D512 packed prefill is validated on gfx938")
    device = torch.device("cuda:0")
    rng = torch.Generator(device=device).manual_seed(42)
    query_lengths = [17, 33]
    kv_lengths = [41, 519]
    page_counts = [math.ceil(n / BLOCK_SIZE) for n in kv_lengths]
    page_ids = torch.randperm(sum(page_counts), generator=rng, device=device)
    block_table = torch.full(
        (2, max(page_counts)), -1, dtype=torch.int32, device=device
    )
    block_table[0, : page_counts[0]] = page_ids[: page_counts[0]]
    block_table[1] = page_ids[page_counts[0] :]

    packed = torch.randn(
        (sum(page_counts), NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD_SIZE),
        generator=rng,
        dtype=torch.bfloat16,
        device=device,
    )
    if layout == "NHD":
        packed = packed.transpose(1, 2).contiguous().transpose(1, 2)
    query = torch.randn(
        (sum(query_lengths), NUM_HEADS, HEAD_SIZE), generator=rng, device=device
    )
    query = (query / math.sqrt(HEAD_SIZE)).bfloat16()
    output = torch.full_like(query, float("nan"))
    metadata = _metadata(query_lengths, kv_lengths, block_table)
    assert attention._can_use_fa(query, packed, metadata, output, None, None)

    attention.forward(None, query, None, None, packed, metadata, output)

    refs = []
    query_start = 0
    for row, (q_len, kv_len, pages) in enumerate(
        zip(query_lengths, kv_lengths, page_counts)
    ):
        # Gather in PyTorch by logical indices, then apply a bottom-right causal mask.
        cache = packed.index_select(0, block_table[row, :pages].long())
        cache = cache.transpose(1, 2).reshape(-1, NUM_KV_HEADS, 2 * HEAD_SIZE)
        k, v = cache[:kv_len].float().split(HEAD_SIZE, dim=-1)
        k = k.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
        v = v.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
        q = query[query_start : query_start + q_len].float()
        scores = torch.einsum("qhd,khd->hqk", q, k)
        causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=device)
        causal = causal.tril(kv_len - q_len)
        scores.masked_fill_(~causal, float("-inf"))
        refs.append(torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v))
        query_start += q_len

    torch.testing.assert_close(output.float(), torch.cat(refs), atol=0.03, rtol=0.03)
