# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import (
    patch_qwen4_exp_qsa_dense_prefill as dense_prefill_patch,
)


class _Indexer:
    output_width = 7

    def forward(self, hidden_states, positions, out=None):
        del positions
        self.forward_calls = getattr(self, "forward_calls", 0) + 1
        if out is None:
            out = torch.empty(
                hidden_states.shape[0], self.output_width, dtype=torch.int32
            )
        return self._select(hidden_states, self.metadata, out)

    def _select(self, q, metadata, out):
        del q, metadata
        self.select_calls = getattr(self, "select_calls", 0) + 1
        out.fill_(3)
        return out


class _Impl:
    kv_cache_dtype = "auto"
    num_kv_heads = 1
    head_size = 2

    def forward_qsa(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        token_to_req,
        output_scale=None,
        output_block_scale=None,
    ):
        del (
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            token_to_req,
            output_scale,
            output_block_scale,
        )
        self.qsa_calls = getattr(self, "qsa_calls", 0) + 1
        output.fill_(5)
        return output


class _Backend:
    @staticmethod
    def get_supported_kernel_block_sizes():
        return ["upstream"]


class _MetadataBuilder:
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        del common_prefix_len, fast_build
        return SimpleNamespace(
            max_seq_len=int(
                common_attn_metadata.seq_lens_cpu_upper_bound.max().item()
            )
        )


def _module() -> ModuleType:
    module = ModuleType(dense_prefill_patch.TARGET_MODULE)
    module.QSAIndexer = _Indexer
    module.Qwen4ExpQSAFlashAttentionImpl = _Impl
    module.Qwen4ExpQSAFlashAttentionBackend = _Backend
    module.Qwen4ExpQSAMetadataBuilder = _MetadataBuilder
    return module


def _side_metadata(
    *,
    decode_tokens: int,
    prefill_tokens: int,
    max_query_len: int | None = None,
    max_seq_len: int | None = None,
):
    if max_query_len is None:
        max_query_len = prefill_tokens or decode_tokens
    if max_seq_len is None:
        max_seq_len = max_query_len
    return SimpleNamespace(
        num_actual_tokens=decode_tokens + prefill_tokens,
        num_decode_tokens=decode_tokens,
        num_prefill_tokens=prefill_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        seq_lens=torch.tensor([max_seq_len], dtype=torch.int32),
    )


@pytest.fixture(autouse=True)
def _enable_feature(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dense_prefill_patch, "_RUNTIME_ENABLED", True)


def test_pure_prefill_skips_qsa_selection_and_uses_dense_varlen(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _module()
    dense_calls = []
    monkeypatch.setattr(
        dense_prefill_patch,
        "run_qsa_dense_varlen_prefill",
        lambda *args, **kwargs: dense_calls.append((args, kwargs)) or args[7].fill_(9),
    )
    dense_prefill_patch.apply_to_module(module)

    indexer = module.QSAIndexer()
    indexer.metadata = _side_metadata(decode_tokens=0, prefill_tokens=4)
    topk = torch.empty(4, indexer.output_width, dtype=torch.int32)
    selected = indexer.forward(torch.empty(4, 2), torch.empty(4), topk)

    impl = module.Qwen4ExpQSAFlashAttentionImpl()
    layer = SimpleNamespace(indexer=indexer)
    output = torch.empty(4, 2)
    result = impl.forward_qsa(
        layer,
        torch.empty(4, 2),
        torch.empty(4, 2),
        torch.empty(4, 2),
        torch.empty(1),
        SimpleNamespace(num_actual_tokens=4),
        output,
        torch.empty(4, dtype=torch.int32),
    )

    assert selected is topk
    assert not hasattr(indexer, "select_calls")
    assert len(dense_calls) == 1
    assert not hasattr(impl, "qsa_calls")
    assert result is output
    assert torch.all(output == 9)


def test_prefill_with_history_keeps_qsa_to_avoid_multisequence_vmfault(
    monkeypatch: pytest.MonkeyPatch,
):
    """The vendor fixed-stride varlen ABI faults for batched history."""

    module = _module()
    dense_calls = []
    monkeypatch.setattr(
        dense_prefill_patch,
        "run_qsa_dense_varlen_prefill",
        lambda *args, **kwargs: dense_calls.append((args, kwargs)) or args[7].fill_(9),
    )
    dense_prefill_patch.apply_to_module(module)

    indexer = module.QSAIndexer()
    indexer.metadata = _side_metadata(
        decode_tokens=0,
        prefill_tokens=8,
        max_query_len=4,
        max_seq_len=8,
    )
    topk = torch.empty(8, indexer.output_width, dtype=torch.int32)
    selected = indexer.forward(torch.empty(8, 2), torch.empty(8), topk)

    impl = module.Qwen4ExpQSAFlashAttentionImpl()
    layer = SimpleNamespace(indexer=indexer)
    output = torch.empty(8, 2)
    result = impl.forward_qsa(
        layer,
        torch.empty(8, 2),
        torch.empty(8, 2),
        torch.empty(8, 2),
        torch.empty(1),
        SimpleNamespace(
            num_actual_tokens=8,
            seq_lens=torch.tensor([8, 8], dtype=torch.int32),
            max_seq_len=8,
            _vllm_hcu_qsa_context_free_prefill=False,
        ),
        output,
        torch.empty(8, dtype=torch.int32),
    )

    assert selected is topk
    assert indexer.select_calls == 1
    assert dense_calls == []
    assert impl.qsa_calls == 1
    assert result is output
    assert torch.all(output == 5)


def test_prefill_with_unsafe_history_falls_back_to_qsa(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _module()
    dense_calls = []
    monkeypatch.setattr(
        dense_prefill_patch,
        "run_qsa_dense_varlen_prefill",
        lambda *args, **kwargs: dense_calls.append((args, kwargs)),
    )
    dense_prefill_patch.apply_to_module(module)

    indexer = module.QSAIndexer()
    indexer.metadata = _side_metadata(
        decode_tokens=0,
        prefill_tokens=4,
        max_query_len=4,
        max_seq_len=32768,
    )
    topk = torch.empty(4, indexer.output_width, dtype=torch.int32)
    indexer.forward(torch.empty(4, 2), torch.empty(4), topk)

    impl = module.Qwen4ExpQSAFlashAttentionImpl()
    output = torch.empty(4, 2)
    result = impl.forward_qsa(
        SimpleNamespace(indexer=indexer),
        torch.empty(4, 2),
        torch.empty(4, 2),
        torch.empty(4, 2),
        torch.empty(1),
        SimpleNamespace(
            num_actual_tokens=4,
            seq_lens=torch.tensor([32768] + [1] * 31, dtype=torch.int32),
            max_seq_len=32768,
            _vllm_hcu_qsa_context_free_prefill=False,
        ),
        output,
        torch.empty(4, dtype=torch.int32),
    )

    assert indexer.select_calls == 1
    assert impl.qsa_calls == 1
    assert dense_calls == []
    assert result is output
    assert torch.all(output == 5)


@pytest.mark.parametrize(
    ("decode_tokens", "prefill_tokens"),
    [(4, 0), (1, 3)],
)
def test_decode_and_mixed_batches_keep_existing_qsa_path(
    monkeypatch: pytest.MonkeyPatch,
    decode_tokens: int,
    prefill_tokens: int,
):
    module = _module()
    dense_calls = []
    monkeypatch.setattr(
        dense_prefill_patch,
        "run_qsa_dense_varlen_prefill",
        lambda *args, **kwargs: dense_calls.append((args, kwargs)),
    )
    dense_prefill_patch.apply_to_module(module)

    num_tokens = decode_tokens + prefill_tokens
    indexer = module.QSAIndexer()
    indexer.metadata = _side_metadata(
        decode_tokens=decode_tokens,
        prefill_tokens=prefill_tokens,
    )
    topk = torch.empty(num_tokens, indexer.output_width, dtype=torch.int32)
    indexer.forward(torch.empty(num_tokens, 2), torch.empty(num_tokens), topk)

    impl = module.Qwen4ExpQSAFlashAttentionImpl()
    layer = SimpleNamespace(indexer=indexer)
    output = torch.empty(num_tokens, 2)
    impl.forward_qsa(
        layer,
        torch.empty(num_tokens, 2),
        torch.empty(num_tokens, 2),
        torch.empty(num_tokens, 2),
        torch.empty(1),
        SimpleNamespace(num_actual_tokens=num_tokens),
        output,
        torch.empty(num_tokens, dtype=torch.int32),
    )

    assert indexer.select_calls == 1
    assert impl.qsa_calls == 1
    assert dense_calls == []
    assert torch.all(output == 5)


def test_master_or_child_switch_off_is_a_complete_noop(
    monkeypatch: pytest.MonkeyPatch,
):
    dense_prefill_patch.configure(False)
    module = _module()
    dense_calls = []
    monkeypatch.setattr(
        dense_prefill_patch,
        "run_qsa_dense_varlen_prefill",
        lambda *args, **kwargs: dense_calls.append((args, kwargs)),
    )
    dense_prefill_patch.apply_to_module(module)

    indexer = module.QSAIndexer()
    indexer.metadata = _side_metadata(decode_tokens=0, prefill_tokens=4)
    topk = torch.empty(4, indexer.output_width, dtype=torch.int32)
    indexer.forward(torch.empty(4, 2), torch.empty(4), topk)

    impl = module.Qwen4ExpQSAFlashAttentionImpl()
    impl.forward_qsa(
        SimpleNamespace(indexer=indexer),
        torch.empty(4, 2),
        torch.empty(4, 2),
        torch.empty(4, 2),
        torch.empty(1),
        SimpleNamespace(num_actual_tokens=4),
        torch.empty(4, 2),
        torch.empty(4, dtype=torch.int32),
    )

    assert indexer.select_calls == 1
    assert impl.qsa_calls == 1
    assert dense_calls == []

    builder = module.Qwen4ExpQSAMetadataBuilder()
    common_metadata = SimpleNamespace(
        num_reqs=1,
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([4], dtype=torch.int32),
    )
    metadata = builder.build(0, common_metadata)
    assert not hasattr(metadata, "_vllm_hcu_qsa_context_free_prefill")


def test_dense_prefill_preserves_qsa_manager_pages(
    monkeypatch: pytest.MonkeyPatch,
):
    """The BLNHC QSA cache cannot be zero-copy split into kernel pages."""

    module = _module()
    dense_prefill_patch.apply_to_module(module)

    assert module.Qwen4ExpQSAFlashAttentionBackend.get_supported_kernel_block_sizes() == [
        "upstream"
    ]

    dense_prefill_patch.configure(False)
    assert module.Qwen4ExpQSAFlashAttentionBackend.get_supported_kernel_block_sizes() == [
        "upstream"
    ]


def test_metadata_builder_checks_context_for_every_prefill_request():
    """A global max equality must not hide history on a shorter request."""

    module = _module()
    dense_prefill_patch.apply_to_module(module)
    builder = module.Qwen4ExpQSAMetadataBuilder()

    mixed_context = SimpleNamespace(
        num_reqs=2,
        query_start_loc_cpu=torch.tensor([0, 3, 4], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([3, 3], dtype=torch.int32),
    )
    metadata = builder.build(0, mixed_context)

    assert (
        getattr(metadata, "_vllm_hcu_qsa_context_free_prefill", None) is False
    )

    no_context = SimpleNamespace(
        num_reqs=2,
        query_start_loc_cpu=torch.tensor([0, 3, 4], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([3, 1], dtype=torch.int32),
    )
    metadata = builder.build(0, no_context)

    assert (
        getattr(metadata, "_vllm_hcu_qsa_context_free_prefill", None) is True
    )

    ragged_lens = torch.tensor([100] + [1] * 7, dtype=torch.int32)
    highly_ragged = SimpleNamespace(
        num_reqs=8,
        query_start_loc_cpu=torch.arange(9, dtype=torch.int32),
        seq_lens_cpu_upper_bound=ragged_lens,
    )
    metadata = builder.build(0, highly_ragged)

    assert (
        getattr(metadata, "_vllm_hcu_qsa_context_free_prefill", None) is False
    )


@pytest.mark.parametrize("cache_dtype", ["auto", "fp8_e5m2"])
def test_dense_varlen_uses_current_bf16_kv_for_context_free_prefill(
    monkeypatch: pytest.MonkeyPatch,
    cache_dtype: str,
):
    calls = []

    def fake_flash_attn_varlen_func(**kwargs):
        calls.append(kwargs)
        kwargs["out"].fill_(11)

    monkeypatch.setattr(
        dense_prefill_patch,
        "_load_dense_flash_attention_symbols",
        lambda: (fake_flash_attn_varlen_func, lambda: 2),
    )
    impl = SimpleNamespace(
        alibi_slopes=None,
        sinks=None,
        sliding_window=(-1, -1),
        head_size=4,
        kv_cache_dtype=cache_dtype,
        num_kv_heads=1,
        supports_quant_query_input=False,
        scale=0.5,
        logits_soft_cap=0.0,
        vllm_flash_attn_version=2,
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(1.0),
        _k_scale=torch.tensor(2.0),
        _v_scale=torch.tensor(3.0),
    )
    metadata = SimpleNamespace(
        num_actual_tokens=3,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        max_query_len=3,
        seq_lens=torch.tensor([3], dtype=torch.int32),
        max_seq_len=3,
        causal=True,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        use_cascade=False,
        scheduler_metadata=None,
        _vllm_hcu_qsa_context_free_prefill=True,
    )
    query = torch.zeros(4, 2, 4, dtype=torch.bfloat16)
    key = torch.full((4, 1, 4), 2, dtype=torch.bfloat16)
    value = torch.full((4, 1, 4), 3, dtype=torch.bfloat16)
    kv_cache = torch.zeros(1, 1, 4, 8, dtype=torch.uint8)
    output = torch.full((4, 2, 4), -1, dtype=torch.bfloat16)

    result = dense_prefill_patch.run_qsa_dense_varlen_prefill(
        impl,
        layer,
        query,
        key,
        value,
        kv_cache,
        metadata,
        output,
    )

    assert result is output
    assert len(calls) == 1
    call = calls[0]
    assert call["q"].shape[0] == 3
    assert call["k"].data_ptr() == key.data_ptr()
    assert call["v"].data_ptr() == value.data_ptr()
    assert call["k"].shape == (3, 1, 4)
    assert call["v"].shape == (3, 1, 4)
    assert call["k"].dtype == torch.bfloat16
    assert call["v"].dtype == torch.bfloat16
    assert call["block_table"] is None
    assert call["seqused_k"] is None
    assert call["cu_seqlens_k"] is metadata.query_start_loc
    assert call["q_descale"] is None
    assert call["k_descale"] is None
    assert call["v_descale"] is None
    assert call["fa_version"] == 2
    assert torch.all(output[:3] == 11)
    assert torch.all(output[3:] == 0)


def test_dense_varlen_rejects_prefill_with_history():
    impl = SimpleNamespace(
        alibi_slopes=None,
        sinks=None,
        sliding_window=(-1, -1),
        head_size=4,
        kv_cache_dtype="auto",
        scale=0.5,
        logits_soft_cap=0.0,
    )
    metadata = SimpleNamespace(
        num_actual_tokens=4,
        query_start_loc=torch.tensor([0, 3, 4], dtype=torch.int32),
        max_query_len=3,
        seq_lens=torch.tensor([3, 2], dtype=torch.int32),
        max_seq_len=3,
        causal=True,
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        _vllm_hcu_qsa_context_free_prefill=False,
    )
    query = torch.zeros(4, 2, 4, dtype=torch.bfloat16)
    current_key = torch.full((4, 1, 4), 9, dtype=torch.bfloat16)
    current_value = torch.full((4, 1, 4), 10, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, 1, 4, 8, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="requires a context-free batch"):
        dense_prefill_patch.run_qsa_dense_varlen_prefill(
            impl,
            SimpleNamespace(),
            query,
            current_key,
            current_value,
            kv_cache,
            metadata,
            torch.empty_like(query),
        )


@pytest.mark.hcu
def test_real_qwen4_exp_classes_receive_dense_prefill_patch():
    env = os.environ.copy()
    env["VLLM_PLUGINS"] = "hcu"
    env["VLLM_HCU_USE_CUSTOM_OPS"] = "1"
    env["VLLM_HCU_QSA_DENSE_PREFILL"] = "1"
    code = """
import torch
import vllm

from vllm_hcu.patch.worker import prepare_worker_patches

prepare_worker_patches()
import vllm.models.qwen4_exp.amd.model
from vllm.models.qwen4_exp.amd.qsa import (
    QSAIndexer,
    Qwen4ExpQSAMetadataBuilder,
    Qwen4ExpQSAFlashAttentionImpl,
)

assert getattr(
    QSAIndexer.forward,
    "_vllm_hcu_qsa_dense_prefill_indexer_forward",
    False,
)
assert getattr(
    QSAIndexer._select,
    "_vllm_hcu_qsa_dense_prefill_indexer_select",
    False,
)
assert getattr(
    Qwen4ExpQSAFlashAttentionImpl.forward_qsa,
    "_vllm_hcu_qsa_dense_prefill_attention_forward",
    False,
)
assert getattr(
    Qwen4ExpQSAMetadataBuilder.build,
    "_vllm_hcu_qsa_dense_prefill_metadata_build",
    False,
)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
