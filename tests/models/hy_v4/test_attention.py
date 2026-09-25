# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import KVQuantMode


def test_sparse_e4m3_uses_ds_mla_layout_without_changing_dense_cache():
    from vllm_hcu.models.hy_v4.attention import _normalize_hy_v4_kv_cache_dtype

    assert _normalize_hy_v4_kv_cache_dtype(
        "fp8_e4m3", use_sparse=True
    ) == "fp8_ds_mla"
    assert _normalize_hy_v4_kv_cache_dtype(
        "fp8_e4m3", use_sparse=False
    ) == "fp8_e4m3"


def test_unsafe_generic_fp8_cache_is_rejected():
    from vllm_hcu.models.hy_v4.attention import _require_accuracy_safe_kv_cache_dtype

    with pytest.raises(RuntimeError, match="--kv-cache-dtype fp8_e4m3"):
        _require_accuracy_safe_kv_cache_dtype("fp8")


def test_unvalidated_context_parallel_paths_fail_closed():
    from vllm_hcu.models.hy_v4.attention import (
        _require_supported_hy_v4_parallelism,
    )

    for pcp, dcp in ((2, 1), (1, 2)):
        with pytest.raises(RuntimeError, match="TP-only"):
            _require_supported_hy_v4_parallelism(
                SimpleNamespace(
                    prefill_context_parallel_size=pcp,
                    decode_context_parallel_size=dcp,
                )
            )

    _require_supported_hy_v4_parallelism(
        SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        )
    )


def test_shared_indexer_weights_are_skipped_but_full_producer_is_loaded():
    from vllm_hcu.models.hy_v4.attention import (
        compute_skip_topk_layers,
        is_skip_topk_indexer_weight,
    )

    config = SimpleNamespace(
        index_topk=64,
        num_hidden_layers=6,
        indexer_types=["full", "shared", "shared", "full", "shared", "shared"],
    )
    skipped = compute_skip_topk_layers(config)

    assert skipped == {1, 2, 4, 5}
    assert is_skip_topk_indexer_weight(
        "model.layers.2.self_attn.indexer.wq_b.weight", skipped
    )
    assert not is_skip_topk_indexer_weight(
        "model.layers.3.self_attn.indexer.wq_b.weight", skipped
    )


def test_shared_indexer_cannot_precede_its_full_producer():
    from vllm_hcu.models.hy_v4.attention import compute_skip_topk_layers

    config = SimpleNamespace(
        index_topk=64,
        num_hidden_layers=3,
        indexer_types=["shared", "shared", "full"],
    )
    with pytest.raises(ValueError, match="preceding 'full'"):
        compute_skip_topk_layers(config)


def test_sparse_e4m3_cache_spec_has_quantized_page_geometry():
    from vllm_hcu.models.hy_v4.attention import HYV4MLAAttentionLayer

    attention = object.__new__(HYV4MLAAttentionLayer)
    attention.kv_cache_dtype = "fp8_ds_mla"
    attention.head_size = 576
    attention.sliding_window = None
    attention.non_causal_multi_token_decode = False

    resolved = attention.get_kv_cache_spec(
        SimpleNamespace(
            model_config=None,
            cache_config=SimpleNamespace(block_size=64),
        )
    )

    assert resolved.kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
    assert resolved.state_content_size_bytes == 656
    assert resolved.page_size_bytes == 64 * 656


def test_hyv4_sparse_backend_keeps_flashmla_name_and_sink_capability():
    from vllm_hcu.models.hy_v4.hcu_sparse import HYV4FlashMLASparseBackend

    assert HYV4FlashMLASparseBackend.get_name() == "FLASHMLA_SPARSE"
    assert HYV4FlashMLASparseBackend.is_sparse()
    assert HYV4FlashMLASparseBackend.supports_sink()


def test_sparse_mqa_backend_accepts_flashmla_and_rejects_dense():
    from vllm_hcu.models.hy_v4.attention import _require_sparse_mqa_backend
    from vllm_hcu.models.hy_v4.hcu_sparse import HYV4FlashMLASparseBackend

    _require_sparse_mqa_backend(HYV4FlashMLASparseBackend)

    class DenseBackend:
        @staticmethod
        def get_impl_cls():
            return object

        @staticmethod
        def get_name():
            return "DENSE"

    with pytest.raises(RuntimeError, match="sparse MQA"):
        _require_sparse_mqa_backend(DenseBackend)


def _bare_sparse_impl(sinks: torch.Tensor):
    from vllm_hcu.models.hy_v4.hcu_sparse import HYV4FlashMLASparseImpl

    impl = object.__new__(HYV4FlashMLASparseImpl)
    impl.sinks = sinks
    impl.num_heads = sinks.numel()
    impl.prefill_padding = 64
    impl.fp8_decode_padded_heads = 64
    impl.softmax_scale = 0.5
    impl.dcp_world_size = 1
    impl.dcp_rank = 0
    impl.head_size = 576
    impl.kv_lora_rank = 512
    return impl


def test_bf16_sparse_kernel_returns_target_output_lse_pair(monkeypatch):
    from vllm_hcu.models.hy_v4 import hcu_sparse

    impl = _bare_sparse_impl(torch.arange(4, dtype=torch.float32))
    captured = {}

    def fake_sparse_fwd(q, kv, indices, scale, attn_sink=None, topk_length=None):
        captured["sink"] = attn_sink
        return (
            torch.zeros(q.shape[0], q.shape[1], 512),
            None,
            torch.zeros(q.shape[0], q.shape[1]),
        )

    monkeypatch.setattr(hcu_sparse, "flash_mla_sparse_fwd", fake_sparse_fwd)
    output, lse = impl._bf16_flash_mla_kernel(
        torch.zeros(2, 4, 576),
        torch.zeros(8, 576),
        torch.zeros(2, 4, dtype=torch.int32),
        None,
        4,
    )

    assert output.shape == (2, 4, 512)
    assert lse.shape == (2, 4)
    assert torch.equal(captured["sink"][:4], impl.sinks)
    assert torch.isneginf(captured["sink"][4:]).all()


def test_fp8_sparse_kernel_forwards_sink_and_slices_target_lse(monkeypatch):
    from vllm_hcu.models.hy_v4 import hcu_sparse

    impl = _bare_sparse_impl(torch.arange(4, dtype=torch.float32))
    captured = {}

    def fake_flashmla(**kwargs):
        captured.update(kwargs)
        return torch.zeros(1, 2, 64, 512), torch.zeros(1, 64, 2)

    monkeypatch.setattr(hcu_sparse, "flash_mla_with_kvcache", fake_flashmla)
    metadata = SimpleNamespace(
        dummy_block_table=torch.zeros(1, 1, dtype=torch.int32),
        cache_lens=torch.tensor([2], dtype=torch.int32),
        scheduler_metadata=None,
    )
    output, lse = impl._fp8_flash_mla_kernel(
        torch.zeros(1, 2, 4, 576),
        torch.zeros(1, 656, dtype=torch.uint8),
        torch.zeros(1, 2, 4, dtype=torch.int32),
        metadata,
    )

    assert output.shape == (1, 2, 4, 512)
    assert lse.shape == (1, 4, 2)
    assert torch.equal(captured["attn_sink"][:4], impl.sinks)
    assert torch.isneginf(captured["attn_sink"][4:]).all()
