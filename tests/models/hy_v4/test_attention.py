# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.models.hy_v4 import hcu_sparse
from vllm_hcu.models.hy_v4.attention import (
    _require_sparse_mqa_backend,
    compute_skip_topk_layers,
    is_skip_topk_indexer_weight,
    require_hyv4_sink_backend,
)
from vllm_hcu.models.hy_v4.hcu_sparse import (
    HYV4FlashMLASparseBackend,
    HYV4FlashMLASparseImpl,
)


def test_full_and_shared_indexer_pattern() -> None:
    config = SimpleNamespace(
        index_topk=64,
        num_hidden_layers=6,
        indexer_types=["full", "shared", "shared", "full", "shared", "shared"],
    )

    assert compute_skip_topk_layers(config) == {1, 2, 4, 5}
    assert is_skip_topk_indexer_weight(
        "model.layers.2.self_attn.indexer.wq_b.weight",
        {1, 2, 4, 5},
    )
    assert not is_skip_topk_indexer_weight(
        "model.layers.3.self_attn.indexer.wq_b.weight",
        {1, 2, 4, 5},
    )


def test_sink_incapable_backend_fails_closed() -> None:
    class SinkIncapableSparseBackend:
        @classmethod
        def supports_sink(cls) -> bool:
            return False

        @classmethod
        def is_sparse(cls) -> bool:
            return True

        @classmethod
        def get_name(cls) -> str:
            return "SINK_INCAPABLE"

    with pytest.raises(RuntimeError, match="learnable sink"):
        require_hyv4_sink_backend(SinkIncapableSparseBackend)


def test_hcu_backend_advertises_sink_support() -> None:
    assert HYV4FlashMLASparseBackend.supports_sink()
    assert HYV4FlashMLASparseBackend.is_sparse()
    assert HYV4FlashMLASparseBackend.get_name() == "FLASHMLA_SPARSE"
    assert HYV4FlashMLASparseBackend.get_impl_cls() is HYV4FlashMLASparseImpl


def test_sink_prefill_requires_sparse_mqa_impl_without_global_config_flag() -> None:
    _require_sparse_mqa_backend(HYV4FlashMLASparseBackend)

    class DenseBackend:
        @staticmethod
        def get_impl_cls():
            return object

        @staticmethod
        def get_name() -> str:
            return "DENSE"

    with pytest.raises(RuntimeError, match="sparse MQA"):
        _require_sparse_mqa_backend(DenseBackend)


@pytest.mark.parametrize(
    "sinks",
    [
        torch.zeros(4, dtype=torch.bfloat16),
        torch.zeros(3, dtype=torch.float32),
        torch.zeros((4, 1), dtype=torch.float32),
    ],
)
def test_sink_validation_rejects_kernel_incompatible_layouts(
    sinks: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        HYV4FlashMLASparseImpl._validate_sinks(sinks, num_heads=4)


def _bare_impl(sinks: torch.Tensor | None) -> HYV4FlashMLASparseImpl:
    impl = object.__new__(HYV4FlashMLASparseImpl)
    impl.sinks = sinks
    impl.num_heads = 4
    impl.prefill_padding = 64
    impl.fp8_decode_padded_heads = 64
    impl.softmax_scale = 0.5
    return impl


def test_sink_padding_uses_negative_infinity() -> None:
    sinks = torch.arange(4, dtype=torch.float32)
    impl = _bare_impl(sinks)

    padded = impl._sinks_for_query(
        torch.zeros(2, 4, 576),
        head_dim=1,
        kernel_heads=64,
    )

    assert padded is not None
    assert torch.equal(padded[:4], sinks)
    assert torch.isneginf(padded[4:]).all()


def test_bf16_prefill_forwards_live_sink(monkeypatch) -> None:
    sinks = torch.arange(4, dtype=torch.float32)
    impl = _bare_impl(sinks)
    captured: dict[str, torch.Tensor | None] = {}

    def fake_sparse_fwd(q, kv, indices, scale, attn_sink=None, topk_length=None):
        del kv, indices, scale, topk_length
        captured["attn_sink"] = attn_sink
        return (torch.zeros(q.shape[0], q.shape[1], 512),)

    monkeypatch.setattr(hcu_sparse, "flash_mla_sparse_fwd", fake_sparse_fwd)
    impl._bf16_flash_mla_kernel(
        q=torch.zeros(2, 4, 576),
        kv_c_and_k_pe_cache=torch.zeros(8, 576),
        topk_indices=torch.zeros(2, 4, dtype=torch.int32),
    )

    forwarded = captured["attn_sink"]
    assert forwarded is not None
    assert forwarded.shape == (64,)
    assert torch.equal(forwarded[:4], sinks)


def test_fp8_decode_forwards_live_sink(monkeypatch) -> None:
    sinks = torch.arange(4, dtype=torch.float32)
    impl = _bare_impl(sinks)
    captured: dict[str, torch.Tensor | None] = {}

    def fake_with_kvcache(**kwargs):
        captured["attn_sink"] = kwargs["attn_sink"]
        q = kwargs["q"]
        return torch.zeros(q.shape[0], q.shape[1], q.shape[2], 512), torch.zeros(1)

    monkeypatch.setattr(hcu_sparse, "flash_mla_with_kvcache", fake_with_kvcache)
    metadata = SimpleNamespace(
        dummy_block_table=torch.zeros(1, 1, dtype=torch.int32),
        cache_lens=torch.zeros(1, dtype=torch.int32),
        scheduler_metadata=None,
    )
    impl._fp8_flash_mla_kernel(
        q=torch.zeros(1, 2, 4, 576),
        kv_c_and_k_pe_cache=torch.zeros(8, 656, dtype=torch.uint8),
        topk_indices=torch.zeros(1, 2, 4, dtype=torch.int32),
        kernel_metadata=metadata,
    )

    forwarded = captured["attn_sink"]
    assert forwarded is not None
    assert forwarded.shape == (64,)
    assert torch.equal(forwarded[:4], sinks)
