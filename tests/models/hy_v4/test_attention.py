# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import KVQuantMode, MLAAttentionSpec

from vllm_hcu.models.hy_v4 import hcu_sparse
from vllm_hcu.models.hy_v4.attention import (
    HYV4MLAAttentionLayer,
    Indexer,
    _normalize_hy_v4_kv_cache_dtype,
    _require_accuracy_safe_kv_cache_dtype,
    _require_sparse_mqa_backend,
    compute_skip_topk_layers,
    is_skip_topk_indexer_weight,
    require_local_indexer_producer,
    require_hyv4_sink_backend,
)


class _LinearResult(torch.nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("value", value)

    def forward(self, _input: torch.Tensor):
        return self.value.clone(), None


class _ReturnIndexerQuery(torch.nn.Module):
    def forward(self, _hidden_states, query, _key, _weights):
        return query


def test_hy_v4_lightop_prepare_reorders_rope_and_norm_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch feeding PTM's PE-last layout to LightOp's PE-first kernel."""
    assert hasattr(Indexer, "prepare_inputs_lightop"), (
        "Hy4 Indexer must expose the LightOp preparation path"
    )
    indexer = object.__new__(Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.n_head = 2
    indexer.head_dim = 4
    indexer.rope_dim = 2
    indexer.softmax_scale = 0.5
    indexer.wq_b = _LinearResult(
        torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.float32)
    )
    indexer.wk_weights_proj = _LinearResult(
        torch.tensor([[10, 11, 12, 13, 20, 21]], dtype=torch.float32)
    )
    indexer.k_norm = torch.nn.LayerNorm(4)
    indexer.k_norm.weight.data.copy_(torch.tensor([1, 2, 3, 4]))
    indexer.k_norm.bias.data.copy_(torch.tensor([5, 6, 7, 8]))
    indexer.register_buffer("_lightop_k_norm_weight", None, persistent=False)
    indexer.register_buffer("_lightop_k_norm_bias", None, persistent=False)

    def fake_fuse_layernorm_rope(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
        weight_q,
        bias_q,
        weight_k,
        bias_k,
        residual_q,
        residual_k,
        epsilon,
    ) -> None:
        del (
            positions,
            head_size,
            cos_sin_cache,
            is_neox,
            weight_q,
            bias_q,
            residual_q,
            residual_k,
            epsilon,
        )
        query.add_(100)
        key.mul_(weight_k).add_(bias_k)

    monkeypatch.setattr(
        "lightop.attention.fuse_layernorm_rotary_embedding",
        fake_fuse_layernorm_rope,
    )
    rotary_emb = SimpleNamespace(cos_sin_cache=torch.empty(1, 2))

    _, query, key, weights = indexer.prepare_inputs_lightop(
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        torch.tensor([0], dtype=torch.int64),
        rotary_emb,
    )

    assert torch.equal(
        query,
        torch.tensor([[[102, 103, 100, 101], [106, 107, 104, 105]]]),
    )
    assert torch.equal(key, torch.tensor([[43, 60, 15, 28]]))
    assert torch.allclose(
        weights,
        torch.tensor([[20, 21]]) * (0.5 * 2**-0.5),
    )


@pytest.mark.parametrize(
    ("use_lightop", "expected"),
    [(True, 2.0), (False, 1.0)],
)
def test_hy_v4_indexer_routes_both_lightop_stages_together(
    use_lightop: bool,
    expected: float,
) -> None:
    """Catch mixing PE-first LightOp preparation with the old cache writer."""
    indexer = object.__new__(Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.use_lightop_indexer = use_lightop
    indexer.indexer_op = _ReturnIndexerQuery()

    def old_prepare(self, hidden_states, qr, positions, rotary_emb):
        del self, qr, positions, rotary_emb
        return hidden_states, torch.tensor(1.0), None, None

    def lightop_prepare(self, hidden_states, qr, positions, rotary_emb):
        del self, qr, positions, rotary_emb
        return hidden_states, torch.tensor(2.0), None, None

    indexer.prepare_inputs = MethodType(old_prepare, indexer)
    indexer.prepare_inputs_lightop = MethodType(lightop_prepare, indexer)

    result = indexer.forward(
        torch.empty(0), torch.empty(0), torch.empty(0), SimpleNamespace()
    )

    assert result.item() == expected
from vllm_hcu.models.hy_v4.hcu_sparse import (
    HYV4FlashMLASparseBackend,
    HYV4FlashMLASparseImpl,
)


@pytest.mark.parametrize("cache_dtype", ["fp8"])
def test_hy_v4_rejects_accuracy_unsafe_kv_cache_dtype(
    cache_dtype: str,
) -> None:
    with pytest.raises(RuntimeError, match="--kv-cache-dtype fp8_e4m3"):
        _require_accuracy_safe_kv_cache_dtype(cache_dtype)


@pytest.mark.parametrize(
    "cache_dtype", ["auto", "bfloat16", "fp8_e4m3", "fp8_ds_mla"]
)
def test_hy_v4_accepts_accuracy_safe_kv_cache_dtype(cache_dtype: str) -> None:
    _require_accuracy_safe_kv_cache_dtype(cache_dtype)


def test_hy_v4_normalizes_fp8_e4m3_for_sparse_flashmla_selection() -> None:
    assert (
        _normalize_hy_v4_kv_cache_dtype("fp8_e4m3", use_sparse=True)
        == "fp8_ds_mla"
    )


def test_hy_v4_preserves_fp8_e4m3_for_dense_flashmla_selection() -> None:
    assert (
        _normalize_hy_v4_kv_cache_dtype("fp8_e4m3", use_sparse=False)
        == "fp8_e4m3"
    )


@pytest.mark.parametrize("cache_dtype", ["auto", "bfloat16", "fp8_ds_mla"])
def test_hy_v4_preserves_native_kv_cache_dtype(cache_dtype: str) -> None:
    assert (
        _normalize_hy_v4_kv_cache_dtype(cache_dtype, use_sparse=True)
        == cache_dtype
    )


def test_hy_v4_mla_cache_spec_marks_fp8_as_quantized(monkeypatch) -> None:
    spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.MLAAttention.get_kv_cache_spec",
        lambda self, vllm_config: spec,
    )
    attention = object.__new__(HYV4MLAAttentionLayer)
    attention.kv_cache_dtype = "fp8_ds_mla"

    resolved = attention.get_kv_cache_spec(SimpleNamespace())

    assert resolved.kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
    assert resolved.page_size_bytes == 64 * 656


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


def test_shared_indexer_pattern_requires_a_preceding_full_producer() -> None:
    config = SimpleNamespace(
        index_topk=64,
        num_hidden_layers=3,
        indexer_types=["shared", "shared", "full"],
    )

    with pytest.raises(ValueError, match="preceding 'full'"):
        compute_skip_topk_layers(config)


def test_pipeline_stage_must_start_with_a_local_full_indexer() -> None:
    config = SimpleNamespace(
        index_topk=64,
        num_hidden_layers=6,
        indexer_types=["full", "shared", "shared", "full", "shared", "shared"],
    )

    require_local_indexer_producer(config, start_layer=0, end_layer=3)
    require_local_indexer_producer(config, start_layer=3, end_layer=6)
    with pytest.raises(ValueError, match="pipeline stage starts at shared"):
        require_local_indexer_producer(config, start_layer=2, end_layer=5)


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


def test_hcu_backend_advertises_sink_and_pcp_support() -> None:
    impl_cls = HYV4FlashMLASparseBackend.get_impl_cls()
    assert HYV4FlashMLASparseBackend.supports_sink()
    assert HYV4FlashMLASparseBackend.is_sparse()
    assert HYV4FlashMLASparseBackend.get_name() == "FLASHMLA_SPARSE"
    assert impl_cls is HYV4FlashMLASparseImpl
    assert impl_cls.supports_pcp is True


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
