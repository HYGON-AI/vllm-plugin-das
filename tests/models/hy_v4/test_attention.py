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


from vllm_hcu.models.hy_v4.hcu_sparse import (
    HYV4FlashMLASparseBackend,
    HYV4FlashMLASparseImpl,
)


@pytest.mark.hcu
@pytest.mark.parametrize("use_ue8m0", [False, True])
def test_hyv4_group_fp8_quant_returns_values_and_fp32_scales(use_ue8m0):
    from vllm_hcu.models.hy_v4.attention import per_token_group_quant_fp8

    if not torch.cuda.is_available():
        pytest.skip("a live HCU/ROCm device is required")
    x = torch.linspace(-3, 3, 4 * 256, device="cuda").reshape(4, 256).bfloat16()
    x[0].zero_()
    q, scale = per_token_group_quant_fp8(x, group_size=128, use_ue8m0=use_ue8m0)
    assert q.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    assert scale.dtype == torch.float32 and scale.shape == (4, 2)
    assert torch.isfinite(q.float()).all() and torch.isfinite(scale).all()
    assert (scale > 0).all()
    actual = q.float() * scale.repeat_interleave(128, dim=-1)
    torch.testing.assert_close(actual, x.float(), rtol=0.08, atol=0.08)
    if use_ue8m0:
        torch.testing.assert_close(scale.log2(), scale.log2().round())


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
    with pytest.raises(ValueError, match="full"):
        require_local_indexer_producer(config, start_layer=2, end_layer=5)
    with pytest.raises(ValueError, match="Invalid HY V4 pipeline layer range"):
        require_local_indexer_producer(config, start_layer=0, end_layer=99)


@pytest.mark.parametrize(
    "layer_types,indexer_types,start_layer,end_layer,missing_producer",
    [
        (["full_attention", "sparse_attention", "sparse_attention"],
         ["full", "shared", "shared"], 0, 3, True),
        (["sparse_attention", "full_attention", "sparse_attention"],
         ["full", "full", "shared"], 1, 3, True),
        (["sparse_attention", "full_attention", "sparse_attention"],
         ["full", "full", "shared"], 0, 3, False),
        (["full_attention", "sparse_attention", "sparse_attention"],
         ["full", "full", "shared"], 0, 3, False),
        (["full_attention", "full_attention", "sparse_attention"],
         ["full", "shared", "full"], 1, 3, False),
        (["full_attention", "full_attention", "full_attention"],
         ["full", "shared", "shared"], 1, 3, False),
    ],
)
def test_mixed_dense_sparse_stage_requires_actual_local_indexer_producer(
    layer_types, indexer_types, start_layer, end_layer, missing_producer,
):
    config = SimpleNamespace(
        index_topk=64,
        num_hidden_layers=3,
        indexer_types=indexer_types,
        layer_types=layer_types,
    )
    if missing_producer:
        with pytest.raises(ValueError, match="local.*full.*sparse.*producer"):
            require_local_indexer_producer(
                config, start_layer=start_layer, end_layer=end_layer)
    else:
        require_local_indexer_producer(
            config, start_layer=start_layer, end_layer=end_layer)


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

    with pytest.raises(ValueError, match="attention sink"):
        require_hyv4_sink_backend(SinkIncapableSparseBackend)


def test_hcu_backend_advertises_sink_support() -> None:
    impl_cls = HYV4FlashMLASparseBackend.get_impl_cls()
    assert HYV4FlashMLASparseBackend.supports_sink()
    assert HYV4FlashMLASparseBackend.is_sparse()
    assert HYV4FlashMLASparseBackend.get_name() == "FLASHMLA_SPARSE"
    assert impl_cls is HYV4FlashMLASparseImpl
    assert require_hyv4_sink_backend(HYV4FlashMLASparseBackend) is HYV4FlashMLASparseBackend


def test_hyv4_sparse_backend_passes_current_pcp_capability_gate(monkeypatch):
    from vllm.v1.worker import cp_utils

    impl = object.__new__(HYV4FlashMLASparseBackend.get_impl_cls())
    monkeypatch.setattr(cp_utils, "get_layers_from_vllm_config", lambda *args: {
        "model.layers.41.self_attn.attn": SimpleNamespace(impl=impl),
    })
    cp_utils.check_attention_cp_compatibility(SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=4, decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        speculative_config=None,
    ))


def test_hyv4_pp2_pcp4_partition_retains_a_local_full_indexer_producer():
    config = SimpleNamespace(
        index_topk=64, num_hidden_layers=78,
        layer_types=["sparse_attention"] * 78,
        indexer_types=["full"] + ["shared"] * 40 + ["full"] + ["shared"] * 36,
    )
    require_local_indexer_producer(config, start_layer=0, end_layer=41)
    require_local_indexer_producer(config, start_layer=41, end_layer=78)
    with pytest.raises(ValueError, match="local.*full.*sparse.*producer"):
        require_local_indexer_producer(config, start_layer=40, end_layer=78)
    config.layer_types[41] = "full_attention"
    with pytest.raises(ValueError, match="local.*full.*sparse.*producer"):
        require_local_indexer_producer(config, start_layer=41, end_layer=78)


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


def test_mtp_topk_union_stats_reports_per_request_reuse() -> None:
    topk_indices = torch.tensor(
        [
            [1, 2, 3, -1],
            [2, 3, 4, -1],
            [1, 4, 5, -1],
            [1, 2, 5, -1],
            [10, 11, -1, -1],
            [10, 12, -1, -1],
        ],
        dtype=torch.int32,
    )
    query_start_loc = torch.tensor([0, 4, 6], dtype=torch.int32)

    valid_counts, union_sizes = hcu_sparse._mtp_topk_union_stats(
        topk_indices, query_start_loc
    )

    assert valid_counts == [12, 4]
    assert union_sizes == [5, 3]


def test_mtp_topk_union_stats_rejects_invalid_request_boundaries() -> None:
    with pytest.raises(ValueError, match="query_start_loc"):
        hcu_sparse._mtp_topk_union_stats(
            torch.zeros(4, 8, dtype=torch.int32),
            torch.tensor([1, 5], dtype=torch.int32),
        )


def test_mtp_topk_union_profiler_skips_then_records(monkeypatch) -> None:
    impl = _bare_impl(None)
    impl._mtp_topk_profile_steps = 1
    impl._mtp_topk_profile_skip_steps = 1
    impl.topk_indices_buffer = torch.tensor(
        [[1, 2], [2, 3], [1, 3], [1, 2]], dtype=torch.int32
    )
    metadata = SimpleNamespace(
        max_query_len=4,
        num_actual_tokens=4,
        num_reqs=1,
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
    )
    logged: list[tuple[object, ...]] = []
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        hcu_sparse.logger,
        "info",
        lambda _message, *args: logged.append(args),
    )

    impl._maybe_profile_mtp_topk_union(metadata)
    assert logged == []
    assert impl._mtp_topk_profile_skip_steps == 0

    impl._maybe_profile_mtp_topk_union(metadata)
    assert logged == [("unknown", [4], [8], [3], 8 / 3)]
    assert impl._mtp_topk_profile_steps == 0


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


def test_sink_changes_softmax_denominator_with_torch_reference(monkeypatch):
    sinks = torch.tensor([0.0, 1.0986122886681098, float("-inf"), 0.0])
    impl = _bare_impl(sinks)
    def reference_kernel(q, kv, indices, scale, attn_sink=None, topk_length=None):
        # A single zero-score token has value 4. Appending the sink to
        # softmax gives output 4 / (1 + exp(sink)).
        scores = torch.stack([torch.zeros_like(attn_sink), attn_sink], dim=-1)
        value_weights = torch.softmax(scores, dim=-1)[:, 0]
        return (4 * value_weights[None, :, None].expand(q.shape[0], -1, 512),)
    monkeypatch.setattr(hcu_sparse, "flash_mla_sparse_fwd", reference_kernel)
    actual = impl._bf16_flash_mla_kernel(
        torch.zeros(1, 4, 576), torch.zeros(1, 576), torch.zeros(1, 1, dtype=torch.int32))
    torch.testing.assert_close(actual[0, :, 0], torch.tensor([2.0, 1.0, 4.0, 2.0]))


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
