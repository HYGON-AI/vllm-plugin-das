# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import math
from types import MethodType, SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
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
    linear_gate_pcp_block_tokens,
    linear_gate_pcp_chunking_enabled,
    linear_gate_pcp_shard_enabled,
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


def test_linear_gate_pcp_chunking_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_HCU_LINEAR_GATE_PCP_CHUNKING", "0")
    assert linear_gate_pcp_chunking_enabled() is False


def test_linear_gate_pcp_block_tokens_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_HCU_LINEAR_GATE_PCP_BLOCK_TOKENS", "1024")
    assert linear_gate_pcp_block_tokens() == 1024


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_linear_gate_pcp_rejects_invalid_block_tokens(monkeypatch, value) -> None:
    monkeypatch.setenv("VLLM_HCU_LINEAR_GATE_PCP_BLOCK_TOKENS", value)
    with pytest.raises(ValueError, match="positive integer"):
        linear_gate_pcp_block_tokens()


def test_linear_gate_pcp_sharding_flag_remains_independent(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_HCU_ENABLE_LINEAR_GATE_PCP_SHARD", "1")
    assert linear_gate_pcp_shard_enabled() is True


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


def test_hy_v4_mla_layer_runs_backend_post_load_hook(monkeypatch) -> None:
    events: list[object] = []

    def fake_layer_process(self, act_dtype):
        events.append(("layer", self, act_dtype))

    class FakeImpl:
        def process_weights_after_loading(self, act_dtype):
            events.append(("impl", self, act_dtype))

    monkeypatch.setattr(
        MLAAttention,
        "process_weights_after_loading",
        fake_layer_process,
    )
    layer = object.__new__(HYV4MLAAttentionLayer)
    layer.impl = FakeImpl()

    layer.process_weights_after_loading(torch.bfloat16)

    assert events == [
        ("layer", layer, torch.bfloat16),
        ("impl", layer.impl, torch.bfloat16),
    ]


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
    assert impl_cls.can_return_lse_for_decode is True
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
    impl._dcp_sinks = None
    impl.num_heads = 4
    impl.prefill_padding = 64
    impl.fp8_decode_padded_heads = 64
    impl.softmax_scale = 0.5
    impl.dcp_world_size = 1
    impl.dcp_rank = 0
    impl.kv_cache_dtype = "auto"
    impl.head_size = 576
    impl.kv_lora_rank = 512
    impl.tokens_per_request = 1
    return impl


def test_dcp_gathers_sink_once_after_weights_load(monkeypatch) -> None:
    sinks = torch.arange(4, dtype=torch.float32)
    impl = _bare_impl(sinks)
    impl.dcp_world_size = 2
    events: list[object] = []

    def fake_parent_process(self, act_dtype):
        events.append(("parent", self, act_dtype))

    class FakeDcpGroup:
        def all_gather(self, tensor, dim):
            events.append(("gather", tensor, dim))
            return torch.cat((tensor, tensor + 10), dim=dim)

    monkeypatch.setattr(
        hcu_sparse.FlashMLASparseImpl,
        "process_weights_after_loading",
        fake_parent_process,
    )
    monkeypatch.setattr(hcu_sparse, "get_dcp_group", lambda: FakeDcpGroup())

    impl.process_weights_after_loading(torch.bfloat16)

    assert events == [
        ("parent", impl, torch.bfloat16),
        ("gather", sinks, 0),
    ]
    assert impl._dcp_sinks is not None
    torch.testing.assert_close(
        impl._dcp_sinks,
        torch.tensor([0, 1, 2, 3, 10, 11, 12, 13], dtype=torch.float32),
    )

    padded = impl._sinks_for_query(
        torch.zeros(2, 8, 576),
        head_dim=1,
        kernel_heads=64,
    )
    assert padded is not None
    torch.testing.assert_close(
        padded[:8],
        impl._dcp_sinks - math.log(2),
    )
    assert torch.isneginf(padded[8:]).all()


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
        return (
            torch.zeros(q.shape[0], q.shape[1], 512),
            torch.empty(0),
            torch.zeros(q.shape[0], q.shape[1]),
        )

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
        return (
            4 * value_weights[None, :, None].expand(q.shape[0], -1, 512),
            torch.empty(0),
            torch.zeros(q.shape[0], q.shape[1]),
        )
    monkeypatch.setattr(hcu_sparse, "flash_mla_sparse_fwd", reference_kernel)
    actual = impl._bf16_flash_mla_kernel(
        torch.zeros(1, 4, 576), torch.zeros(1, 576), torch.zeros(1, 1, dtype=torch.int32))
    torch.testing.assert_close(actual[0, :, 0], torch.tensor([2.0, 1.0, 4.0, 2.0]))


def test_bf16_dcp_kernel_preserves_and_slices_lse(monkeypatch) -> None:
    impl = _bare_impl(torch.arange(4, dtype=torch.float32))
    impl.dcp_world_size = 2
    impl._dcp_sinks = torch.arange(8, dtype=torch.float32)
    captured: dict[str, torch.Tensor | None] = {}

    def fake_sparse_fwd(q, kv, indices, scale, attn_sink=None, topk_length=None):
        del kv, indices, scale, topk_length
        captured["attn_sink"] = attn_sink
        return (
            torch.arange(q.shape[0] * q.shape[1] * 3, dtype=torch.float32).view(
                q.shape[0], q.shape[1], 3
            ),
            torch.empty(0),
            torch.arange(q.shape[0] * q.shape[1], dtype=torch.float32).view(
                q.shape[0], q.shape[1]
            ),
        )

    monkeypatch.setattr(hcu_sparse, "flash_mla_sparse_fwd", fake_sparse_fwd)
    output, lse = impl._bf16_flash_mla_kernel_with_lse(
        q=torch.zeros(2, 8, 576),
        kv_c_and_k_pe_cache=torch.zeros(8, 576),
        topk_indices=torch.zeros(2, 4, dtype=torch.int32),
        topk_length=torch.tensor([4, 2], dtype=torch.int32),
    )

    assert output.shape == (2, 8, 3)
    assert lse.shape == (2, 8)
    assert captured["attn_sink"] is not None
    raw_lse = torch.arange(2 * 64, dtype=torch.float32).view(2, 64)[:, :8]
    normalized_sink = impl._dcp_sinks - math.log(2)
    torch.testing.assert_close(
        lse,
        torch.logaddexp(raw_lse, normalized_sink.view(1, -1)),
    )
    torch.testing.assert_close(
        captured["attn_sink"][:8],
        normalized_sink,
    )


def test_bf16_dcp_kernel_rejects_missing_lse(monkeypatch) -> None:
    impl = _bare_impl(torch.arange(4, dtype=torch.float32))

    def fake_sparse_fwd(q, *args, **kwargs):
        del args, kwargs
        return torch.zeros(q.shape[0], q.shape[1], 3), torch.empty(0), None

    monkeypatch.setattr(hcu_sparse, "flash_mla_sparse_fwd", fake_sparse_fwd)
    with pytest.raises(RuntimeError, match="did not return LSE"):
        impl._bf16_flash_mla_kernel_with_lse(
            q=torch.zeros(2, 4, 576),
            kv_c_and_k_pe_cache=torch.zeros(8, 576),
            topk_indices=torch.zeros(2, 4, dtype=torch.int32),
        )


def test_fp8_dcp_localizes_dequantizes_and_masks_empty_rows(monkeypatch) -> None:
    impl = _bare_impl(torch.arange(4, dtype=torch.float32))
    impl.dcp_world_size = 2
    impl.dcp_rank = 1
    impl.kv_cache_dtype = "fp8_ds_mla"
    impl._dcp_sinks = torch.arange(8, dtype=torch.float32)
    impl.topk_indices_buffer = torch.tensor(
        [[0, 1, 2, 3], [0, 2, 4, 6]], dtype=torch.int32
    )
    localized_indices = torch.tensor(
        [[112, 113, -1, -1], [-1, -1, -1, -1]], dtype=torch.int32
    )
    topk_length = torch.tensor([2, 0], dtype=torch.int32)
    calls: dict[str, object] = {}

    def fake_filter(req_ids, block_table, indices, **kwargs):
        calls["filter"] = (req_ids, block_table, indices, kwargs)
        return localized_indices, topk_length

    def fake_dequant(cache, indices, kv_lora_rank, rope_dim, tokens_per_request):
        calls["dequant"] = (
            cache,
            indices,
            kv_lora_rank,
            rope_dim,
            tokens_per_request,
        )
        return torch.zeros(8, 576), torch.tensor(
            [[0, 1, -1, -1], [-1, -1, -1, -1]], dtype=torch.int32
        )

    kernel_output = torch.arange(48, dtype=torch.float32).view(2, 8, 3)
    kernel_lse = torch.arange(16, dtype=torch.float32).view(2, 8)

    def fake_kernel(self, q, cache, indices, topk_length=None):
        calls["kernel"] = (q, cache, indices, topk_length)
        return kernel_output.clone(), kernel_lse.clone()

    monkeypatch.setattr(hcu_sparse, "triton_filter_and_convert_dcp_index", fake_filter)
    monkeypatch.setattr(
        hcu_sparse,
        "gather_dequantize_fp8_ds_mla_cache",
        fake_dequant,
    )
    impl._bf16_flash_mla_kernel_with_lse = MethodType(fake_kernel, impl)
    q = torch.zeros(2, 8, 576)
    fp8_cache = torch.zeros(16, 656, dtype=torch.uint8)
    metadata = SimpleNamespace(
        req_id_per_token=torch.tensor([0, 1], dtype=torch.int32),
        block_table=torch.tensor([[7], [11]], dtype=torch.int32),
        block_size=64,
        cp_kv_cache_interleave_size=1,
    )

    output, lse = impl.forward_mqa(q, fp8_cache, metadata, object())

    filter_call = calls["filter"]
    assert filter_call[3] == {
        "dcp_size": 2,
        "dcp_rank": 1,
        "cp_kv_cache_interleave_size": 1,
        "BLOCK_SIZE": 64,
        "NUM_TOPK_TOKENS": 4,
        "return_valid_counts": True,
    }
    assert calls["dequant"][1] is localized_indices
    assert calls["kernel"][3] is topk_length
    torch.testing.assert_close(output[0], kernel_output[0])
    torch.testing.assert_close(output[1], torch.zeros_like(output[1]))
    torch.testing.assert_close(lse[0], kernel_lse[0])
    torch.testing.assert_close(lse[1], kernel_lse[1])

    impl.sinks = None
    impl._dcp_sinks = None
    _, sink_free_lse = impl.forward_mqa(q, fp8_cache, metadata, object())
    assert torch.isneginf(sink_free_lse[1]).all()


def test_dcp_size_one_delegates_to_upstream_forward(monkeypatch) -> None:
    impl = _bare_impl(torch.arange(4, dtype=torch.float32))
    expected = (torch.ones(1, 4, 3), None)
    calls: list[tuple[object, ...]] = []

    def fake_parent_forward(self, q, cache, metadata, layer):
        calls.append((self, q, cache, metadata, layer))
        return expected

    monkeypatch.setattr(
        hcu_sparse.FlashMLASparseImpl,
        "forward_mqa",
        fake_parent_forward,
    )
    q = torch.zeros(1, 4, 576)
    cache = torch.zeros(2, 576)
    metadata = object()
    layer = object()

    assert impl.forward_mqa(q, cache, metadata, layer) is expected
    assert calls == [(impl, q, cache, metadata, layer)]


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
