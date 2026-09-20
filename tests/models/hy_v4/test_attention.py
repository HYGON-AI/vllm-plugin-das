# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import math
from types import MethodType, SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.v1.kv_cache_interface import KVQuantMode, MLAAttentionSpec

from vllm_hcu.models.hy_v4 import fp8_kv_dequant, hcu_sparse
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
    linear_gate_dp_block_tokens,
    linear_gate_dp_chunking_enabled,
    linear_gate_dp_shard_enabled,
    linear_gate_pcp_block_tokens,
    linear_gate_pcp_chunking_enabled,
    linear_gate_pcp_shard_enabled,
    _linear_gate_dp_group_size,
    _linear_gate_pcp_group_size,
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


def test_linear_gate_pcp_group_size_defaults_to_full_pcp(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_HCU_LINEAR_GATE_PCP_GROUP_SIZE", raising=False)
    assert _linear_gate_pcp_group_size(32) == 32
    assert _linear_gate_pcp_group_size(4) == 4


def test_linear_gate_pcp_group_size_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_HCU_LINEAR_GATE_PCP_GROUP_SIZE", "8")
    assert _linear_gate_pcp_group_size(32) == 8
    monkeypatch.setenv("VLLM_HCU_LINEAR_GATE_PCP_GROUP_SIZE", "4")
    assert _linear_gate_pcp_group_size(16) == 4
    monkeypatch.setenv("VLLM_HCU_LINEAR_GATE_PCP_GROUP_SIZE", "2")
    assert _linear_gate_pcp_group_size(8) == 2


@pytest.mark.parametrize("value", ["1", "3", "16", "not-an-integer"])
def test_linear_gate_pcp_rejects_invalid_group_size(monkeypatch, value) -> None:
    monkeypatch.setenv("VLLM_HCU_LINEAR_GATE_PCP_GROUP_SIZE", value)
    with pytest.raises(ValueError):
        _linear_gate_pcp_group_size(32)


def test_linear_gate_pcp_subgroups_split_pcp32_into_groups_of_8() -> None:
    from vllm_hcu.models.hy_v4.attention import _linear_gate_pcp_group_ranks

    groups = _linear_gate_pcp_group_ranks(
        world=32, dp_size=1, pp_size=1, pcp_size=32, tp_size=1, group_size=8
    )
    assert groups == [
        list(range(0, 8)),
        list(range(8, 16)),
        list(range(16, 24)),
        list(range(24, 32)),
    ]


def test_linear_gate_pcp_subgroups_support_pcp16_group4() -> None:
    from vllm_hcu.models.hy_v4.attention import _linear_gate_pcp_group_ranks

    groups = _linear_gate_pcp_group_ranks(
        world=16, dp_size=1, pp_size=1, pcp_size=16, tp_size=1, group_size=4
    )
    assert groups == [
        list(range(0, 4)),
        list(range(4, 8)),
        list(range(8, 12)),
        list(range(12, 16)),
    ]


def test_linear_gate_dp_sharding_flag_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_HCU_ENABLE_LINEAR_GATE_DP_SHARD", raising=False)
    assert linear_gate_dp_shard_enabled() is False
    monkeypatch.setenv("VLLM_HCU_ENABLE_LINEAR_GATE_DP_SHARD", "1")
    assert linear_gate_dp_shard_enabled() is True


def test_linear_gate_dp_group_size_is_configurable(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_HCU_LINEAR_GATE_DP_GROUP_SIZE", raising=False)
    assert _linear_gate_dp_group_size() == 8
    monkeypatch.setenv("VLLM_HCU_LINEAR_GATE_DP_GROUP_SIZE", "4")
    assert _linear_gate_dp_group_size() == 4
    monkeypatch.setenv("VLLM_HCU_LINEAR_GATE_DP_GROUP_SIZE", "2")
    assert _linear_gate_dp_group_size() == 2


@pytest.mark.parametrize("value", ["1", "3", "16", "not-an-integer"])
def test_linear_gate_dp_rejects_invalid_group_size(monkeypatch, value) -> None:
    monkeypatch.setenv("VLLM_HCU_LINEAR_GATE_DP_GROUP_SIZE", value)
    with pytest.raises(ValueError):
        _linear_gate_dp_group_size()


def test_linear_gate_dp_chunking_and_block_tokens(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_HCU_LINEAR_GATE_DP_CHUNKING", "1")
    monkeypatch.setenv("VLLM_HCU_LINEAR_GATE_DP_BLOCK_TOKENS", "1024")
    assert linear_gate_dp_chunking_enabled() is True
    assert linear_gate_dp_block_tokens() == 1024


def test_linear_gate_dp_subgroups_split_dp32_into_groups_of_8() -> None:
    from vllm_hcu.models.hy_v4.attention import _linear_gate_dp_group_ranks

    groups = _linear_gate_dp_group_ranks(
        world=32, dp_size=32, pp_size=1, pcp_size=1, tp_size=1, group_size=8
    )
    assert groups == [
        list(range(0, 8)),
        list(range(8, 16)),
        list(range(16, 24)),
        list(range(24, 32)),
    ]


def test_linear_gate_dp_subgroups_support_dp16_group4() -> None:
    from vllm_hcu.models.hy_v4.attention import _linear_gate_dp_group_ranks

    groups = _linear_gate_dp_group_ranks(
        world=16, dp_size=16, pp_size=1, pcp_size=1, tp_size=1, group_size=4
    )
    assert groups == [
        list(range(0, 4)),
        list(range(4, 8)),
        list(range(8, 12)),
        list(range(12, 16)),
    ]


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
    return impl


@pytest.mark.parametrize(
    ("num_tokens", "num_reqs", "expected"),
    [(8, 2, 4), (8, 8, 1)],
)
def test_uniform_tokens_per_request(
    num_tokens: int,
    num_reqs: int,
    expected: int,
) -> None:
    assert hcu_sparse._uniform_tokens_per_request(num_tokens, num_reqs) == expected


@pytest.mark.parametrize(("num_tokens", "num_reqs"), [(7, 2), (8, 0)])
def test_uniform_tokens_per_request_rejects_invalid_shape(
    num_tokens: int,
    num_reqs: int,
) -> None:
    with pytest.raises(ValueError, match="uniform query width|at least one request"):
        hcu_sparse._uniform_tokens_per_request(num_tokens, num_reqs)


def test_fp8_kv_dequant_prefers_lightop_and_passes_mtp3_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    def lightop_gather(
        cache,
        indices,
        output,
        valid_lengths,
        compact_indices,
        tokens_per_request,
    ):
        calls.append(
            (
                cache,
                indices,
                output,
                valid_lengths,
                compact_indices,
                tokens_per_request,
            )
        )
        output.fill_(2)
        compact_indices.copy_(
            torch.arange(indices.numel(), dtype=torch.int32).view_as(indices)
        )

    monkeypatch.setattr(
        fp8_kv_dequant,
        "_resolve_lightop_gather",
        lambda: lightop_gather,
        raising=False,
    )
    monkeypatch.setattr(
        fp8_kv_dequant,
        "_gather_dequantize_fp8_ds_mla_kernel",
        pytest.fail,
    )
    cache = torch.zeros((2, 64, 656), dtype=torch.uint8)
    indices = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [6, -1]], dtype=torch.int64
    )

    output, compact_indices = (
        fp8_kv_dequant.gather_dequantize_fp8_ds_mla_cache(
            cache,
            indices,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            tokens_per_request=4,
        )
    )

    assert len(calls) == 1
    assert calls[0][1].dtype == torch.int32
    assert calls[0][3].tolist() == [2, 2, 2, 2]
    assert calls[0][5] == 4
    assert output.shape == (8, 576)
    assert output.eq(2).all()
    torch.testing.assert_close(
        compact_indices,
        torch.arange(8, dtype=torch.int32).view(4, 2),
    )


def test_fp8_kv_dequant_reuses_lightop_mapping_for_shared_indexer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, bool]] = []

    def lightop_gather(
        cache,
        indices,
        output,
        valid_lengths,
        compact_indices,
        tokens_per_request=1,
        dedup_table=None,
        reuse_compact_indices=False,
    ):
        del cache, valid_lengths, dedup_table
        calls.append((compact_indices.data_ptr(), reuse_compact_indices))
        output.fill_(len(calls))
        if not reuse_compact_indices:
            compact_indices.copy_(
                torch.arange(indices.numel(), dtype=torch.int32).view_as(indices)
            )

    monkeypatch.setattr(
        fp8_kv_dequant,
        "_resolve_lightop_gather",
        lambda: lightop_gather,
    )
    cache = torch.zeros((2, 64, 656), dtype=torch.uint8)
    indices = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [6, -1]], dtype=torch.int32
    )
    state = fp8_kv_dequant.LightOpKVReuseState(
        compact_indices=torch.empty((8, 2), dtype=torch.int32)
    )

    first_output, first_indices = fp8_kv_dequant.gather_dequantize_fp8_ds_mla_cache(
        cache,
        indices,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        tokens_per_request=4,
        reuse_state=state,
        allow_mapping_reuse=False,
        mapping_reuse_group_size=4,
    )
    second_output, second_indices = (
        fp8_kv_dequant.gather_dequantize_fp8_ds_mla_cache(
            cache,
            indices,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            tokens_per_request=4,
            reuse_state=state,
            allow_mapping_reuse=True,
            mapping_reuse_group_size=4,
        )
    )

    assert calls == [(first_indices.data_ptr(), False), (first_indices.data_ptr(), True)]
    assert second_indices.data_ptr() == first_indices.data_ptr()
    assert first_output.eq(1).all()
    assert second_output.eq(2).all()
    torch.testing.assert_close(
        second_indices,
        torch.arange(8, dtype=torch.int32).view(4, 2),
    )


def test_fp8_kv_dequant_decode_resets_lightop_mapping_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reuse_flags: list[bool] = []

    def lightop_gather(
        cache,
        indices,
        output,
        valid_lengths,
        compact_indices,
        tokens_per_request=1,
        dedup_table=None,
        reuse_compact_indices=False,
    ):
        del cache, valid_lengths, dedup_table
        reuse_flags.append(reuse_compact_indices)
        output.zero_()
        compact_indices.copy_(
            torch.arange(indices.numel(), dtype=torch.int32).view_as(indices)
        )

    monkeypatch.setattr(
        fp8_kv_dequant,
        "_resolve_lightop_gather",
        lambda: lightop_gather,
    )
    cache = torch.zeros((1, 64, 656), dtype=torch.uint8)
    indices = torch.tensor([[0, 1]], dtype=torch.int32)
    state = fp8_kv_dequant.LightOpKVReuseState(
        compact_indices=torch.empty((4, 2), dtype=torch.int32)
    )
    state.dedup_key = (1, cache.numel(), 1)

    fp8_kv_dequant.gather_dequantize_fp8_ds_mla_cache(
        cache,
        indices,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        tokens_per_request=1,
        reuse_state=state,
        allow_mapping_reuse=True,
    )

    assert reuse_flags == [False]
    assert state.dedup_key is None


def test_fp8_kv_dequant_falls_back_to_old_lightop_mapping_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def old_lightop_gather(
        cache,
        indices,
        output,
        valid_lengths,
        compact_indices,
        tokens_per_request=1,
    ):
        del cache, valid_lengths
        calls.append(tokens_per_request)
        output.zero_()
        compact_indices.copy_(
            torch.arange(indices.numel(), dtype=torch.int32).view_as(indices)
        )

    monkeypatch.setattr(
        fp8_kv_dequant,
        "_resolve_lightop_gather",
        lambda: old_lightop_gather,
    )
    cache = torch.zeros((1, 64, 656), dtype=torch.uint8)
    indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    state = fp8_kv_dequant.LightOpKVReuseState(
        compact_indices=torch.empty((2, 2), dtype=torch.int32),
        dedup_key=(2, cache.numel(), 2),
    )

    fp8_kv_dequant.gather_dequantize_fp8_ds_mla_cache(
        cache,
        indices,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        tokens_per_request=2,
        reuse_state=state,
        allow_mapping_reuse=True,
        mapping_reuse_group_size=2,
    )

    assert calls == [2]
    assert state.supports_mapping_reuse is False
    assert state.dedup_key is None


def test_lightop_kv_reuse_state_preallocates_from_topk_buffer() -> None:
    topk_indices = torch.empty((8, 2048), dtype=torch.int32)

    state = fp8_kv_dequant.LightOpKVReuseState.from_topk_buffer(topk_indices)

    assert state.compact_indices.shape == topk_indices.shape
    assert state.compact_indices.dtype == torch.int32
    assert state.compact_indices.device == topk_indices.device
    assert state.compact_indices.data_ptr() != topk_indices.data_ptr()
    assert state.dedup_key is None


@pytest.mark.parametrize(
    ("is_prefilling", "num_tokens", "expected_group_size"),
    [
        ([False, False], 8, 4),
        ([True, True], 8, 1),
        ([False, True], 8, 1),
        ([False, False], 7, 1),
    ],
)
def test_lightop_mapping_reuse_only_marks_uniform_target_verify(
    is_prefilling: list[bool],
    num_tokens: int,
    expected_group_size: int,
) -> None:
    metadata = SimpleNamespace(
        num_reqs=2,
        max_query_len=4,
        num_actual_tokens=num_tokens,
    )

    group_size = hcu_sparse._lightop_mapping_reuse_group_size(
        metadata,
        torch.tensor(is_prefilling, dtype=torch.bool),
    )

    assert group_size == expected_group_size


@pytest.mark.parametrize(
    ("batch_size", "query_len", "expected_tokens_per_request"),
    [(2, 4, 4), (8, 1, 1)],
)
def test_fp8_kv_dequant_uses_runtime_query_width(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    query_len: int,
    expected_tokens_per_request: int,
) -> None:
    impl = _bare_impl(None)
    observed: list[int] = []

    def gather(cache, indices, kv_lora_rank, rope_dim, tokens_per_request):
        observed.append(tokens_per_request)
        num_rows = indices.numel()
        return (
            torch.zeros((num_rows, kv_lora_rank + rope_dim), dtype=torch.bfloat16),
            torch.arange(num_rows, dtype=torch.int32).view_as(indices),
        )

    monkeypatch.setattr(hcu_sparse, "gather_dequantize_fp8_ds_mla_cache", gather)
    monkeypatch.setattr(
        impl,
        "_bf16_flash_mla_kernel",
        lambda q, cache, indices: torch.zeros(
            (q.shape[0], q.shape[1], impl.kv_lora_rank), dtype=q.dtype
        ),
    )
    q = torch.zeros((batch_size, query_len, 4, 576), dtype=torch.bfloat16)
    cache = torch.zeros((1, 64, 656), dtype=torch.uint8)
    indices = torch.zeros((batch_size, query_len, 2), dtype=torch.int32)

    impl._dequant_bf16_attn(q, cache, indices)

    assert observed == [expected_tokens_per_request]


@pytest.mark.parametrize(
    ("is_indexer_producer", "expected_allow_reuse"),
    [(True, False), (False, True)],
)
def test_fp8_dequant_only_allows_mapping_reuse_for_shared_indexer(
    monkeypatch: pytest.MonkeyPatch,
    is_indexer_producer: bool,
    expected_allow_reuse: bool,
) -> None:
    impl = _bare_impl(None)
    state = object()
    impl._lightop_kv_reuse_state = state
    impl._is_indexer_producer = is_indexer_producer
    observed: list[tuple[object, bool, int]] = []

    def gather(
        cache,
        indices,
        kv_lora_rank,
        rope_dim,
        tokens_per_request,
        *,
        reuse_state=None,
        allow_mapping_reuse=False,
        mapping_reuse_group_size=1,
    ):
        del cache, kv_lora_rank, rope_dim, tokens_per_request
        observed.append(
            (reuse_state, allow_mapping_reuse, mapping_reuse_group_size)
        )
        num_rows = indices.numel()
        return (
            torch.zeros((num_rows, 576), dtype=torch.bfloat16),
            torch.arange(num_rows, dtype=torch.int32).view_as(indices),
        )

    monkeypatch.setattr(hcu_sparse, "gather_dequantize_fp8_ds_mla_cache", gather)
    monkeypatch.setattr(
        impl,
        "_bf16_flash_mla_kernel",
        lambda q, cache, indices: torch.zeros(
            (q.shape[0], q.shape[1], impl.kv_lora_rank), dtype=q.dtype
        ),
    )

    impl._dequant_bf16_attn(
        torch.zeros((2, 4, 4, 576), dtype=torch.bfloat16),
        torch.zeros((1, 64, 656), dtype=torch.uint8),
        torch.zeros((2, 4, 2), dtype=torch.int32),
        mapping_reuse_group_size=4,
    )

    assert observed == [(state, expected_allow_reuse, 4)]


@pytest.mark.parametrize(
    ("num_reqs", "max_query_len", "num_tokens", "expected_width"),
    [(2, 4, 8, 4), (2, 4, 7, 1)],
)
def test_fp8_mixed_batch_passes_request_width_to_dequant(
    monkeypatch: pytest.MonkeyPatch,
    num_reqs: int,
    max_query_len: int,
    num_tokens: int,
    expected_width: int,
) -> None:
    impl = _bare_impl(None)
    observed: list[int | None] = []

    monkeypatch.setattr(hcu_sparse.henvs, "VLLM_HCU_HYV4_FP8_KV_DEQUANT", True)
    monkeypatch.setattr(
        hcu_sparse,
        "triton_convert_req_index_to_global_index",
        lambda req_ids, block_table, indices, **kwargs: indices,
    )

    def fake_dequant(
        self,
        q,
        cache,
        indices,
        tokens_per_request=None,
        mapping_reuse_group_size=1,
    ):
        assert mapping_reuse_group_size == 1
        observed.append(tokens_per_request)
        return torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2], self.kv_lora_rank),
            dtype=q.dtype,
        ), None

    impl._dequant_bf16_attn = MethodType(fake_dequant, impl)
    q = torch.zeros((num_tokens, 4, 576), dtype=torch.bfloat16)
    cache = torch.zeros((1, 64, 656), dtype=torch.uint8)
    indices = torch.zeros((num_tokens, 2), dtype=torch.int32)
    metadata = SimpleNamespace(
        num_reqs=num_reqs,
        max_query_len=max_query_len,
        num_actual_tokens=num_tokens,
        req_id_per_token=torch.arange(num_tokens, dtype=torch.int32),
        block_table=torch.zeros((num_reqs, 1), dtype=torch.int32),
        block_size=64,
    )

    output = impl._forward_fp8_kv_mixed_batch(q, cache, indices, metadata)

    assert output.shape == (num_tokens, 4, impl.kv_lora_rank)
    assert observed == [expected_width]


def test_fp8_kv_dequant_falls_back_when_lightop_gather_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launches: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class FakeKernel:
        def __getitem__(self, grid):
            assert grid == (4,)

            def launch(*args, **kwargs):
                launches.append((args, kwargs))
                args[2].fill_(3)

            return launch

    monkeypatch.setattr(fp8_kv_dequant, "_resolve_lightop_gather", lambda: None)
    monkeypatch.setattr(
        fp8_kv_dequant,
        "_gather_dequantize_fp8_ds_mla_kernel",
        FakeKernel(),
    )
    cache = torch.zeros((1, 64, 656), dtype=torch.uint8)
    indices = torch.tensor([[3, -1], [7, 8]], dtype=torch.int64)

    output, compact_indices = (
        fp8_kv_dequant.gather_dequantize_fp8_ds_mla_cache(
            cache,
            indices,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            tokens_per_request=4,
        )
    )

    assert len(launches) == 1
    assert launches[0][0][1].dtype == torch.int32
    assert output.shape == (4, 576)
    assert output.eq(3).all()
    torch.testing.assert_close(
        compact_indices,
        torch.tensor([[0, -1], [2, 3]], dtype=torch.int32),
    )


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
    state = object()
    impl._lightop_kv_reuse_state = state
    impl._is_indexer_producer = False
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

    def fake_dequant(
        cache,
        indices,
        kv_lora_rank,
        rope_dim,
        tokens_per_request,
        *,
        reuse_state=None,
        allow_mapping_reuse=False,
        mapping_reuse_group_size=1,
    ):
        calls["dequant"] = (
            cache,
            indices,
            kv_lora_rank,
            rope_dim,
            tokens_per_request,
            reuse_state,
            allow_mapping_reuse,
            mapping_reuse_group_size,
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
        num_reqs=2,
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
    assert calls["dequant"][4] == 1
    assert calls["dequant"][5:] == (state, False, 1)
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
