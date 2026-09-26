# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import KVQuantMode


def test_hyv4_mla_backend_post_load_hook_runs_once(monkeypatch):
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm_hcu.models.hy_v4.attention import HYV4MLAAttentionLayer

    layer = object.__new__(HYV4MLAAttentionLayer)
    torch.nn.Module.__init__(layer)
    calls = []
    layer.impl = SimpleNamespace(
        process_weights_after_loading=lambda dtype: calls.append(dtype)
    )
    monkeypatch.setattr(
        MLAAttention,
        "process_weights_after_loading",
        lambda self, dtype: self.impl.process_weights_after_loading(dtype),
    )
    layer.process_weights_after_loading(torch.bfloat16)
    assert calls == [torch.bfloat16]


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


@pytest.mark.parametrize(
    "tp,pcp,dcp,pp,dp,ep,supported",
    [
        (8, 1, 1, 1, 1, False, True),
        (4, 1, 1, 1, 2, True, True),
        (1, 1, 1, 1, 8, True, True),
        (4, 2, 1, 1, 1, True, True),
        (1, 4, 1, 2, 1, True, True),
        (8, 1, 2, 1, 1, True, True),
        (2, 1, 2, 1, 4, True, True),
        (2, 2, 1, 1, 1, True, False),
        (8, 2, 1, 1, 1, True, False),
        (4, 2, 2, 1, 1, True, False),
        (8, 2, 2, 1, 1, True, False),
        (4, 1, 2, 1, 1, True, False),
        (8, 1, 4, 1, 1, True, False),
        (8, 1, 2, 2, 1, True, False),
        (8, 1, 2, 1, 2, True, False),
        (4, 2, 1, 1, 2, True, False),
        (1, 4, 1, 2, 2, True, False),
        (4, 2, 1, 1, 1, False, False),
        (1, 4, 1, 2, 1, False, False),
    ],
)
def test_hy4_context_parallel_topology_contract(tp, pcp, dcp, pp, dp, ep, supported):
    from vllm_hcu.models.hy_v4.attention import (
        _require_supported_hy_v4_parallelism,
    )

    config = SimpleNamespace(
        tensor_parallel_size=tp,
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=dcp,
        pipeline_parallel_size=pp,
        data_parallel_size=dp,
        enable_expert_parallel=ep,
    )
    if supported:
        _require_supported_hy_v4_parallelism(config)
    else:
        with pytest.raises(RuntimeError, match="context parallel"):
            _require_supported_hy_v4_parallelism(config)


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


def test_hy4_pp2_stage_41_starts_with_local_sparse_indexer_producer() -> None:
    from vllm_hcu.models.hy_v4.attention import require_local_indexer_producer

    indexer_types = ["full"] * 78
    indexer_types[40] = "shared"
    indexer_types[42] = "shared"
    config = SimpleNamespace(
        index_topk=2048,
        num_hidden_layers=78,
        indexer_types=indexer_types,
        layer_types=["deepseek_sparse_attention"] * 78,
    )

    require_local_indexer_producer(config, start_layer=41, end_layer=78)
    with pytest.raises(ValueError, match="preceding local 'full'"):
        require_local_indexer_producer(config, start_layer=40, end_layer=78)


def test_hy4_dcp_sparse_e4m3_cache_spec_keeps_tp8_page_geometry():
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
            parallel_config=SimpleNamespace(
                tensor_parallel_size=8,
                decode_context_parallel_size=2,
                cp_kv_cache_interleave_size=1,
            ),
        )
    )

    assert resolved.kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
    assert resolved.state_content_size_bytes == 656
    assert resolved.page_size_bytes == 64 * 656


def test_hyv4_sparse_backend_keeps_flashmla_name_and_sink_capability():
    from vllm_hcu.models.hy_v4.hcu_sparse import HYV4FlashMLASparseBackend

    impl = HYV4FlashMLASparseBackend.get_impl_cls()
    assert HYV4FlashMLASparseBackend.get_name() == "FLASHMLA_SPARSE"
    assert HYV4FlashMLASparseBackend.is_sparse()
    assert HYV4FlashMLASparseBackend.supports_sink()
    assert impl.supports_pcp is True
    assert impl.can_return_lse_for_decode is True


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


def test_hyv4_sink_keeps_short_prefill_on_sparse_mqa(monkeypatch):
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm_hcu.models.hy_v4 import attention as hy_attention

    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(sparse_mla_force_mqa=False)
    )
    monkeypatch.setattr(
        hy_attention, "get_current_vllm_config", lambda: vllm_config
    )
    mla = object.__new__(MLAAttention)
    torch.nn.Module.__init__(mla)
    mla._vllm_config = vllm_config
    mla.prefill_backend = None
    short_prefill = SimpleNamespace(prefill=SimpleNamespace(use_dense_mha=True))
    assert mla._use_sparse_mha(short_prefill)

    attention = object.__new__(hy_attention.HYV4MLAAttention)
    torch.nn.Module.__init__(attention)
    attention._force_sparse_mqa()

    assert not mla._use_sparse_mha(short_prefill)


@pytest.mark.parametrize("learnable_sink", [False, True])
@pytest.mark.parametrize("dcp", [1, 2])
def test_hyv4_constructor_forces_sparse_mqa_only_with_sink(
    monkeypatch, learnable_sink, dcp
):
    from vllm_hcu.models.hy_v4 import attention as hy_attention

    class StubModule(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.kwargs = kwargs

    vllm_config = SimpleNamespace(
        use_v2_model_runner=True,
        attention_config=SimpleNamespace(sparse_mla_force_mqa=False),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=True,
            dcp_q_replicate=False,
            dcp_comm_backend="ag_rs",
            cp_kv_cache_interleave_size=1,
        ),
        model_config=SimpleNamespace(
            architectures=["HYV4ForCausalLM"],
            use_mla=True,
            hf_config=SimpleNamespace(num_attention_heads=64),
            is_multimodal_model=False,
        ),
        cache_config=SimpleNamespace(
            cache_dtype="fp8_e4m3", kv_offloading_size=None
        ),
        kernel_config=SimpleNamespace(moe_backend="aiter"),
        speculative_config=None,
        lora_config=None,
        kv_transfer_config=None,
        additional_config={"hcu": {}},
    )
    config = SimpleNamespace(
        layer_types=["sparse_attention"],
        index_topk=64,
        indexer_types=["full"],
        num_hidden_layers=1,
        rms_norm_eps=1e-6,
        rope_parameters={},
        learnable_sink=learnable_sink,
        gated_mla=False,
    )
    monkeypatch.setattr(
        hy_attention, "get_current_vllm_config", lambda: vllm_config
    )
    monkeypatch.setattr(
        hy_attention, "get_tensor_model_parallel_world_size", lambda: 8
    )
    monkeypatch.setenv("VLLM_DCP_Q_REPLICATE", "0")
    monkeypatch.setattr(
        hy_attention, "get_attn_backend", lambda **kwargs: StubModule
    )
    monkeypatch.setattr(
        hy_attention.HYV4MLAAttention,
        "_resolve_sink_backend", lambda self, _: StubModule
    )
    monkeypatch.setattr(
        hy_attention, "_require_sparse_mqa_backend", lambda _: None
    )
    monkeypatch.setattr(
        hy_attention, "get_rope", lambda *args, **kwargs: StubModule()
    )
    for name in (
        "MergedColumnParallelLinear", "ColumnParallelLinear", "RowParallelLinear",
        "RMSNorm", "Indexer", "HYV4MLAAttentionLayer",
    ):
        monkeypatch.setattr(hy_attention, name, StubModule)

    attention = hy_attention.HYV4MLAAttention(
        vllm_config=vllm_config,
        config=config,
        hidden_size=16,
        num_heads=64,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        q_lora_rank=8,
        kv_lora_rank=8,
        prefix="model.layers.0.self_attn",
    )

    assert vllm_config.attention_config.sparse_mla_force_mqa is learnable_sink
    assert ("sinks" in attention.mla_attn.kwargs) is learnable_sink
    assert hy_attention.dcp_q_replication_enabled() is False
    assert attention.num_local_heads == 8
    assert type(attention.q_b_proj) is StubModule


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


def test_hy4_pcp_short_prefills_and_decode_keep_sink_and_finite_lse(
    monkeypatch,
) -> None:
    from vllm_hcu.models.hy_v4 import hcu_sparse

    impl = _bare_sparse_impl(torch.arange(4, dtype=torch.float32))
    impl.kv_cache_dtype = "bfloat16"
    impl.need_to_return_lse_for_decode = True
    impl.topk_indices_buffer = torch.zeros(3, 4, dtype=torch.int32)
    seen_sinks = []

    def fake_sparse_fwd(q, kv, indices, scale, attn_sink=None, topk_length=None):
        seen_sinks.append(attn_sink)
        return (
            torch.ones(q.shape[0], q.shape[1], 512),
            None,
            torch.ones(q.shape[0], q.shape[1]),
        )

    def forward_bf16(q, cache, indices, metadata, actual_num_heads):
        return impl._bf16_flash_mla_kernel(
            q, cache, indices, actual_num_heads=actual_num_heads
        )

    monkeypatch.setattr(hcu_sparse, "flash_mla_sparse_fwd", fake_sparse_fwd)
    impl._forward_bf16_kv = forward_bf16
    cache = torch.zeros(8, 576)
    for num_tokens in (2, 2, 1):
        output, lse = impl.forward_mqa(
            torch.zeros(num_tokens, 4, 576), cache, SimpleNamespace(), None
        )
        assert output.shape == (num_tokens, 4, 512)
        assert lse.shape == (num_tokens, 4)
        assert torch.isfinite(lse).all()

    assert len(seen_sinks) == 3
    for sink in seen_sinks:
        torch.testing.assert_close(sink[:4], impl.sinks)


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
    assert "topk_length" not in captured


def test_hyv4_dcp_gathers_loaded_sinks_once(monkeypatch):
    from vllm_hcu.models.hy_v4 import hcu_sparse

    impl = _bare_sparse_impl(torch.arange(8, dtype=torch.float32))
    impl.dcp_world_size = 2
    other = torch.arange(8, 16, dtype=torch.float32)
    calls = []

    class Group:
        def all_gather(self, value, dim):
            calls.append((value.clone(), dim))
            return torch.cat((value, other), dim=0)

    monkeypatch.setattr(hcu_sparse, "get_dcp_group", lambda: Group(), raising=False)
    impl.sinks.add_(10)
    impl.process_weights_after_loading(torch.bfloat16)

    assert len(calls) == 1 and calls[0][1] == 0
    torch.testing.assert_close(
        calls[0][0], torch.arange(8, dtype=torch.float32) + 10
    )
    torch.testing.assert_close(
        impl._dcp_gathered_sinks,
        torch.cat((torch.arange(8, dtype=torch.float32) + 10, other)),
    )

    no_dcp = _bare_sparse_impl(torch.arange(8, dtype=torch.float32))
    no_dcp.process_weights_after_loading(torch.bfloat16)
    assert no_dcp._dcp_gathered_sinks is None
    assert len(calls) == 1


def test_hyv4_dcp_kernel_passes_gathered_sink_on_rank0_only(monkeypatch):
    from vllm_hcu.models.hy_v4 import hcu_sparse

    impl = _bare_sparse_impl(torch.arange(8, dtype=torch.float32))
    impl.dcp_world_size = 2
    impl._dcp_gathered_sinks = torch.arange(16, dtype=torch.float32)
    received = []

    def fake_flashmla(**kwargs):
        received.append(kwargs["attn_sink"])
        return torch.zeros(1, 2, 64, 512), torch.zeros(1, 64, 2)

    monkeypatch.setattr(hcu_sparse, "flash_mla_with_kvcache", fake_flashmla)
    metadata = SimpleNamespace(
        dummy_block_table=torch.zeros(1, 1, dtype=torch.int32),
        cache_lens=torch.tensor([2], dtype=torch.int32),
        scheduler_metadata=None,
    )
    args = (
        torch.zeros(1, 2, 16, 576),
        torch.zeros(1, 656, dtype=torch.uint8),
        torch.zeros(1, 2, 4, dtype=torch.int32),
        metadata,
    )

    impl.dcp_rank = 0
    impl._fp8_flash_mla_kernel(*args)
    torch.testing.assert_close(received[0][:16], impl._dcp_gathered_sinks)
    assert torch.isneginf(received[0][16:]).all()

    impl.dcp_rank = 1
    impl._fp8_flash_mla_kernel(*args)
    assert received[1] is None


def test_hyv4_dcp_rank0_lse_counts_sink_after_empty_mask():
    from vllm_hcu.models.hy_v4.hcu_sparse import HYV4FlashMLASparseImpl

    raw_lse = torch.tensor([[0.0] * 16, [float("-inf")] * 16])
    raw_out = torch.ones(2, 16, 512)
    raw_out[1].zero_()
    impl = object.__new__(HYV4FlashMLASparseImpl)
    impl.sinks = torch.ones(8, dtype=torch.float32)
    impl._dcp_gathered_sinks = torch.ones(16, dtype=torch.float32)
    impl.dcp_world_size = 2

    impl.dcp_rank = 0
    out, lse = impl._add_single_dcp_sink_to_lse(
        raw_out.clone(), raw_lse.clone()
    )
    torch.testing.assert_close(
        lse[0], torch.logaddexp(raw_lse[0], impl._dcp_gathered_sinks)
    )
    torch.testing.assert_close(lse[1], impl._dcp_gathered_sinks)
    assert torch.count_nonzero(out[1]) == 0

    impl.dcp_rank = 1
    other_out, other_lse = impl._add_single_dcp_sink_to_lse(
        raw_out.clone(), raw_lse.clone()
    )
    torch.testing.assert_close(other_out, raw_out)
    torch.testing.assert_close(other_lse, raw_lse)


def test_hyv4_uses_target_sparse_mqa_implementation():
    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl
    from vllm_hcu.models.hy_v4.hcu_sparse import HYV4FlashMLASparseImpl
    from vllm_hcu.v1.attention.backends.mla.flashmla_sparse import (
        HcuFlashMLASparseImpl,
    )

    assert HYV4FlashMLASparseImpl.forward_mqa is FlashMLASparseImpl.forward_mqa
    assert (
        HYV4FlashMLASparseImpl.forward_mqa
        is not HcuFlashMLASparseImpl.forward_mqa
    )
