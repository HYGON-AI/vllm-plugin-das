# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_hcu.models.hy_v4.fp8_kv_dequant import LightOpKVReuseState
from vllm_hcu.models.index_sharing import compute_skip_topk_layers
from vllm_hcu.v1.attention.backends.mla import flashmla_sparse


def _bare_impl() -> flashmla_sparse.HcuFlashMLASparseImpl:
    impl = object.__new__(flashmla_sparse.HcuFlashMLASparseImpl)
    impl.dcp_world_size = 1
    impl.dcp_rank = 0
    impl.kv_cache_dtype = "fp8_ds_mla"
    impl.head_size = 576
    impl.kv_lora_rank = 512
    impl.num_heads = 64
    impl.prefill_padding = 64
    impl.softmax_scale = 0.5
    impl.topk_indices_buffer = torch.arange(20, dtype=torch.int32).view(5, 4)
    return impl


@pytest.mark.parametrize(
    ("num_tokens", "num_reqs", "max_query_len", "expected"),
    [(8, 2, 4, 4), (5, 2, 3, 1)],
)
def test_lightop_request_width_supports_uniform_and_ragged_batches(
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
    expected: int,
) -> None:
    assert (
        flashmla_sparse._lightop_tokens_per_request(
            num_tokens,
            num_reqs,
            max_query_len,
        )
        == expected
    )


def test_lightop_request_width_rejects_invalid_runtime_metadata() -> None:
    with pytest.raises(ValueError, match="positive max_query_len"):
        flashmla_sparse._lightop_tokens_per_request(1, 1, 0)


def test_glm_fp8_lightop_route_is_disabled_by_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = _bare_impl()
    expected = torch.ones(1)
    calls: list[tuple[object, ...]] = []

    def fake_forward(self, q, cache, metadata, layer):
        calls.append((self, q, cache, metadata, layer))
        return expected, None

    monkeypatch.setattr(
        flashmla_sparse.henvs,
        "VLLM_HCU_HYV4_FP8_KV_DEQUANT",
        False,
    )
    monkeypatch.setattr(
        flashmla_sparse.FlashMLASparseImpl,
        "forward_mqa",
        fake_forward,
    )
    q = torch.zeros(1, 64, 576)
    cache = torch.zeros(1, 64, 656, dtype=torch.uint8)
    metadata = object()
    layer = object()

    output, lse = impl.forward_mqa(q, cache, metadata, layer)

    assert calls == [(impl, q, cache, metadata, layer)]
    assert output is expected
    assert lse is None


@pytest.mark.parametrize("is_prefilling", [True, False])
def test_glm_fp8_lightop_route_covers_prefill_and_decode(
    monkeypatch: pytest.MonkeyPatch,
    is_prefilling: bool,
) -> None:
    del is_prefilling  # The environment switch intentionally controls both phases.
    impl = _bare_impl()
    state = LightOpKVReuseState.from_topk_buffer(impl.topk_indices_buffer)
    impl._lightop_kv_reuse_state = state
    impl._is_indexer_producer = True

    monkeypatch.setattr(
        flashmla_sparse.henvs,
        "VLLM_HCU_HYV4_FP8_KV_DEQUANT",
        True,
    )
    localized_indices = torch.arange(100, 120, dtype=torch.int32).view(5, 4)
    topk_length = torch.full((5,), 4, dtype=torch.int32)
    compact_indices = torch.arange(20, dtype=torch.int32).view(5, 4)
    gathered_cache = torch.zeros(20, 576)
    calls: dict[str, object] = {}

    def fake_convert(req_ids, block_table, indices, **kwargs):
        calls["convert"] = (req_ids, block_table, indices, kwargs)
        return localized_indices, topk_length

    def fake_gather(
        cache,
        indices,
        kv_lora_rank,
        rope_dim,
        tokens_per_request,
        **kwargs,
    ):
        calls["gather"] = (
            cache,
            indices,
            kv_lora_rank,
            rope_dim,
            tokens_per_request,
            kwargs,
        )
        return gathered_cache, compact_indices

    expected = torch.arange(5 * 64 * 3, dtype=torch.float32).view(5, 64, 3)

    def fake_kernel(self, q, cache, indices, topk_length=None):
        calls["kernel"] = (q, cache, indices, topk_length)
        return expected.clone()

    monkeypatch.setattr(
        flashmla_sparse,
        "triton_convert_req_index_to_global_index",
        fake_convert,
    )
    monkeypatch.setattr(
        flashmla_sparse,
        "gather_dequantize_fp8_ds_mla_cache",
        fake_gather,
    )
    impl._bf16_flash_mla_kernel = MethodType(fake_kernel, impl)

    metadata = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=5,
        max_query_len=3,
        req_id_per_token=torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32),
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        block_size=64,
        lightop_kv_group_size=1,
    )
    cache = torch.zeros(16, 656, dtype=torch.uint8)
    output, lse = impl.forward_mqa(
        torch.zeros(5, 64, 576),
        cache,
        metadata,
        object(),
    )

    gather_call = calls["gather"]
    assert gather_call[0] is cache
    assert gather_call[1] is localized_indices
    assert gather_call[2:5] == (512, 64, 1)
    assert gather_call[5] == {
        "reuse_state": state,
        "allow_mapping_reuse": False,
        "mapping_reuse_group_size": 1,
    }
    kernel_call = calls["kernel"]
    assert kernel_call[0].shape == (5, 64, 576)
    assert kernel_call[1] is gathered_cache
    assert kernel_call[2] is compact_indices
    assert kernel_call[3] is topk_length
    torch.testing.assert_close(output, expected)
    assert lse is None


def test_glm_fp8_lightop_dcp_localizes_dequantizes_and_returns_lse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = _bare_impl()
    impl.dcp_world_size = 2
    impl.dcp_rank = 1
    state = LightOpKVReuseState.from_topk_buffer(impl.topk_indices_buffer)
    impl._lightop_kv_reuse_state = state

    monkeypatch.setattr(
        flashmla_sparse.henvs,
        "VLLM_HCU_HYV4_FP8_KV_DEQUANT",
        True,
    )
    localized_indices = torch.tensor(
        [
            [112, 113, -1, -1],
            [114, 115, -1, -1],
            [116, 117, -1, -1],
            [-1, -1, -1, -1],
        ],
        dtype=torch.int32,
    )
    topk_length = torch.tensor([2, 2, 2, 0], dtype=torch.int32)
    compact_indices = torch.tensor(
        [
            [0, 1, -1, -1],
            [2, 3, -1, -1],
            [4, 5, -1, -1],
            [-1, -1, -1, -1],
        ],
        dtype=torch.int32,
    )
    gathered_cache = torch.zeros(16, 576)
    calls: dict[str, object] = {}

    def fake_filter(req_ids, block_table, indices, **kwargs):
        calls["filter"] = (req_ids, block_table, indices, kwargs)
        return localized_indices, topk_length

    def fake_gather(
        cache,
        indices,
        kv_lora_rank,
        rope_dim,
        tokens_per_request,
        **kwargs,
    ):
        calls["gather"] = (
            cache,
            indices,
            kv_lora_rank,
            rope_dim,
            tokens_per_request,
            kwargs,
        )
        return gathered_cache, compact_indices

    kernel_output = torch.arange(4 * 64 * 3, dtype=torch.float32).view(4, 64, 3)
    kernel_lse = torch.arange(4 * 64, dtype=torch.float32).view(4, 64)

    def fake_kernel(q, cache, indices, scale, *, topk_length):
        calls["kernel"] = (q, cache, indices, scale, topk_length)
        return kernel_output.clone(), torch.empty(0), kernel_lse.clone()

    from vllm.v1.attention.backends.mla import sparse_utils
    from vllm_hcu.v1.attention.ops import flashmla as flashmla_ops

    monkeypatch.setattr(
        sparse_utils,
        "triton_filter_and_convert_dcp_index",
        fake_filter,
    )
    monkeypatch.setattr(
        flashmla_sparse,
        "gather_dequantize_fp8_ds_mla_cache",
        fake_gather,
    )
    monkeypatch.setattr(flashmla_ops, "flash_mla_sparse_fwd", fake_kernel)

    metadata = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=4,
        max_query_len=4,
        req_id_per_token=torch.zeros(4, dtype=torch.int32),
        block_table=torch.tensor([[7]], dtype=torch.int32),
        block_size=16,
        cp_kv_cache_interleave_size=1,
    )
    cache = torch.zeros(16, 656, dtype=torch.uint8)
    output, lse = impl.forward_mqa(
        torch.zeros(4, 64, 576),
        cache,
        metadata,
        object(),
    )

    filter_call = calls["filter"]
    assert filter_call[3] == {
        "dcp_size": 2,
        "dcp_rank": 1,
        "cp_kv_cache_interleave_size": 1,
        "BLOCK_SIZE": 16,
        "NUM_TOPK_TOKENS": 4,
        "return_valid_counts": True,
    }
    gather_call = calls["gather"]
    assert gather_call[0] is cache
    assert gather_call[1] is localized_indices
    assert gather_call[2:5] == (512, 64, 4)
    assert gather_call[5] == {
        "reuse_state": state,
        "allow_mapping_reuse": False,
        "mapping_reuse_group_size": 1,
    }
    kernel_call = calls["kernel"]
    assert kernel_call[1].shape == (16, 1, 576)
    torch.testing.assert_close(kernel_call[2][:, 0], compact_indices)
    assert kernel_call[4] is topk_length
    torch.testing.assert_close(output[:-1], kernel_output[:-1])
    torch.testing.assert_close(output[-1], torch.zeros_like(output[-1]))
    torch.testing.assert_close(lse[:-1], kernel_lse[:-1])
    assert torch.isneginf(lse[-1]).all()


def test_glm_shared_indexer_reuses_compact_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = _bare_impl()
    consumer = _bare_impl()
    state = LightOpKVReuseState.from_topk_buffer(producer.topk_indices_buffer)
    producer._lightop_kv_reuse_state = state
    producer._is_indexer_producer = True
    consumer._lightop_kv_reuse_state = state
    consumer._is_indexer_producer = False
    consumer.topk_indices_buffer = producer.topk_indices_buffer

    monkeypatch.setattr(
        flashmla_sparse.henvs,
        "VLLM_HCU_HYV4_FP8_KV_DEQUANT",
        True,
    )
    localized_indices = torch.arange(20, dtype=torch.int32).view(5, 4)
    topk_length = torch.full((5,), 4, dtype=torch.int32)
    reuse_calls: list[tuple[bool, int]] = []

    monkeypatch.setattr(
        flashmla_sparse,
        "triton_convert_req_index_to_global_index",
        lambda *args, **kwargs: (localized_indices, topk_length),
    )

    def fake_gather(*args, **kwargs):
        reuse_calls.append(
            (
                kwargs["allow_mapping_reuse"],
                kwargs["mapping_reuse_group_size"],
            )
        )
        return torch.zeros(20, 576), localized_indices

    monkeypatch.setattr(
        flashmla_sparse,
        "gather_dequantize_fp8_ds_mla_cache",
        fake_gather,
    )
    for impl in (producer, consumer):
        impl._bf16_flash_mla_kernel = MethodType(
            lambda self, q, cache, indices, topk_length=None: torch.zeros(
                q.shape[0], q.shape[1], 512
            ),
            impl,
        )

    metadata = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=8,
        max_query_len=4,
        req_id_per_token=torch.arange(8, dtype=torch.int32),
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        block_size=64,
        lightop_kv_group_size=4,
    )
    for impl in (producer, consumer):
        impl.topk_indices_buffer = torch.zeros(8, 4, dtype=torch.int32)
        impl.forward_mqa(
            torch.zeros(8, 64, 576),
            torch.zeros(16, 656, dtype=torch.uint8),
            metadata,
            object(),
        )

    assert reuse_calls == [(False, 4), (True, 4)]


def test_glm_attention_binds_shared_mapping_state_to_backend() -> None:
    impl = SimpleNamespace()
    wrapper = SimpleNamespace(mla_attn=SimpleNamespace(impl=impl))
    state = object()

    flashmla_sparse.bind_lightop_kv_reuse_state(
        wrapper,
        state,
        is_indexer_producer=False,
    )

    assert impl._lightop_kv_reuse_state is state
    assert impl._is_indexer_producer is False


def test_glm_indexer_types_mark_shared_layers() -> None:
    config = SimpleNamespace(
        num_hidden_layers=6,
        indexer_types=["full", "shared", "shared", "full", "shared", "shared"],
    )

    assert compute_skip_topk_layers(config) == {1, 2, 4, 5}


def test_glm_indexer_types_reject_leading_shared_layer() -> None:
    config = SimpleNamespace(
        num_hidden_layers=2,
        indexer_types=["shared", "full"],
    )

    with pytest.raises(ValueError, match="preceding 'full'"):
        compute_skip_topk_layers(config)
