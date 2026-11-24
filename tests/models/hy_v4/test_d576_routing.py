# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.models.hy_v4 import hcu_sparse
from vllm_hcu.models.hy_v4.hcu_sparse import HYV4FlashMLASparseImpl
from vllm_hcu.v1.attention.backends.mla.flashmla_sparse import HcuFlashMLASparseImpl


@pytest.mark.parametrize(
    ("prefix", "num_decodes", "mixed", "expected"),
    [
        ([2048, 79050], 0, False, True),
        ([127, 79050], 0, False, False),
        ([2047], 0, False, False),
        ([79050], 1, False, False),
        ([79050], 0, True, False),
        (None, 0, False, False),
    ],
)
def test_prefill_uses_host_prefix_before_this_step(prefix, num_decodes, mixed, expected):
    n = len(prefix) if prefix is not None else 1
    metadata = SimpleNamespace(
        fp8_extra_metadata=SimpleNamespace(num_decodes=num_decodes, num_prefills=n - num_decodes),
        fp8_use_mixed_batch=mixed,
        topk_tokens=2048,
        num_reqs=n,
    )
    common = SimpleNamespace(
        _num_computed_tokens_cpu=None if prefix is None else torch.tensor(prefix),
        seq_lens_cpu_upper_bound=torch.full((n,), 85000),
    )
    assert hcu_sparse._long_context_prefill(metadata, common) == expected
    common._num_computed_tokens_cpu = torch.empty(n, device="meta")
    assert not hcu_sparse._long_context_prefill(metadata, common)


@pytest.mark.parametrize(
    ("enabled", "eligible", "width", "kv_rows", "route"),
    [
        (True, True, 2048, 2048, "mls"),
        (True, False, 2048, 2048, "original"),
        (False, True, 2048, 2048, "original"),
        (True, True, 1024, 2048, "original"),
        (True, True, 2048, 128, "original"),
    ],
)
def test_hyv4_forward_routes_only_eligible_calls(monkeypatch, enabled, eligible, width, kv_rows, route):
    impl = object.__new__(HYV4FlashMLASparseImpl)
    impl.dcp_world_size = 1
    impl._resident_d576 = enabled
    impl.sinks = None
    impl.prefill_padding = 64
    impl.softmax_scale = 576**-0.5
    q = torch.zeros(1, 64, 576, dtype=torch.bfloat16)
    kv = torch.zeros(kv_rows, 1, 576, dtype=q.dtype)
    indices = torch.zeros(1, width, dtype=torch.int32)
    lengths = torch.full((1,), width, dtype=torch.int32)
    calls = []

    def original(query, cache, slots, scale, *, attn_sink, topk_length):
        assert topk_length is lengths
        calls.append("original")
        return query[..., :512], None, torch.zeros(1, 64)

    def resident(query, cache, slots, scale, sink, counts):
        assert counts is lengths
        calls.append("mls")
        return query[..., :512], torch.zeros(1, 64)

    def parent(self, query, cache, metadata, layer):
        return self._bf16_flash_mla_kernel_with_lse(query, cache, indices, lengths)

    monkeypatch.setattr(hcu_sparse, "flash_mla_sparse_fwd", original)
    monkeypatch.setattr(hcu_sparse, "resident_d576_with_sink", resident)
    monkeypatch.setattr(HcuFlashMLASparseImpl, "forward_mqa", parent)
    impl.forward_mqa(q, kv, SimpleNamespace(hyv4_long_d576_prefill=eligible), None)
    # Missing metadata on a later call must never reuse the previous decision.
    impl.forward_mqa(q, kv, SimpleNamespace(), None)
    assert calls == [route, "original"]


@pytest.mark.parametrize(
    ("eager", "dcp", "enabled", "expected"),
    [(True, 1, True, True), (False, 1, True, False),
     (True, 2, True, False), (True, 1, False, False)],
)
def test_builder_propagates_d576_decision(monkeypatch, eager, dcp, enabled, expected):
    builder = object.__new__(hcu_sparse.HYV4FlashMLASparseMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=eager),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
    )
    metadata = SimpleNamespace(
        fp8_extra_metadata=SimpleNamespace(num_decodes=0, num_prefills=1),
        fp8_use_mixed_batch=False, topk_tokens=2048, num_reqs=1,
        max_query_len=4, num_actual_tokens=4,
    )
    common = SimpleNamespace(
        _num_computed_tokens_cpu=torch.tensor([79050]),
        is_prefilling=torch.tensor([True]),
    )
    monkeypatch.setenv("VLLM_HCU_HYV4_D576_MLS", str(int(enabled)))
    monkeypatch.setattr(hcu_sparse.HcuFlashMLASparseMetadataBuilder, "build",
                        lambda *args, **kwargs: metadata)
    built = builder.build(0, common)
    assert getattr(built, "hyv4_long_d576_prefill", False) is expected
