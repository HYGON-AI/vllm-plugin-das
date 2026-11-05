# SPDX-License-Identifier: Apache-2.0
"""Bitwise compatibility with vLLM's packed FP8 DS-MLA gather operator."""

from types import FunctionType, SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm_hcu.v1.attention.backends.mla.flashmla_sparse import _with_plugin_cp_gather
from vllm_hcu.v1.attention.ops.cp_gather_fp8 import (
    _supports_fp8_e4m3,
    cp_gather_and_upconvert_fp8_kv_cache,
)


pytestmark = pytest.mark.hcu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("lengths", [
    [0, 0, 0], [1], [7, 0, 1, 9], [31, 1, 0, 65], [0, 127, 257, 4097],
    [1] * 33,
])
@pytest.mark.parametrize("strided", [False, True])
def test_cp_gather_bitwise(monkeypatch, block_size, lengths, strided):
    if not _supports_fp8_e4m3(torch.device("cuda:0")):
        pytest.skip("the plugin kernel targets gfx938 FP8 E4M3")
    monkeypatch.setenv("VLLM_HCU_ENABLE_TRITON_CP_GATHER", "1")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        count = len(lengths)
        blocks = (max(lengths) + block_size - 1) // block_size
        storage = torch.empty(max(count * blocks, 1), block_size,
                              672 if strided else 656,
                              dtype=torch.uint8, device="cuda")
        cache = storage[:, :, :656]
        cache[:, :, :512].copy_(torch.arange(512, device="cuda").to(torch.uint8))
        scales = cache[:, :, 512:528].view(torch.float32)
        scales.copy_(torch.tensor([0.0, -0.125, 1.0, 19.75], device="cuda"))
        rope = cache[:, :, 528:].view(torch.int16)
        rope.copy_(torch.arange(rope.numel(), device="cuda").to(torch.int16).reshape_as(rope))
        table_storage = torch.zeros(count, blocks + 5,
                                    dtype=torch.int32, device="cuda")
        table = table_storage[:, :blocks]
        table.copy_(torch.randperm(count * blocks, device="cuda",
                                  dtype=torch.int32).reshape_as(table))
        seq_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        starts = torch.tensor([sum(lengths[:i]) for i in range(count)],
                              dtype=torch.int32, device="cuda")
        dst_storage = torch.full((sum(lengths), 592 if strided else 576), -123,
                                 dtype=torch.bfloat16, device="cuda")
        dst = dst_storage[:, :576]
        expected = torch.empty_like(dst)
        ops.cp_gather_and_upconvert_fp8_kv_cache(
            cache, expected, table, seq_lens, starts, count)
        monkeypatch.setattr(
            ops, "cp_gather_and_upconvert_fp8_kv_cache",
            lambda *args: pytest.fail("candidate unexpectedly used native fallback"),
        )
        cp_gather_and_upconvert_fp8_kv_cache(
            cache, dst, table, seq_lens, starts, count)
        assert torch.equal(dst.view(torch.int16), expected.view(torch.int16))
        if strided:
            assert (dst_storage[:, 576:] == -123).all()


def test_prefill_binding_is_local_and_delegates_other_ops(monkeypatch):
    import vllm_hcu.v1.attention.backends.mla.flashmla_sparse as sparse

    old_ops = SimpleNamespace(cp_gather_and_upconvert_fp8_kv_cache=lambda: "old",
                              other=lambda: "other")
    namespace = {"ops": old_ops}
    exec("def forward(): return ops.cp_gather_and_upconvert_fp8_kv_cache(), ops.other()", namespace)
    original = namespace["forward"]
    monkeypatch.setattr(sparse, "cp_gather_and_upconvert_fp8_kv_cache", lambda: "new")
    bound = _with_plugin_cp_gather(original)
    assert isinstance(bound, FunctionType)
    assert bound() == ("new", "other")
    assert original() == ("old", "other")
    assert original.__globals__["ops"] is old_ops
    old_ops.other = lambda: "updated"
    assert bound() == ("new", "updated")


def test_disabled_kernel_retains_upstream_call(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_ENABLE_TRITON_CP_GATHER", "0")
    arguments = []
    monkeypatch.setattr(ops, "cp_gather_and_upconvert_fp8_kv_cache",
                        lambda *args: arguments.append(args))
    values = [object() for _ in range(5)]
    cp_gather_and_upconvert_fp8_kv_cache(*values, 3)
    assert arguments == [(*values, 3)]


def test_hyv4_binding_keeps_upstream_operators():
    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl
    from vllm_hcu.models.hy_v4.hcu_sparse import HYV4FlashMLASparseImpl
    from vllm_hcu.v1.attention.backends.mla.flashmla_sparse import HcuFlashMLASparseImpl

    assert (HYV4FlashMLASparseImpl._forward_fp8_kv_separate_prefill_decode
            is HcuFlashMLASparseImpl._forward_fp8_kv_separate_prefill_decode)
    proxy = HcuFlashMLASparseImpl._forward_fp8_kv_no_dedup.__globals__["ops"]
    assert proxy is not ops
    assert proxy.cp_gather_and_upconvert_fp8_kv_cache is cp_gather_and_upconvert_fp8_kv_cache
    assert (FlashMLASparseImpl._forward_fp8_kv_separate_prefill_decode
            .__globals__["ops"] is ops)


def test_existing_dedup_dispatch_is_preserved(monkeypatch):
    from vllm_hcu.v1.attention.backends.mla import workspace_dedup
    from vllm_hcu.v1.attention.backends.mla.flashmla_sparse import HcuFlashMLASparseImpl

    monkeypatch.setattr(HcuFlashMLASparseImpl, "_forward_fp8_kv_no_dedup",
                        lambda *args: "flat")
    monkeypatch.setattr(workspace_dedup, "forward_workspace_dedup",
                        lambda *args: "dedup")
    impl = HcuFlashMLASparseImpl.__new__(HcuFlashMLASparseImpl)
    for plan, expected in [(None, "flat"), (object(), "dedup")]:
        metadata = SimpleNamespace(hcu_workspace_dedup=plan)
        assert impl._forward_fp8_kv_separate_prefill_decode(
            None, None, None, metadata) == expected
