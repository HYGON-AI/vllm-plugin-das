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


def _fill_cache(cache):
    device = cache.device
    generator = torch.Generator(device=device).manual_seed(42)
    cache[:, :, :512].random_(0, 256, generator=generator)
    cache[0, 0, :256].copy_(
        torch.arange(256, device=device).to(torch.uint8))
    token = torch.arange(cache.shape[0] * cache.shape[1], device=device,
                         dtype=torch.float32).view(*cache.shape[:2], 1)
    base = torch.tensor([0.0, -0.125, 1.0, 19.75], device=device,
                        dtype=torch.float32)
    step = torch.tensor([1.0, -2.0, 3.0, -4.0], device=device,
                        dtype=torch.float32)
    cache[:, :, 512:528].view(torch.float32).copy_(base + token * step / 4096)
    rope = cache[:, :, 528:].view(torch.int16)
    rope.copy_(torch.arange(rope.numel(), device=device).to(torch.int16)
               .reshape_as(rope))


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
        _fill_cache(cache)
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


@pytest.mark.multi_hcu
@pytest.mark.hcu_count(2)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two GPUs required")
@pytest.mark.parametrize("current_device,cache_device", [(0, 1), (1, 0)])
def test_cp_gather_uses_cache_device_stream(monkeypatch, current_device,
                                          cache_device):
    import vllm_hcu.v1.attention.ops.cp_gather_fp8 as gather

    device = torch.device("cuda", cache_device)
    if not _supports_fp8_e4m3(device):
        pytest.skip("the plugin kernel targets gfx938 FP8 E4M3")
    monkeypatch.setenv("VLLM_HCU_ENABLE_TRITON_CP_GATHER", "1")
    stream = torch.cuda.Stream(device=device)
    original_kernel = gather._cp_gather_fp8_flat
    launches = []

    class CheckedKernel:
        def __getitem__(self, grid):
            assert torch.cuda.current_device() == cache_device
            assert torch.cuda.current_stream(device).cuda_stream == stream.cuda_stream
            launches.append(grid)
            return original_kernel[grid]

    with torch.cuda.stream(stream):
        cache = torch.empty((3, 64, 656), dtype=torch.uint8, device=device)
        _fill_cache(cache)
        table = torch.tensor([[2, 0], [0, 1]], dtype=torch.int32, device=device)
        lengths = torch.tensor([7, 70], dtype=torch.int32, device=device)
        starts = torch.tensor([0, 7], dtype=torch.int32, device=device)
        output = torch.full((77, 576), -123, dtype=torch.bfloat16, device=device)
        expected = torch.empty_like(output)
        ops.cp_gather_and_upconvert_fp8_kv_cache(
            cache, expected, table, lengths, starts, 2)
        monkeypatch.setattr(
            ops, "cp_gather_and_upconvert_fp8_kv_cache",
            lambda *args: pytest.fail("candidate unexpectedly used native fallback"),
        )
        monkeypatch.setattr(gather, "_cp_gather_fp8_flat", CheckedKernel())
        with torch.cuda.device(current_device):
            cp_gather_and_upconvert_fp8_kv_cache(
                cache, output, table, lengths, starts, 2)
            assert torch.cuda.current_device() == current_device
    stream.synchronize()
    assert len(launches) == 1
    assert torch.equal(output.view(torch.int16), expected.view(torch.int16))


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


def _valid_gather_arguments():
    return [
        torch.zeros((1, 32, 656), dtype=torch.uint8, device="cuda:0"),
        torch.empty((1, 576), dtype=torch.bfloat16, device="cuda:0"),
        torch.zeros((1, 1), dtype=torch.int32, device="cuda:0"),
        torch.ones((1,), dtype=torch.int32, device="cuda:0"),
        torch.zeros((1,), dtype=torch.int32, device="cuda:0"),
        1,
    ]


def _assert_native_rejection(monkeypatch, arguments):
    import vllm_hcu.v1.attention.ops.cp_gather_fp8 as gather

    monkeypatch.setenv("VLLM_HCU_ENABLE_TRITON_CP_GATHER", "1")
    native = ops.cp_gather_and_upconvert_fp8_kv_cache
    with pytest.raises(RuntimeError) as reference:
        native(*arguments)
    calls = []

    def fallback(*values):
        calls.append(values)
        return native(*values)

    class UnexpectedKernel:
        def __getitem__(self, grid):
            pytest.fail("invalid inputs reached the Triton kernel")

    monkeypatch.setattr(ops, "cp_gather_and_upconvert_fp8_kv_cache", fallback)
    monkeypatch.setattr(gather, "_cp_gather_fp8_flat", UnexpectedKernel())
    with pytest.raises(type(reference.value)):
        cp_gather_and_upconvert_fp8_kv_cache(*arguments)
    assert len(calls) == 1
    assert all(actual is expected
               for actual, expected in zip(calls[0], arguments))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("argument", [2, 3, 4],
                         ids=["block_table", "seq_lens", "workspace_starts"])
def test_cp_gather_invalid_metadata_dtype(monkeypatch, argument):
    arguments = _valid_gather_arguments()
    arguments[argument] = arguments[argument].to(torch.int64)
    _assert_native_rejection(monkeypatch, arguments)


@pytest.mark.multi_hcu
@pytest.mark.hcu_count(2)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two GPUs required")
@pytest.mark.parametrize("argument", [1, 2, 3, 4],
                         ids=["dst", "block_table", "seq_lens", "workspace_starts"])
def test_cp_gather_rejects_mixed_devices(monkeypatch, argument):
    arguments = _valid_gather_arguments()
    arguments[argument] = arguments[argument].to("cuda:1")
    _assert_native_rejection(monkeypatch, arguments)
