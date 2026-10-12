# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU connector regressions and opt-in native HCU cache read-back checks."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode, MLAAttentionSpec

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "vllm_hcu/distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py"
)
spec = importlib.util.spec_from_file_location("hcu_decode_bench_under_test", SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def configs(specs, extra=None, layers=None, dcp=2, zeroing=False, dp=0):
    groups = [
        NS(layer_names=[name], kv_cache_spec=spec) for name, spec in specs.items()
    ]
    cache_config = NS(kv_cache_groups=groups, needs_kv_cache_zeroing=zeroing)
    options = extra or {}
    config = NS(
        kv_transfer_config=NS(
            get_from_extra_config=lambda key, default: options.get(key, default)
        ),
        cache_config=NS(block_size=16, cache_dtype="fp8_e4m3"),
        parallel_config=NS(decode_context_parallel_size=dcp, data_parallel_rank=dp),
        compilation_config=NS(static_forward_context=layers or {}),
    )
    return config, cache_config


def mla_spec(width=576, cache_dtype="fp8_ds_mla", block_size=16):
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=width,
        dtype=torch.uint8,
        cache_dtype_str=cache_dtype,
    )


def worker(specs, extra=None, layers=None):
    config, cache_config = configs(specs, extra, layers)
    result = mod.DecodeBenchConnectorWorker(config, cache_config)
    # CPU tests avoid initializing a distributed group.
    result._generator = lambda name, device: result.generators.setdefault(
        name, torch.Generator(device=device).manual_seed(42)
    )
    return result


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
    ],
)
@pytest.mark.parametrize("std", [0.0, 0.1])
def test_floating_samples(dtype, std):
    values = mod.make_fill_values(
        (8192,), "cpu", dtype, 0.015, std, torch.Generator().manual_seed(0)
    ).float()
    assert torch.isfinite(values).all()
    if std:
        assert abs(values.mean().item() - 0.015) < 0.005
        assert abs(values.std().item() - std) < 0.005
    else:
        expected = torch.tensor(0.015).to(dtype).float()
        torch.testing.assert_close(values, expected.expand_as(values))


@pytest.mark.parametrize("std", [0.0, 0.1])
def test_plain_fp8_byte_cache(std):
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )
    w = worker({"attn": spec}, {"fill_std": std})
    cache = torch.full((4, 16, 8), 123, dtype=torch.uint8)
    w.register_kv_caches({"attn": cache})
    w.start_fill_kv(mod.DecodeBenchConnectorMetadata({"r": (([-1, 1, 1, 99],), 16)}))
    assert torch.all(cache[[0, 2, 3]] == 123)
    actual = cache[1].view(mod.current_platform.fp8_dtype()).float()
    assert torch.isfinite(actual).all()
    assert (
        actual.std() > 0
        if std
        else torch.all(
            actual == torch.tensor(0.015).to(mod.current_platform.fp8_dtype()).float()
        )
    )


@pytest.mark.parametrize("startup", [False, True])
@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_scheduler_cache_groups_and_dcp(startup, dcp):
    config, cc = configs(
        {"mla": mla_spec(), "indexer": mla_spec(132, None)},
        {"startup_fill": startup},
        dcp=dcp,
    )
    s = mod.DecodeBenchConnectorScheduler(config, cc)
    s.update_state_after_alloc(
        NS(request_id="r", num_computed_tokens=0),
        NS(get_block_ids=lambda: ([10, 11, 12], [20, 21, 22])),
        16 * dcp,
    )
    if startup:
        assert not s._pending_fills
    else:
        assert s._pending_fills["r"] == (([10], [20]), 16 * dcp)
    assert "r" in s._filled_requests


def test_prefix_fill_preserves_completed_blocks():
    config, cc = configs({"mla": mla_spec()})
    s = mod.DecodeBenchConnectorScheduler(config, cc)
    s.update_state_after_alloc(
        NS(request_id="r", num_computed_tokens=32),
        NS(get_block_ids=lambda: ([10, 11, 12],)),
        32,
    )
    assert s._pending_fills["r"] == (([11],), 32)


def test_startup_zeroed_groups_remain_per_request():
    config, cc = configs({"mla": mla_spec()}, {"startup_fill": True}, zeroing=True)
    s = mod.DecodeBenchConnectorScheduler(config, cc)
    s.update_state_after_alloc(
        NS(request_id="r", num_computed_tokens=0),
        NS(get_block_ids=lambda: ([1, 2],)),
        32,
    )
    assert s._pending_fills["r"] == (([1],), 32)


def test_startup_fill_and_strided_blocks():
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32
    )
    w = worker(
        {"attn": spec}, {"startup_fill": True, "fill_std": 0.1, "fill_chunk_tokens": 16}
    )
    storage = torch.full((4, 16, 16), 123.0)
    cache = storage[:, :, ::2]
    w.register_kv_caches({"attn": cache})
    assert torch.isfinite(cache).all() and cache.std() > 0
    assert torch.all(storage[:, :, 1::2] == 123.0)
    saved = cache.clone()
    w.start_fill_kv(mod.DecodeBenchConnectorMetadata({}))
    torch.testing.assert_close(cache, saved)


@pytest.mark.parametrize(
    "options",
    [
        {"fill_std": -1},
        {"fill_std": float("nan")},
        {"fill_mean": float("inf")},
        {"fill_chunk_tokens": 0},
    ],
)
def test_invalid_fill_config(options):
    with pytest.raises(ValueError):
        worker({"mla": mla_spec()}, options)


def test_distinct_dcp_streams_and_reproducibility(monkeypatch):
    import vllm.distributed

    config, cc = configs({"mla": mla_spec()})
    streams = []
    for rank in (0, 1, 0):
        monkeypatch.setattr(
            vllm.distributed, "get_dcp_group", lambda rank=rank: NS(rank_in_group=rank)
        )
        w = mod.DecodeBenchConnectorWorker(config, cc)
        streams.append(torch.randn(100, generator=w._generator("mla", "cpu")))
    torch.testing.assert_close(streams[0], streams[2])
    assert not torch.equal(streams[0], streams[1])


@pytest.mark.parametrize("layout", ["mla", "indexer"])
def test_native_writer_slots_chunking_and_isolation(monkeypatch, layout):
    from vllm import _custom_ops as ops

    indexer_layer = NS(
        get_attn_backend=lambda: NS(get_name=lambda: "DEEPSEEK_V32_INDEXER")
    )
    spec = mla_spec() if layout == "mla" else mla_spec(132, None)
    w = worker(
        {"cache": spec},
        {"fill_std": 0.1, "fill_chunk_tokens": 16},
        {"cache": indexer_layer} if layout == "indexer" else {},
    )
    cache = torch.full((4, 16, 656 if layout == "mla" else 132), 123, dtype=torch.uint8)
    calls = []

    def write(values, *args):
        if layout == "mla":
            rope, dest, slots = args
            assert rope.shape == (16, 64)
        else:
            dest, slots, block, fmt = args
            assert block == 128 and fmt == "ue8m0"
        assert values.dtype == torch.bfloat16 and values.std() > 0
        calls.append(slots.clone())
        dest[slots // 16] = 0

    if layout == "mla":
        monkeypatch.setattr(mod, "write_mla_cache", write)
    else:
        monkeypatch.setattr(ops, "indexer_k_quant_and_cache", write)
    w.register_kv_caches({"cache": cache})
    w._fill_layer("cache", [-1, 1, 1, 3, 99])
    assert len(calls) == 2
    assert torch.equal(calls[0], torch.arange(16, 32))
    assert torch.equal(calls[1], torch.arange(48, 64))
    assert torch.all(cache[[0, 2]] == 123)


@pytest.mark.skipif(
    os.getenv("RUN_HCU_CACHE_TESTS") != "1", reason="opt-in native HCU kernel test"
)
@pytest.mark.hcu
@pytest.mark.parametrize("std", [0.0, 0.1])
def test_native_packed_readback(std):
    # Real ops + independent byte-layout decoding, on an explicitly released HCU.
    import vllm._C_stable_libtorch  # noqa: F401
    from vllm import _custom_ops as ops

    block = 64
    indexer = NS(get_attn_backend=lambda: NS(get_name=lambda: "DEEPSEEK_V32_INDEXER"))
    w = worker(
        {"mla": mla_spec(block_size=block), "indexer": mla_spec(132, None, block)},
        {"fill_std": std},
        {"indexer": indexer},
    )
    mla = torch.full((3, block, 656), 123, device="cuda", dtype=torch.uint8)
    idx = torch.full((3, block, 132), 123, device="cuda", dtype=torch.uint8)
    w.register_kv_caches({"mla": mla, "indexer": idx})
    w._fill_layer("mla", [1])
    w._fill_layer("indexer", [1])
    torch.cuda.synchronize()
    assert torch.all(mla[[0, 2]] == 123) and torch.all(idx[[0, 2]] == 123)
    row = mla[1].cpu()
    scales = row[:, 512:528].contiguous().view(torch.float32)
    vals = (
        row[:, :512]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
        .reshape(block, 4, 128)
    )
    nope = (vals * scales.unsqueeze(-1)).reshape(block, 512)
    rope = row[:, 528:].contiguous().view(torch.bfloat16).float()
    assert torch.isfinite(nope).all() and torch.isfinite(rope).all()
    assert torch.all(scales > 0)
    # Use the installed gather kernel, which independently interprets HCU's
    # swizzled indexer payload and its block-tail scales.
    keys = torch.empty(
        (block, 128), device="cuda", dtype=mod.current_platform.fp8_dtype()
    )
    kscales = torch.empty((block, 4), device="cuda", dtype=torch.uint8)
    ops.cp_gather_indexer_k_quant_cache(
        idx,
        keys,
        kscales,
        torch.tensor([[1]], device="cuda", dtype=torch.int32),
        torch.tensor([0, block], device="cuda", dtype=torch.int32),
    )
    restored = keys.float() * kscales.view(torch.float32)
    assert torch.isfinite(restored).all()
    assert torch.all(kscales.view(torch.float32) > 0)
    for values in (nope, rope, restored.cpu()):
        assert abs(values.mean().item() - 0.015) < 0.008
        assert (
            values.std() > 0.05
            if std
            else torch.allclose(values, torch.full_like(values, 0.015), atol=0.001)
        )
