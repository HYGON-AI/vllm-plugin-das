# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def paged():
    path = Path(__file__).resolve().parents[2] / "vllm_hcu/model_executor/layers/step5_paged_kv.py"
    spec = importlib.util.spec_from_file_location("step5_paged_kv_tested", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("padding", [0, 32])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_paged_write_and_flat_read_preserve_heads_blocks_and_padding(paged, heads, padding, dtype):
    blocks, page, dim = 5, 8, 4
    stride = heads * page * 2 * dim + padding
    raw = torch.zeros((blocks * stride,), dtype=torch.bfloat16).to(dtype)
    cache = torch.as_strided(raw, (blocks, heads, page, 2 * dim),
                             (stride, page * 2 * dim, 2 * dim, 1))
    slots = torch.tensor([2 * page + 3, page + 2, -1])
    keys = torch.arange(3 * heads * dim).reshape(3, heads, dim).to(torch.bfloat16) / 4
    values = -keys
    k, v = paged.write_paged_kv_and_get_flat(cache, keys, values, slots)
    expected_k = cache[..., :dim].transpose(1, 2).contiguous().view(blocks * page, heads, dim)
    expected_v = cache[..., dim:].transpose(1, 2).contiguous().view(blocks * page, heads, dim)
    assert torch.equal(k.view(torch.uint8), expected_k.view(torch.uint8))
    assert torch.equal(v.view(torch.uint8), expected_v.view(torch.uint8))
    for row, slot in enumerate(slots[:2].tolist()):
        assert torch.equal(k[slot].view(torch.uint8), keys[row].to(dtype).view(torch.uint8))
        assert torch.equal(v[slot].view(torch.uint8), values[row].to(dtype).view(torch.uint8))
    assert (cache[0].float() == 0).all()
    for block in range(blocks):
        assert (raw[block * stride + heads * page * 2 * dim:(block + 1) * stride].float() == 0).all()


def test_only_referenced_physical_pages_are_materialized(paged):
    cache = torch.zeros((7, 4, 8, 8), dtype=torch.bfloat16)
    cache[5].fill_(3)
    cache[2].fill_(4)
    md = SimpleNamespace(seq_lens_cpu=torch.tensor([11]), block_table_np=[[5, 2, 6]],
                         num_decodes=0, num_prefills=1)
    k, v = paged.write_paged_kv_and_get_flat(cache, torch.zeros((0, 4, 4)),
                                           torch.zeros((0, 4, 4)), torch.empty(0, dtype=torch.long), metadata=md)
    assert md._step5_flat_read_blocks[1].tolist() == [2, 5]
    assert (k[5 * 8:(5 + 1) * 8] == 3).all()
    assert (v[2 * 8:(2 + 1) * 8] == 4).all()


@pytest.mark.parametrize("dbo_threshold", [0, 8192])
def test_interleaved_microbatch_metadata_reads_its_own_pages(paged, monkeypatch, dbo_threshold):
    monkeypatch.setenv("HCU_DBO_MIN_TOKENS", str(dbo_threshold))
    cache = torch.zeros((7, 4, 8, 8), dtype=torch.bfloat16)
    empty = torch.empty((0, 4, 4))
    slots = torch.empty(0, dtype=torch.long)
    batches = [
        SimpleNamespace(seq_lens_cpu=torch.tensor([9]), block_table_np=[[5, 2]],
                        num_decodes=0, num_prefills=1),
        SimpleNamespace(seq_lens_cpu=torch.tensor([9]), block_table_np=[[1, 4]],
                        num_decodes=0, num_prefills=1),
    ]
    # Alternate microbatches across layers, as the eager HCU DBO runner does.
    for layer in range(2):
        for batch, pages in zip(batches, [[2, 5], [1, 4]]):
            cache.fill_(layer + 1)
            cache[pages] = layer + 8
            k, v = paged.write_paged_kv_and_get_flat(cache, empty, empty, slots, metadata=batch)
            assert batch._step5_flat_read_blocks[1].tolist() == pages
            for page in pages:
                expected = cache[page].transpose(0, 1)
                assert torch.equal(k[page * 8:(page + 1) * 8], expected[..., :4])
                assert torch.equal(v[page * 8:(page + 1) * 8], expected[..., 4:])


def test_quantization_scales_and_saturation_are_preserved(paged):
    cache = torch.zeros((3, 4, 8, 8), dtype=torch.uint8)
    keys = torch.full((1, 4, 4), 500, dtype=torch.bfloat16)
    values = torch.full_like(keys, -500)
    k, v = paged.write_paged_kv_and_get_flat(cache, keys, values, torch.tensor([17]), 0.5, 2.0)
    assert (k[17].float() == 448).all()
    assert torch.equal(v[17].view(torch.uint8), torch.full((4, 4), -250).to(torch.float8_e4m3fn).view(torch.uint8))


@pytest.mark.parametrize("heads", [1, 4])
def test_gpu_fp8_paged_storage_matches_reference(paged, heads):
    if not torch.cuda.is_available():
        pytest.skip("GPU is required")
    blocks, page, dim = 5, 64, 192
    stride = heads * page * 2 * dim + 256
    storage = torch.zeros((128 + blocks * stride,), device="cuda", dtype=torch.uint8)
    cache = torch.as_strided(storage, (blocks, heads, page, 2 * dim),
                             (stride, page * 2 * dim, 2 * dim, 1), 128)
    key = torch.randn((7, heads, dim), device="cuda", dtype=torch.bfloat16) * 2
    value = torch.randn_like(key)
    slots = torch.tensor([3, 127, 128, 257, -1, 20, 203], device="cuda")
    k, v = paged.write_paged_kv_and_get_flat(cache, key, value, slots)
    for row, slot in enumerate(slots.cpu().tolist()):
        if slot < 0:
            continue
        assert torch.equal(k[slot].view(torch.uint8), key[row].to(torch.float8_e4m3fn).view(torch.uint8))
        assert torch.equal(v[slot].view(torch.uint8), value[row].to(torch.float8_e4m3fn).view(torch.uint8))
    logical = cache.view(torch.float8_e4m3fn)
    expected = logical[..., :dim].transpose(1, 2).contiguous().view(blocks * page, heads, dim)
    assert torch.equal(k.view(torch.uint8), expected.view(torch.uint8))
