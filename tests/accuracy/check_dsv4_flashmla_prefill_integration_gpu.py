# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Exercise real DSv4 _forward_prefill gather/combiner/adapter on HCU.

Run: python tests/accuracy/check_dsv4_flashmla_prefill_integration_gpu.py
No model weights are required. Batched execution is also compared with each
request executed independently, so shared request-offset bugs cannot hide in
the AITER-versus-FlashMLA comparison. This is not a multi-GPU model test.
"""

from __future__ import annotations

import os
from types import ModuleType, SimpleNamespace

import torch

from vllm.models.deepseek_v4.amd import rocm
from vllm.models.deepseek_v4.common.ops import quantize_and_insert_k_cache
from vllm.platforms import current_platform
from vllm.v1.worker.workspace import init_workspace_manager
from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_rocm_flashmla_sparse as patch


DEVICE = "cuda:0"
TOL = 0.015625
QUERY_LENS = (17, 33, 65)
SEQ_LENS = (173, 301, 521)
WINDOW = 128
BLOCK = 256
DECODE_TOKENS = 2


def make_cache(lengths, block_size, use_fnuz):
    """Use disjoint, reversed physical pages for each request."""
    counts = [(length + block_size - 1) // block_size for length in lengths]
    blocks = sum(counts)
    pages = torch.arange(blocks - 1, -1, -1, device=DEVICE, dtype=torch.int32)
    table = torch.zeros(len(lengths), max(counts), device=DEVICE, dtype=torch.int32)
    cursor = 0
    for row, count in enumerate(counts):
        table[row, :count] = pages[cursor:cursor + count]
        cursor += count
    cache = torch.zeros(blocks, block_size, 584, device=DEVICE, dtype=torch.uint8)
    keys = torch.randn(blocks * block_size, 512, device=DEVICE, dtype=torch.bfloat16)
    quantize_and_insert_k_cache(
        keys, cache.view(blocks, -1),
        torch.arange(blocks * block_size, device=DEVICE, dtype=torch.int64),
        block_size=block_size, use_fnuz=use_fnuz,
    )
    return cache, table


def run_prefill(kernel, q, sink, ratio, topk, swa_cache, swa_table,
                compressed_cache, compressed_table, requests, chunk_size):
    lengths = [QUERY_LENS[i] for i in requests]
    # A preceding decode request makes every prefill query_start_loc nonzero.
    starts = [0, DECODE_TOKENS]
    for length in lengths:
        starts.append(starts[-1] + length)
    starts_cpu = torch.tensor(starts, dtype=torch.int32)
    table_prefix = torch.zeros_like(swa_table[:1])
    swa = SimpleNamespace(
        num_prefills=len(requests), num_prefill_tokens=sum(lengths),
        num_decodes=1, num_decode_tokens=DECODE_TOKENS,
        prefill_seq_lens=torch.tensor([SEQ_LENS[i] for i in requests],
                                     device=DEVICE, dtype=torch.int32),
        prefill_gather_lens=torch.tensor(
            [min(SEQ_LENS[i], QUERY_LENS[i] + WINDOW - 1) for i in requests],
            device=DEVICE, dtype=torch.int32),
        query_start_loc_cpu=starts_cpu, query_start_loc=starts_cpu.to(DEVICE),
        block_table=torch.cat((table_prefix, swa_table[list(requests)])),
        block_size=BLOCK,
    )
    attn = None if ratio == 1 else SimpleNamespace(
        block_table=torch.cat((torch.zeros_like(compressed_table[:1]),
                               compressed_table[list(requests)])),
        block_size=BLOCK, c128a_prefill_topk_indices=topk,
    )
    attention = SimpleNamespace(
        compress_ratio=ratio, max_model_len=1024,
        max_num_batched_tokens=sum(QUERY_LENS), window_size=WINDOW,
        PREFILL_CHUNK_SIZE=chunk_size,
        topk_indices_buffer=torch.cat((torch.full(
            (DECODE_TOKENS, topk.shape[1]), -1, device=DEVICE, dtype=torch.int32), topk)),
        attn_sink=sink, scale=512 ** -0.5,
        head_dim=512, nope_head_dim=448, rope_head_dim=64,
    )
    output = torch.full_like(q, float("nan"))
    original = rocm.rocm_sparse_attn_prefill
    try:
        rocm.rocm_sparse_attn_prefill = kernel
        rocm.DeepseekV4ROCMAiterMLAAttention._forward_prefill(
            attention, q, torch.empty(0, device=DEVICE, dtype=torch.int64),
            compressed_cache, swa_cache, output, attn, swa,
        )
    finally:
        rocm.rocm_sparse_attn_prefill = original
    torch.cuda.synchronize()
    assert torch.isfinite(output).all(), (ratio, requests, chunk_size)
    return output


def compare(label, actual, expected):
    error = (actual.float() - expected.float()).abs()
    maximum, mean = error.max().item(), error.mean().item()
    print(label, "max", maximum, "mean", mean, flush=True)
    assert maximum <= TOL, (label, maximum, mean)


def main():
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("This integration check requires an HCU/ROCm GPU")
    os.environ["VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL"] = "1"
    os.environ["VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL_HEAD_PADDING"] = "1"
    original = rocm.rocm_sparse_attn_prefill
    assert not getattr(original, patch._PREFILL_MARKER, False)
    module = ModuleType(patch.TARGET_MODULE)
    module.rocm_sparse_attn_prefill = original
    assert patch._apply_prefill_to_module(module)
    adapter = module.rocm_sparse_attn_prefill
    init_workspace_manager(torch.device(DEVICE))
    for heads in (16, 32, 96, 64, 128):
        for ratio in (1, 4, 128):
            torch.manual_seed(20261012 + heads + ratio)
            q = torch.randn(sum(QUERY_LENS), heads, 512,
                            device=DEVICE, dtype=torch.bfloat16)
            sink = torch.randn(heads, device=DEVICE, dtype=torch.float32)
            topk = torch.arange(16, device=DEVICE, dtype=torch.int32).repeat(q.shape[0], 1)
            topk[:, 1] = -1  # Interior hole, including nonempty C128A prefixes.
            swa_cache, swa_table = make_cache(
                SEQ_LENS, BLOCK, current_platform.is_fp8_fnuz())
            compressed_cache = compressed_table = None
            if ratio != 1:
                compressed_cache, compressed_table = make_cache(
                    [length // ratio for length in SEQ_LENS], BLOCK // ratio, False)
            common = (swa_cache, swa_table, compressed_cache, compressed_table)
            independent = []
            offset = 0
            for request, length in enumerate(QUERY_LENS):
                independent.append(run_prefill(
                    original, q[offset:offset + length], sink, ratio,
                    topk[offset:offset + length], *common, (request,), 1))
                offset += length
            expected = torch.cat(independent)
            # 2 crosses a chunk boundary; 3 gathers every request together.
            for chunk_size in (2, 3):
                reference = run_prefill(original, q, sink, ratio, topk,
                                        *common, (0, 1, 2), chunk_size)
                actual = run_prefill(adapter, q, sink, ratio, topk,
                                     *common, (0, 1, 2), chunk_size)
                label = f"heads={heads} ratio={ratio} chunk={chunk_size}"
                compare(label + " AITER batch/single", reference, expected)
                compare(label + " adapter/AITER", actual, reference)
                compare(label + " adapter/single", actual, expected)
    print("PREFILL INTEGRATION CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
