# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""GPU parity checks for DeepSeek-V4 ROCm FlashMLA sparse attention."""

from __future__ import annotations

import os

# Capture native AITER callables before applying the adapter manually below.
# Installed wheels auto-load the HCU plugin during vLLM imports, which would
# patch dsv4_rocm before we can save the reference implementation.
os.environ["VLLM_PLUGINS"] = ""

import torch
from flash_mla import (
    flash_mla_with_kvcache,
    get_mla_metadata,
)
import vllm.models.deepseek_v4.amd.rocm as dsv4_rocm
from vllm.models.deepseek_v4.common.ops import quantize_and_insert_k_cache
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    rocm_sparse_attn_decode,
)
from vllm_hcu.patch.worker.core_fix import (
    patch_deepseek_v4_rocm_flashmla_sparse as sparse_patch,
)

dev = 'cuda:0'
heads = 4
TOL = 0.01


def cache(blocks, size):
    c = torch.zeros(blocks, size, 584, device=dev, dtype=torch.uint8)
    k = torch.randn(blocks * size, 512, device=dev, dtype=torch.bfloat16)
    slots = torch.arange(blocks * size, device=dev, dtype=torch.int64)
    quantize_and_insert_k_cache(k, c.view(blocks, -1), slots, block_size=size, use_fnuz=False)
    return c


def make_inputs(batch, length, ratio):
    torch.manual_seed(13 + batch + length + ratio)
    q = torch.randn(batch, heads, 512, device=dev, dtype=torch.bfloat16)
    swa = cache(batch, 128)
    idx = torch.arange(128, device=dev, dtype=torch.int32)
    idx = idx.view(1, 1, 128).expand(batch, 1, -1).clone()
    idx += torch.arange(batch, device=dev, dtype=torch.int32).view(batch, 1, 1) * 128
    lengths = torch.full((batch,), min(length, 128), device=dev, dtype=torch.int32)
    extra = extra_idx = extra_lens = None
    if ratio != 1:
        extra = cache(batch, 64)
        extra_idx = torch.arange(64, device=dev, dtype=torch.int32)
        extra_idx = extra_idx.view(1, 1, 64).expand(batch, 1, -1).clone()
        extra_idx += torch.arange(batch, device=dev, dtype=torch.int32).view(batch, 1, 1) * 64
        extra_lens = torch.full((batch,), min(length // ratio, 64), device=dev, dtype=torch.int32)
    sink = torch.randn(heads, device=dev, dtype=torch.float32)
    return q, swa, idx, lengths, extra, extra_idx, extra_lens, sink


def reference(q, swa, idx, lengths, extra, extra_idx, extra_lens, sink, ratio):
    a = torch.empty_like(q)
    rocm_sparse_attn_decode(q, extra, swa, ratio == 1, extra_idx, extra_lens, idx,
                            lengths, None, None, None, None, sink, 512 ** -0.5, 512, 448, 64, a)
    return a


def flashmla(q, swa, idx, lengths, extra, extra_idx, extra_lens, sink, sched=None):
    f, _ = flash_mla_with_kvcache(
        q.unsqueeze(1), swa.unsqueeze(2), None, None, 512,
        sched if sched is not None else get_mla_metadata()[0],
        softmax_scale=512 ** -0.5, is_fp8_kvcache=True,
        indices=idx, topk_length=lengths, attn_sink=sink,
        extra_k_cache=extra.unsqueeze(2) if extra is not None else None,
        extra_indices_in_kvcache=extra_idx, extra_topk_length=extra_lens,
    )
    return f.squeeze(1)


# Eager comparison. length=6000 makes the C128A compressed side nonempty
# (6000 // 128 = 46 of 64 slots), which lengths 17/93 cannot reach.
for batch in (1, 8):
    for length in (17, 93, 6000):
        for ratio in (1, 4, 128):
            args = make_inputs(batch, length, ratio)
            a = reference(*args, ratio)
            f = flashmla(*args)
            err = (f.float() - a.float()).abs()
            extra_lens = args[6]
            n_extra = int(extra_lens[0]) if extra_lens is not None else 0
            print(batch, length, ratio, f'extra_len={n_extra}',
                  'max', err.max().item(), 'mean', err.mean().item(),
                  'refmax', a.abs().max().item(), flush=True)
            assert err.max().item() <= TOL, (batch, length, ratio, err.max().item())

# CUDA graph: capture once, then mutate lengths and indices in place and
# replay. FULL_DECODE_ONLY replays the captured planner kernel with new
# metadata values, so the plan must follow the device buffers, not the
# values seen at capture.
batch = 8
for ratio in (1, 4, 128):
    q, swa, idx, lengths, extra, extra_idx, extra_lens, sink = make_inputs(batch, 6000, ratio)
    # Warm up lazy HIP/kernel state with a THROWAWAY sched. The captured sched
    # must be fresh (have_initialized=False) so the planner kernels are recorded
    # inside the graph and re-run on every replay, re-reading the topk_length
    # device buffers. This mirrors vLLM, where the metadata builder allocates a
    # new FlashMLASchedMeta each step, so the capture-step planner lands in the
    # graph. A sched pre-initialized outside the graph freezes the plan and
    # VM-faults on replay once the lengths change.
    flashmla(q, swa, idx, lengths, extra, extra_idx, extra_lens, sink, get_mla_metadata()[0])
    torch.cuda.synchronize()
    sched = get_mla_metadata()[0]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = flashmla(q, swa, idx, lengths, extra, extra_idx, extra_lens, sink, sched)

    # Replay 1: same values as capture (sanity).
    graph.replay()
    torch.cuda.synchronize()
    a = reference(q, swa, idx, lengths, extra, extra_idx, extra_lens, sink, ratio)
    err = (captured.float() - a.float()).abs().max().item()
    print('graph same-values', ratio, err, flush=True)
    assert err <= TOL, (ratio, 'same-values', err)

    # Replay 2: new lengths and new query, written into the SAME buffers.
    torch.manual_seed(999 + ratio)
    q.copy_(torch.randn_like(q))
    lengths.fill_(31)
    if extra_lens is not None:
        extra_lens.fill_(9)
    graph.replay()
    torch.cuda.synchronize()
    a = reference(q, swa, idx, lengths, extra, extra_idx, extra_lens, sink, ratio)
    err = (captured.float() - a.float()).abs().max().item()
    print('graph changed-lengths', ratio, err, flush=True)
    assert err <= TOL, (
        ratio, 'changed-lengths', err,
        'the captured tile-scheduler plan does not track topk_length; '
        'FULL_DECODE_ONLY replay would compute wrong attention',
    )

    # Replay 3: grow lengths past the captured values.
    lengths.fill_(128)
    if extra_lens is not None:
        extra_lens.fill_(46)
    graph.replay()
    torch.cuda.synchronize()
    a = reference(q, swa, idx, lengths, extra, extra_idx, extra_lens, sink, ratio)
    err = (captured.float() - a.float()).abs().max().item()
    print('graph grown-lengths', ratio, err, flush=True)
    assert err <= TOL, (ratio, 'grown-lengths', err)

print('DECODE CHECKS PASSED', flush=True)

DEVICE = "cuda:0"
D_QK = 512
D_V = 512
SCALE = D_QK**-0.5


def run_case(
    backend: str,
    name: str,
    heads: int,
    num_q: int,
    compressed_len: int,
    swa_len: int,
) -> None:
    torch.manual_seed(20260923 + heads + num_q + compressed_len)
    total_kv = compressed_len + swa_len
    q = torch.randn(num_q, heads, D_QK, device=DEVICE, dtype=torch.bfloat16)
    kv = torch.randn(total_kv, 1, D_QK, device=DEVICE, dtype=torch.bfloat16)
    width = total_kv
    indices = torch.arange(width, device=DEVICE, dtype=torch.int32)
    indices = indices.view(1, width).expand(num_q, width).clone()
    # Exercise variable top-k lengths while keeping every row non-empty.
    lengths = torch.linspace(
        max(1, width // 4), width, num_q, device=DEVICE, dtype=torch.float32,
    ).to(torch.int32)
    # Keep a hole inside every valid top-k prefix. Both AITER and the adapter
    # must treat -1 as an invalid gathered-KV slot rather than reading KV[-1].
    rows = torch.arange(num_q, device=DEVICE)
    indices[rows, lengths.to(torch.int64) // 2] = -1
    sink = torch.randn(heads, device=DEVICE, dtype=torch.float32)
    reference = torch.empty(num_q, heads, D_V, device=DEVICE, dtype=torch.bfloat16)
    aiter_prefill(
        q=q,
        kv=kv,
        indices=indices,
        topk_length=lengths,
        scale=SCALE,
        head_dim=D_QK,
        nope_head_dim=448,
        rope_head_dim=64,
        attn_sink=sink,
        output=reference,
    )
    actual = torch.full_like(reference, float("nan"))
    os.environ["VLLM_HCU_DEEPSEEK_V4_ROCM_SPARSE_PREFILL_BACKEND"] = backend
    sparse_patch._require_flashmla_prefill_ready.cache_clear()
    adapter_prefill(
        q=q,
        kv=kv,
        indices=indices,
        topk_length=lengths,
        scale=SCALE,
        head_dim=D_QK,
        nope_head_dim=448,
        rope_head_dim=64,
        attn_sink=sink,
        output=actual,
    )
    torch.cuda.synchronize()
    error = (actual.float() - reference.float()).abs()
    max_error = error.max().item()
    mean_error = error.mean().item()
    print(
        backend, name, f"heads={heads}", f"tokens={num_q}",
        f"topk={lengths.tolist()}", max_error, mean_error, flush=True,
    )
    assert torch.isfinite(actual).all(), (backend, name, heads, "non-finite output")
    assert max_error <= 0.015625, (backend, name, heads, max_error, mean_error)


os.environ["VLLM_HCU_USE_CUSTOM_OPS"] = "1"
os.environ["VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_DECODE"] = "0"
os.environ["VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL"] = "1"
os.environ["VLLM_HCU_DEEPSEEK_V4_ROCM_FLASHMLA_PREFILL_HEAD_PADDING"] = "1"
aiter_prefill = dsv4_rocm.rocm_sparse_attn_prefill
assert sparse_patch.apply_to_module(dsv4_rocm)
adapter_prefill = dsv4_rocm.rocm_sparse_attn_prefill

for provider in ("flashmla", "boltops"):
    # 64/128 exercise native kernel widths; 8/16/32 and 96 exercise padding
    # to 64 and 128 respectively.
    for local_heads in (8, 16, 32, 64, 96, 128):
        for tokens in (17, 257):
            run_case(provider, "swa_only", local_heads, tokens, 0, 128)
            run_case(provider, "c4a", local_heads, tokens, 64, 128)
            run_case(provider, "c128a", local_heads, tokens, 32, 128)

print("PREFILL CHECKS PASSED", flush=True)
