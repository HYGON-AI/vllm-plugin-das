# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""FP8 PA regression on real fused pages, including changed-input Graph replay."""

import pytest
import torch

pytestmark = [
    pytest.mark.hcu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.version.hip is None,
        reason="requires an HCU runtime",
    ),
]


@pytest.mark.parametrize("max_q", [1, 4, 32])
@pytest.mark.parametrize("quantized_query", [False, True])
def test_e4m3_d256_strided_pages_and_changed_graph_replay(
    monkeypatch, max_q, quantized_query,
):
    from vllm_hcu.v1.attention import kv_cache_layout
    from vllm_hcu.v1.attention.backends import fa_utils

    monkeypatch.setattr(kv_cache_layout, "_worker_kv_cache_layout", "HND")
    torch.manual_seed(553)
    device = "cuda"
    batch, heads, qheads, dim, page = 3, 4, 24, 256, 64
    storage = (torch.randn(10, 2, heads, page, dim, device=device) * 0.4).to(
        torch.float8_e4m3fn
    )
    key, value = storage.permute(0, 1, 3, 2, 4).unbind(1)
    dtype = torch.float8_e4m3fn if quantized_query else torch.bfloat16
    query = (torch.randn(batch * max_q, qheads, dim * 2, device=device) * 0.3).to(
        dtype
    )[..., ::2]
    output = torch.empty(batch * max_q, qheads, dim * 2, device=device,
                         dtype=torch.bfloat16)[..., ::2]
    table = torch.tensor([[8, 3, 6], [5, 1, 7], [9, 0, 4]],
                         device=device, dtype=torch.int32)
    lengths = torch.tensor([63, 129, 190], device=device, dtype=torch.int32)
    cuq = torch.arange(batch + 1, device=device, dtype=torch.int32) * max_q
    qscale = torch.tensor([0.75 if quantized_query else 1.0], device=device)
    kscale = torch.tensor([0.5], device=device)
    vscale = torch.tensor([0.25], device=device)

    def run():
        return fa_utils.flash_attn_varlen_func(
            q=query, k=key, v=value, out=output, cu_seqlens_q=cuq,
            max_seqlen_q=max_q, max_seqlen_k=190, seqused_k=lengths,
            block_table=table, softmax_scale=dim ** -0.5, causal=True,
            q_descale=qscale if quantized_query else None,
            k_descale=kscale.expand(batch, heads),
            v_descale=vscale.expand(batch, heads),
        )

    def reference(sequence_lengths):
        expected = []
        for b, length in enumerate(sequence_lengths):
            k = key[table[b].long()].float().flatten(0, 1)[:length]
            v = value[table[b].long()].float().flatten(0, 1)[:length]
            k = k.repeat_interleave(qheads // heads, dim=1) * kscale
            v = v.repeat_interleave(qheads // heads, dim=1) * vscale
            q = query[b * max_q:(b + 1) * max_q].float() * qscale
            logits = torch.einsum("qhd,lhd->hql", q, k) * dim ** -0.5
            allowed = torch.arange(length, device=device)[None, :] <= (
                length - max_q + torch.arange(max_q, device=device)
            )[:, None]
            logits.masked_fill_(~allowed[None], float("-inf"))
            expected.append(torch.einsum("hql,lhd->qhd", logits.softmax(-1), v))
        return torch.cat(expected)

    assert run() is output
    torch.testing.assert_close(output.float(), reference([63, 129, 190]),
                               atol=0.003, rtol=0.04)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Neither cached lengths, block IDs nor query values may be frozen by capture.
    table.copy_(table.roll(1, dims=1))
    lengths.copy_(torch.tensor([59, 120, 183], device=device, dtype=torch.int32))
    query.copy_((query.float() + 0.05).to(dtype))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), reference([59, 120, 183]),
                               atol=0.003, rtol=0.04)