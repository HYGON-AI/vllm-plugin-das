# SPDX-License-Identifier: Apache-2.0
"""Exact BF16 greedy Markov pointwise; retain the native bias projection."""
import torch
import triton
import triton.language as tl


@triton.jit
def _first_max(values, ids, valid):
    nan_id = tl.min(tl.where(valid & (values != values), ids, 2147483647), 0)
    maximum = tl.max(tl.where(valid & (values == values), values, -float('inf')), 0)
    index = tl.min(tl.where(valid & (values == maximum), ids, 2147483647), 0)
    return tl.where(nan_id != 2147483647, float('nan'), maximum), tl.where(
        nan_id != 2147483647, nan_id, index)


@triton.jit
def _add_argmax_partials(base, bias, maxima, indices,
                         V: tl.constexpr, BASE_STRIDE: tl.constexpr,
                         NB: tl.constexpr, TILE: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    cols = block * TILE + tl.arange(0, TILE)
    a = tl.load(base + row * BASE_STRIDE + cols, cols < V, 0).to(tl.float32)
    b = tl.load(bias + row * V + cols, cols < V, 0).to(tl.float32)
    # Native add stores BF16 before torch.argmax observes its input.
    logits = (a + b).to(tl.bfloat16).to(tl.float32)
    value, index = _first_max(logits, cols, cols < V)
    tl.store(maxima + row * NB + block, value)
    tl.store(indices + row * NB + block, index)


@triton.jit
def _argmax_embedding(maxima, indices, table, tokens, embedding,
                      NB: tl.constexpr, NBP: tl.constexpr, R: tl.constexpr,
                      TOKEN_STRIDE: tl.constexpr, GATHER: tl.constexpr):
    row = tl.program_id(0)
    block = tl.arange(0, NBP)
    maximum = tl.load(maxima + row * NB + block, block < NB, -float('inf'))
    ids = tl.load(indices + row * NB + block, block < NB, 2147483647)
    _, index = _first_max(maximum, ids, block < NB)
    tl.store(tokens + row * TOKEN_STRIDE, index.to(tl.int64))
    if GATHER:
        k = tl.arange(0, R)
        value = tl.load(table + index.to(tl.int64) * R + k)
        tl.store(embedding + row * R + k, value)


def step(base, bias, table, tokens, embedding, maxima, indices, *, tile=2048, gather=True):
    m, v = base.shape
    nb = triton.cdiv(v, tile)
    _add_argmax_partials[(m, nb)](
        base, bias, maxima, indices, V=v, BASE_STRIDE=base.stride(0),
        NB=nb, TILE=tile, num_warps=4)
    _argmax_embedding[(m,)](
        maxima, indices, table, tokens, embedding, NB=nb,
        NBP=triton.next_power_of_2(nb), R=table.shape[1],
        TOKEN_STRIDE=tokens.stride(0), GATHER=gather, num_warps=4)


@torch.inference_mode()
def greedy(w1, w2, base, anchors, *, tile=2048, bias_fn=None, embed_fn=None, out=None):
    m, positions, v = base.shape
    assert positions > 0 and w1.shape == w2.shape == (v, 256)
    assert w1.dtype == w2.dtype == base.dtype == torch.bfloat16
    assert all(t.is_cuda and t.is_contiguous() for t in (w1, w2, base, anchors))
    assert anchors.shape == (m,) and anchors.dtype in (torch.int32, torch.int64)
    nb = triton.cdiv(v, tile)
    maxima = torch.empty((m, nb), device=base.device, dtype=torch.float32)
    indices = torch.empty((m, nb), device=base.device, dtype=torch.int32)
    embeddings = [torch.empty((m, 256), device=base.device, dtype=torch.bfloat16)
                  for _ in range(2)]
    tokens = (torch.empty((m, positions), device=base.device, dtype=torch.int64)
              if out is None else out)
    assert tokens.shape == (m, positions) and tokens.dtype == torch.int64
    embed = (torch.nn.functional.embedding(anchors, w1)
             if embed_fn is None else embed_fn(anchors))
    for i in range(positions):
        bias = (torch.nn.functional.linear(embed, w2)
                if bias_fn is None else bias_fn(embed))
        next_embed = embeddings[i % 2]
        step(base[:, i], bias, w1, tokens[:, i], next_embed, maxima, indices,
             tile=tile, gather=(i != positions - 1))
        embed = next_embed
    return tokens
