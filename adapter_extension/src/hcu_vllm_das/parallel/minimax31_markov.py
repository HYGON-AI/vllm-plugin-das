# SPDX-License-Identifier: Apache-2.0
"""MiniMax DSpark greedy proposal with a tiled BF16 Markov epilogue."""

import torch
import triton
import triton.language as tl


_RANK = 256
_TILE_VOCAB = 128
_DRAFT_WIDTH = 7


@triton.jit
def _first_max(a, a_id, b, b_id):
    # torch.argmax treats NaN as a maximum and returns the first index on ties.
    a_nan = a != a
    b_nan = b != b
    tied = (a == b) | (a_nan & b_nan)
    take_a = (a_nan & ~b_nan) | (a > b) | (tied & (a_id <= b_id))
    return tl.where(take_a, a, b), tl.where(take_a, a_id, b_id)


@triton.jit
def _markov_tile_top1(
    Base, Prev, Embed, Weight, TileValue, TileID,
    B: tl.constexpr, V: tl.constexpr, PREV_STRIDE: tl.constexpr,
    BASE_ROW_STRIDE: tl.constexpr, EMBED_ROW_STRIDE: tl.constexpr,
    WEIGHT_ROW_STRIDE: tl.constexpr, TILES: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, R: tl.constexpr,
):
    vocab_tile = tl.program_id(0)
    row = tl.program_id(1) * M + tl.arange(0, M)
    token = tl.load(Prev + row * PREV_STRIDE, row < B, other=0)
    column = vocab_tile * N + tl.arange(0, N)
    inner = tl.arange(0, K)
    accum = tl.zeros((M, N), dtype=tl.float32)
    for part in range(R // K):
        offset = part * K + inner
        embedding = tl.load(
            Embed + token[:, None] * EMBED_ROW_STRIDE + offset[None, :],
            (row[:, None] < B) & (token[:, None] >= 0) & (token[:, None] < V),
            other=0,
        )
        weight = tl.load(
            Weight + column[None, :] * WEIGHT_ROW_STRIDE + offset[:, None],
            column[None, :] < V, other=0,
        )
        accum = tl.dot(embedding, weight, accum)
    base = tl.load(
        Base + row[:, None] * BASE_ROW_STRIDE + column[None, :],
        (row[:, None] < B) & (column[None, :] < V), other=0,
    ).to(tl.float32)
    # Preserve the separate BF16 rounding of the projection and the addition.
    bias = accum.to(tl.bfloat16).to(tl.float32)
    corrected = (base + bias).to(tl.bfloat16).to(tl.float32)
    corrected = tl.where(column[None, :] < V, corrected, float("-inf"))
    indices = tl.broadcast_to(tl.minimum(column, V)[None, :], (M, N))
    value, index = tl.reduce((corrected, indices), 1, _first_max)
    tl.store(TileValue + row * TILES + vocab_tile, value, row < B)
    tl.store(TileID + row * TILES + vocab_tile, index, row < B)


@triton.jit
def _merge_tile_top1(
    TileValue, TileID, Output,
    TILES: tl.constexpr, V: tl.constexpr, OUT_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    value = tl.load(TileValue + row * TILES + offset, offset < TILES, other=float("-inf"))
    index = tl.load(TileID + row * TILES + offset, offset < TILES, other=V)
    _, token = tl.reduce((value, index), 0, _first_max)
    tl.store(Output + row * OUT_STRIDE, token)


def _supported(base, seed, embedding, projection):
    if base.ndim != 3 or base.shape[1] != _DRAFT_WIDTH:
        return False
    batch, _, vocab = base.shape
    return (
        1 <= batch <= 32
        and 1 <= vocab <= 262144
        and base.stride(2) == 1
        and base.is_cuda
        and base.dtype == torch.bfloat16
        and seed.shape == (batch,)
        and seed.dtype == torch.int64
        and seed.device == base.device
        and all(
            weight is not None
            and weight.shape == (vocab, _RANK)
            and weight.stride(1) == 1
            and weight.stride(0) >= _RANK
            and weight.dtype == torch.bfloat16
            and weight.device == base.device
            for weight in (embedding, projection)
        )
    )


def fused_markov_greedy(base, seed, embedding, projection, *, out=None):
    """Run seven dependent greedy steps from valid target-vocabulary seed IDs."""
    if not _supported(base, seed, embedding, projection):
        raise ValueError("unsupported MiniMax DSpark Markov inputs")
    batch, width, vocab = base.shape
    if out is None:
        out = torch.empty((batch, width), dtype=torch.int64, device=base.device)
    if (
        out.shape != (batch, width) or out.dtype != torch.int64
        or out.device != base.device or out.stride(1) != 1 or out.stride(0) < width
    ):
        raise ValueError("invalid MiniMax DSpark Markov output")

    tiles = triton.cdiv(vocab, _TILE_VOCAB)
    tile_values = torch.empty((batch, tiles), dtype=torch.float32, device=base.device)
    tile_ids = torch.empty((batch, tiles), dtype=torch.int32, device=base.device)
    block_rows = 16 if batch <= 16 else 32
    previous = seed
    for step in range(width):
        output = out[:, step]
        _markov_tile_top1[(tiles, 1)](
            base[:, step], previous, embedding, projection, tile_values, tile_ids,
            batch, vocab, previous.stride(0), base.stride(0),
            embedding.stride(0), projection.stride(0), tiles,
            block_rows, _TILE_VOCAB, 64, _RANK,
            num_warps=4, num_stages=2, matrix_instr_nonkdim=16,
        )
        _merge_tile_top1[(batch,)](
            tile_values, tile_ids, output, tiles, vocab, output.stride(0),
            triton.next_power_of_2(tiles), num_warps=4,
        )
        previous = output
    return out


def greedy_markov_tokens(model, base, seed, *, out=None, fused=False):
    """Offline greedy entry point with opt-in fusion for the shipped full vocabulary."""
    if fused:
        processor = model.logits_processor
        head = model.model.markov_head
        embedding = head.markov_w1.weight
        projection = head.markov_w2.weight
        if (
            model.draft_id_to_target_id is None
            and processor.vocab_size == base.shape[-1]
            and processor.scale == 1.0
            and processor.soft_cap is None
            and not processor.logits_as_input
            and processor.head_dtype in (None, torch.bfloat16)
            and head.markov_w2.tp_size == 1
            and type(head.markov_w2.quant_method).__name__ in (
                "UnquantizedEmbeddingMethod", "UnquantizedLinearMethod"
            )
            and _supported(base, seed, embedding, projection)
        ):
            return fused_markov_greedy(base, seed, embedding, projection, out=out)

    previous = seed
    tokens = []
    for step in range(base.shape[1]):
        bias = model.markov_bias(model.markov_embed(previous))
        previous = model.map_draft_to_target((base[:, step] + bias).argmax(-1))
        if out is None:
            tokens.append(previous)
        else:
            out[:, step].copy_(previous)
    return torch.stack(tokens, dim=1) if out is None else out
