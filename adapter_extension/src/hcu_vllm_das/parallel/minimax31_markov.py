# SPDX-License-Identifier: Apache-2.0
"""MiniMax DSpark greedy proposal with a tiled BF16 Markov epilogue."""

from functools import wraps

import torch
import triton
import triton.language as tl


_RANK = 256
_TILE_VOCAB = 128
_DRAFT_WIDTH = 7


@triton.jit
def _first_max(a, a_id, b, b_id):
    take_a = (a > b) | ((a == b) & (a_id <= b_id))
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
    # Match vLLM greedy_argmax after both BF16 roundings.
    corrected = tl.where(
        (column[None, :] < V) & (corrected == corrected), corrected, float("-inf")
    )
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
        and seed.dtype in (torch.int32, torch.int64)
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


def _greedy_argmax(logits):
    # vLLM 0.33 still uses torch.argmax in the D-Spark loop.
    return torch.nan_to_num(
        logits, nan=float("-inf"), posinf=float("inf"), neginf=float("-inf")
    ).argmax(dim=-1)


def after_minimax31_markov_load(context):
    """Keep the MiniMax instance's vLLM 0.33 fallback on the same greedy policy."""
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    from ..models.minimax31_dspark import DSparkMiniMaxDraftModel

    speculator = getattr(context.worker.model_runner, "speculator", None)
    if not isinstance(speculator, DSparkSpeculator) or not isinstance(
        speculator.model, DSparkMiniMaxDraftModel
    ):
        return

    original = speculator._sample_logits
    if getattr(original, "_minimax31_greedy_nan_mask", False):
        return

    @wraps(original)
    def sample_logits(logits, idx_map, sample_pos, step):
        if (
            speculator.draft_logits is not None
            or speculator._draft_topk is not None
            or not isinstance(speculator.model, DSparkMiniMaxDraftModel)
        ):
            return original(logits, idx_map, sample_pos, step)
        speculator._maybe_predict_acceptance(logits, idx_map, speculator._step_cols[step])
        return speculator.model.map_draft_to_target(_greedy_argmax(logits))

    sample_logits._minimax31_greedy_nan_mask = True
    speculator._sample_logits = sample_logits


def _greedy_markov_weights(model, vocab):
    processor = model.logits_processor
    head = model.model.markov_head
    if (
        head is None
        or model.draft_id_to_target_id is not None
        or processor.vocab_size != vocab
        or getattr(processor, "org_vocab_size", vocab) != vocab
        or processor.scale != 1.0
        or processor.soft_cap is not None
        or processor.logits_as_input
        or processor.head_dtype not in (None, torch.bfloat16)
    ):
        return None
    projection = head.markov_w2
    if (
        projection.tp_size != 1
        or getattr(projection, "bias", None) is not None
        or type(projection.quant_method).__name__ not in (
            "UnquantizedEmbeddingMethod", "UnquantizedLinearMethod"
        )
    ):
        return None
    return head.markov_w1.weight, projection.weight


def sample_online_markov(model, speculator, num_reqs, head_hidden):
    """Use the native DSpark model hook for qualified greedy MiniMax batches."""
    if (
        not 1 <= num_reqs <= 32
        or speculator.num_speculative_steps != _DRAFT_WIDTH
        or speculator.draft_logits is not None
        or speculator._draft_topk is not None
        or speculator.draft_watermarker is not None
        or speculator.enable_adaptive_verification
        or speculator.use_confidence_head
        or speculator.acceptance_estimator is not None
        or hasattr(model, "apply_output_logit_transform")
        or torch.is_grad_enabled()
        or torch.is_autocast_enabled("cuda")
        or torch.version.hip is None
        or not head_hidden.is_cuda
        or head_hidden.dtype != torch.bfloat16
        or torch.cuda.get_device_properties(head_hidden.device).gcnArchName.split(":")[0]
        != "gfx938"
    ):
        return False

    vocab = model.logits_processor.vocab_size
    weights = _greedy_markov_weights(model, vocab)
    if weights is None:
        return False

    num_sample = num_reqs * _DRAFT_WIDTH
    sample_hidden = head_hidden[speculator.sample_indices[:num_sample]]
    logits = model.compute_draft_logits(sample_hidden)
    if logits is None or logits.shape != (num_sample, vocab) or logits.stride(1) != 1:
        return False

    base = logits.view(num_reqs, _DRAFT_WIDTH, vocab)
    seed = speculator.input_buffers.input_ids[speculator._anchor_idx[:num_reqs]]
    if not _supported(base, seed, *weights):
        return False

    fused_markov_greedy(base, seed, *weights, out=speculator.draft_tokens[:num_reqs])
    return True


def greedy_markov_tokens(model, base, seed, *, out=None, fused=False):
    """Offline greedy entry point with opt-in fusion for the shipped full vocabulary."""
    if fused:
        weights = _greedy_markov_weights(model, base.shape[-1])
        if weights is not None and _supported(base, seed, *weights):
            return fused_markov_greedy(base, seed, *weights, out=out)

    previous = seed
    tokens = []
    for step in range(base.shape[1]):
        bias = model.markov_bias(model.markov_embed(previous))
        previous = model.map_draft_to_target(_greedy_argmax(base[:, step] + bias))
        if out is None:
            tokens.append(previous)
        else:
            out[:, step].copy_(previous)
    return torch.stack(tokens, dim=1) if out is None else out
