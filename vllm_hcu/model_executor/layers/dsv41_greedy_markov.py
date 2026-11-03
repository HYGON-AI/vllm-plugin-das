"""Greedy full-vocabulary DSpark transitions with BF16 rounding."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _greedy_markov_step_kernel(
    previous_max, previous_ids, next_max, next_ids,
    input_ids, anchor_indices, draft_tokens,
    embedding_weight, projection_weight, base_logits,
    token_stride: tl.constexpr, base_stride: tl.constexpr,
    step_stride: tl.constexpr, projection_stride: tl.constexpr,
    rows, steps: tl.constexpr, vocab: tl.constexpr,
    partitions: tl.constexpr, reduce_block: tl.constexpr,
    step: tl.constexpr, row_block: tl.constexpr, vocab_block: tl.constexpr,
):
    partition = tl.program_id(0)
    requests = tl.arange(0, row_block)
    if step == 0:
        anchors = tl.load(anchor_indices + requests, requests < rows, other=0)
        previous = tl.load(input_ids + anchors, requests < rows, other=0).to(tl.int64)
    else:
        parts = tl.arange(0, reduce_block)
        maxima = tl.load(
            previous_max + requests[:, None] * partitions + parts[None, :],
            (requests[:, None] < rows) & (parts[None, :] < partitions),
            other=float("-inf"),
        )
        winners = tl.argmax(maxima, axis=1, tie_break_left=True)
        previous = tl.load(
            previous_ids + requests * partitions + winners,
            requests < rows, other=0,
        ).to(tl.int64)
        if partition == 0:
            tl.store(draft_tokens + requests * token_stride + step - 1, previous, requests < rows)
    vocabulary = partition * vocab_block + tl.arange(0, vocab_block)
    accumulator = tl.zeros((row_block, vocab_block), tl.float32)
    for chunk in tl.static_range(4):
        ranks = chunk * 64 + tl.arange(0, 64)
        embeddings = tl.load(
            embedding_weight + previous[:, None] * 256 + ranks[None, :],
            requests[:, None] < rows, other=0,
        )
        projection = tl.load(
            projection_weight + vocabulary[None, :].to(tl.int64) * projection_stride + ranks[:, None],
            vocabulary[None, :] < vocab, other=0,
        )
        accumulator = tl.dot(embeddings, projection, accumulator)
    bias = accumulator.to(tl.bfloat16).to(tl.float32)
    base = tl.load(
        base_logits + requests[:, None] * base_stride + step * step_stride + vocabulary[None, :],
        (requests[:, None] < rows) & (vocabulary[None, :] < vocab), other=0,
    )
    scores = (base.to(tl.float32) + bias).to(tl.bfloat16).to(tl.float32)
    scores = tl.where(vocabulary[None, :] < vocab, scores, float("-inf"))
    best = tl.argmax(scores, axis=1, tie_break_left=True)
    maximum = tl.max(scores, axis=1)
    tl.store(next_max + requests * partitions + partition, maximum, requests < rows)
    tl.store(next_ids + requests * partitions + partition, partition * vocab_block + best, requests < rows)


@triton.jit
def _greedy_markov_finish_kernel(
    maxima, indices, tokens,
    partitions: tl.constexpr, reduce_block: tl.constexpr,
    token_stride: tl.constexpr, last_step: tl.constexpr,
):
    request = tl.program_id(0)
    parts = tl.arange(0, reduce_block)
    scores = tl.load(maxima + request * partitions + parts, parts < partitions, other=float("-inf"))
    winner = tl.argmax(scores, axis=0, tie_break_left=True)
    token = tl.load(indices + request * partitions + winner)
    tl.store(tokens + request * token_stride + last_step, token)


def greedy_markov(
    base_logits: torch.Tensor,
    embedding_weight: torch.Tensor,
    projection_weight: torch.Tensor,
    input_ids: torch.Tensor,
    anchor_indices: torch.Tensor,
    draft_tokens: torch.Tensor,
) -> None:
    rows, steps, vocab = base_logits.shape
    vocab_block = 256
    row_block = max(16, triton.next_power_of_2(rows))
    partitions = triton.cdiv(vocab, vocab_block)
    maxima = torch.empty((2, row_block, partitions), device=base_logits.device, dtype=torch.float32)
    indices = torch.empty((2, row_block, partitions), device=base_logits.device, dtype=torch.int32)
    for step in range(steps):
        _greedy_markov_step_kernel[(partitions,)](
            maxima[(step - 1) % 2], indices[(step - 1) % 2],
            maxima[step % 2], indices[step % 2],
            input_ids, anchor_indices, draft_tokens,
            embedding_weight, projection_weight, base_logits,
            draft_tokens.stride(0), base_logits.stride(0), base_logits.stride(1),
            projection_weight.stride(0), rows, steps, vocab,
            partitions, triton.next_power_of_2(partitions), step,
            row_block, vocab_block,
            num_warps=4,
        )
    _greedy_markov_finish_kernel[(rows,)](
        maxima[(steps - 1) % 2], indices[(steps - 1) % 2], draft_tokens,
        partitions, triton.next_power_of_2(partitions), draft_tokens.stride(0), steps - 1,
    )
