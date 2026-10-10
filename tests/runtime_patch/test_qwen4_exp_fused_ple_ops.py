# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import math
from itertools import accumulate

import pytest
import torch

from vllm_hcu.models.qwen4_exp.common.ops import ple as fused_ple


def test_fused_ple_ops_export_upstream_interfaces() -> None:
    assert list(inspect.signature(fused_ple.ple_ngram_ids).parameters) == [
        "input_ids",
        "query_start_loc",
        "ngram_context",
        "layer_multipliers",
        "ngram_heads_vocab_sizes",
        "ngram_heads_offsets",
        "eos_token_id",
        "heads_per_ngram",
        "output",
    ]
    assert list(inspect.signature(fused_ple.ple_gate).parameters) == [
        "key",
        "value",
        "hidden",
        "norm_key_w",
        "norm_query_w",
        "norm_conv_w",
        "eps",
    ]
    assert list(inspect.signature(fused_ple.ple_conv).parameters) == [
        "inputs",
        "residual",
        "conv_state",
        "conv_weights",
        "state_indices",
        "outer_residual",
        "mode",
        "dilation",
        "query_start_loc",
        "num_accepted_tokens",
        "has_initial_states",
        "spec_query_len",
        "token_indices",
    ]


def _reference_ngram_ids(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    multipliers: torch.Tensor,
    sizes: torch.Tensor,
    offsets: torch.Tensor,
    eos_token_id: int,
    heads_per_ngram: int,
) -> torch.Tensor:
    tokens = input_ids.cpu().tolist()
    starts = query_start_loc.cpu().tolist()
    contexts = ngram_context.cpu().tolist()
    multipliers = multipliers.cpu()
    sizes = sizes.cpu()
    offsets = offsets.cpu()
    rows: list[torch.Tensor] = []
    context_len = ngram_context.shape[1]
    for req, (start, end) in enumerate(zip(starts, starts[1:])):
        history = contexts[req] + tokens[start:end]
        for pos in range(context_len, len(history)):
            shifted = [history[pos]]
            crossed_eos = False
            for shift in range(1, context_len + 1):
                token = eos_token_id if crossed_eos else history[pos - shift]
                shifted.append(token)
                crossed_eos |= token == eos_token_id

            mixed = torch.tensor(shifted[0], dtype=torch.int64) * multipliers[0]
            row: list[torch.Tensor] = []
            for ngram_order in range(2, context_len + 2):
                mixed ^= shifted[ngram_order - 1] * multipliers[ngram_order - 1]
                head_start = (ngram_order - 2) * heads_per_ngram
                for head in range(head_start, head_start + heads_per_ngram):
                    row.append(torch.remainder(mixed, sizes[head]) + offsets[head])
            rows.append(torch.stack(row))
    return torch.stack(rows).to(input_ids.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused PLE needs an accelerator")
def test_ple_ngram_ids_matches_reference() -> None:
    device = torch.device("cuda")
    query_lens = (3, 0, 2)
    query_start_loc = torch.tensor(
        [0, *accumulate(query_lens)], dtype=torch.int32, device=device
    )
    input_ids = torch.tensor([20, 251, 22, 30, 31], dtype=torch.int32, device=device)
    ngram_context = torch.tensor(
        [[11, 12, 13], [14, 15, 16], [251, 17, 18]],
        dtype=torch.int32,
        device=device,
    )
    heads_per_ngram = 2
    multipliers = torch.tensor(
        [18_014_398_509_481_983, 17_114_398_509_481_981,
         16_214_398_509_481_979, 15_314_398_509_481_977],
        dtype=torch.int64,
        device=device,
    )
    sizes = torch.tensor([101, 103, 107, 109, 113, 127],
                         dtype=torch.int64, device=device)
    offsets = torch.zeros_like(sizes)
    offsets[1:] = torch.cumsum(sizes, dim=0)[:-1]
    expected = _reference_ngram_ids(
        input_ids,
        query_start_loc,
        ngram_context,
        multipliers,
        sizes,
        offsets,
        251,
        heads_per_ngram,
    )
    output = torch.full_like(expected, -1)

    actual = fused_ple.ple_ngram_ids(
        input_ids,
        query_start_loc,
        ngram_context,
        multipliers,
        sizes,
        offsets,
        251,
        heads_per_ngram,
        output=output,
    )

    assert actual is output
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused PLE needs an accelerator")
@pytest.mark.parametrize("strided_kv", [False, True])
def test_ple_gate_matches_reference(strided_kv: bool) -> None:
    device = torch.device("cuda")
    num_tokens, hc, h = 3, 2, 64
    generator = torch.Generator(device=device).manual_seed(7)
    kv = torch.randn(
        num_tokens,
        hc * h + h,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    key, value = kv[:, : hc * h], kv[:, hc * h :]
    if not strided_kv:
        key, value = key.contiguous(), value.contiguous()
    hidden = torch.randn(
        num_tokens,
        hc * h,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    weights = [
        torch.randn(hc * h, device=device, dtype=torch.bfloat16,
                    generator=generator) * 0.1
        for _ in range(3)
    ]

    def grouped_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        grouped = x.float().reshape(num_tokens, hc, h)
        normalized = grouped * torch.rsqrt(
            grouped.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        return (normalized.flatten(-2) * (1 + weight.float())).to(x.dtype)

    gated, normed = fused_ple.ple_gate(
        key, value, hidden, weights[0], weights[1], weights[2], 1e-6
    )
    key_norm = grouped_norm(key, weights[0]).reshape(num_tokens, hc, h)
    query_norm = grouped_norm(hidden, weights[1]).reshape(num_tokens, hc, h)
    dot = ((key_norm * query_norm).sum(-1, keepdim=True) / math.sqrt(h)).to(
        torch.bfloat16
    )
    gate = torch.sigmoid(dot.sign() * dot.abs().clamp_min(1e-6).sqrt()).to(
        torch.bfloat16
    )
    expected_gated = (gate * value.unsqueeze(-2)).flatten(-2)
    expected_normed = grouped_norm(expected_gated, weights[2])
    assert torch.equal(gated, expected_gated)
    torch.testing.assert_close(normed, expected_normed, atol=1e-2, rtol=1e-2)


def _conv_step(
    token: torch.Tensor,
    state: torch.Tensor,
    weights: torch.Tensor,
    dilation: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    history = torch.cat((state, token[:, None]), dim=1)
    taps = torch.stack(
        [history[:, k * dilation] for k in range(weights.shape[1])], dim=1
    )
    conv = (taps.float() * weights.float()).sum(dim=1).to(token.dtype)
    return torch.nn.functional.silu(conv.float()).to(token.dtype), history[:, 1:]


def _reference_conv(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    outer_residual: torch.Tensor,
    state: torch.Tensor,
    weights: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    mode: str,
    dilation: int,
    query_start_loc: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    has_initial_states: torch.Tensor | None,
    spec_query_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = residual.clone()
    expected_state = state.clone()
    if mode == "decode":
        ranges = [(req, req + 1) for req in range(inputs.shape[0])]
    else:
        assert query_start_loc is not None
        starts = query_start_loc.cpu().tolist()
        ranges = list(zip(starts, starts[1:]))

    state_len = (weights.shape[1] - 1) * dilation
    for req, (start, end) in enumerate(ranges):
        slot = int(state_indices[req])
        if mode == "prefill":
            assert has_initial_states is not None
            current = expected_state[slot, :, :state_len].clone()
            if not bool(has_initial_states[req]):
                current.zero_()
        elif mode == "spec":
            assert num_accepted_tokens is not None
            offset = min(max(int(num_accepted_tokens[req]) - 1, 0),
                         spec_query_len - 1)
            current = expected_state[slot, :, offset:offset + state_len].clone()
        else:
            use_state = has_initial_states is None or bool(has_initial_states[req])
            current = expected_state[slot, :, :state_len].clone()
            if not use_state:
                current.zero_()

        original_state = expected_state[slot].clone()
        initial = current.clone()
        for token_idx in range(start, end):
            conv, current = _conv_step(inputs[token_idx], current, weights, dilation)
            ple = (output[token_idx] + conv).to(output.dtype)
            output[token_idx] = (outer_residual[token_idx].float() + ple.float()).to(
                output.dtype
            )

        if mode == "spec":
            history = torch.cat(
                (initial, inputs[start:end].transpose(0, 1)), dim=1
            )
            write_count = state_len + (end - start) - 1
            expected_state[slot, :, :write_count] = history[:, 1:1 + write_count]
            expected_state[slot, :, write_count:] = original_state[:, write_count:]
        elif end > start or mode == "decode":
            expected_state[slot, :, :state_len] = current
    return output, expected_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused PLE needs an accelerator")
@pytest.mark.parametrize("state_layout", ["channels_first", "window_first"])
@pytest.mark.parametrize("mode", ["decode", "prefill", "spec"])
def test_ple_conv_matches_reference(mode: str, state_layout: str) -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(11)
    channels, kernel_size, dilation, spec_query_len = 16, 3, 2, 3
    state_len = (kernel_size - 1) * dilation
    state_width = state_len + spec_query_len - 1
    storage_shape = (
        (8, channels, state_width)
        if state_layout == "channels_first"
        else (8, state_width, channels)
    )
    storage = torch.randn(
        storage_shape,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    conv_state = storage if state_layout == "channels_first" else storage.transpose(1, 2)
    state_reference = conv_state.clone()
    weights = torch.randn(
        channels, kernel_size, device=device, dtype=torch.bfloat16,
        generator=generator,
    )

    if mode == "decode":
        query_start_loc = num_accepted = None
        state_indices = torch.tensor([1, 2], dtype=torch.int32, device=device)
        has_initial = torch.tensor([True, False], device=device)
        token_count = 2
    elif mode == "prefill":
        query_start_loc = torch.tensor([0, 3, 3, 5], dtype=torch.int32, device=device)
        num_accepted = None
        state_indices = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
        has_initial = torch.tensor([True, False, False], device=device)
        token_count = 5
    else:
        query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
        num_accepted = torch.tensor([2, 1], dtype=torch.int32, device=device)
        state_indices = torch.tensor([1, 2], dtype=torch.int32, device=device)
        has_initial = None
        token_count = 5

    inputs = torch.randn(
        token_count, channels, device=device, dtype=torch.bfloat16,
        generator=generator,
    )
    residual = torch.randn(
        inputs.shape, device=device, dtype=torch.bfloat16, generator=generator
    )
    outer = torch.randn(
        inputs.shape, device=device, dtype=torch.bfloat16, generator=generator
    )
    expected_output, expected_state = _reference_conv(
        inputs,
        residual,
        outer,
        state_reference,
        weights,
        state_indices,
        mode=mode,
        dilation=dilation,
        query_start_loc=query_start_loc,
        num_accepted_tokens=num_accepted,
        has_initial_states=has_initial,
        spec_query_len=spec_query_len,
    )
    actual_output = residual.clone()

    fused_ple.ple_conv(
        inputs,
        actual_output,
        conv_state,
        weights,
        state_indices,
        outer,
        mode=mode,
        dilation=dilation,
        query_start_loc=query_start_loc,
        num_accepted_tokens=num_accepted,
        has_initial_states=has_initial,
        spec_query_len=spec_query_len,
    )

    torch.testing.assert_close(actual_output, expected_output, atol=2e-2, rtol=2e-2)
    assert torch.equal(conv_state, expected_state)
