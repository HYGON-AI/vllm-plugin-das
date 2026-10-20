# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts for the reference (plain-matmul) expert kernel.

The expected values are recomputed here token-by-token and slot-by-slot from
the weight tensors, independent of the implementation under test. This keeps
the test meaningful when the production code changes its chunking or
accumulation strategy.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_hcu.model_executor.layers.fused_moe import reference_moe
from vllm_hcu.model_executor.layers.fused_moe.reference_moe import (
    apply_reference_moe,
)

NUM_EXPERTS = 4
HIDDEN = 8
INTERMEDIATE = 6
TOP_K = 2
NUM_TOKENS = 5

_TOP_K_IDS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (1, 3),
)

_DTYPE_TOLERANCES = (
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 5e-3, 5e-3),
    (torch.bfloat16, 2e-2, 2e-2),
)


def _make_problem(
    dtype: torch.dtype = torch.float32,
    gated: bool = True,
    seed: int = 0,
):
    generator = torch.Generator().manual_seed(seed)
    width = 2 * INTERMEDIATE if gated else INTERMEDIATE
    hidden_states = torch.randn(
        NUM_TOKENS, HIDDEN, generator=generator, dtype=torch.float32
    ).to(dtype)
    w1 = torch.randn(
        NUM_EXPERTS, width, HIDDEN, generator=generator, dtype=torch.float32
    ).to(dtype)
    w2 = torch.randn(
        NUM_EXPERTS, HIDDEN, INTERMEDIATE, generator=generator, dtype=torch.float32
    ).to(dtype)
    topk_weights = torch.rand(
        NUM_TOKENS, TOP_K, generator=generator, dtype=torch.float32
    )
    topk_ids = torch.tensor(_TOP_K_IDS, dtype=torch.int32)
    return hidden_states, w1, w2, topk_weights, topk_ids


def _activation_name(activation) -> str:
    return str(getattr(activation, "value", activation)).lower()


def _reference_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    if activation in ("relu2", "relu_squared", "relu2_no_mul"):
        return torch.square(F.relu(x))
    if activation == "relu":
        return F.relu(x)
    if activation == "silu":
        return F.silu(x)
    if activation in ("gelu", "gelu_tanh"):
        return F.gelu(x, approximate="tanh")
    raise AssertionError(f"test reference does not know activation {activation!r}")


def _expected_output(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation,
    expert_map: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """Token-by-token reference in FP32."""

    name = _activation_name(activation)
    gated = w1.shape[1] == 2 * w2.shape[2]
    expected = torch.zeros(
        hidden_states.shape[0], w2.shape[1], dtype=torch.float32
    )
    for token in range(hidden_states.shape[0]):
        x = hidden_states[token].to(torch.float32)
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot])
            if expert < 0:
                continue
            if expert_map is None:
                local = expert
            else:
                local = int(expert_map[expert])
            if local < 0:
                continue

            weight = float(topk_weights[token, slot])
            activated_input = x * weight if apply_router_weight_on_input else x
            gate_up = w1[local].to(torch.float32) @ activated_input
            if gated:
                gate, up = gate_up.chunk(2)
                act = _reference_activation(gate, name) * up
            else:
                act = _reference_activation(gate_up, name)

            contribution = w2[local].to(torch.float32) @ act
            if not apply_router_weight_on_input:
                contribution = contribution * weight
            expected[token] += contribution
    return expected


def _run_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation="relu2",
    expert_map: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = NUM_EXPERTS,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if output is None:
        # NaN pre-fill proves the kernel writes every output element.
        output = torch.full(
            (hidden_states.shape[0], w2.shape[1]),
            float("nan"),
            dtype=hidden_states.dtype,
        )
    result = apply_reference_moe(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=activation,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    assert result is None
    return output


def _assert_matches(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
) -> None:
    torch.testing.assert_close(
        actual.to(torch.float32), expected, rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("activation", ("relu2", "relu_squared", "relu2_no_mul"))
def test_relu2_non_gated(activation: str) -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=False)
    actual = _run_reference(hidden, w1, w2, weights, ids, activation=activation)
    expected = _expected_output(
        hidden, w1, w2, weights, ids, activation=activation
    )
    _assert_matches(actual, expected, 1e-5, 1e-5)
    # relu2 is non-negative before the down projection; a corrupted kernel
    # manifests as NaN or huge magnitudes, so bound the routed output too.
    assert torch.isfinite(actual).all()


def test_silu_gated() -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=True)
    actual = _run_reference(hidden, w1, w2, weights, ids, activation="silu")
    expected = _expected_output(hidden, w1, w2, weights, ids, activation="silu")
    _assert_matches(actual, expected, 1e-5, 1e-5)


@pytest.mark.parametrize("activation", ("gelu", "gelu_tanh"))
def test_gelu_gated(activation: str) -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=True)
    actual = _run_reference(hidden, w1, w2, weights, ids, activation=activation)
    expected = _expected_output(
        hidden, w1, w2, weights, ids, activation=activation
    )
    _assert_matches(actual, expected, 1e-5, 1e-5)


def test_activation_enum_like_value_is_used() -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=False)
    enum_like = SimpleNamespace(value="relu2")
    actual = _run_reference(hidden, w1, w2, weights, ids, activation=enum_like)
    expected = _expected_output(hidden, w1, w2, weights, ids, activation="relu2")
    _assert_matches(actual, expected, 1e-5, 1e-5)


@pytest.mark.parametrize(("dtype", "rtol", "atol"), _DTYPE_TOLERANCES)
def test_dtype_is_preserved_and_every_row_written(
    dtype: torch.dtype, rtol: float, atol: float
) -> None:
    hidden, w1, w2, weights, ids = _make_problem(dtype=dtype)
    actual = _run_reference(hidden, w1, w2, weights, ids)
    assert actual.dtype is dtype
    expected = _expected_output(hidden, w1, w2, weights, ids, activation="relu2")
    _assert_matches(actual, expected, rtol, atol)


def test_expert_map_global_to_local() -> None:
    # Global 0 -> local 0, global 1 absent, global 2 -> local 1, global 3 absent.
    expert_map = torch.tensor([0, -1, 1, -1], dtype=torch.int32)
    hidden, w1, w2, weights, ids = _make_problem(gated=False)
    w1 = w1[:2]
    w2 = w2[:2]

    actual = _run_reference(
        hidden, w1, w2, weights, ids, expert_map=expert_map
    )
    expected = _expected_output(
        hidden, w1, w2, weights, ids, activation="relu2", expert_map=expert_map
    )
    _assert_matches(actual, expected, 1e-5, 1e-5)

    # Token 0 selects global experts 0 and 1; only expert 0 exists locally.
    assert actual[0].abs().sum() > 0
    # Token 2 selects global experts 2 and 3; only expert 2 exists locally.
    assert actual[2].abs().sum() > 0


def test_absent_local_experts_write_zeros() -> None:
    expert_map = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32)
    hidden, w1, w2, weights, ids = _make_problem(gated=False)
    w1 = w1[:2]
    w2 = w2[:2]

    actual = _run_reference(
        hidden, w1, w2, weights, ids, expert_map=expert_map
    )
    assert torch.equal(actual, torch.zeros_like(actual))


def test_negative_topk_ids_are_skipped() -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=False)
    ids = ids.clone()
    ids[1, 0] = -1

    actual = _run_reference(hidden, w1, w2, weights, ids)
    expected = _expected_output(hidden, w1, w2, weights, ids, activation="relu2")
    _assert_matches(actual, expected, 1e-5, 1e-5)


def test_apply_router_weight_on_input() -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=False)

    on_input = _run_reference(
        hidden,
        w1,
        w2,
        weights,
        ids,
        apply_router_weight_on_input=True,
    )
    on_output = _run_reference(
        hidden,
        w1,
        w2,
        weights,
        ids,
        apply_router_weight_on_input=False,
    )
    expected_on_input = _expected_output(
        hidden,
        w1,
        w2,
        weights,
        ids,
        activation="relu2",
        apply_router_weight_on_input=True,
    )
    expected_on_output = _expected_output(
        hidden,
        w1,
        w2,
        weights,
        ids,
        activation="relu2",
        apply_router_weight_on_input=False,
    )

    _assert_matches(on_input, expected_on_input, 1e-5, 1e-5)
    _assert_matches(on_output, expected_on_output, 1e-5, 1e-5)
    # relu2 is non-linear, so the two weighting points must differ.
    assert not torch.allclose(on_input, on_output, rtol=1e-4, atol=1e-4)


def test_chunking_matches_single_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=True)
    baseline = _run_reference(hidden, w1, w2, weights, ids, activation="silu")

    monkeypatch.setattr(reference_moe, "_MAX_ROWS_PER_CHUNK", 2)
    chunked = _run_reference(hidden, w1, w2, weights, ids, activation="silu")

    _assert_matches(chunked, baseline.to(torch.float32), 1e-6, 1e-6)


def test_requires_precomputed_routing() -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=False)
    with pytest.raises(ValueError, match="precomputed top-k routing"):
        apply_reference_moe(
            output=torch.zeros(NUM_TOKENS, HIDDEN),
            hidden_states=hidden,
            w1=w1,
            w2=w2,
            topk_weights=None,
            topk_ids=ids,
            activation="relu2",
            global_num_experts=NUM_EXPERTS,
            expert_map=None,
            apply_router_weight_on_input=False,
        )


def test_mismatched_topk_shapes_raise() -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=False)
    with pytest.raises(ValueError, match="same 2-D shape"):
        _run_reference(hidden, w1, w2, weights[:, :1], ids)


def test_non_2d_topk_ids_raise() -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=False)
    with pytest.raises(ValueError, match="same 2-D shape"):
        _run_reference(hidden, w1, w2, weights, ids.reshape(-1))


def test_quantized_weights_raise() -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=False)
    with pytest.raises(TypeError, match="unquantized weights"):
        _run_reference(hidden, w1.to(torch.int8), w2, weights, ids)


@pytest.mark.parametrize("dtype", (torch.float8_e4m3fn, torch.int8))
def test_unsupported_hidden_dtype_raises(dtype: torch.dtype) -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=False)
    with pytest.raises(TypeError, match="fp16/bf16/fp32"):
        _run_reference(
            hidden.to(dtype),
            w1.to(dtype),
            w2.to(dtype),
            weights,
            ids,
        )


@pytest.mark.parametrize("activation", ("swigluoai", "swiglustep", "situ"))
def test_unsupported_gated_activation_raises(activation: str) -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=True)
    with pytest.raises(ValueError, match="does not support"):
        _run_reference(hidden, w1, w2, weights, ids, activation=activation)


@pytest.mark.parametrize("activation", ("tanh", "sigmoid"))
def test_unsupported_plain_activation_raises(activation: str) -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=False)
    with pytest.raises(ValueError, match="does not support"):
        _run_reference(hidden, w1, w2, weights, ids, activation=activation)


def test_accepts_caller_supplied_output_buffer() -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=False)
    output = torch.empty(NUM_TOKENS, HIDDEN, dtype=hidden.dtype)
    returned = _run_reference(hidden, w1, w2, weights, ids, output=output)
    assert returned is output