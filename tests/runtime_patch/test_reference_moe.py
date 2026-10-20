# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts for the reference (plain-matmul) expert kernel.

Expected values are recomputed here token-by-token and slot-by-slot from the
weight tensors, independently of the implementation under test.  That keeps the
test meaningful if processing code changes its chunking or accumulation order.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_hcu.model_executor.layers.fused_moe import reference_moe
from vllm_hcu.model_executor.layers.fused_moe.reference_moe import (
    apply_reference_moe,
)
from vllm_hcu.patch.worker.op_opt.moe import patch_reference_moe

NUM_EXPERTS = 4
HIDDEN = 8
INTERMEDIATE = 6
TOP_K = 2
NUM_TOKENS = 5

_TOPK_IDS = (
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
    topk_ids = torch.tensor(_TOPK_IDS, dtype=torch.int32)
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
    """Token-by-token reference computed in FP32."""

    name = _activation_name(activation)
    gated = w1.shape[1] == 2 * w2.shape[2]
    expected = torch.zeros(hidden_states.shape[0], w2.shape[1], dtype=torch.float32)
    for token in range(hidden_states.shape[0]):
        x = hidden_states[token].to(torch.float32)
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot])
            if expert < 0:
                continue
            if expert_map is None:
                local = expert
            elif expert >= expert_map.numel():
                continue
            else:
                local = int(expert_map[expert])
            if local < 0 or local >= w1.shape[0]:
                continue

            weight = float(topk_weights[token, slot])
            activated_input = x * weight if apply_router_weight_on_input else x
            gate_up = w1[local].to(torch.float32) @ activated_input
            if gated:
                gate, up = gate_up.chunk(2)
                contribution = (
                    w2[local].to(torch.float32)
                    @ (_reference_activation(gate, name) * up)
                )
            else:
                contribution = (
                    w2[local].to(torch.float32)
                    @ _reference_activation(gate_up, name)
                )
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
        if hidden_states.is_floating_point():
            # NaN pre-fill proves every output element is written.
            output = torch.full(
                (hidden_states.shape[0], w2.shape[1]),
                float("nan"),
                dtype=hidden_states.dtype,
            )
        else:
            output = torch.zeros(
                (hidden_states.shape[0], w2.shape[1]),
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
    actual: torch.Tensor, expected: torch.Tensor, rtol: float, atol: float
) -> None:
    torch.testing.assert_close(
        actual.to(torch.float32), expected, rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("activation", ("relu2", "relu_squared", "relu2_no_mul"))
def test_relu2_non_gated(activation: str) -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=False)
    actual = _run_reference(hidden, w1, w2, weights, ids, activation=activation)
    expected = _expected_output(hidden, w1, w2, weights, ids, activation=activation)
    _assert_matches(actual, expected, 1e-5, 1e-5)
    assert torch.isfinite(actual).all()
    # A corrupted kernel shows up as NaN or ~1e36 magnitudes.
    assert actual.abs().max() < 1e3


@pytest.mark.parametrize("activation", ("silu",))
def test_silu_gated(activation: str) -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=True)
    actual = _run_reference(hidden, w1, w2, weights, ids, activation=activation)
    expected = _expected_output(
        hidden, w1, w2, weights, ids, activation=activation
    )
    _assert_matches(actual, expected, 1e-5, 1e-5)


@pytest.mark.parametrize("activation", ("relu2", "relu_squared"))
def test_relu2_gated(activation: str) -> None:
    """The gated relu^2 layout (``relu2(gate) * up``) is used by NemotronH."""

    hidden, w1, w2, weights, ids = _make_problem(gated=True)
    actual = _run_reference(hidden, w1, w2, weights, ids, activation=activation)
    expected = _expected_output(
        hidden, w1, w2, weights, ids, activation=activation
    )
    _assert_matches(actual, expected, 1e-5, 1e-5)


@pytest.mark.parametrize("activation", ("gelu", "gelu_tanh"))
def test_gelu_gated(activation: str) -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=True)
    actual = _run_reference(hidden, w1, w2, weights, ids, activation=activation)
    expected = _expected_output(hidden, w1, w2, weights, ids, activation=activation)
    _assert_matches(actual, expected, 1e-5, 1e-5)


def test_activation_enum_like_value_is_used() -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=False)
    actual = _run_reference(
        hidden, w1, w2, weights, ids, activation=SimpleNamespace(value="relu2")
    )
    expected = _expected_output(hidden, w1, w2, weights, ids, activation="relu2")
    _assert_matches(actual, expected, 1e-5, 1e-5)


@pytest.mark.parametrize(("dtype", "rtol", "atol"), _DTYPE_TOLERANCES)
def test_dtype_is_preserved_and_every_row_written(
    dtype: torch.dtype, rtol: float, atol: float
) -> None:
    hidden, w1, w2, weights, ids = _make_problem(dtype=dtype, gated=False)
    actual = _run_reference(hidden, w1, w2, weights, ids)
    assert actual.dtype is dtype
    assert torch.isfinite(actual).all()
    expected = _expected_output(hidden, w1, w2, weights, ids, activation="relu2")
    _assert_matches(actual, expected, rtol, atol)


def test_expert_map_global_to_local() -> None:
    # Global 0 -> local 0, global 2 -> local 1; global 1 and 3 are remote.
    expert_map = torch.tensor([0, -1, 1, -1], dtype=torch.int32)

    # All-positive weights keep every relu2 activation non-zero, so the
    # assertions below test the mapping rather than the activation.
    hidden = torch.ones(NUM_TOKENS, HIDDEN)
    w1 = torch.ones(2, INTERMEDIATE, HIDDEN)
    w2 = torch.ones(2, HIDDEN, INTERMEDIATE)
    topk_weights = torch.full((NUM_TOKENS, TOP_K), 0.5)
    ids = torch.tensor(_TOPK_IDS, dtype=torch.int32)

    actual = _run_reference(
        hidden, w1, w2, topk_weights, ids, expert_map=expert_map
    )
    expected = _expected_output(
        hidden,
        w1,
        w2,
        topk_weights,
        ids,
        activation="relu2",
        expert_map=expert_map,
    )
    _assert_matches(actual, expected, 1e-5, 1e-5)

    # Token 0 routes to global 0 (local) and 1 (remote).
    assert actual[0].abs().sum() > 0
    # Token 2 routes to global 2 (local) and 3 (remote).
    assert actual[2].abs().sum() > 0
    # Token 4 routes to global 1 and 3, both remote on this rank.
    assert torch.equal(actual[4], torch.zeros(HIDDEN))


def test_absent_local_experts_write_zeros() -> None:
    expert_map = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32)
    hidden, w1, w2, weights, ids = _make_problem(gated=False)
    actual = _run_reference(
        hidden, w1[:2], w2[:2], weights, ids, expert_map=expert_map
    )
    assert torch.equal(actual, torch.zeros_like(actual))


def test_expert_map_shorter_than_global_ids_is_skipped() -> None:
    """Ids beyond the map length cannot be local and must be ignored."""

    expert_map = torch.tensor([0, 1], dtype=torch.int32)
    hidden, w1, w2, weights, ids = _make_problem(gated=False)

    actual = _run_reference(hidden, w1, w2, weights, ids, expert_map=expert_map)
    expected = _expected_output(
        hidden, w1, w2, weights, ids, activation="relu2", expert_map=expert_map
    )
    _assert_matches(actual, expected, 1e-5, 1e-5)


def test_local_expert_map_reads_device_tensor_once() -> None:
    """The mapping is resolved with a single host read, not one per expert."""

    reads = 0
    real = reference_moe._local_expert_map

    def counting(expert_map, num_local_experts):
        nonlocal reads
        reads += 1
        return real(expert_map, num_local_experts)

    hidden, w1, w2, weights, ids = _make_problem(gated=False)
    original = reference_moe._local_expert_map
    reference_moe._local_expert_map = counting
    try:
        _run_reference(
            hidden,
            w1,
            w2,
            weights,
            ids,
            expert_map=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )
    finally:
        reference_moe._local_expert_map = original
    # The value used by production code is bound at call time.
    assert reads == 1


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
        hidden, w1, w2, weights, ids, apply_router_weight_on_input=True
    )
    on_output = _run_reference(
        hidden, w1, w2, weights, ids, apply_router_weight_on_input=False
    )
    _assert_matches(
        on_input,
        _expected_output(
            hidden,
            w1,
            w2,
            weights,
            ids,
            activation="relu2",
            apply_router_weight_on_input=True,
        ),
        1e-5,
        1e-5,
    )
    _assert_matches(
        on_output,
        _expected_output(
            hidden,
            w1,
            w2,
            weights,
            ids,
            activation="relu2",
            apply_router_weight_on_input=False,
        ),
        1e-5,
        1e-5,
    )
    # relu2 is non-linear, so the two weighting points must differ.
    assert not torch.allclose(on_input, on_output, rtol=1e-4, atol=1e-4)


def test_chunking_matches_single_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=True)
    baseline = _run_reference(hidden, w1, w2, weights, ids, activation="silu")

    monkeypatch.setattr(reference_moe, "_MAX_ROWS_PER_CHUNK", 2)
    chunked = _run_reference(hidden, w1, w2, weights, ids, activation="silu")

    _assert_matches(chunked, baseline.to(torch.float32), 1e-6, 1e-6)


def test_accepts_caller_supplied_output_buffer() -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=False)
    output = torch.empty(NUM_TOKENS, HIDDEN, dtype=hidden.dtype)
    assert _run_reference(hidden, w1, w2, weights, ids, output=output) is output


def test_empty_token_batch_writes_empty_output() -> None:
    hidden, w1, w2, weights, ids = _make_problem(gated=False)
    empty_hidden = hidden[:0]
    output = torch.full((0, HIDDEN), float("nan"), dtype=hidden.dtype)
    _run_reference(
        empty_hidden,
        w1,
        w2,
        weights[:0],
        ids[:0],
        output=output,
    )
    assert output.shape == (0, HIDDEN)


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
            hidden.to(dtype), w1.to(dtype), w2.to(dtype), weights, ids
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


# --------------------------------------------------------------------------
# Adapter contract: a fake module keeps this CPU-only and hermetic.
# --------------------------------------------------------------------------


def _adapter_module(calls: list[str]) -> ModuleType:
    module = ModuleType(patch_reference_moe.TARGET_MODULE)

    class TritonExperts:
        def apply(
            self,
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            a1q_scale,
            a2_scale,
            workspace13,
            workspace2,
            expert_tokens_meta,
            apply_router_weight_on_input,
        ):
            calls.append("original")
            output.zero_()
            return None

    module.TritonExperts = TritonExperts
    return module


def _call_apply(experts, hidden_states, w1, w2, weights, ids, **overrides):
    kwargs = {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "topk_weights": weights,
        "topk_ids": ids,
        "activation": "relu2",
        "global_num_experts": NUM_EXPERTS,
        "expert_map": None,
        "a1q_scale": None,
        "a2_scale": None,
        "workspace13": None,
        "workspace2": None,
        "expert_tokens_meta": None,
        "apply_router_weight_on_input": False,
    }
    kwargs.update(overrides)
    output = torch.full(
        (hidden_states.shape[0], HIDDEN), float("nan"), dtype=hidden_states.dtype
    )
    experts.apply(output=output, **kwargs)
    return output


def _install(monkeypatch: pytest.MonkeyPatch, enabled: bool, calls):
    monkeypatch.setenv(
        "VLLM_HCU_USE_REFERENCE_MOE", "1" if enabled else "0"
    )
    module = _adapter_module(calls)
    assert patch_reference_moe.apply_to_module(module) is True
    return module.TritonExperts()


def test_env_off_calls_original(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    experts = _install(monkeypatch, False, calls)
    hidden, w1, w2, weights, ids = _make_problem(gated=False)

    output = _call_apply(experts, hidden, w1, w2, weights, ids)

    assert calls == ["original"]
    assert torch.equal(output, torch.zeros_like(output))


def test_env_on_writes_reference_output(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    experts = _install(monkeypatch, True, calls)
    hidden, w1, w2, weights, ids = _make_problem(gated=False)

    output = _call_apply(experts, hidden, w1, w2, weights, ids)

    assert calls == []
    _assert_matches(
        output,
        _expected_output(hidden, w1, w2, weights, ids, activation="relu2"),
        1e-5,
        1e-5,
    )


def test_apply_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setenv("VLLM_HCU_USE_REFERENCE_MOE", "1")
    module = _adapter_module(calls)

    assert patch_reference_moe.apply_to_module(module) is True
    assert patch_reference_moe.apply_to_module(module) is False


def test_quantized_input_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    experts = _install(monkeypatch, True, calls)
    hidden, w1, w2, weights, ids = _make_problem(gated=False)

    with pytest.raises(RuntimeError, match="quantized"):
        _call_apply(
            experts,
            hidden,
            w1,
            w2,
            weights,
            ids,
            a1q_scale=torch.ones(1),
        )


def test_batched_expert_metadata_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    experts = _install(monkeypatch, True, calls)
    hidden, w1, w2, weights, ids = _make_problem(gated=False)

    with pytest.raises(RuntimeError, match="batched expert token metadata"):
        _call_apply(
            experts,
            hidden,
            w1,
            w2,
            weights,
            ids,
            expert_tokens_meta=object(),
        )


def test_signature_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_hcu.patch.worker.op_opt._common import PatchCompatibilityError

    monkeypatch.setenv("VLLM_HCU_USE_REFERENCE_MOE", "1")
    module = ModuleType(patch_reference_moe.TARGET_MODULE)

    class TritonExperts:
        def apply(self, output, hidden_states):
            return None

    module.TritonExperts = TritonExperts

    with pytest.raises(PatchCompatibilityError):
        patch_reference_moe.apply_to_module(module)


def test_eager_moe_alias_enables_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.delenv("VLLM_HCU_USE_REFERENCE_MOE", raising=False)
    monkeypatch.setenv("VLLM_HCU_EAGER_MOE", "1")
    assert henvs.VLLM_HCU_USE_REFERENCE_MOE is True

    monkeypatch.setenv("VLLM_HCU_EAGER_MOE", "0")
    assert henvs.VLLM_HCU_USE_REFERENCE_MOE is False

    # The canonical name wins over the deprecated alias.
    monkeypatch.setenv("VLLM_HCU_EAGER_MOE", "1")
    monkeypatch.setenv("VLLM_HCU_USE_REFERENCE_MOE", "0")
    assert henvs.VLLM_HCU_USE_REFERENCE_MOE is False
