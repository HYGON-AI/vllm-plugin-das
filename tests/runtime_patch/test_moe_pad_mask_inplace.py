# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""VLLM_HCU_MOE_PAD_MASK_INPLACE: in-place padding mask after moe_fused_gate."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm import envs, forward_context

from vllm_hcu.model_executor.layers.fused_moe import router_runtime
from vllm_hcu.model_executor.layers.fused_moe.router_runtime import (
    make_hcu_grouped_topk_router,
)

ENV_VAR = "VLLM_HCU_MOE_PAD_MASK_INPLACE"

WEIGHTS = torch.tensor([[0.25, 0.75], [0.4, 0.6], [0.3, 0.7], [0.45, 0.55]])
IDS = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 0]], dtype=torch.int32)


@pytest.fixture
def gate_outputs():
    """Fresh tensors returned by the fake LightOp gate on every call."""
    return []


@pytest.fixture
def router(monkeypatch, gate_outputs):
    from vllm_hcu.platforms import envs as henvs

    class BaseRouter:
        def _compute_routing(self, *args, **kwargs):
            raise AssertionError("LightOp success should not use official fallback")

    def moe_fused_gate(*args, **kwargs):
        outputs = (WEIGHTS.clone(), IDS.clone())
        gate_outputs.append(outputs)
        return outputs

    lightop = ModuleType("lightop")
    lightop.__path__ = []
    moe = ModuleType("lightop.moe")
    moe.moe_fused_gate = moe_fused_gate
    lightop.moe = moe
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.moe", moe)
    monkeypatch.setattr(router_runtime, "lightop_moe_gate_kwargs", lambda *args: {})
    monkeypatch.setattr(henvs, "optional_custom_op_enabled", lambda *args: True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_FUSE_MOE_GATE", True)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setattr(envs, "VLLM_MOE_SKIP_PADDING", True)
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)

    instance = make_hcu_grouped_topk_router(BaseRouter)()
    instance.num_expert_group = 2
    instance.topk_group = 1
    instance.top_k = 2
    instance.e_score_correction_bias = torch.ones(4)
    instance.routed_scaling_factor = 1.0
    instance.scoring_func = "sigmoid"
    instance.renormalize = True
    return instance


def _set_padding(monkeypatch, mask):
    is_padding = None if mask is None else torch.tensor(mask)
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(is_padding=is_padding),
    )


def _expected_masked(padding):
    pad = torch.tensor(padding[: WEIGHTS.shape[0]])[:, None]
    return WEIGHTS.masked_fill(pad, 0.0), IDS.masked_fill(pad, -1)


def _route(router, indices_type=None):
    return router._compute_routing(None, torch.ones((4, 4)), indices_type)


def test_switch_disabled_by_default(monkeypatch):
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.delenv(ENV_VAR, raising=False)
    assert henvs.VLLM_HCU_MOE_PAD_MASK_INPLACE is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), ("True", True), ("0", False), ("no", False)],
)
def test_switch_parsing(monkeypatch, value, expected):
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setenv(ENV_VAR, value)
    assert henvs.VLLM_HCU_MOE_PAD_MASK_INPLACE is expected


@pytest.mark.parametrize(
    "padding",
    [
        [False, True, False, True],
        # Longer graph-padded mask: only the prefix applies.
        [True, False, False, False, True, True],
    ],
)
def test_inplace_masks_gate_outputs_without_copy(monkeypatch, router, gate_outputs,
                                                 padding):
    monkeypatch.setenv(ENV_VAR, "1")
    _set_padding(monkeypatch, padding)

    weights, ids = _route(router)

    (gate_weights, gate_ids), = gate_outputs
    # No clone: the gate's own buffers are returned, already masked.
    assert weights is gate_weights
    assert ids is gate_ids
    expected_weights, expected_ids = _expected_masked(padding)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(ids, expected_ids)


def test_out_of_place_when_switch_disabled(monkeypatch, router, gate_outputs):
    padding = [False, True, False, True]
    monkeypatch.setenv(ENV_VAR, "0")
    _set_padding(monkeypatch, padding)

    weights, ids = _route(router)

    (gate_weights, gate_ids), = gate_outputs
    assert weights is not gate_weights
    assert ids is not gate_ids
    # The gate's buffers keep their unmasked values.
    torch.testing.assert_close(gate_weights, WEIGHTS)
    torch.testing.assert_close(gate_ids, IDS)
    expected_weights, expected_ids = _expected_masked(padding)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(ids, expected_ids)


def test_out_of_place_when_custom_ops_disabled(monkeypatch, router, gate_outputs):
    padding = [True, False, False, False]
    monkeypatch.setenv(ENV_VAR, "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    _set_padding(monkeypatch, padding)

    weights, ids = _route(router)

    (gate_weights, gate_ids), = gate_outputs
    assert weights is not gate_weights
    assert ids is not gate_ids
    torch.testing.assert_close(gate_ids, IDS)
    expected_weights, expected_ids = _expected_masked(padding)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(ids, expected_ids)


def test_inplace_and_out_of_place_agree(monkeypatch, router):
    padding = [True, True, False, True]
    _set_padding(monkeypatch, padding)

    monkeypatch.setenv(ENV_VAR, "0")
    ref_weights, ref_ids = _route(router)
    monkeypatch.setenv(ENV_VAR, "1")
    weights, ids = _route(router)

    assert torch.equal(weights, ref_weights)
    assert torch.equal(ids, ref_ids)


def test_inplace_after_indices_type_conversion(monkeypatch, router, gate_outputs):
    padding = [False, False, True, False]
    monkeypatch.setenv(ENV_VAR, "1")
    _set_padding(monkeypatch, padding)

    weights, ids = _route(router, indices_type=torch.int64)

    (gate_weights, gate_ids), = gate_outputs
    assert ids.dtype == torch.int64
    # .to() produced a fresh tensor; the gate's int32 ids stay unmasked.
    torch.testing.assert_close(gate_ids, IDS)
    assert weights is gate_weights
    expected_weights, expected_ids = _expected_masked(padding)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(ids, expected_ids.to(torch.int64))


@pytest.mark.parametrize("switch", ["0", "1"])
@pytest.mark.parametrize("padding", [None, [False, True]])
def test_missing_or_short_mask_returns_gate_outputs(monkeypatch, router, gate_outputs,
                                                    switch, padding):
    monkeypatch.setenv(ENV_VAR, switch)
    _set_padding(monkeypatch, padding)

    weights, ids = _route(router)

    (gate_weights, gate_ids), = gate_outputs
    assert weights is gate_weights
    assert ids is gate_ids
    torch.testing.assert_close(weights, WEIGHTS)
    torch.testing.assert_close(ids, IDS)


@pytest.mark.parametrize("switch", ["0", "1"])
def test_skip_padding_disabled_leaves_outputs(monkeypatch, router, gate_outputs,
                                              switch):
    monkeypatch.setenv(ENV_VAR, switch)
    monkeypatch.setattr(envs, "VLLM_MOE_SKIP_PADDING", False)

    def unexpected_context():
        raise AssertionError("padding mask must not be read")

    monkeypatch.setattr(forward_context, "get_forward_context", unexpected_context)

    weights, ids = _route(router)

    (gate_weights, gate_ids), = gate_outputs
    assert weights is gate_weights
    assert ids is gate_ids
    torch.testing.assert_close(ids, IDS)
