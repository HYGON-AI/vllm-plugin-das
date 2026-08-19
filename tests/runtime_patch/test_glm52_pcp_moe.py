# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""PCP-aware MoE dispatch contracts for the HCU-owned runner."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

# This test imports the HCU-owned runner directly, outside the runtime patch
# coordinator that owns the upstream replacement target.
os.environ["VLLM_PLUGINS"] = "__disabled__"
from vllm.utils import torch_utils

_register_custom_op = torch_utils.direct_register_custom_op


def _register_without_duplicate_moe_ops(op_name, *args, **kwargs):
    if op_name in {"moe_forward", "moe_forward_shared"}:
        return None
    return _register_custom_op(op_name, *args, **kwargs)


torch_utils.direct_register_custom_op = _register_without_duplicate_moe_ops

from vllm_hcu.model_executor.layers.fused_moe import moe_runner

torch_utils.direct_register_custom_op = _register_custom_op


def make_hidden() -> torch.Tensor:
    return torch.tensor([[10.0, 11.0], [20.0, 21.0]])


def make_logits() -> torch.Tensor:
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]])


class _PCPCollectives:
    """Two-rank in-memory PCP collective with observable token ordering."""

    def __init__(self) -> None:
        self.all_gather_count = 0
        self.reduce_scatter_count = 0
        self._peer_tensors = [
            torch.tensor([[30.0, 31.0], [40.0, 41.0]]),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        ]

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        assert dim == 0
        peer = self._peer_tensors[self.all_gather_count]
        self.all_gather_count += 1
        return torch.cat((tensor, peer), dim=dim)

    def reduce_scatter(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        assert dim == 0
        self.reduce_scatter_count += 1
        return tensor[: tensor.shape[0] // 2]


@pytest.fixture
def make_runner(monkeypatch: pytest.MonkeyPatch):
    def make(pcp: int, use_all2all_kernels: bool):
        runner = object.__new__(moe_runner.MoERunner)
        runner.moe_config = SimpleNamespace(
            pcp_size=pcp,
            dp_size=1,
            is_sequence_parallel=False,
            moe_parallel_config=SimpleNamespace(
                use_all2all_kernels=use_all2all_kernels,
            ),
        )
        runner.routed_experts = SimpleNamespace(
            quant_method=SimpleNamespace(supports_internal_mk=False),
        )
        runner._shared_experts = None
        group = _PCPCollectives()
        monkeypatch.setattr(moe_runner, "get_pcp_group", lambda: group)
        return runner, group

    return make


@pytest.mark.parametrize(
    ("pcp", "use_all2all_kernels", "gathers", "reduce_scatters"),
    [(1, False, 0, 0), (2, False, 2, 1), (2, True, 0, 0)],
)
def test_moe_uses_exactly_one_pcp_dispatch_path(
    make_runner, pcp, use_all2all_kernels, gathers, reduce_scatters
) -> None:
    """An all-to-all kernel must not be wrapped in fallback PCP collectives."""

    runner, group = make_runner(pcp, use_all2all_kernels)
    hidden, logits = runner._maybe_dispatch(make_hidden(), make_logits())
    combined = runner._maybe_combine(None, hidden)

    expected_dispatched_hidden = (
        torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]])
        if pcp == 2 and not use_all2all_kernels
        else make_hidden()
    )
    expected_dispatched_logits = (
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        if pcp == 2 and not use_all2all_kernels
        else make_logits()
    )
    torch.testing.assert_close(hidden, expected_dispatched_hidden)
    torch.testing.assert_close(logits, expected_dispatched_logits)
    torch.testing.assert_close(combined, make_hidden())
    assert group.all_gather_count == gathers
    assert group.reduce_scatter_count == reduce_scatters


@pytest.mark.parametrize("use_all2all_kernels", [False, True])
def test_pcp_dispatch_requires_router_logits(
    make_runner, use_all2all_kernels: bool
) -> None:
    """Removing the PCP routing guard would admit unusable preselected input."""

    runner, _ = make_runner(2, use_all2all_kernels)

    with pytest.raises(RuntimeError, match="without router_logits"):
        runner._maybe_dispatch(make_hidden(), None)


def test_shared_and_routed_outputs_keep_local_token_order_before_addition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reordering either local output before the shared+routed add is a bug."""

    runner = object.__new__(moe_runner.MoERunner)
    runner.moe_config = SimpleNamespace(hidden_dim_unpadded=2)
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(has_unpadded_output=False),
    )
    runner.routed_input_transform = None
    runner.routed_output_transform = None
    runner.routed_scaling_factor = 1.0
    runner._shared_experts = object()
    runner.router = object()
    runner.__dict__["_forward_entry"] = lambda *_args: (
        torch.tensor([[100.0, 101.0], [200.0, 201.0]]),
        torch.tensor([[10.0, 11.0], [20.0, 21.0]]),
    )
    monkeypatch.setattr(
        moe_runner.MoERunner,
        "_maybe_pad_hidden_states",
        lambda self, shared, hidden: (hidden, None, None),
    )
    monkeypatch.setattr(
        moe_runner.MoERunner,
        "_encode_layer_name",
        lambda self: "test-layer",
    )
    monkeypatch.setattr(
        moe_runner.MoERunner,
        "_maybe_reduce_shared_expert_output",
        lambda self, shared: shared,
    )
    monkeypatch.setattr(
        moe_runner.MoERunner,
        "_maybe_reduce_final_output",
        lambda self, output, _truncate: output,
    )
    monkeypatch.setattr(
        moe_runner.MoERunner,
        "_maybe_add_zero_expert_output",
        lambda self, output: output,
    )

    output = runner.forward(make_hidden(), make_logits())

    torch.testing.assert_close(
        output,
        torch.tensor([[110.0, 112.0], [220.0, 222.0]]),
    )
