# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""PCP-aware MoE dispatch contracts for the HCU-owned runner."""

from __future__ import annotations

import importlib
import os
import warnings
from types import ModuleType, SimpleNamespace

import pytest
import torch

# Defer the replacement import until fixture setup.  Pytest imports every test
# module before running earlier platform fixtures, so a collection-time direct
# import would race the coordinator that owns the canonical replacement target.
os.environ["VLLM_PLUGINS"] = "__disabled__"
from vllm.utils import torch_utils


@pytest.fixture(scope="module")
def moe_op_registrations() -> dict[str, dict]:
    return {}


@pytest.fixture(scope="module")
def moe_runner_module(moe_op_registrations: dict[str, dict]) -> ModuleType:
    register_custom_op = torch_utils.direct_register_custom_op

    def register_without_duplicate_moe_ops(op_name, *args, **kwargs):
        if op_name in {
            "moe_forward",
            "moe_forward_shared",
            "moe_forward_shared_inplace",
        }:
            moe_op_registrations[op_name] = kwargs
            return None
        return register_custom_op(op_name, *args, **kwargs)

    torch_utils.direct_register_custom_op = register_without_duplicate_moe_ops
    try:
        module = importlib.import_module(
            "vllm_hcu.model_executor.layers.fused_moe.moe_runner"
        )
    finally:
        torch_utils.direct_register_custom_op = register_custom_op
    assert module.MoERunner.__module__.startswith("vllm_hcu.")
    return module


def test_inplace_moe_forward_shared_satisfies_aot_mutation_contract(
    monkeypatch: pytest.MonkeyPatch,
    moe_runner_module: ModuleType,
    moe_op_registrations: dict[str, dict],
) -> None:
    """AOT must preserve the input mutation without returning an input alias."""

    class Layer:
        def _forward_impl(self, hidden_states, *_args, **_kwargs):
            hidden_states.add_(1)
            return torch.full_like(hidden_states, 7), hidden_states

    monkeypatch.setattr(
        moe_runner_module,
        "get_layer_from_name",
        lambda _name: Layer(),
    )
    registration = moe_op_registrations["moe_forward_shared_inplace"]
    test_library = torch.library.Library("vllm_hcu_test_moe", "FRAGMENT")
    torch_utils.direct_register_custom_op(
        op_name="moe_forward_shared_inplace",
        op_func=registration["op_func"],
        mutates_args=registration["mutates_args"],
        fake_impl=registration["fake_impl"],
        target_lib=test_library,
        dispatch_key="CPU",
        tags=registration["tags"],
    )
    torch.library.opcheck(
        torch.ops.vllm_hcu_test_moe.moe_forward_shared_inplace.default,
        (
            torch.zeros((2, 2)),
            None,
            torch.zeros((2, 2)),
            None,
            None,
            None,
            None,
            None,
            "test-layer",
            0,
        ),
        test_utils=("test_schema",),
    )

    def invoke(hidden_states):
        return torch.ops.vllm_hcu_test_moe.moe_forward_shared_inplace(
            hidden_states,
            None,
            hidden_states,
            None,
            None,
            None,
            None,
            None,
            "test-layer",
            0,
        )

    compiled = torch.compile(invoke, backend="aot_eager", fullgraph=True)
    hidden_states = torch.zeros((2, 2))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=r".*moe_forward_shared.*custom operator.*",
            category=UserWarning,
        )
        shared_output = compiled(hidden_states)

    torch.testing.assert_close(hidden_states, torch.ones_like(hidden_states))
    torch.testing.assert_close(shared_output, torch.full_like(hidden_states, 7))
    assert not torch._C._is_alias_of(shared_output, hidden_states)


def make_hidden() -> torch.Tensor:
    return torch.tensor([[10.0, 11.0], [20.0, 21.0]])


def make_logits() -> torch.Tensor:
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.parametrize(
    ("supports_inplace", "dp_size", "sequence_parallel", "pcp_size", "expected"),
    [
        (True, 1, False, 1, True),
        (False, 1, False, 1, False),
        (True, 2, False, 1, False),
        (True, 1, True, 1, False),
        (True, 1, False, 2, False),
    ],
)
def test_inplace_shared_output_requires_local_inplace_kernel(
    moe_runner_module: ModuleType,
    supports_inplace: bool,
    dp_size: int,
    sequence_parallel: bool,
    pcp_size: int,
    expected: bool,
) -> None:
    """Dispatch or an out-of-place kernel must retain the tuple-returning op."""

    runner = object.__new__(moe_runner_module.MoERunner)
    runner._shared_experts = object()
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(
            supports_inplace_output=supports_inplace,
            supports_internal_mk=False,
        )
    )
    runner.moe_config = SimpleNamespace(
        dp_size=dp_size,
        is_sequence_parallel=sequence_parallel,
        pcp_size=pcp_size,
    )

    assert runner._can_use_inplace_shared_output() is expected


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
def make_runner(
    monkeypatch: pytest.MonkeyPatch,
    moe_runner_module: ModuleType,
):
    def make(pcp: int, use_all2all_kernels: bool):
        runner = object.__new__(moe_runner_module.MoERunner)
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
        monkeypatch.setattr(moe_runner_module, "get_pcp_group", lambda: group)
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


@pytest.mark.parametrize("uses_mutated_output", [False, True])
def test_shared_and_routed_outputs_keep_local_token_order_before_addition(
    monkeypatch: pytest.MonkeyPatch,
    moe_runner_module: ModuleType,
    uses_mutated_output: bool,
) -> None:
    """Reordering either local output before the shared+routed add is a bug."""

    runner = object.__new__(moe_runner_module.MoERunner)
    runner.moe_config = SimpleNamespace(hidden_dim_unpadded=2)
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(has_unpadded_output=False),
    )
    runner.routed_input_transform = None
    runner.routed_output_transform = None
    runner.routed_scaling_factor = 1.0
    runner._shared_experts = object()
    runner.router = object()
    shared_output = torch.tensor([[100.0, 101.0], [200.0, 201.0]])
    fused_output = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
    runner._forward_uses_mutated_hidden_states = uses_mutated_output
    if uses_mutated_output:

        def forward_entry(hidden_states, *_args):
            hidden_states.copy_(fused_output)
            return shared_output

        runner.__dict__["_forward_entry"] = forward_entry
    else:
        runner.__dict__["_forward_entry"] = lambda *_args: (
            shared_output,
            fused_output,
        )
    monkeypatch.setattr(
        moe_runner_module.MoERunner,
        "_maybe_pad_hidden_states",
        lambda self, shared, hidden: (hidden, None, None),
    )
    monkeypatch.setattr(
        moe_runner_module.MoERunner,
        "_encode_layer_name",
        lambda self: "test-layer",
    )
    monkeypatch.setattr(
        moe_runner_module.MoERunner,
        "_maybe_reduce_shared_expert_output",
        lambda self, shared: shared,
    )
    monkeypatch.setattr(
        moe_runner_module.MoERunner,
        "_maybe_reduce_final_output",
        lambda self, output, _truncate: output,
    )
    monkeypatch.setattr(
        moe_runner_module.MoERunner,
        "_maybe_add_zero_expert_output",
        lambda self, output: output,
    )

    output = runner.forward(make_hidden(), make_logits())

    torch.testing.assert_close(
        output,
        torch.tensor([[110.0, 112.0], [220.0, 222.0]]),
    )
