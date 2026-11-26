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


def test_fallback_moe_forward_shared_satisfies_aot_mutation_contract(
    monkeypatch: pytest.MonkeyPatch,
    moe_runner_module: ModuleType,
    moe_op_registrations: dict[str, dict],
) -> None:
    """The tuple fallback must clone the mutated routed output."""

    class Layer:
        _ep_fused_combine = False

        def _forward_impl(self, hidden_states, *_args, **_kwargs):
            hidden_states.add_(1)
            return torch.full_like(hidden_states, 7), hidden_states

    monkeypatch.setattr(
        moe_runner_module,
        "get_layer_from_name",
        lambda _name: Layer(),
    )
    registration = moe_op_registrations["moe_forward_shared"]
    test_library = torch.library.Library(
        "vllm_hcu_test_moe_fallback", "FRAGMENT"
    )
    torch_utils.direct_register_custom_op(
        op_name="moe_forward_shared",
        op_func=registration["op_func"],
        mutates_args=registration["mutates_args"],
        fake_impl=registration["fake_impl"],
        target_lib=test_library,
        dispatch_key="CPU",
        tags=registration["tags"],
    )
    torch.library.opcheck(
        torch.ops.vllm_hcu_test_moe_fallback.moe_forward_shared.default,
        (
            torch.zeros((2, 2)),
            None,
            None,
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
        return torch.ops.vllm_hcu_test_moe_fallback.moe_forward_shared(
            hidden_states,
            None,
            None,
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
        shared_output, fused_output = compiled(hidden_states)

    torch.testing.assert_close(hidden_states, torch.ones_like(hidden_states))
    torch.testing.assert_close(shared_output, torch.full_like(hidden_states, 7))
    torch.testing.assert_close(fused_output, torch.ones_like(hidden_states))
    assert not torch._C._is_alias_of(shared_output, hidden_states)
    assert not torch._C._is_alias_of(fused_output, hidden_states)


def make_hidden() -> torch.Tensor:
    return torch.tensor([[10.0, 11.0], [20.0, 21.0]])


def make_logits() -> torch.Tensor:
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.parametrize(
    ("use_deepep", "use_aiter", "expected"),
    [
        (False, False, True),
        (True, False, False),
        (False, True, False),
        (True, True, False),
    ],
)
def test_slimquant_marlin_only_advertises_local_inplace_output(
    monkeypatch: pytest.MonkeyPatch,
    use_deepep: bool,
    use_aiter: bool,
    expected: bool,
) -> None:
    module = importlib.import_module(
        "vllm_hcu.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe_marlin"
    )
    method = object.__new__(module.CompressedTensorsW8A8FP8MarlinMoEMethod)
    method.moe = object()
    method.use_deepep = use_deepep
    monkeypatch.setattr(
        module,
        "_is_hcu_aiter_w8a8_moe_requested",
        lambda moe: use_aiter,
    )

    assert method.supports_inplace_output is expected


def test_forward_entry_stays_on_official_shared_op_after_reconfiguration(
    monkeypatch: pytest.MonkeyPatch,
    moe_runner_module: ModuleType,
) -> None:
    """AITER capabilities must not select a plugin-only custom op."""

    class RoutedExperts:
        def __init__(self) -> None:
            self.quant_method = SimpleNamespace(
                supports_inplace_output=True,
                supports_internal_mk=False,
            )

        def _replace_quant_method(self, quant_method) -> None:
            self.quant_method = quant_method

        def _set_moe_config(self, _moe_config) -> None:
            return None

    class SharedExperts:
        def _set_moe_config(self, _moe_config) -> None:
            return None

    monkeypatch.setattr(moe_runner_module.current_platform, "is_cpu", lambda: True)
    monkeypatch.setattr(moe_runner_module.current_platform, "is_tpu", lambda: False)

    runner = object.__new__(moe_runner_module.MoERunner)
    runner._ep_fused_combine = False
    runner.routed_experts = RoutedExperts()
    runner._shared_experts = SharedExperts()
    runner.moe_config = SimpleNamespace(
        dp_size=1,
        is_sequence_parallel=False,
        pcp_size=1,
    )

    runner._forward_entry = runner._select_forward()
    assert runner._forward_entry is moe_runner_module._moe_forward_shared

    runner._replace_quant_method(
        SimpleNamespace(
            supports_inplace_output=False,
            supports_internal_mk=False,
        )
    )
    assert runner._forward_entry is moe_runner_module._moe_forward_shared

    runner._replace_quant_method(
        SimpleNamespace(
            supports_inplace_output=True,
            supports_internal_mk=False,
        )
    )
    assert runner._forward_entry is moe_runner_module._moe_forward_shared

    runner._set_moe_config(
        SimpleNamespace(
            dp_size=1,
            is_sequence_parallel=False,
            pcp_size=2,
        )
    )
    assert runner._forward_entry is moe_runner_module._moe_forward_shared


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
        runner._ep_fused_combine = False
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


def test_shared_and_routed_outputs_keep_local_token_order_before_addition(
    monkeypatch: pytest.MonkeyPatch,
    moe_runner_module: ModuleType,
) -> None:
    """Reordering either local output before the shared+routed add is a bug."""

    runner = object.__new__(moe_runner_module.MoERunner)
    runner._ep_fused_combine = False
    runner.moe_config = SimpleNamespace(hidden_dim_unpadded=2)
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(has_unpadded_output=False, moe_kernel=None),
    )
    runner.routed_input_transform = None
    runner.routed_output_transform = None
    runner.routed_scaling_factor = 1.0
    runner._shared_experts = object()
    runner.router = object()
    shared_output = torch.tensor([[100.0, 101.0], [200.0, 201.0]])
    fused_output = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
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
        lambda self, output, _truncate, _is_reduced=False: output,
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


@pytest.mark.parametrize(
    "pcp_size,use_all2all_kernels", [(1, False), (2, False), (2, True)]
)
@pytest.mark.parametrize("shape", [(2, 2), (1, 2, 2)])
def test_hy4_gate_runs_once_before_runner_dispatch(
    monkeypatch, moe_runner_module, pcp_size, use_all2all_kernels, shape
):
    """Catch duplicate outer gating using the real HCU runner gate/dispatch path.

    Only communication and expert kernels are CPU doubles; this is not a
    hardware accuracy or performance test.
    """
    from vllm_hcu.models.hy_v4.moe import HYV4MoEFused

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden):
            self.calls += 1
            return torch.nn.functional.linear(hidden.float(), torch.eye(2)), None

    class Collectives:
        def all_gather(self, value, dim=0):
            assert value is not None
            return torch.cat((value, value), dim=dim)

        def reduce_scatter(self, value, dim=0):
            return value.chunk(2, dim=dim)[0]

    runner = object.__new__(moe_runner_module.MoERunner)
    torch.nn.Module.__init__(runner)
    runner._ep_fused_combine = False
    runner.gate = Gate()
    runner._fse_fuse_gate = False
    runner._shared_experts = None
    runner.moe_config = SimpleNamespace(
        dp_size=1,
        pcp_size=pcp_size,
        is_sequence_parallel=False,
        moe_parallel_config=SimpleNamespace(use_all2all_kernels=use_all2all_kernels),
    )
    runner.routed_experts = SimpleNamespace(
        _ensure_moe_quant_config_init=lambda: None,
        quant_method=SimpleNamespace(supports_internal_mk=False),
    )
    monkeypatch.setattr(moe_runner_module, "get_pcp_group", lambda: Collectives())
    monkeypatch.setattr(
        moe_runner_module,
        "get_forward_context",
        lambda: SimpleNamespace(dp_metadata=None),
    )

    def cpu_experts(*, hidden_states, router_logits, **kwargs):
        assert router_logits is not None
        assert router_logits.dtype == torch.float32
        torch.testing.assert_close(router_logits, hidden_states.float())
        return None, hidden_states * router_logits.sigmoid()

    monkeypatch.setattr(runner, "_apply_quant_method", cpu_experts)

    class Experts(torch.nn.Module):
        def forward(self, *, hidden_states, router_logits):
            return runner._forward_impl(hidden_states, router_logits, None)

    layer = HYV4MoEFused.__new__(HYV4MoEFused)
    torch.nn.Module.__init__(layer)
    layer.gate = runner.gate
    layer.experts = Experts()
    hidden = torch.tensor([[0.0, 1.0], [2.0, 3.0]]).reshape(shape)

    actual = layer(hidden)

    assert runner.gate.calls == 1
    assert actual.shape == hidden.shape
    torch.testing.assert_close(actual, hidden * hidden.sigmoid())


@pytest.mark.parametrize("pcp", [2, 8])
def test_hash_router_receives_gathered_auxiliaries_in_real_forward_impl(
    monkeypatch, moe_runner_module, pcp
):
    from contextlib import nullcontext

    runner = object.__new__(moe_runner_module.MoERunner)
    torch.nn.Module.__init__(runner)
    runner.gate = None
    runner._ep_fused_combine = False
    runner._shared_experts = None
    runner.moe_config = SimpleNamespace(
        pcp_size=pcp, dp_size=1, is_sequence_parallel=False,
        moe_parallel_config=SimpleNamespace(use_all2all_kernels=False),
    )
    runner.routed_experts = SimpleNamespace(
        _ensure_moe_quant_config_init=lambda: None,
        quant_method=SimpleNamespace(supports_internal_mk=False),
    )
    padding = torch.tensor([False, True])
    context = SimpleNamespace(is_padding=padding)
    group = SimpleNamespace(
        all_gather=lambda tensor, dim=0: torch.cat([tensor] * pcp, dim=dim),
        reduce_scatter=lambda tensor, dim=0: tensor.chunk(pcp, dim=dim)[0],
    )
    monkeypatch.setattr(moe_runner_module, "get_pcp_group", lambda: group)
    monkeypatch.setattr(moe_runner_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(moe_runner_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(runner, "_sequence_parallel_context", nullcontext)
    monkeypatch.setattr(runner, "_maybe_sync_shared_experts_stream", lambda *a, **k: None)
    ids = torch.tensor([23, 24])

    def route_and_apply(*, hidden_states, router_logits, input_ids, **kwargs):
        assert hidden_states.shape[0] == router_logits.shape[0] == 2 * pcp
        torch.testing.assert_close(input_ids, ids.repeat(pcp))
        torch.testing.assert_close(context.is_padding, padding.repeat(pcp))
        return None, hidden_states

    monkeypatch.setattr(runner, "_apply_quant_method", route_and_apply)
    result = runner._forward_impl(make_hidden(), make_logits(), None, ids)
    torch.testing.assert_close(result, make_hidden())
    assert context.is_padding is padding


@pytest.mark.parametrize("with_ids", [False, True])
@pytest.mark.parametrize("with_padding", [False, True])
@pytest.mark.parametrize("raise_in_router", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("pcp_size", [1, 2])
def test_naive_dispatch_aligns_routing_auxiliaries(
    monkeypatch, moe_runner_module, with_ids, with_padding,
    raise_in_router, sequence_parallel, pcp_size,
):
    """DP ranks may have different row counts and different padding bits."""
    # This matrix models DP/PCP only. Real TP>1 SP coordinates are below.
    monkeypatch.setattr(moe_runner_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    from contextlib import nullcontext

    runner = object.__new__(moe_runner_module.MoERunner)
    torch.nn.Module.__init__(runner)
    runner._ep_fused_combine = False
    runner.gate = None
    runner._shared_experts = None
    runner.moe_config = SimpleNamespace(
        dp_size=4, tp_size=1, pcp_size=pcp_size,
        is_sequence_parallel=sequence_parallel,
        moe_parallel_config=SimpleNamespace(use_all2all_kernels=False),
    )
    runner.routed_experts = SimpleNamespace(
        _ensure_moe_quant_config_init=lambda: None,
        quant_method=SimpleNamespace(supports_internal_mk=False),
    )
    # Ranks have [2, 1, 3, 2] rows, with nonidentical padding patterns.
    masks = [torch.tensor(bits) for bits in (
        [False, True], [False], [True, False, True], [False, False],
    )]
    ids = [torch.arange(10 * rank, 10 * rank + len(mask))
           for rank, mask in enumerate(masks)]
    # The runner's backing buffer may extend beyond this layer's local rows.
    original_mask = torch.tensor([False, True, True, True]) if with_padding else None
    context = SimpleNamespace(is_padding=original_mask)
    expected_mask = torch.cat(masks)
    hidden = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    logits = hidden + 100
    calls = []

    class Group:
        def dispatch_router_logits(self, x, gate, sp, extra_tensors=None):
            assert sp == sequence_parallel
            calls.append(extra_tensors)
            torch.testing.assert_close(x, hidden[:2])
            torch.testing.assert_close(gate, logits[:2])
            if extra_tensors is None:
                return hidden, logits
            gathered = []
            for tensor in extra_tensors:
                if tensor.dtype == torch.int32:
                    torch.testing.assert_close(tensor, masks[0].to(torch.int32))
                    gathered.append(torch.cat(masks).to(torch.int32))
                else:
                    torch.testing.assert_close(tensor, ids[0])
                    gathered.append(torch.cat(ids))
            return hidden, logits, gathered

        def combine(self, tensor, sp):
            assert sp == sequence_parallel
            return tensor[:2 * pcp_size]

    class PCPGroup:
        def all_gather(self, tensor, dim=0):
            assert tensor.dtype != torch.bool
            peer = ~tensor.to(torch.bool) if tensor.dtype == torch.int32 else tensor + 200
            if tensor.dtype == torch.int32:
                peer = peer.to(torch.int32)
            return torch.cat((tensor, peer), dim=dim)

        def reduce_scatter(self, tensor, dim=0):
            return tensor[:2]

    if pcp_size == 2:
        expected_mask = torch.cat((expected_mask, ~expected_mask))
    expected_hidden = hidden if pcp_size == 1 else torch.cat((hidden, hidden + 200))
    expected_logits = logits if pcp_size == 1 else torch.cat((logits, logits + 200))
    expected_ids = torch.cat(ids)
    if pcp_size == 2:
        expected_ids = torch.cat((expected_ids, expected_ids + 200))
    monkeypatch.setattr(moe_runner_module, "get_pcp_group", lambda: PCPGroup())
    monkeypatch.setattr(moe_runner_module, "get_ep_group", lambda: Group())
    monkeypatch.setattr(moe_runner_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(moe_runner_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(runner, "_sequence_parallel_context", nullcontext)
    monkeypatch.setattr(runner, "_maybe_sync_shared_experts_stream", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_use_int8_allgatherv_dispatch", lambda _x: False)

    def route_and_apply(*, hidden_states, router_logits, input_ids, **kwargs):
        torch.testing.assert_close(hidden_states, expected_hidden)
        torch.testing.assert_close(router_logits, expected_logits)
        if with_padding:
            assert context.is_padding.shape[0] == router_logits.shape[0]
            torch.testing.assert_close(context.is_padding, expected_mask)
        else:
            assert context.is_padding is None
        if with_ids:
            torch.testing.assert_close(input_ids, expected_ids)
        else:
            assert input_ids is None
        if raise_in_router:
            raise RuntimeError("router failure")
        return None, hidden_states

    monkeypatch.setattr(runner, "_apply_quant_method", route_and_apply)
    if raise_in_router:
        with pytest.raises(RuntimeError, match="router failure"):
            runner._forward_impl(hidden[:2], logits[:2], None, ids[0] if with_ids else None)
    else:
        result = runner._forward_impl(hidden[:2], logits[:2], None, ids[0] if with_ids else None)
        torch.testing.assert_close(result, hidden[:2])
    assert context.is_padding is original_mask
    assert len(calls) == 1


@pytest.mark.parametrize("use_ep", [False, True])
@pytest.mark.parametrize("raise_in_router", [False, True])
@pytest.mark.parametrize("tp_size,bits,bitmap_scope", [
    pytest.param(tp, bits, scope, id=f"tp{tp}-{name}-{scope}")
    for tp in (2, 4)
    for name, bits in (
        ("tail", [False, False, True, True]),
        ("interior", [False, True, False, False]),
        ("uneven", [False, True, False]),
        ("single", [False]),
        ("empty", []),
    )
    for scope in ("tp_global", "tp_local", "inferred_local")
    if scope != "inferred_local" or (len(bits) + tp - 1) // tp > 1
])
def test_sequence_parallel_padding_uses_activation_coordinates(
    monkeypatch, moe_runner_module, tp_size, use_ep, bitmap_scope,
    raise_in_router, bits,
):
    """Use Core's real SP chunk; double only the rank-ordered collective.

    The one-row case has identical local/global shapes. Such local producers
    must declare their coordinate domain rather than rely on shape inference.
    """
    from contextlib import nullcontext
    from vllm.model_executor.models import utils as model_utils

    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    global_mask = torch.tensor(bits, dtype=torch.bool)
    global_hidden = torch.arange(2 * len(bits), dtype=torch.float32).reshape(-1, 2)
    local_rows = (len(bits) + tp_size - 1) // tp_size
    expected_mask = torch.cat((
        global_mask, torch.ones(tp_size * local_rows - len(bits), dtype=torch.bool),
    ))
    rank_holder = [0]
    for module in (model_utils, moe_runner_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: tp_size)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank_holder[0], raising=False)
    chunks = []
    for rank in range(tp_size):
        rank_holder[0] = rank
        chunks.append(model_utils.sequence_parallel_chunk_impl(global_hidden))
    expected_hidden = torch.cat(chunks)
    expected_logits = expected_hidden + 100

    runner = object.__new__(moe_runner_module.MoERunner)
    torch.nn.Module.__init__(runner)
    runner.moe_config = SimpleNamespace(
        dp_size=1, pcp_size=1, sp_size=tp_size,
        # EP collapses expert TP, but not the activation SP partition.
        tp_size=1 if use_ep else tp_size,
        is_sequence_parallel=True,
        moe_parallel_config=SimpleNamespace(use_all2all_kernels=False),
    )
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(supports_internal_mk=False),
    )
    for rank in range(tp_size):
        rank_holder[0] = rank
        runner.moe_config.tp_rank = 0 if use_ep else rank
        start = rank * local_rows
        local_mask = expected_mask[start:start + local_rows]
        original_mask = global_mask if bitmap_scope == "tp_global" else local_mask.clone()
        additional_kwargs = (
            {"hcu_moe_padding_is_local": True} if bitmap_scope == "tp_local" else {}
        )
        context = SimpleNamespace(is_padding=original_mask, additional_kwargs=additional_kwargs)
        before = original_mask.clone()
        calls = []

        class Group:
            def dispatch_router_logits(self, x, gate, sp, extra_tensors=None):
                assert sp
                torch.testing.assert_close(x, chunks[rank])
                torch.testing.assert_close(gate, chunks[rank] + 100)
                assert len(extra_tensors) == 1
                wire_mask, = extra_tensors
                assert wire_mask.dtype == torch.int32
                # Other ranks contribute independent expected slices. This
                # rank must contribute what the real runner actually sent.
                gathered = expected_mask.to(torch.int32)
                gathered[start:start + local_rows] = wire_mask
                calls.append(wire_mask)
                return expected_hidden, expected_logits, [gathered]

        monkeypatch.setattr(moe_runner_module, "get_ep_group", lambda: Group())
        monkeypatch.setattr(moe_runner_module, "is_forward_context_available", lambda: True)
        monkeypatch.setattr(moe_runner_module, "get_forward_context", lambda: context)
        error = pytest.raises(RuntimeError, match="router failure") if raise_in_router else nullcontext()
        with error:
            with runner._dispatched_routing_inputs(chunks[rank], chunks[rank] + 100, None) as routed:
                routed_hidden, routed_logits, routed_ids = routed
                torch.testing.assert_close(routed_hidden, expected_hidden)
                torch.testing.assert_close(routed_logits, expected_logits)
                assert routed_ids is None
                torch.testing.assert_close(context.is_padding, expected_mask)
                # Model the native router sentinel, including interior masks.
                ids = torch.arange(expected_mask.numel())
                torch.testing.assert_close(
                    ids.masked_fill(context.is_padding, -1),
                    ids.masked_fill(expected_mask, -1),
                )
                if raise_in_router:
                    raise RuntimeError("router failure")
        assert len(calls) == 1
        assert context.is_padding is original_mask
        assert context.additional_kwargs is additional_kwargs
        torch.testing.assert_close(original_mask, before)


@pytest.mark.parametrize(("tp_size", "num_tokens"), [(2, 2), (4, 2), (4, 4)])
@pytest.mark.parametrize("raise_in_router", [False, True])
@pytest.mark.parametrize("initial_padding", [False, True])
def test_hc_sp_producer_single_row_bitmap_reaches_runner(
    monkeypatch, moe_runner_module, tp_size, num_tokens,
    raise_in_router, initial_padding,
):
    """Exercise the real HC-SP producer, not a hand-marked local bitmap."""
    from contextlib import nullcontext
    from vllm_hcu.patch.worker.core_fix import patch_qwen4_exp_sp as sp

    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    rank_holder = [0]
    for module in (sp, moe_runner_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: tp_size)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank_holder[0])
        monkeypatch.setattr(module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(sp, "envs", SimpleNamespace(VLLM_MOE_SKIP_PADDING=True))
    runner = object.__new__(moe_runner_module.MoERunner)
    torch.nn.Module.__init__(runner)
    runner.moe_config = SimpleNamespace(
        dp_size=2, pcp_size=1, sp_size=tp_size, tp_size=1,
        is_sequence_parallel=True,
        moe_parallel_config=SimpleNamespace(use_all2all_kernels=False),
    )
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(supports_internal_mk=False),
    )
    expected = torch.tensor([False] * num_tokens + [True] * (tp_size - num_tokens))
    expected = expected.repeat(2)
    hidden = torch.arange(num_tokens * 2, dtype=torch.float32).view(num_tokens, 2)
    gathered_masks = []
    for rank in range(tp_size):
        rank_holder[0] = rank
        context = SimpleNamespace(
            is_padding=torch.zeros(num_tokens, dtype=torch.bool) if initial_padding else None,
            additional_kwargs={},
        )
        for module in (sp, moe_runner_module):
            monkeypatch.setattr(module, "get_forward_context", lambda: context)
        local_hidden = sp.shard_tokens_for_hc_sp(hidden)
        original_mask = context.is_padding

        class Group:
            def dispatch_router_logits(self, x, logits, is_sp, extra_tensors=None):
                assert is_sp
                assert len(extra_tensors) == 1
                # Only collective/native boundaries are doubled. The current
                # rank's real producer/runner payload replaces its expected row.
                bitmap = expected.to(extra_tensors[0].dtype)
                bitmap[rank] = extra_tensors[0][0]
                gathered_masks.append(bitmap.to(torch.bool))
                return x.repeat(tp_size * 2, 1), logits.repeat(tp_size * 2, 1), [bitmap]

        monkeypatch.setattr(moe_runner_module, "get_ep_group", lambda: Group())
        error = pytest.raises(RuntimeError, match="router failure") if raise_in_router else nullcontext()
        with error:
            with runner._dispatched_routing_inputs(local_hidden, local_hidden + 1, None):
                torch.testing.assert_close(context.is_padding, expected)
                if raise_in_router:
                    raise RuntimeError("router failure")
        assert context.is_padding is original_mask
    assert len(gathered_masks) == tp_size


@pytest.mark.parametrize("skip_padding", ["0", "1"])
@pytest.mark.parametrize("with_ids", [False, True])
@pytest.mark.parametrize("rank", [0, 3])
def test_routing_bitmap_preserves_ipc_provider_selection(
    monkeypatch, moe_runner_module, skip_padding, with_ids, rank,
):
    """Real runner -> Core AGRS -> HCU wrapper -> real IPC eligibility."""
    from vllm.distributed.device_communicators import all2all
    from tests.runtime_patch.test_ipc_all_gather import (
        _logical_gather, _patched_communicator,
    )

    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", skip_padding)
    ag = _logical_gather(rank=rank)
    ag.world_size = 4
    decisions = []

    def native_gather(tensors, sizes):
        # Native communication only: record the wrapper's provider decision
        # and emulate rank-ordered rows with distinct per-rank padding masks.
        decisions.append("ipc")
        outputs = [torch.cat([t] * 4) for t in tensors]
        if len(tensors) > 2:
            assert tensors[-1].dtype == torch.int32
            assert tensors[-1].shape == (2,)
            outputs[-1] = torch.tensor([0, 1, 1, 0, 0, 0, 1, 1], dtype=torch.int32)
        return outputs

    monkeypatch.setattr(ag, "gather", native_gather)
    comm = _patched_communicator(monkeypatch, ag)
    comm.world_size, comm.rank_in_group = 4, rank
    masks = torch.tensor([False, True, True, False, False, False, True, True])
    original = masks[rank * 2:rank * 2 + 2].clone()
    context = SimpleNamespace(
        is_padding=original, additional_kwargs={},
        dp_metadata=SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda: [2] * 4),
    )
    for module in (all2all, moe_runner_module):
        monkeypatch.setattr(module, "get_forward_context", lambda: context)
    monkeypatch.setattr(all2all, "get_dp_group", lambda: comm)
    manager = object.__new__(all2all.AgRsAll2AllManager)
    manager.dp_world_size = 4
    monkeypatch.setattr(moe_runner_module, "get_ep_group", lambda: manager)
    monkeypatch.setattr(moe_runner_module, "is_forward_context_available", lambda: True)
    runner = object.__new__(moe_runner_module.MoERunner)
    torch.nn.Module.__init__(runner)
    runner.moe_config = SimpleNamespace(
        dp_size=4, pcp_size=1, tp_size=1, is_sequence_parallel=False,
        moe_parallel_config=SimpleNamespace(use_all2all_kernels=False),
    )
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(supports_internal_mk=False),
    )
    hidden = torch.zeros(2, 64, dtype=torch.bfloat16)
    logits = torch.zeros(2, 512)
    manager.dispatch_router_logits(hidden, logits)  # eligible baseline
    assert decisions == ["ipc"]
    ids = torch.arange(2, dtype=torch.int64) if with_ids else None
    with runner._dispatched_routing_inputs(hidden, logits, ids) as (x, gate, routed_ids):
        assert decisions == ["ipc", "ipc"]
        assert not comm.events  # never entered original all_gatherv fallback
        assert x.shape == (8, 64) and gate.shape == (8, 512)
        torch.testing.assert_close(context.is_padding, masks)
        if with_ids:
            torch.testing.assert_close(routed_ids, ids.repeat(4))
    assert context.is_padding is original
