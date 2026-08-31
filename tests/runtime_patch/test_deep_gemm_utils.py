# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _load_permute_function():
    source_path = (
        Path(__file__).parents[2]
        / "vllm_hcu/model_executor/layers/fused_moe/deep_gemm_utils.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "deepgemm_moe_permute"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    namespace: dict[str, object] = {
        "torch": torch,
        "mk": SimpleNamespace(ExpertTokensMetadata=object),
        "_HCU_TOKEN_ALIGNMENT": 256,
        "round_up": lambda value, multiple: (
            (value + multiple - 1) // multiple * multiple
        ),
    }
    exec(compile(ast.fix_missing_locations(module), source_path, "exec"), namespace)
    return namespace


def test_rocm_permute_uses_hcu_alignment_without_upstream_query():
    namespace = _load_permute_function()
    namespace["current_platform"] = SimpleNamespace(is_rocm=lambda: True)
    namespace["get_mk_alignment_for_contiguous_layout"] = lambda: (
        _ for _ in ()
    ).throw(AssertionError("upstream query invoked"))
    namespace["count_expert_num_tokens"] = lambda *_args: torch.ones(
        2, dtype=torch.int32
    )
    namespace["compute_aligned_M_and_alignment"] = (
        lambda **kwargs: (512, kwargs["alignment"])
    )
    scatter: dict[str, object] = {}
    namespace["ep_scatter"] = lambda **kwargs: scatter.update(kwargs)

    result = namespace["deepgemm_moe_permute"](
        aq=torch.zeros((2, 4), dtype=torch.int8),
        aq_scale=torch.ones((2, 1), dtype=torch.float32),
        topk_ids=torch.tensor([[0], [1]], dtype=torch.int32),
        local_num_experts=2,
        expert_map=None,
        expert_tokens_meta=None,
    )

    assert result[-1] == 256
    assert scatter["align_m"] == 256


def _load_deep_gemm_apply():
    source_path = (
        Path(__file__).parents[2]
        / "vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    experts = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DeepGemmExperts"
    )
    function = next(
        node
        for node in experts.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    namespace: dict[str, object] = {
        "torch": torch,
        "mk": SimpleNamespace(ExpertTokensMetadata=object),
        "MoEActivation": SimpleNamespace(SILU="silu"),
        "nullcontext": nullcontext,
        "_HCU_HT_EP_TOKEN_ALIGNMENT": 256,
        "compute_aligned_M_and_alignment": lambda **_kwargs: (2, 256),
        "get_mk_alignment_for_contiguous_layout": lambda: (128, 128),
        "_resize_cache": lambda tensor, shape: torch.zeros(
            shape, dtype=tensor.dtype
        ),
        "deepgemm_moe_permute": lambda **kwargs: (
            kwargs["aq"],
            kwargs["aq_scale"],
            torch.zeros(2, dtype=torch.int32),
            torch.zeros_like(kwargs["topk_ids"], dtype=torch.int32),
            256,
        ),
        "deepgemm_unpermute_and_reduce": lambda **_kwargs: None,
        "m_grouped_fp8_gemm_nt_contiguous": lambda *_args, **_kwargs: None,
    }
    exec(compile(ast.fix_missing_locations(module), source_path, "exec"), namespace)
    return namespace


def _run_w8a8_apply(
    namespace: dict[str, object], *, packed_weights: bool = False
) -> None:
    logical_n = 16 if packed_weights else 4
    logical_k = 64 if packed_weights else 4
    experts = SimpleNamespace(
        block_shape=(128, 128),
        quant_config=SimpleNamespace(use_int8_w8a8=True, use_fp8_w8a8=False),
        w1_scale=torch.ones((1, 1), dtype=torch.float32),
        w2_scale=torch.ones((1, 1), dtype=torch.float32),
        _hcu_logical_n=logical_n,
        _hcu_logical_k=logical_k,
        mxfp8=False,
        adjust_N_for_activation=lambda size, _activation: size // 2,
    )
    namespace["apply"](
        experts,
        output=torch.empty((2, logical_k)),
        hidden_states=torch.zeros((2, logical_k), dtype=torch.int8),
        w1=(
            torch.zeros((1, 1, 1, 4, 16, 16), dtype=torch.int8)
            if packed_weights
            else torch.zeros((1, 4, 4), dtype=torch.int8)
        ),
        w2=(
            torch.zeros((1, 1, 4, 4, 16, 16), dtype=torch.int8)
            if packed_weights
            else torch.zeros((1, 4, 4), dtype=torch.int8)
        ),
        topk_weights=torch.ones((2, 1)),
        topk_ids=torch.zeros((2, 1), dtype=torch.int32),
        activation="silu",
        global_num_experts=1,
        expert_map=None,
        a1q_scale=torch.ones((2, 1)),
        a2_scale=None,
        workspace13=torch.empty(32, dtype=torch.int8),
        workspace2=torch.empty(32),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )


def test_w8a8_apply_skips_alignment_scope_only_on_rocm(monkeypatch):
    deepgemm = ModuleType("deepgemm")
    deepgemm.m_grouped_i8_gemm_nt_contiguous = lambda *_args: None
    monkeypatch.setitem(sys.modules, "deepgemm", deepgemm)
    lightop = ModuleType("lightop")
    lightop.m_grouped_w8a8_gemm_nt_contig_asm = lambda *_args: None
    lightop.fuse_silu_mul_quant = lambda tensor, **kwargs: (
        kwargs["output"],
        torch.ones((tensor.shape[0], 1)),
    )
    monkeypatch.setitem(sys.modules, "lightop", lightop)

    hcu = _load_deep_gemm_apply()
    hcu["current_platform"] = SimpleNamespace(is_rocm=lambda: True)
    hcu["mk_alignment_scope"] = lambda _alignment: (_ for _ in ()).throw(
        AssertionError("upstream alignment scope invoked")
    )
    _run_w8a8_apply(hcu)

    events: list[object] = []

    class RecordingScope:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_args):
            events.append("exit")

    non_hcu = _load_deep_gemm_apply()
    non_hcu["current_platform"] = SimpleNamespace(is_rocm=lambda: False)
    non_hcu["mk_alignment_scope"] = lambda alignment: (
        events.append(("scope", alignment)) or RecordingScope()
    )
    _run_w8a8_apply(non_hcu)

    assert events == [("scope", 256), "enter", "exit"]


def test_w8a8_apply_uses_architecture_aware_deepgemm_contiguous_api(
    monkeypatch,
):
    lightop = ModuleType("lightop")

    def reject_raw_lightop(*_args):
        raise AssertionError("raw LightOP contiguous GEMM invoked")

    lightop.m_grouped_w8a8_gemm_nt_contig_asm = reject_raw_lightop
    lightop.fuse_silu_mul_quant = lambda tensor, **kwargs: (
        kwargs["output"],
        torch.ones((tensor.shape[0], 1)),
    )
    monkeypatch.setitem(sys.modules, "lightop", lightop)

    calls: list[tuple[object, ...]] = []
    deepgemm = ModuleType("deepgemm")
    deepgemm.m_grouped_i8_gemm_nt_contiguous = (
        lambda *args: calls.append(args)
    )
    monkeypatch.setitem(sys.modules, "deepgemm", deepgemm)
    hcu = _load_deep_gemm_apply()
    hcu["current_platform"] = SimpleNamespace(is_rocm=lambda: True)

    _run_w8a8_apply(hcu, packed_weights=True)

    assert len(calls) == 2


def _load_batched_deep_gemm_apply():
    source_path = (
        Path(__file__).parents[2]
        / "vllm_hcu/model_executor/layers/fused_moe/experts/"
        "batched_deep_gemm_moe.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    experts = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchedDeepGemmExperts"
    )
    function = next(
        node
        for node in experts.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    namespace: dict[str, object] = {
        "torch": torch,
        "mk": SimpleNamespace(ExpertTokensMetadata=object),
        "MoEActivation": SimpleNamespace(SILU="silu"),
        "_resize_cache": lambda tensor, shape: torch.zeros(
            shape, dtype=tensor.dtype
        ),
    }
    exec(compile(ast.fix_missing_locations(module), source_path, "exec"), namespace)
    return namespace


def test_w8a8_batched_apply_uses_masked_int8_deepgemm_api(
    monkeypatch,
):
    lightop = ModuleType("lightop")

    def reject_raw_lightop(*_args):
        raise AssertionError("raw LightOP masked GEMM invoked")

    lightop.m_grouped_w8a8_gemm_nt_masked = reject_raw_lightop
    lightop.fuse_silu_mul_quant_ep = lambda tensor, _counts: (
        tensor,
        torch.ones((*tensor.shape[:-1], 1)),
    )
    monkeypatch.setitem(sys.modules, "lightop", lightop)

    calls: list[tuple[object, ...]] = []
    deepgemm = ModuleType("deepgemm")
    deepgemm.__path__ = []
    deepgemm.m_grouped_i8_gemm_nt_masked = lambda *args: calls.append(args)
    m_group_gemm = ModuleType("deepgemm.m_group_gemm")

    def reject_w6_api(*_args):
        raise AssertionError("W6 low-latency masked API invoked")

    m_group_gemm.m_grouped_w8a8_gemm_nt_masked_ll = reject_w6_api
    monkeypatch.setitem(sys.modules, "deepgemm", deepgemm)
    monkeypatch.setitem(sys.modules, "deepgemm.m_group_gemm", m_group_gemm)
    hcu = _load_batched_deep_gemm_apply()
    hcu["current_platform"] = SimpleNamespace(is_rocm=lambda: True)
    experts = SimpleNamespace(
        block_shape=(128, 128),
        quant_config=SimpleNamespace(use_int8_w8a8=True, use_fp8_w8a8=False),
        w1_scale=torch.ones((1, 16), dtype=torch.float32),
        w2_scale=torch.ones((1, 64), dtype=torch.float32),
        _hcu_logical_n=16,
        _hcu_logical_k=64,
        moe_problem_size=lambda *_args: (1, 2, 16, 64, 1),
        estimate_expected_m=lambda **_kwargs: 2,
    )

    hcu["apply"](
        experts,
        output=torch.empty((1, 2, 64)),
        hidden_states=torch.zeros((1, 2, 64), dtype=torch.int8),
        w1=torch.zeros((1, 1, 1, 4, 16, 16), dtype=torch.int8),
        w2=torch.zeros((1, 1, 4, 4, 16, 16), dtype=torch.int8),
        topk_weights=torch.ones((2, 1)),
        topk_ids=torch.zeros((2, 1), dtype=torch.int32),
        activation="silu",
        global_num_experts=1,
        expert_map=None,
        a1q_scale=torch.ones((1, 2, 1)),
        a2_scale=None,
        workspace13=torch.empty(128),
        workspace2=torch.empty(128),
        expert_tokens_meta=SimpleNamespace(
            expert_num_tokens=torch.ones(1, dtype=torch.int32)
        ),
        apply_router_weight_on_input=False,
    )

    assert len(calls) == 2


def _make_w4a8_expert_layer() -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(
        torch.arange(2 * 8 * 2, dtype=torch.int8).reshape(2, 8, 2),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.arange(2 * 4 * 2, dtype=torch.int8).reshape(2, 4, 2),
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.ones((2, 8, 1), dtype=torch.float32),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.ones((2, 4, 1), dtype=torch.float32),
        requires_grad=False,
    )
    return layer


def test_w4a8_contiguous_packs_once_without_replacing_canonical_parameters(
    monkeypatch: pytest.MonkeyPatch,
):
    """Derived HIPC layouts must never take ownership from checkpoint params."""

    from vllm_hcu.model_executor.layers.quantization import (
        slimquant_w4a8_deepgemm_runtime as module,
    )

    layer = _make_w4a8_expert_layer()
    canonical_w13 = layer.w13_weight
    canonical_w2 = layer.w2_weight
    expected_w13 = canonical_w13.detach().clone()
    expected_w2 = canonical_w2.detach().clone()
    packed: list[torch.Tensor] = []

    def pack(weight: torch.Tensor) -> torch.Tensor:
        weight.fill_(41 + len(packed))
        packed.append(weight)
        return weight

    monkeypatch.setattr(module, "pack_w4a8_moe_hipc_weight", pack)

    experts = object.__new__(module.DeepEPDeepGemmW4A8ContiguousExperts)
    experts._deepgemm_w13 = None
    experts._deepgemm_w2 = None
    experts.process_weights_after_loading(layer)

    assert layer.w13_weight is canonical_w13
    assert layer.w2_weight is canonical_w2
    torch.testing.assert_close(layer.w13_weight, expected_w13)
    torch.testing.assert_close(layer.w2_weight, expected_w2)
    assert len(packed) == 2
    assert experts._deepgemm_w13.ndim == 3
    assert experts._deepgemm_w2.ndim == 3
    assert experts._deepgemm_w13.untyped_storage().data_ptr() != (
        layer.w13_weight.untyped_storage().data_ptr()
    )

    replacement = object.__new__(
        module.DeepEPDeepGemmW4A8ContiguousExperts
    )
    replacement._deepgemm_w13 = None
    replacement._deepgemm_w2 = None
    replacement.process_weights_after_loading(layer)

    assert len(packed) == 2
    assert replacement._deepgemm_w13 is experts._deepgemm_w13
    assert replacement._deepgemm_w2 is experts._deepgemm_w2


def test_w4a8_contiguous_rejects_invalid_channel_scale_before_packing(
    monkeypatch: pytest.MonkeyPatch,
):
    """A non-channel scale must fail before a derived layout is allocated."""

    from vllm_hcu.model_executor.layers.quantization import (
        slimquant_w4a8_deepgemm_runtime as module,
    )

    layer = _make_w4a8_expert_layer()
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.ones((2, 7, 1), dtype=torch.float32),
        requires_grad=False,
    )
    pack_calls = 0

    def pack(_weight: torch.Tensor) -> torch.Tensor:
        nonlocal pack_calls
        pack_calls += 1
        raise AssertionError("invalid channel scales reached the packer")

    monkeypatch.setattr(module, "pack_w4a8_moe_hipc_weight", pack)
    experts = object.__new__(module.DeepEPDeepGemmW4A8ContiguousExperts)
    experts._deepgemm_w13 = None
    experts._deepgemm_w2 = None

    with pytest.raises(RuntimeError, match="w13_weight_scale"):
        experts.process_weights_after_loading(layer)

    assert pack_calls == 0


def test_w4a8_contiguous_runs_two_hipc_gemms_with_expert_map_and_scales(
    monkeypatch: pytest.MonkeyPatch,
):
    """Both GEMM stages must retain scale and expert-map ownership."""

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm_hcu.model_executor.layers.quantization import (
        slimquant_w4a8_deepgemm_runtime as module,
    )

    w13_scale = torch.ones((2, 8, 1), dtype=torch.float32)
    w2_scale = torch.full((2, 4, 1), 2.0, dtype=torch.float32)
    experts = object.__new__(module.DeepEPDeepGemmW4A8ContiguousExperts)
    experts._deepgemm_w13 = torch.full((2, 1, 1, 1, 8, 2), 13, dtype=torch.int8)
    experts._deepgemm_w2 = torch.full((2, 1, 1, 1, 4, 2), 17, dtype=torch.int8)
    experts.quant_config = SimpleNamespace(
        w1_scale=w13_scale,
        w2_scale=w2_scale,
    )
    experts.adjust_N_for_activation = lambda n, _activation: n // 2

    hidden_states = torch.arange(8, dtype=torch.int8).reshape(2, 4)
    input_scale = torch.ones((2, 1), dtype=torch.float32)
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)
    topk_weights = torch.ones((2, 1), dtype=torch.float32)
    expert_map = torch.tensor([1, 0], dtype=torch.int32)
    m_indices = torch.tensor([0, 1], dtype=torch.int64)
    inv_perm = torch.tensor([1, 0], dtype=torch.int32)
    expert_tokens_meta = SimpleNamespace(
        expert_num_tokens=torch.tensor([1, 1], dtype=torch.int32),
        expert_num_tokens_cpu=torch.tensor([1, 1], dtype=torch.int32),
    )
    permute_call: dict[str, object] = {}
    reduce_call: dict[str, object] = {}
    gemm_calls: list[tuple[object, object, torch.Tensor]] = []

    monkeypatch.setattr(module, "compute_aligned_M", lambda **_kwargs: 2)
    monkeypatch.setattr(
        module,
        "_resize_cache",
        lambda tensor, shape: torch.empty(shape, dtype=tensor.dtype),
    )

    def permute(**kwargs: object):
        permute_call.update(kwargs)
        return hidden_states, input_scale, m_indices, inv_perm, 256

    monkeypatch.setattr(module, "deepgemm_moe_permute", permute)

    def grouped_gemm(a, b, output, passed_m_indices):
        gemm_calls.append((a, b, passed_m_indices))
        output.fill_(len(gemm_calls))

    monkeypatch.setattr(
        module,
        "m_grouped_w4a8_gemm_nt_contiguous_hipc",
        grouped_gemm,
    )

    def quantize(gate_up: torch.Tensor, **kwargs: object):
        del gate_up
        quant_output = kwargs["output"]
        assert isinstance(quant_output, torch.Tensor)
        quant_output.fill_(7)
        return quant_output, torch.full((2, 1), 0.5, dtype=torch.float32)

    monkeypatch.setattr(module, "fuse_silu_mul_quant", quantize)

    def unpermute(**kwargs: object) -> None:
        reduce_call.update(kwargs)
        output = kwargs["output"]
        down = kwargs["a"]
        assert isinstance(output, torch.Tensor)
        assert isinstance(down, torch.Tensor)
        output.copy_(down[:2])

    monkeypatch.setattr(module, "deepgemm_unpermute_and_reduce", unpermute)

    output = torch.empty((2, 4), dtype=torch.bfloat16)
    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=torch.empty((2, 8, 2), dtype=torch.int8),
        w2=torch.empty((2, 4, 2), dtype=torch.int8),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=2,
        expert_map=expert_map,
        a1q_scale=input_scale,
        a2_scale=None,
        workspace13=torch.empty(32, dtype=torch.int8),
        workspace2=torch.empty(32, dtype=torch.bfloat16),
        expert_tokens_meta=expert_tokens_meta,
        apply_router_weight_on_input=False,
    )

    assert len(gemm_calls) == 2
    assert gemm_calls[0][1] == (experts._deepgemm_w13, w13_scale)
    assert gemm_calls[1][1] == (experts._deepgemm_w2, w2_scale)
    assert gemm_calls[0][2].dtype == torch.int32
    assert gemm_calls[1][2] is gemm_calls[0][2]
    assert permute_call["expert_map"] is expert_map
    assert permute_call["expert_tokens_meta"] is expert_tokens_meta
    assert reduce_call["expert_map"] is expert_map
    torch.testing.assert_close(output, torch.full_like(output, 2))


def test_w4a8_contiguous_zero_token_output_skips_all_kernels(
    monkeypatch: pytest.MonkeyPatch,
):
    """An empty HT dispatch is a valid no-op with an empty output."""

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm_hcu.model_executor.layers.quantization import (
        slimquant_w4a8_deepgemm_runtime as module,
    )

    experts = object.__new__(module.DeepEPDeepGemmW4A8ContiguousExperts)
    experts._deepgemm_w13 = torch.empty((2, 1), dtype=torch.int8)
    experts._deepgemm_w2 = torch.empty((2, 1), dtype=torch.int8)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("empty contiguous dispatch launched a kernel")

    monkeypatch.setattr(
        module, "m_grouped_w4a8_gemm_nt_contiguous_hipc", forbidden
    )
    monkeypatch.setattr(module, "deepgemm_moe_permute", forbidden)
    output = torch.empty((0, 4), dtype=torch.bfloat16)

    experts.apply(
        output=output,
        hidden_states=torch.empty((0, 4), dtype=torch.int8),
        w1=torch.empty((2, 8, 2), dtype=torch.int8),
        w2=torch.empty((2, 4, 2), dtype=torch.int8),
        topk_weights=torch.empty((0, 1), dtype=torch.float32),
        topk_ids=torch.empty((0, 1), dtype=torch.int32),
        activation=MoEActivation.SILU,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=torch.empty((0, 1), dtype=torch.float32),
        a2_scale=None,
        workspace13=torch.empty(0, dtype=torch.int8),
        workspace2=torch.empty(0, dtype=torch.bfloat16),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert output.shape == (0, 4)


def test_w4a8_masked_packs_once_into_n32_view_without_replacing_parameters(
    monkeypatch: pytest.MonkeyPatch,
):
    """LL experts cache the documented six-dimensional N32 weight view."""

    from vllm_hcu.model_executor.layers.quantization import (
        slimquant_w4a8_deepgemm_runtime as module,
    )

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(
        torch.arange(128 * 32, dtype=torch.int8).reshape(1, 128, 32),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.arange(64 * 32, dtype=torch.int8).reshape(1, 64, 32),
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.ones((1, 128, 1), dtype=torch.float32),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.ones((1, 64, 1), dtype=torch.float32),
        requires_grad=False,
    )
    canonical_w13 = layer.w13_weight
    canonical_w2 = layer.w2_weight
    expected_w13 = canonical_w13.detach().clone()
    expected_w2 = canonical_w2.detach().clone()
    pack_calls: list[torch.Tensor] = []
    view_shapes: list[tuple[int, ...]] = []

    def pack(weight: torch.Tensor) -> torch.Tensor:
        weight.fill_(53 + len(pack_calls))
        pack_calls.append(weight)
        return weight

    def n32_view(weight: torch.Tensor) -> torch.Tensor:
        experts, n, k_half = weight.shape
        viewed = weight.view(experts, k_half // 32, n // 32, 4, 32, 8)
        view_shapes.append(tuple(viewed.shape))
        return viewed

    monkeypatch.setattr(module, "pack_w4a8_moe_hipc_weight", pack)
    monkeypatch.setattr(
        module, "view_w4a8_moe_hipc_weight_n32_layout", n32_view
    )

    experts = object.__new__(module.DeepEPDeepGemmW4A8BatchedExperts)
    experts._deepgemm_w13 = None
    experts._deepgemm_w2 = None
    experts.process_weights_after_loading(layer)

    assert layer.w13_weight is canonical_w13
    assert layer.w2_weight is canonical_w2
    torch.testing.assert_close(layer.w13_weight, expected_w13)
    torch.testing.assert_close(layer.w2_weight, expected_w2)
    assert len(pack_calls) == 2
    assert view_shapes == [(1, 1, 4, 4, 32, 8), (1, 1, 2, 4, 32, 8)]
    assert tuple(experts._deepgemm_w13.shape) == view_shapes[0]
    assert tuple(experts._deepgemm_w2.shape) == view_shapes[1]

    replacement = object.__new__(module.DeepEPDeepGemmW4A8BatchedExperts)
    replacement._deepgemm_w13 = None
    replacement._deepgemm_w2 = None
    replacement.process_weights_after_loading(layer)

    assert len(pack_calls) == 2
    assert replacement._deepgemm_w13 is experts._deepgemm_w13
    assert replacement._deepgemm_w2 is experts._deepgemm_w2


def test_w4a8_masked_batched_apply_propagates_scales_and_token_counts(
    monkeypatch: pytest.MonkeyPatch,
):
    """The int4 branch must use HIPC masked GEMMs for gate/up and down."""

    lightop = ModuleType("lightop")
    activation_call: dict[str, object] = {}

    def quantize(tensor: torch.Tensor, tokens_per_expert: torch.Tensor):
        activation_call.update(
            tensor=tensor,
            tokens_per_expert=tokens_per_expert,
        )
        return (
            torch.full((1, 2, 64), 7, dtype=torch.int8),
            torch.full((1, 2, 1), 0.5, dtype=torch.float32),
        )

    lightop.fuse_silu_mul_quant_ep = quantize
    monkeypatch.setitem(sys.modules, "lightop", lightop)

    gemm_calls: list[tuple[object, ...]] = []
    deepgemm = ModuleType("deepgemm")
    deepgemm.__path__ = []
    deepgemm.m_grouped_w4a8_gemm_nt_masked_hipc = (
        lambda *args: gemm_calls.append(args)
    )
    deepgemm.m_grouped_i8_gemm_nt_masked = lambda *_args: (_ for _ in ()).throw(
        AssertionError("W8A8 masked API invoked for W4A8 weights")
    )
    monkeypatch.setitem(sys.modules, "deepgemm", deepgemm)

    hcu = _load_batched_deep_gemm_apply()
    hcu["current_platform"] = SimpleNamespace(is_rocm=lambda: True)
    w13_scale = torch.ones((1, 128, 1), dtype=torch.float32)
    w2_scale = torch.full((1, 64, 1), 2.0, dtype=torch.float32)
    derived_w13 = torch.full((1, 1, 4, 4, 32, 8), 13, dtype=torch.int8)
    derived_w2 = torch.full((1, 1, 2, 4, 32, 8), 17, dtype=torch.int8)
    expert_num_tokens = torch.tensor([2], dtype=torch.int32)
    experts = SimpleNamespace(
        block_shape=None,
        quant_config=SimpleNamespace(
            use_int8_w8a8=True,
            use_fp8_w8a8=False,
            weight_quant_dtype="int4",
        ),
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        _deepgemm_w13=derived_w13,
        _deepgemm_w2=derived_w2,
        _hcu_logical_n=128,
        _hcu_logical_k=64,
        moe_problem_size=lambda *_args: (1, 2, 128, 64, 1),
        estimate_expected_m=lambda **_kwargs: 2,
    )

    output = torch.empty((1, 2, 64), dtype=torch.bfloat16)
    hcu["apply"](
        experts,
        output=output,
        hidden_states=torch.zeros((1, 2, 64), dtype=torch.int8),
        w1=torch.zeros((1, 128, 32), dtype=torch.int8),
        w2=torch.zeros((1, 64, 32), dtype=torch.int8),
        topk_weights=torch.ones((2, 1)),
        topk_ids=torch.zeros((2, 1), dtype=torch.int32),
        activation="silu",
        global_num_experts=1,
        expert_map=None,
        a1q_scale=torch.ones((1, 2, 1), dtype=torch.float32),
        a2_scale=None,
        workspace13=torch.empty(256, dtype=torch.bfloat16),
        workspace2=torch.empty(256, dtype=torch.bfloat16),
        expert_tokens_meta=SimpleNamespace(
            expert_num_tokens=expert_num_tokens
        ),
        apply_router_weight_on_input=False,
    )

    assert len(gemm_calls) == 2
    assert gemm_calls[0][1] == (derived_w13, w13_scale)
    assert gemm_calls[1][1] == (derived_w2, w2_scale)
    assert gemm_calls[0][3] is expert_num_tokens
    assert gemm_calls[1][3] is expert_num_tokens
    assert gemm_calls[0][4] == 2
    assert gemm_calls[1][4] == 2
    assert activation_call["tokens_per_expert"] is expert_num_tokens


def test_w4a8_masked_empty_dispatch_skips_all_kernels(
    monkeypatch: pytest.MonkeyPatch,
):
    """An empty LL dispatch must not launch a masked GEMM."""

    lightop = ModuleType("lightop")
    lightop.fuse_silu_mul_quant_ep = lambda *_args: (_ for _ in ()).throw(
        AssertionError("empty masked dispatch launched activation")
    )
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    deepgemm = ModuleType("deepgemm")
    deepgemm.__path__ = []
    deepgemm.m_grouped_w4a8_gemm_nt_masked_hipc = lambda *_args: (
        _ for _ in ()
    ).throw(AssertionError("empty masked dispatch launched DeepGEMM"))
    monkeypatch.setitem(sys.modules, "deepgemm", deepgemm)

    hcu = _load_batched_deep_gemm_apply()
    hcu["current_platform"] = SimpleNamespace(is_rocm=lambda: True)
    experts = SimpleNamespace(
        block_shape=None,
        quant_config=SimpleNamespace(
            use_int8_w8a8=True,
            use_fp8_w8a8=False,
            weight_quant_dtype="int4",
        ),
        w1_scale=torch.ones((1, 128, 1), dtype=torch.float32),
        w2_scale=torch.ones((1, 64, 1), dtype=torch.float32),
        _deepgemm_w13=torch.empty((1, 1, 4, 4, 32, 8), dtype=torch.int8),
        _deepgemm_w2=torch.empty((1, 1, 2, 4, 32, 8), dtype=torch.int8),
        _hcu_logical_n=128,
        _hcu_logical_k=64,
        moe_problem_size=lambda *_args: (1, 0, 128, 64, 1),
        estimate_expected_m=lambda **_kwargs: 0,
    )
    output = torch.empty((1, 0, 64), dtype=torch.bfloat16)

    hcu["apply"](
        experts,
        output=output,
        hidden_states=torch.empty((1, 0, 64), dtype=torch.int8),
        w1=torch.empty((1, 128, 32), dtype=torch.int8),
        w2=torch.empty((1, 64, 32), dtype=torch.int8),
        topk_weights=torch.empty((0, 1)),
        topk_ids=torch.empty((0, 1), dtype=torch.int32),
        activation="silu",
        global_num_experts=1,
        expert_map=None,
        a1q_scale=torch.empty((1, 0, 1), dtype=torch.float32),
        a2_scale=None,
        workspace13=torch.empty(0, dtype=torch.bfloat16),
        workspace2=torch.empty(0, dtype=torch.bfloat16),
        expert_tokens_meta=SimpleNamespace(
            expert_num_tokens=torch.zeros(1, dtype=torch.int32)
        ),
        apply_router_weight_on_input=False,
    )

    assert output.shape == (1, 0, 64)
