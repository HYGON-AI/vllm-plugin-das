# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Runtime contracts for LightOp's categorized activation and GEMM APIs."""

from __future__ import annotations

import ast
import copy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


REPO = Path(__file__).resolve().parents[2]


class _WarningLogger:
    """Small behavioral stand-in for vLLM's message-deduplicating logger."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning_once(self, message: str) -> None:
        if message not in self.messages:
            self.messages.append(message)


def _module(relative: str) -> ast.Module:
    return ast.parse((REPO / relative).read_text(encoding="utf-8"))


def _function(relative: str, name: str, namespace: dict[str, object]):
    function = copy.deepcopy(
        next(
            node
            for node in _module(relative).body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, relative, "exec"), namespace)
    return namespace[name]


def _method(
    relative: str,
    class_name: str,
    method_name: str,
    namespace: dict[str, object],
):
    klass = next(
        node
        for node in _module(relative).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = copy.deepcopy(
        next(
            node
            for node in klass.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, relative, "exec"), namespace)
    return namespace[method_name]


def _eager_activation_import(
    namespace: dict[str, object],
) -> None:
    """Execute dpsk's real eager categorized-import boundary."""
    block = copy.deepcopy(
        next(
            node
            for node in _module(
                "vllm_hcu/model_executor/layers/fused_moe/experts/"
                "dpsk_v4_deep_gemm_moe.py"
            ).body
            if isinstance(node, ast.Try)
            and any(
                isinstance(item, ast.ImportFrom)
                and item.module == "lightop.activation"
                for item in node.body
            )
        )
    )
    module = ast.Module(body=[block], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "dpsk_v4_deep_gemm_moe", "exec"), namespace)


def _install_lightop(
    monkeypatch: pytest.MonkeyPatch,
    *,
    activation: ModuleType | None = None,
    gemm_ops: ModuleType | None = None,
    legacy: ModuleType | None = None,
    op: ModuleType | None = None,
) -> None:
    lightop = legacy or ModuleType("lightop")
    lightop.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    for name, module in (
        ("lightop.activation", activation),
        ("lightop.gemm_ops", gemm_ops),
        ("lightop.op", op),
    ):
        if module is None:
            monkeypatch.delitem(sys.modules, name, raising=False)
        else:
            monkeypatch.setitem(sys.modules, name, module)


def test_fuse_silu_quant_prefers_categorized_kernel_and_consumes_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing the categorized import must make this real adapter fail."""
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    activation = ModuleType("lightop.activation")

    def categorized(
        input: torch.Tensor, *, output: torch.Tensor, scales: torch.Tensor
    ) -> None:
        calls.append((input, output, scales))
        output.fill_(11)
        scales.fill_(0.25)

    activation.fuse_silu_mul_per_token_quant = categorized
    _install_lightop(monkeypatch, activation=activation)
    real = _function(
        "vllm_hcu/ops/fuse_silu_mul_quant.py",
        "fuse_silu_mul_quant_real",
        {"torch": torch, "logger": _WarningLogger()},
    )

    output, scales = real(torch.ones((2, 6)), torch.int8)

    assert len(calls) == 1
    assert calls[0][0].shape == (2, 6)
    assert output.shape == (2, 3)
    assert torch.equal(output, torch.full_like(output, 11))
    assert torch.equal(scales, torch.full_like(scales, 0.25))


def test_fuse_silu_quant_uses_legacy_abi_once_when_categorized_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = ModuleType("lightop")
    observed: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def legacy_quant(
        input: torch.Tensor, *, output: torch.Tensor, scales: torch.Tensor
    ) -> None:
        observed.append((input, output, scales))
        output.fill_(7)
        scales.fill_(0.5)

    legacy.fuse_silu_mul_per_token_quant = legacy_quant
    logger = _WarningLogger()
    _install_lightop(monkeypatch, legacy=legacy)
    real = _function(
        "vllm_hcu/ops/fuse_silu_mul_quant.py",
        "fuse_silu_mul_quant_real",
        {"torch": torch, "logger": logger},
    )

    first = real(torch.ones((1, 4)), torch.int8)
    second = real(torch.ones((1, 4)), torch.int8)

    assert len(observed) == 2
    assert torch.equal(first[0], torch.full_like(first[0], 7))
    assert torch.equal(second[1], torch.full_like(second[1], 0.5))
    assert len(logger.messages) == 1


def test_fuse_silu_quant_does_not_fallback_after_kernel_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = ModuleType("lightop.activation")
    activation.fuse_silu_mul_per_token_quant = lambda *_args, **_kwargs: (
        _ for _ in ()
    ).throw(RuntimeError("kernel failed"))
    legacy = ModuleType("lightop")
    legacy_calls: list[object] = []
    legacy.fuse_silu_mul_per_token_quant = lambda *_args, **_kwargs: legacy_calls.append(
        "called"
    )
    _install_lightop(monkeypatch, activation=activation, legacy=legacy)
    real = _function(
        "vllm_hcu/ops/fuse_silu_mul_quant.py",
        "fuse_silu_mul_quant_real",
        {"torch": torch, "logger": _WarningLogger()},
    )

    with pytest.raises(RuntimeError, match="kernel failed"):
        real(torch.ones((1, 4)), torch.int8)
    assert legacy_calls == []


def test_silu_and_mul_selects_categorized_module_and_returns_its_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = ModuleType("lightop.activation")

    def categorized(output: torch.Tensor, input: torch.Tensor) -> None:
        assert input.shape == (1, 6)
        output.fill_(3)

    activation.silu_and_mul_opt = categorized
    _install_lightop(monkeypatch, activation=activation)
    namespace: dict[str, object] = {"torch": torch, "logger": _WarningLogger()}
    _execute_silu_module_selection(namespace)
    impl = _function(
        "vllm_hcu/ops/silu_and_mul.py",
        "silu_and_mul_opt_lightop_impl",
        namespace,
    )

    result = impl(torch.ones((1, 6)))

    assert torch.equal(result, torch.full_like(result, 3))


def _execute_silu_module_selection(namespace: dict[str, object]) -> None:
    selection = copy.deepcopy(
        next(
            node
            for node in _module("vllm_hcu/ops/silu_and_mul.py").body
            if isinstance(node, ast.Try)
            and any(
                isinstance(item, ast.ImportFrom)
                and item.module == "lightop"
                and any(alias.name == "activation" for alias in item.names)
                for item in node.body
            )
        )
    )
    module = ast.Module(body=[selection], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "silu_and_mul", "exec"), namespace)


def test_silu_and_mul_uses_legacy_module_once_when_categorized_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_op = ModuleType("lightop.op")
    legacy_op.silu_and_mul_opt = lambda output, _input: output.fill_(5)
    logger = _WarningLogger()
    _install_lightop(monkeypatch, op=legacy_op)
    namespace: dict[str, object] = {"torch": torch, "logger": logger}

    _execute_silu_module_selection(namespace)
    _execute_silu_module_selection(namespace)
    impl = _function(
        "vllm_hcu/ops/silu_and_mul.py",
        "silu_and_mul_opt_lightop_impl",
        namespace,
    )

    assert torch.equal(impl(torch.ones((1, 4))), torch.full((1, 2), 5.0))
    assert len(logger.messages) == 1


def _deep_apply_namespace(logger: _WarningLogger) -> dict[str, object]:
    return {
        "torch": torch,
        "MoEActivation": SimpleNamespace(SILU="silu"),
        "current_platform": SimpleNamespace(is_rocm=lambda: True),
        "compute_aligned_M_and_alignment": lambda **_kwargs: (1, 256),
        "deepgemm_moe_permute": lambda **kwargs: (
            kwargs["aq"],
            kwargs["aq_scale"],
            torch.tensor([0]),
            object(),
            256,
        ),
        "_resize_cache": lambda _cache, shape: torch.empty(shape),
        "mk_alignment_scope": lambda _alignment: nullcontext(),
        "deepgemm_unpermute_and_reduce": lambda **kwargs: kwargs["output"].copy_(
            kwargs["a"]
        ),
        "_HCU_HT_EP_TOKEN_ALIGNMENT": 256,
        "logger": logger,
    }


def _deep_self() -> SimpleNamespace:
    return SimpleNamespace(
        block_shape=None,
        quant_config=SimpleNamespace(use_int8_w8a8=True, use_fp8_w8a8=False),
        w1_scale="w1-scale",
        w2_scale="w2-scale",
        _hcu_logical_n=4,
        _hcu_logical_k=2,
        mxfp8=False,
        adjust_N_for_activation=lambda n, _activation: n // 2,
    )


def _run_deep_int8_apply(apply, output: torch.Tensor) -> None:
    apply(
        _deep_self(),
        output,
        torch.ones((1, 2)),
        torch.empty((1, 4, 2)),
        torch.empty((1, 2, 2)),
        torch.ones((1, 1)),
        torch.zeros((1, 1), dtype=torch.int64),
        "silu",
        1,
        None,
        torch.ones((1, 1)),
        None,
        torch.empty(16),
        torch.empty(16),
        None,
        False,
    )


def test_contiguous_deep_gemm_prefers_categorized_kernels_and_consumes_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = ModuleType("lightop.activation")
    gemm_ops = ModuleType("lightop.gemm_ops")
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def gemm(*args, **kwargs) -> None:
        calls.append(("gemm", args, kwargs))
        args[2].fill_(4 if len(calls) == 1 else 9)

    def quant(input: torch.Tensor, **kwargs):
        calls.append(("quant", (input,), kwargs))
        assert torch.equal(input, torch.full_like(input, 4))
        return torch.full((1, 2), 6, dtype=torch.int8), torch.ones((1, 1))

    activation.fuse_silu_mul_quant = quant
    gemm_ops.m_grouped_w8a8_gemm_nt_contig_asm = gemm
    _install_lightop(monkeypatch, activation=activation, gemm_ops=gemm_ops)
    apply = _method(
        "vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py",
        "DeepGemmExperts",
        "apply",
        _deep_apply_namespace(_WarningLogger()),
    )
    output = torch.empty((1, 2))

    _run_deep_int8_apply(apply, output)

    assert [name for name, _, _ in calls] == ["gemm", "quant", "gemm"]
    assert calls[0][1][1][0].shape == (1, 4, 2)
    assert calls[0][1][1][1] == "w1-scale"
    assert torch.equal(calls[2][1][0][0], torch.full((1, 2), 6, dtype=torch.int8))


def test_contiguous_deep_gemm_legacy_fallback_warns_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = ModuleType("lightop")
    legacy.m_grouped_w8a8_gemm_nt_contig_asm = lambda *_args, **_kwargs: None
    legacy.fuse_silu_mul_quant = lambda *_args, **_kwargs: (
        torch.ones((1, 2), dtype=torch.int8),
        torch.ones((1, 1)),
    )
    logger = _WarningLogger()
    _install_lightop(monkeypatch, legacy=legacy)
    apply = _method(
        "vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py",
        "DeepGemmExperts",
        "apply",
        _deep_apply_namespace(logger),
    )

    _run_deep_int8_apply(apply, torch.empty((1, 2)))
    _run_deep_int8_apply(apply, torch.empty((1, 2)))

    assert len(logger.messages) == 1


def test_masked_deep_gemm_prefers_categorized_kernels_and_consumes_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = ModuleType("lightop.activation")
    gemm_ops = ModuleType("lightop.gemm_ops")
    calls: list[tuple[str, tuple[object, ...]]] = []

    def gemm(*args) -> None:
        calls.append(("gemm", args))
        args[2].fill_(2 if len(calls) == 1 else 8)

    def quant(input: torch.Tensor, expert_tokens: torch.Tensor):
        calls.append(("quant", (input, expert_tokens)))
        assert torch.equal(input, torch.full_like(input, 2))
        return torch.full((1, 1, 2), 7, dtype=torch.int8), torch.ones((1, 1, 1))

    activation.fuse_silu_mul_quant_ep = quant
    gemm_ops.m_grouped_w8a8_gemm_nt_masked = gemm
    _install_lightop(monkeypatch, activation=activation, gemm_ops=gemm_ops)
    namespace = {
        "torch": torch,
        "MoEActivation": SimpleNamespace(SILU="silu"),
        "current_platform": SimpleNamespace(is_rocm=lambda: True),
        "_resize_cache": lambda _cache, shape: torch.empty(shape),
        "logger": _WarningLogger(),
    }
    apply = _method(
        "vllm_hcu/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py",
        "BatchedDeepGemmExperts",
        "apply",
        namespace,
    )
    self = SimpleNamespace(
        block_shape=None,
        quant_config=SimpleNamespace(use_int8_w8a8=True, use_fp8_w8a8=False),
        w1_scale="w1-scale",
        w2_scale="w2-scale",
        _hcu_logical_n=4,
        _hcu_logical_k=2,
        moe_problem_size=lambda *_args: (1, 1, 4, 2, None),
        estimate_expected_m=lambda **_kwargs: 13,
    )
    output = torch.empty((1, 1, 2))
    tokens = torch.tensor([1], dtype=torch.int32)

    apply(
        self,
        output,
        torch.ones((1, 1, 2)),
        torch.empty((1, 4, 2)),
        torch.empty((1, 2, 2)),
        torch.ones((1, 1)),
        torch.zeros((1, 1), dtype=torch.int64),
        "silu",
        1,
        None,
        torch.ones((1, 1, 1)),
        None,
        torch.empty(16),
        torch.empty(16),
        SimpleNamespace(expert_num_tokens=tokens),
        False,
    )

    assert [name for name, _ in calls] == ["gemm", "quant", "gemm"]
    assert calls[0][1][-2:] == (tokens, 13)
    assert torch.equal(calls[2][1][0][0], torch.full((1, 1, 2), 7, dtype=torch.int8))


def test_masked_deep_gemm_legacy_fallback_warns_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = ModuleType("lightop")
    legacy.m_grouped_w8a8_gemm_nt_masked = lambda *_args, **_kwargs: None
    legacy.fuse_silu_mul_quant_ep = lambda *_args, **_kwargs: (
        torch.ones((1, 1, 2), dtype=torch.int8),
        torch.ones((1, 1, 1)),
    )
    _install_lightop(monkeypatch, legacy=legacy)
    logger = _WarningLogger()
    namespace = {
        "torch": torch,
        "MoEActivation": SimpleNamespace(SILU="silu"),
        "current_platform": SimpleNamespace(is_rocm=lambda: True),
        "_resize_cache": lambda _cache, shape: torch.empty(shape),
        "logger": logger,
    }
    apply = _method(
        "vllm_hcu/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py",
        "BatchedDeepGemmExperts",
        "apply",
        namespace,
    )
    self = SimpleNamespace(
        block_shape=None,
        quant_config=SimpleNamespace(use_int8_w8a8=True, use_fp8_w8a8=False),
        w1_scale="w1-scale",
        w2_scale="w2-scale",
        _hcu_logical_n=4,
        _hcu_logical_k=2,
        moe_problem_size=lambda *_args: (1, 1, 4, 2, None),
        estimate_expected_m=lambda **_kwargs: 13,
    )
    kwargs = dict(
        output=torch.empty((1, 1, 2)),
        hidden_states=torch.ones((1, 1, 2)),
        w1=torch.empty((1, 4, 2)),
        w2=torch.empty((1, 2, 2)),
        topk_weights=torch.ones((1, 1)),
        topk_ids=torch.zeros((1, 1), dtype=torch.int64),
        activation="silu",
        global_num_experts=1,
        expert_map=None,
        a1q_scale=torch.ones((1, 1, 1)),
        a2_scale=None,
        workspace13=torch.empty(16),
        workspace2=torch.empty(16),
        expert_tokens_meta=SimpleNamespace(expert_num_tokens=torch.tensor([1])),
        apply_router_weight_on_input=False,
    )

    apply(self, **kwargs)
    apply(self, **kwargs)

    assert len(logger.messages) == 1


def _run_contiguous_fp8_apply(apply, output: torch.Tensor) -> None:
    self = _deep_self()
    self.quant_config = SimpleNamespace(use_int8_w8a8=False, use_fp8_w8a8=True)
    apply(
        self,
        output,
        torch.ones((1, 2)),
        torch.empty((1, 4, 2)),
        torch.empty((1, 2, 2)),
        torch.ones((1, 1)),
        torch.zeros((1, 1), dtype=torch.int64),
        "silu",
        1,
        None,
        torch.ones((1, 1)),
        None,
        torch.empty(16),
        torch.empty(16),
        None,
        False,
    )


def test_contiguous_fp8_deep_gemm_uses_categorized_activation_and_its_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = ModuleType("lightop.activation")
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def gemm(*args, **kwargs) -> None:
        calls.append(("gemm", args, kwargs))
        args[2].fill_(4 if len(calls) == 1 else 9)

    def quant(input: torch.Tensor, **kwargs):
        calls.append(("quant", (input,), kwargs))
        assert torch.equal(input, torch.full_like(input, 4))
        assert kwargs["fp8type"] == 0
        assert kwargs["output"].shape == (1, 2)
        assert torch.equal(kwargs["expert_ids"], torch.tensor([0]))
        return torch.full((1, 2), 6), torch.full((1, 1), 0.5)

    activation.fuse_silu_mul_fp8_quant = quant
    _install_lightop(monkeypatch, activation=activation)
    namespace = _deep_apply_namespace(_WarningLogger())
    namespace["m_grouped_fp8_gemm_nt_contiguous"] = gemm
    apply = _method(
        "vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py",
        "DeepGemmExperts",
        "apply",
        namespace,
    )
    output = torch.empty((1, 2))

    _run_contiguous_fp8_apply(apply, output)

    assert [name for name, _, _ in calls] == ["gemm", "quant", "gemm"]
    assert torch.equal(calls[2][1][0][0], torch.full((1, 2), 6))
    assert torch.equal(calls[2][1][0][1], torch.full((1, 1), 0.5))
    assert torch.equal(output, torch.full_like(output, 9))


def test_contiguous_fp8_deep_gemm_legacy_activation_fallback_warns_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = ModuleType("lightop")
    calls: list[tuple[object, ...]] = []
    gemm_calls: list[tuple[object, ...]] = []

    def legacy_quant(input: torch.Tensor, **kwargs):
        calls.append((input, kwargs))
        return torch.full((1, 2), 8), torch.ones((1, 1))

    legacy.fuse_silu_mul_fp8_quant = legacy_quant
    logger = _WarningLogger()
    _install_lightop(monkeypatch, legacy=legacy)
    namespace = _deep_apply_namespace(logger)

    def gemm(*args, **_kwargs) -> None:
        gemm_calls.append(args)
        args[2].fill_(3)

    namespace["m_grouped_fp8_gemm_nt_contiguous"] = gemm
    apply = _method(
        "vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py",
        "DeepGemmExperts",
        "apply",
        namespace,
    )

    _run_contiguous_fp8_apply(apply, torch.empty((1, 2)))
    _run_contiguous_fp8_apply(apply, torch.empty((1, 2)))

    assert len(calls) == 2
    assert set(calls[0][1]) == {"fp8type", "output", "expert_ids"}
    assert calls[0][1]["fp8type"] == 0
    assert calls[0][1]["output"].shape == (1, 2)
    assert torch.equal(calls[0][1]["expert_ids"], torch.tensor([0]))
    assert torch.equal(gemm_calls[1][0][0], torch.full((1, 2), 8))
    assert len(logger.messages) == 1


def _install_masked_fp8_gemm(
    monkeypatch: pytest.MonkeyPatch, kernel
) -> None:
    deepgemm = ModuleType("deepgemm")
    deepgemm.__path__ = []  # type: ignore[attr-defined]
    m_group_gemm = ModuleType("deepgemm.m_group_gemm")
    m_group_gemm.m_grouped_fp8_gemm_nt_masked_ll = kernel
    monkeypatch.setitem(sys.modules, "deepgemm", deepgemm)
    monkeypatch.setitem(sys.modules, "deepgemm.m_group_gemm", m_group_gemm)


def _masked_fp8_namespace(logger: _WarningLogger) -> dict[str, object]:
    return {
        "torch": torch,
        "MoEActivation": SimpleNamespace(SILU="silu"),
        "current_platform": SimpleNamespace(is_rocm=lambda: True),
        "_resize_cache": lambda _cache, shape: torch.empty(shape),
        "logger": logger,
    }


def _masked_fp8_self() -> SimpleNamespace:
    return SimpleNamespace(
        block_shape=None,
        quant_config=SimpleNamespace(use_int8_w8a8=False, use_fp8_w8a8=True),
        w1_scale="w1-scale",
        w2_scale="w2-scale",
        _hcu_logical_n=4,
        _hcu_logical_k=2,
        moe_problem_size=lambda *_args: (1, 1, 4, 2, None),
        estimate_expected_m=lambda **_kwargs: 13,
    )


def _run_masked_fp8_apply(apply, self: SimpleNamespace) -> None:
    apply(
        self,
        torch.empty((1, 1, 2)),
        torch.ones((1, 1, 2)),
        torch.empty((1, 4, 2)),
        torch.empty((1, 2, 2)),
        torch.ones((1, 1)),
        torch.zeros((1, 1), dtype=torch.int64),
        "silu",
        1,
        None,
        torch.ones((1, 1, 1)),
        None,
        torch.empty(16),
        torch.empty(16),
        SimpleNamespace(expert_num_tokens=torch.tensor([1], dtype=torch.int32)),
        False,
    )


def test_masked_fp8_deep_gemm_uses_categorized_ep_activation_and_its_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = ModuleType("lightop.activation")
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def gemm(*args) -> None:
        calls.append(("gemm", args, {}))
        args[2].fill_(2 if len(calls) == 1 else 8)

    def quant(**kwargs):
        calls.append(("quant", (), kwargs))
        assert torch.equal(kwargs["input"], torch.full_like(kwargs["input"], 2))
        assert kwargs["fp8type"] == 0
        assert torch.equal(kwargs["tokens_per_expert"], torch.tensor([1]))
        return torch.full((1, 1, 2), 7), torch.full((1, 1, 1), 0.25)

    activation.fuse_silu_mul_fp8_quant_ep = quant
    _install_lightop(monkeypatch, activation=activation)
    _install_masked_fp8_gemm(monkeypatch, gemm)
    apply = _method(
        "vllm_hcu/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py",
        "BatchedDeepGemmExperts",
        "apply",
        _masked_fp8_namespace(_WarningLogger()),
    )

    _run_masked_fp8_apply(apply, _masked_fp8_self())

    assert [name for name, _, _ in calls] == ["gemm", "quant", "gemm"]
    assert torch.equal(calls[2][1][0][0], torch.full((1, 1, 2), 7))
    assert torch.equal(calls[2][1][0][1], torch.full((1, 1, 1), 0.25))


def test_masked_fp8_deep_gemm_legacy_ep_activation_fallback_warns_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = ModuleType("lightop")
    calls: list[dict[str, object]] = []
    gemm_calls: list[tuple[object, ...]] = []

    def legacy_quant(**kwargs):
        calls.append(kwargs)
        return torch.ones((1, 1, 2)), torch.ones((1, 1, 1))

    legacy.fuse_silu_mul_fp8_quant_ep = legacy_quant
    logger = _WarningLogger()
    _install_lightop(monkeypatch, legacy=legacy)

    def gemm(*args) -> None:
        gemm_calls.append(args)

    _install_masked_fp8_gemm(monkeypatch, gemm)
    apply = _method(
        "vllm_hcu/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py",
        "BatchedDeepGemmExperts",
        "apply",
        _masked_fp8_namespace(logger),
    )

    _run_masked_fp8_apply(apply, _masked_fp8_self())
    _run_masked_fp8_apply(apply, _masked_fp8_self())

    assert len(calls) == 2
    assert calls[0]["fp8type"] == 0
    assert calls[0]["input"].shape == (1, 1, 4)
    assert torch.equal(calls[0]["tokens_per_expert"], torch.tensor([1]))
    assert torch.equal(gemm_calls[1][0][0], torch.ones((1, 1, 2)))
    assert len(logger.messages) == 1


def _dpsk_namespace(logger: _WarningLogger, calls: list[tuple[str, tuple, dict]]):
    def contiguous_gemm(*args) -> None:
        calls.append(("contiguous-gemm", args, {}))
        args[2].fill_(4 if len(calls) == 1 else 9)

    def masked_gemm(*args) -> None:
        calls.append(("masked-gemm", args, {}))
        args[2].fill_(2 if len(calls) == 4 else 8)

    return {
        "torch": torch,
        "MoEActivation": SimpleNamespace(SILU="silu"),
        "compute_aligned_M": lambda **_kwargs: 1,
        "deepgemm_moe_permute": lambda **kwargs: (
            kwargs["aq"],
            kwargs["aq_scale"],
            torch.tensor([0]),
            object(),
        ),
        "deepgemm_unpermute_and_reduce": lambda **kwargs: kwargs["output"].copy_(
            kwargs["a"]
        ),
        "_resize_cache": lambda _cache, shape: torch.empty(shape),
        "m_grouped_fp8_gemm_nt_contiguous": contiguous_gemm,
        "m_grouped_fp8_gemm_nt_masked_ll": masked_gemm,
        "logger": logger,
    }


def _dpsk_contiguous_self() -> SimpleNamespace:
    return SimpleNamespace(
        w1_scale="w1-scale",
        w2_scale="w2-scale",
        _deepgemm_w13="packed-w13",
        _deepgemm_w2="packed-w2",
        ALIGNMENT=256,
        moe_problem_size=lambda *_args: (1, 1, 4, 2, None),
        _ensure_expert_tokens_meta=lambda **kwargs: kwargs["expert_tokens_meta"],
        _ensure_2d_scale=lambda scale: scale,
    )


def _dpsk_masked_self() -> SimpleNamespace:
    return SimpleNamespace(
        w1_scale=torch.ones((1, 1)),
        w2_scale=torch.ones((1, 1)),
        _deepgemm_w13="packed-w13",
        _deepgemm_w2="packed-w2",
        moe_problem_size=lambda *_args: (1, 1, 4, 2, None),
        _ensure_ll_scale=lambda scale, *_args: scale,
        _ensure_ll_weight_scale=lambda scale: scale,
        adjust_N_for_activation=lambda n, _activation: n // 2,
    )


def _run_dpsk_fp8_exports(
    namespace: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    contiguous = _method(
        "vllm_hcu/model_executor/layers/fused_moe/experts/"
        "dpsk_v4_deep_gemm_moe.py",
        "DeepEPDeepGemmContiguousExperts",
        "_apply_deepgemm_ht",
        namespace,
    )
    contiguous_output = torch.empty((1, 2))
    contiguous(
        _dpsk_contiguous_self(),
        contiguous_output,
        torch.ones((1, 2)),
        torch.empty((1, 4, 2)),
        torch.empty((1, 2, 2)),
        torch.ones((1, 1)),
        torch.zeros((1, 1), dtype=torch.int64),
        "silu",
        1,
        None,
        torch.ones((1, 1)),
        None,
        torch.empty(16),
        torch.empty(16),
        SimpleNamespace(expert_num_tokens=torch.tensor([1])),
        False,
    )

    masked = _method(
        "vllm_hcu/model_executor/layers/fused_moe/experts/"
        "dpsk_v4_deep_gemm_moe.py",
        "DeepEPDeepGemmMaskedExperts",
        "apply",
        namespace,
    )
    masked_output = torch.empty((1, 1, 2))
    masked(
        _dpsk_masked_self(),
        masked_output,
        torch.ones((1, 1, 2)),
        torch.empty((1, 4, 2)),
        torch.empty((1, 2, 2)),
        torch.ones((1, 1)),
        torch.zeros((1, 1), dtype=torch.int64),
        "silu",
        1,
        None,
        torch.ones((1, 1)),
        None,
        torch.empty(16),
        torch.empty(16),
        SimpleNamespace(expert_num_tokens=torch.tensor([1])),
        False,
    )
    return contiguous_output, masked_output


def test_dpsk_eager_fp8_exports_use_categorized_kernels_and_consume_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = ModuleType("lightop.activation")
    calls: list[tuple[str, tuple, dict]] = []

    def quant(input: torch.Tensor, **kwargs):
        calls.append(("quant", (input,), kwargs))
        assert torch.equal(input, torch.full_like(input, 4))
        assert set(kwargs) == {"fp8type", "expert_ids"}
        assert kwargs["fp8type"] == 0
        assert torch.equal(kwargs["expert_ids"], torch.tensor([0]))
        return torch.full((1, 2), 6), torch.full((1, 1), 0.5)

    def quant_ep(input: torch.Tensor, **kwargs):
        calls.append(("quant-ep", (input,), kwargs))
        assert torch.equal(input, torch.full_like(input, 2))
        assert set(kwargs) == {"fp8type", "tokens_per_expert"}
        assert kwargs["fp8type"] == 0
        assert torch.equal(kwargs["tokens_per_expert"], torch.tensor([1]))
        return torch.full((1, 1, 2), 7), torch.ones((1, 1))

    activation.fuse_silu_mul_fp8_quant = quant
    activation.fuse_silu_mul_fp8_quant_ep = quant_ep
    _install_lightop(monkeypatch, activation=activation)
    namespace = _dpsk_namespace(_WarningLogger(), calls)
    _eager_activation_import(namespace)

    contiguous_output, _ = _run_dpsk_fp8_exports(namespace)

    assert [name for name, _, _ in calls] == [
        "contiguous-gemm",
        "quant",
        "contiguous-gemm",
        "masked-gemm",
        "quant-ep",
        "masked-gemm",
    ]
    assert torch.equal(calls[2][1][0][0], torch.full((1, 2), 6))
    assert torch.equal(calls[5][1][0][0], torch.full((1, 1, 2), 7))
    assert torch.equal(contiguous_output, torch.full_like(contiguous_output, 9))


def test_dpsk_eager_fp8_exports_use_legacy_abi_and_warn_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = ModuleType("lightop")
    calls: list[tuple[str, tuple, dict]] = []

    def quant(input: torch.Tensor, **kwargs):
        calls.append(("quant", (input,), kwargs))
        return torch.ones((1, 2)), torch.ones((1, 1))

    def quant_ep(input: torch.Tensor, **kwargs):
        calls.append(("quant-ep", (input,), kwargs))
        return torch.ones((1, 1, 2)), torch.ones((1, 1))

    legacy.fuse_silu_mul_fp8_quant = quant
    legacy.fuse_silu_mul_fp8_quant_ep = quant_ep
    logger = _WarningLogger()
    _install_lightop(monkeypatch, legacy=legacy)
    namespace = _dpsk_namespace(logger, calls)
    _eager_activation_import(namespace)
    _eager_activation_import(namespace)

    _run_dpsk_fp8_exports(namespace)

    assert [name for name, _, _ in calls] == [
        "contiguous-gemm",
        "quant",
        "contiguous-gemm",
        "masked-gemm",
        "quant-ep",
        "masked-gemm",
    ]
    assert calls[1][2]["fp8type"] == 0
    assert torch.equal(calls[1][2]["expert_ids"], torch.tensor([0]))
    assert calls[4][2]["fp8type"] == 0
    assert torch.equal(calls[4][2]["tokens_per_expert"], torch.tensor([1]))
    assert torch.equal(calls[2][1][0][0], torch.ones((1, 2)))
    assert torch.equal(calls[5][1][0][0], torch.ones((1, 1, 2)))
    assert len(logger.messages) == 1
