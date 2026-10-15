# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Quantize-before-dispatch contracts for SlimQuant W4A8 naive DP/EP MoE."""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

# Defer the replacement import until fixture setup; see test_glm52_pcp_moe.
os.environ["VLLM_PLUGINS"] = "__disabled__"
from vllm.utils import torch_utils


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture(scope="module")
def moe_runner_module() -> ModuleType:
    register_custom_op = torch_utils.direct_register_custom_op

    def register_without_duplicate_moe_ops(op_name, *args, **kwargs):
        if op_name in {"moe_forward", "moe_forward_shared"}:
            return None
        return register_custom_op(op_name, *args, **kwargs)

    torch_utils.direct_register_custom_op = register_without_duplicate_moe_ops
    try:
        return importlib.import_module(
            "vllm_hcu.model_executor.layers.fused_moe.moe_runner"
        )
    finally:
        torch_utils.direct_register_custom_op = register_custom_op


@pytest.fixture
def slimquant_w4a8() -> ModuleType:
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    return slimquant_w4a8


@pytest.fixture
def moe_runtime() -> ModuleType:
    from vllm_hcu.model_executor.layers.quantization import (
        compressed_tensors_moe_runtime,
    )

    return compressed_tensors_moe_runtime


# --------------------------------------------------------------------------
# SlimQuant method: which layers quantize before dispatch
# --------------------------------------------------------------------------


def _install_quant(monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    calls: list[torch.Tensor] = []

    def per_token_quant_int8(x: torch.Tensor):
        calls.append(x)
        return x.to(torch.int8), torch.ones(x.shape[0], 1)

    monkeypatch.setitem(
        sys.modules,
        "aiter.fused_moe_c",
        _module("aiter.fused_moe_c", per_token_quant_int8=per_token_quant_int8),
    )
    return calls


def _method(slimquant_w4a8: ModuleType, moe_kernel: object | None = None):
    method = object.__new__(slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod)
    method.moe_kernel = moe_kernel
    return method


def test_moe_c_layer_quantizes_dispatch_input_by_default(
    monkeypatch: pytest.MonkeyPatch, slimquant_w4a8: ModuleType
) -> None:
    monkeypatch.delenv("VLLM_HCU_USE_CUSTOM_OPS", raising=False)
    calls = _install_quant(monkeypatch)
    layer = SimpleNamespace(_hcu_slimquant_post_load_aiter_config=object())
    x = torch.ones(3, 4, dtype=torch.bfloat16)

    quantized = _method(slimquant_w4a8).quantize_dispatch_input(layer, x)

    assert quantized is not None
    assert quantized[0].dtype == torch.int8
    assert len(calls) == 1 and calls[0] is x


@pytest.mark.parametrize(
    ("env", "config", "moe_kernel"),
    [
        ("0", object(), None),  # custom-op master switch off
        ("1", None, None),  # native layout: vLLM/Triton route needs BF16
        ("1", object(), object()),  # deepep_auto modular kernel
    ],
)
def test_non_moe_c_layers_keep_bf16_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    slimquant_w4a8: ModuleType,
    env: str,
    config: object | None,
    moe_kernel: object | None,
) -> None:
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", env)
    calls = _install_quant(monkeypatch)
    layer = SimpleNamespace(_hcu_slimquant_post_load_aiter_config=config)

    quantized = _method(slimquant_w4a8, moe_kernel).quantize_dispatch_input(
        layer, torch.ones(3, 4, dtype=torch.bfloat16)
    )

    assert quantized is None
    assert calls == []


def test_apply_forwards_prequantized_input_to_aiter(
    monkeypatch: pytest.MonkeyPatch,
    slimquant_w4a8: ModuleType,
    moe_runtime: ModuleType,
) -> None:
    """Dropping i_q/i_s (the old ``**_`` signature) must not go unnoticed."""

    calls: dict[str, object] = {}

    def apply_aiter(method, layer, hidden_states, topk_weights, topk_ids, **kw):
        calls.update(hidden_states=hidden_states, **kw)
        return torch.zeros(hidden_states.shape, dtype=torch.bfloat16)

    monkeypatch.setattr(moe_runtime, "apply_aiter_w4a8_moe", apply_aiter)
    method = _method(slimquant_w4a8)
    method.moe = SimpleNamespace(
        moe_backend="auto",
        moe_parallel_config=SimpleNamespace(
            dp_size=4, use_ep=True, all2all_backend="allgather_reducescatter"
        ),
    )
    quanted = torch.ones(3, 4, dtype=torch.int8)
    scale = torch.ones(3, 1)

    method.apply(
        layer=SimpleNamespace(),
        x=quanted,
        topk_weights=torch.ones(3, 2),
        topk_ids=torch.zeros(3, 2, dtype=torch.int32),
        shared_experts=None,
        shared_experts_input=None,
        i_q=quanted,
        i_s=scale,
    )

    assert calls["hidden_states"] is quanted
    assert calls["a1q_scale"] is scale


def test_apply_rejects_prequantized_input_on_triton_backend(
    monkeypatch: pytest.MonkeyPatch,
    slimquant_w4a8: ModuleType,
    moe_runtime: ModuleType,
) -> None:
    monkeypatch.setattr(
        moe_runtime,
        "apply_vllm_w4a8_moe",
        lambda *args, **kwargs: pytest.fail("Triton must not see INT8 input"),
    )
    method = _method(slimquant_w4a8)
    method.moe = SimpleNamespace(
        moe_backend="triton",
        moe_parallel_config=SimpleNamespace(
            dp_size=4, use_ep=True, all2all_backend="allgather_reducescatter"
        ),
    )
    quanted = torch.ones(3, 4, dtype=torch.int8)

    with pytest.raises(RuntimeError, match="requires the AITER backend"):
        method.apply(
            layer=SimpleNamespace(),
            x=quanted,
            topk_weights=torch.ones(3, 2),
            topk_ids=torch.zeros(3, 2, dtype=torch.int32),
            shared_experts=None,
            shared_experts_input=None,
            i_q=quanted,
            i_s=torch.ones(3, 1),
        )


# --------------------------------------------------------------------------
# W4A8 runtime: pre-quantized activations reach AITER MOE_C unchanged
# --------------------------------------------------------------------------


def _w4a8_case(selected: object | None):
    calls: dict[str, object] = {}

    class MoeQuantType:
        W4A8 = "w4a8"

    def get_config(**kwargs: object):
        calls["selector"] = kwargs
        return (selected is not None), selected

    def aiter_moe(**kwargs: object):
        calls["execute"] = kwargs
        return torch.zeros(
            kwargs["hidden_states"].shape, dtype=kwargs["output_dtype"]
        )

    aiter_moe_module = _module(
        "aiter.moe",
        MoeQuantType=MoeQuantType,
        get_aiter_moe_config=get_config,
        aiter_moe=aiter_moe,
    )
    w1 = torch.zeros(2, 8, 2, dtype=torch.int8)
    w2 = torch.zeros(2, 4, 2, dtype=torch.int8)
    method = SimpleNamespace(
        moe=SimpleNamespace(num_experts=2, in_dtype=torch.bfloat16),
        moe_quant_config=SimpleNamespace(
            w1_scale=torch.ones((2, 8, 1)),
            w2_scale=torch.ones((2, 4, 1)),
            a1_scale=None,
            a2_scale=None,
        ),
    )
    layer = SimpleNamespace(
        w13_weight=w1,
        w2_weight=w2,
        global_num_experts=2,
        _expert_map=None,
        expert_mask=None,
    )
    return calls, aiter_moe_module, method, layer


def test_runtime_passes_int8_activation_and_scale_to_moe_c(
    monkeypatch: pytest.MonkeyPatch, moe_runtime: ModuleType
) -> None:
    selected = SimpleNamespace(
        quant_type="w4a8",
        solution_type="moe_c",
        need_shuffle=False,
        need_shuffle_scale=False,
    )
    calls, aiter_moe_module, method, layer = _w4a8_case(selected)
    monkeypatch.setitem(sys.modules, "aiter.moe", aiter_moe_module)
    quanted = torch.ones(3, 4, dtype=torch.int8)
    scale = torch.full((3, 1), 0.5)

    result = moe_runtime.apply_aiter_w4a8_moe(
        method,
        layer,
        quanted,
        torch.ones(3, 2),
        torch.zeros(3, 2, dtype=torch.int32),
        a1q_scale=scale,
    )

    # Config selection keys on the BF16 problem, so tuned configs still match.
    assert calls["selector"]["dtype"] == torch.bfloat16
    execute = calls["execute"]
    assert execute["hidden_states"] is quanted
    assert execute["a1_scale"] is scale
    assert execute["output_dtype"] == torch.bfloat16
    assert result.dtype == torch.bfloat16


def test_runtime_rejects_prequantized_input_without_moe_c(
    monkeypatch: pytest.MonkeyPatch, moe_runtime: ModuleType
) -> None:
    _, aiter_moe_module, method, layer = _w4a8_case(None)
    monkeypatch.setitem(sys.modules, "aiter.moe", aiter_moe_module)

    with pytest.raises(
        moe_runtime.HcuCompressedTensorsMoeError, match="MOE_C solution"
    ):
        moe_runtime.apply_aiter_w4a8_moe(
            method,
            layer,
            torch.ones(3, 4, dtype=torch.int8),
            torch.ones(3, 2),
            torch.zeros(3, 2, dtype=torch.int32),
            a1q_scale=torch.ones(3, 1),
        )


# --------------------------------------------------------------------------
# Runner: quantize, gather INT8 + scale, keep shared experts local
# --------------------------------------------------------------------------


class _EPGroup:
    """Two-rank in-memory naive DP/EP gather."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def dispatch_router_logits(
        self, hidden_states, router_logits, is_sequence_parallel, extra_tensors
    ):
        self.calls.append(
            dict(hidden_states=hidden_states, extra_tensors=extra_tensors)
        )
        gather = lambda t: torch.cat((t, t), dim=0)  # noqa: E731
        return (
            gather(hidden_states),
            gather(router_logits),
            [gather(t) for t in extra_tensors],
        )


def _runner(
    moe_runner_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    quant_method: object,
    dp_size: int = 2,
    pcp_size: int = 1,
):
    runner = object.__new__(moe_runner_module.MoERunner)
    runner.moe_config = SimpleNamespace(
        dp_size=dp_size,
        pcp_size=pcp_size,
        is_sequence_parallel=False,
        moe_parallel_config=SimpleNamespace(use_all2all_kernels=True),
    )
    runner.routed_experts = SimpleNamespace(quant_method=quant_method)
    group = _EPGroup()
    monkeypatch.setattr(moe_runner_module, "get_ep_group", lambda: group)
    return runner, group


class _PrequantMethod:
    supports_internal_mk = False

    def quantize_dispatch_input(self, layer, x):
        del layer
        return x.to(torch.int8), torch.full((x.shape[0], 1), 0.25)


def test_runner_gathers_int8_and_scale(
    monkeypatch: pytest.MonkeyPatch, moe_runner_module: ModuleType
) -> None:
    runner, group = _runner(moe_runner_module, monkeypatch, _PrequantMethod())
    hidden = torch.ones(3, 4, dtype=torch.bfloat16)
    logits = torch.zeros(3, 8)

    quanted, gathered_logits, scale = runner._maybe_dispatch_prequantized(
        hidden, logits
    )

    assert group.calls[0]["hidden_states"].dtype == torch.int8
    assert quanted.dtype == torch.int8 and quanted.shape == (6, 4)
    assert gathered_logits.shape == (6, 8)
    torch.testing.assert_close(scale, torch.full((6, 1), 0.25))


@pytest.mark.parametrize(
    ("dp_size", "pcp_size", "quant_method", "has_logits"),
    [
        (1, 1, _PrequantMethod(), True),  # no naive dispatch
        (2, 2, _PrequantMethod(), True),  # PCP keeps its fallback collective
        (2, 1, SimpleNamespace(supports_internal_mk=False), True),  # no hook
        (
            2,
            1,
            SimpleNamespace(
                supports_internal_mk=False,
                quantize_dispatch_input=lambda layer, x: None,
            ),
            True,
        ),  # hook declines (native layout / master off)
        (2, 1, _PrequantMethod(), False),  # preselected routing guard owns it
    ],
)
def test_runner_keeps_bf16_dispatch_otherwise(
    monkeypatch: pytest.MonkeyPatch,
    moe_runner_module: ModuleType,
    dp_size: int,
    pcp_size: int,
    quant_method: object,
    has_logits: bool,
) -> None:
    runner, group = _runner(
        moe_runner_module, monkeypatch, quant_method, dp_size, pcp_size
    )

    result = runner._maybe_dispatch_prequantized(
        torch.ones(3, 4), torch.zeros(3, 8) if has_logits else None
    )

    assert result is None
    assert group.calls == []


def test_dispatched_int8_never_reaches_shared_experts(
    moe_runner_module: ModuleType,
) -> None:
    seen: dict[str, object] = {}

    class QuantMethod:
        is_monolithic = False
        topk_indices_dtype = None

        def apply(self, layer, x, topk_weights, topk_ids, shared_experts,
                  shared_experts_input, i_q=None, i_s=None):
            seen.update(i_q=i_q, i_s=i_s)
            return torch.zeros(x.shape, dtype=torch.bfloat16)

    class SharedExperts:
        output = torch.zeros(3, 4)

        def __call__(self, x, order, x_and_scale_quanted=None):
            seen.setdefault("shared_quanted", []).append(x_and_scale_quanted)

    runner = object.__new__(moe_runner_module.MoERunner)
    runner.routed_experts = SimpleNamespace(quant_method=QuantMethod())
    runner._shared_experts = SharedExperts()
    quanted = torch.ones(6, 4, dtype=torch.int8)
    scale = torch.ones(6, 1)

    runner._apply_quant_method(
        quanted,
        None,
        torch.ones(3, 4, dtype=torch.bfloat16),
        quanted_hidden_states=quanted,
        scale=scale,
        topk_weights=torch.ones(6, 2),
        topk_ids=torch.zeros(6, 2, dtype=torch.int32),
        shared_experts_use_quanted=False,
    )

    assert seen["i_q"] is quanted and seen["i_s"] is scale
    assert seen["shared_quanted"] == [None, None]
