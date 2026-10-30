# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""SlimQuant W4A8 on fixed deepep_high_throughput / deepep_low_latency (Qwen4Exp)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe import deepep_runtime
from vllm_hcu.model_executor.layers.fused_moe.experts import (
    dpsk_v4_deep_gemm_moe as deepgemm_module,
)
from vllm_hcu.model_executor.layers.quantization import (
    compressed_tensors_moe_runtime,
    slimquant_w4a8,
)

_QWEN = "Qwen4ExpForConditionalGeneration"


def _moe(
    backend: str,
    architecture: str = _QWEN,
    *,
    dp_size: int = 4,
    use_ep: bool = True,
    activation: str = "silu",
    moe_backend: str = "deep_gemm",
    enable_eplb: bool = False,
    auto_kernels: bool = False,
    **extra: object,
) -> SimpleNamespace:
    return SimpleNamespace(
        activation=SimpleNamespace(value=activation),
        moe_backend=moe_backend,
        moe_parallel_config=SimpleNamespace(
            dp_size=dp_size,
            pcp_size=1,
            use_ep=use_ep,
            all2all_backend=backend,
            use_deepep_auto_kernels=auto_kernels,
            enable_eplb=enable_eplb,
            use_deepep_ht_kernels=backend == "deepep_high_throughput",
            use_deepep_ll_kernels=backend == "deepep_low_latency",
        ),
        _hcu_vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(architectures=[architecture])
        ),
        **extra,
    )


@pytest.fixture
def hcu_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        deepep_runtime, "current_platform", SimpleNamespace(is_rocm=lambda: True)
    )
    monkeypatch.setattr(
        deepep_runtime, "_require_slimquant_w4a8_hipc_runtime", lambda: None
    )


@pytest.mark.hcu
@pytest.mark.parametrize(
    ("backend", "expected"),
    [("deepep_high_throughput", False), ("deepep_low_latency", True)],
)
@pytest.mark.parametrize(
    "architecture", ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"]
)
def test_fixed_deepep_role_for_qwen4exp(
    hcu_runtime: None, backend: str, expected: bool, architecture: str
) -> None:
    moe = _moe(backend, architecture)

    assert deepep_runtime.slimquant_w4a8_fixed_deepep_use_low_latency(moe) is expected
    # A supported fixed layout is not a deepep_auto route.
    assert deepep_runtime.slimquant_w4a8_uses_deepep_auto(moe) is False


@pytest.mark.hcu
@pytest.mark.parametrize(
    ("backend", "architecture"),
    [
        ("deepep_auto", _QWEN),
        ("allgather_reducescatter", _QWEN),
        ("deepep_high_throughput", "DeepseekV4ForCausalLM"),
        ("deepep_low_latency", "GlmMoeDsaForCausalLM"),
    ],
)
def test_fixed_deepep_role_is_none_outside_supported_combinations(
    hcu_runtime: None, backend: str, architecture: str
) -> None:
    moe = _moe(backend, architecture, auto_kernels=backend == "deepep_auto")

    assert deepep_runtime.slimquant_w4a8_fixed_deepep_use_low_latency(moe) is None


@pytest.mark.hcu
@pytest.mark.parametrize("backend", ["deepep_high_throughput", "deepep_low_latency"])
@pytest.mark.parametrize("architecture", ["DeepseekV4ForCausalLM", "UnknownModel"])
def test_fixed_deepep_still_rejected_for_other_architectures(
    hcu_runtime: None, backend: str, architecture: str
) -> None:
    with pytest.raises(ValueError, match="requires all2all_backend='deepep_auto'"):
        deepep_runtime.slimquant_w4a8_uses_deepep_auto(_moe(backend, architecture))


@pytest.mark.hcu
@pytest.mark.parametrize("backend", ["deepep_high_throughput", "deepep_low_latency"])
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dp_size": 1}, "requires dp_size > 1 with expert parallelism"),
        ({"use_ep": False}, "requires dp_size > 1 with expert parallelism"),
        ({"enable_eplb": True}, "does not support EPLB"),
        ({"auto_kernels": True}, "incompatible use_deepep_auto_kernels"),
        ({"moe_backend": "aiter"}, "requires moe_backend='auto' or 'deep_gemm'"),
        ({"activation": "situ"}, "supports only SiLU activation"),
    ],
)
def test_fixed_deepep_invalid_qwen4exp_config_fails_closed(
    hcu_runtime: None, backend: str, kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        deepep_runtime.slimquant_w4a8_fixed_deepep_use_low_latency(
            _moe(backend, **kwargs)
        )


@pytest.mark.hcu
def test_fixed_deepep_requires_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        deepep_runtime, "current_platform", SimpleNamespace(is_rocm=lambda: False)
    )

    with pytest.raises(RuntimeError, match="requires the HCU ROCm runtime"):
        deepep_runtime.slimquant_w4a8_fixed_deepep_use_low_latency(
            _moe("deepep_low_latency")
        )


@pytest.mark.hcu
@pytest.mark.parametrize(
    ("backend", "use_low_latency"),
    [("deepep_high_throughput", False), ("deepep_low_latency", True)],
)
def test_fixed_deepep_builds_w4a8_deepgemm_kernel_not_aiter(
    monkeypatch: pytest.MonkeyPatch,
    hcu_runtime: None,
    backend: str,
    use_low_latency: bool,
) -> None:
    factory_calls: list[dict[str, object]] = []
    processed_layers: list[object] = []

    class W4A8Experts:
        def process_weights_after_loading(self, layer: object) -> None:
            processed_layers.append(layer)

    class W4A8Kernel:
        fused_experts = SimpleNamespace(experts=W4A8Experts())

        @staticmethod
        def apply(**kwargs: object) -> torch.Tensor:
            return kwargs["hidden_states"] + 3

    def make_fixed(**kwargs: object) -> W4A8Kernel:
        factory_calls.append(kwargs)
        return W4A8Kernel()

    monkeypatch.setattr(
        deepgemm_module,
        "make_fixed_deepep_deepgemm_w4a8_moe_kernel",
        make_fixed,
    )
    monkeypatch.setattr(
        deepgemm_module,
        "make_deepep_auto_deepgemm_w4a8_moe_kernel",
        lambda **_: pytest.fail("fixed backend must not build the auto kernel"),
    )
    monkeypatch.setattr(
        compressed_tensors_moe_runtime,
        "prewarm_aiter_w4a8_moe",
        lambda *_args: pytest.fail("fixed W4A8 must not prewarm AITER"),
    )
    monkeypatch.setattr(
        compressed_tensors_moe_runtime,
        "apply_aiter_w4a8_moe",
        lambda *_args: pytest.fail("fixed W4A8 must execute its DeepGEMM kernel"),
    )

    moe = _moe(backend)
    method = slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod(object(), moe)
    routing_tables = (object(), object(), object())
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.zeros((2, 8, 2), dtype=torch.int8), requires_grad=False
        ),
        w2_weight=torch.nn.Parameter(
            torch.zeros((2, 4, 2), dtype=torch.int8), requires_grad=False
        ),
        w13_weight_scale=torch.nn.Parameter(torch.ones((2, 8, 1)), requires_grad=False),
        w2_weight_scale=torch.nn.Parameter(torch.ones((2, 4, 1)), requires_grad=False),
        w13_input_scale=None,
        w2_input_scale=None,
        swiglu_limit=10.0,
        _expert_routing_tables=lambda: routing_tables,
    )

    method.process_weights_after_loading(layer)

    assert len(factory_calls) == 1
    call = factory_calls[0]
    assert call["use_low_latency"] is use_low_latency
    assert call["moe_config"] is moe
    assert call["routing_tables"] is routing_tables
    quant_config = call["moe_quant_config"]
    assert quant_config.weight_quant_dtype == "int4"
    assert quant_config.gemm1_clamp_limit == 10.0
    assert processed_layers == [layer]
    assert method.supports_int8_allgatherv_input is False
    x = torch.zeros((2, 4), dtype=torch.bfloat16)
    torch.testing.assert_close(
        method.apply(
            layer,
            x,
            torch.ones((2, 1), dtype=torch.float32),
            torch.zeros((2, 1), dtype=torch.int32),
            None,
            None,
        ),
        x + 3,
    )


def _w4a8_quant_config(layer_scale: torch.Tensor):
    return slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod._get_deepgemm_moe_quant_config(
        SimpleNamespace(
            w13_weight_scale=layer_scale,
            w2_weight_scale=layer_scale,
            w13_input_scale=None,
            w2_input_scale=None,
            swiglu_limit=None,
        )
    )


@pytest.mark.hcu
@pytest.mark.parametrize(
    ("backend", "use_low_latency", "pf_tokens", "expected_tokens"),
    [
        ("deepep_high_throughput", False, None, 8192),
        ("deepep_low_latency", True, 128, 128),
    ],
)
def test_fixed_deepep_factory_uses_role_fixed_experts_and_single_prepare_finalize(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    use_low_latency: bool,
    pf_tokens: int | None,
    expected_tokens: int,
) -> None:
    from vllm.model_executor.layers.fused_moe import all2all_utils

    prepare_finalize = SimpleNamespace(
        max_num_tokens_per_rank=lambda: pf_tokens,
        num_dispatchers=lambda: 4,
    )
    pf_calls: list[dict[str, object]] = []

    def maybe_make_prepare_finalize(**kwargs: object) -> object:
        pf_calls.append(kwargs)
        return prepare_finalize

    expert_calls: list[dict[str, object]] = []

    class FakeExperts:
        def __init__(self, **kwargs: object) -> None:
            expert_calls.append(kwargs)

    kernel_calls: list[tuple[object, object]] = []

    class FakeKernel:
        def __init__(self, pf: object, experts: object) -> None:
            kernel_calls.append((pf, experts))

    monkeypatch.setattr(
        all2all_utils, "maybe_make_prepare_finalize", maybe_make_prepare_finalize
    )
    monkeypatch.setattr(deepgemm_module, "DeepEPAutoW4A8Experts", FakeExperts)
    monkeypatch.setattr(deepgemm_module.mk, "FusedMoEKernel", FakeKernel)

    moe = _moe(backend, max_num_tokens=8192)
    routing_tables = (object(), object(), object())
    quant_config = _w4a8_quant_config(torch.ones((2, 8, 1)))

    kernel = deepgemm_module.make_fixed_deepep_deepgemm_w4a8_moe_kernel(
        quant_config, moe, routing_tables, use_low_latency=use_low_latency
    )

    assert isinstance(kernel, FakeKernel)
    assert [call["moe"] for call in pf_calls] == [moe]
    assert pf_calls[0]["routing_tables"] is routing_tables
    assert pf_calls[0]["use_monolithic"] is False
    assert len(expert_calls) == 1
    assert expert_calls[0]["fixed_use_low_latency"] is use_low_latency
    assert expert_calls[0]["max_num_tokens"] == expected_tokens
    assert expert_calls[0]["num_dispatchers"] == 4
    assert expert_calls[0]["quant_config"] is quant_config
    assert kernel_calls == [(prepare_finalize, kernel_calls[0][1])]


@pytest.mark.hcu
@pytest.mark.parametrize(
    ("backend", "use_low_latency", "match"),
    [
        ("deepep_high_throughput", True, "requires the deepep_low_latency backend"),
        ("deepep_low_latency", False, "requires the deepep_high_throughput backend"),
    ],
)
def test_fixed_deepep_factory_rejects_mismatched_backend(
    backend: str, use_low_latency: bool, match: str
) -> None:
    with pytest.raises(RuntimeError, match=match):
        deepgemm_module.make_fixed_deepep_deepgemm_w4a8_moe_kernel(
            _w4a8_quant_config(torch.ones((2, 8, 1))),
            _moe(backend, max_num_tokens=8192),
            use_low_latency=use_low_latency,
        )


@pytest.mark.hcu
def test_fixed_deepep_factory_requires_llm_token_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe import all2all_utils

    monkeypatch.setattr(
        all2all_utils,
        "maybe_make_prepare_finalize",
        lambda **_: SimpleNamespace(max_num_tokens_per_rank=lambda: None),
    )

    with pytest.raises(RuntimeError, match="LL token capacity is unavailable"):
        deepgemm_module.make_fixed_deepep_deepgemm_w4a8_moe_kernel(
            _w4a8_quant_config(torch.ones((2, 8, 1))),
            _moe("deepep_low_latency", max_num_tokens=8192),
            use_low_latency=True,
        )
