# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from types import SimpleNamespace
import sys

import pytest
import torch

from vllm_hcu.model_executor.layers.quantization import grouped_w4a16_moe_runtime as runtime


def _case():
    quant = SimpleNamespace(
        use_int4_w4a16=True,
        w1_zp=None,
        w2_zp=None,
    )
    experts = SimpleNamespace(
        quant_config=quant, block_shape=[0, 128],
        moe_config=SimpleNamespace(moe_backend="aiter"),
        w1_scale=torch.ones(2, 256, 2), w2_scale=torch.ones(2, 256, 1),
    )
    return dict(
        experts=experts, output=torch.empty(3, 256, dtype=torch.bfloat16),
        hidden_states=torch.ones(3, 256, dtype=torch.bfloat16),
        w1=torch.ones(2, 256, 128, dtype=torch.uint8),
        w2=torch.ones(2, 256, 64, dtype=torch.uint8),
        topk_weights=torch.ones(3, 1), topk_ids=torch.zeros(3, 1, dtype=torch.int32),
        activation="silu", global_num_experts=2, expert_map=None,
        a1q_scale=None, a2_scale=None, expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )


@pytest.mark.parametrize("solution", ["moe_c", "asm", "triton", None])
def test_grouped_dispatch_preserves_group_and_packed_weights(monkeypatch, solution):
    call = _case()
    monkeypatch.setitem(sys.modules, "aiter.moe", SimpleNamespace(
        MoeQuantType=SimpleNamespace(W4A16="w4a16")))
    config = None if solution is None else SimpleNamespace(solution_type=solution, need_shuffle=False)
    selected, executed = [], []
    def select(problem, cache_owner):
        selected.append(problem)
        assert cache_owner is call["w1"]
        return config
    def execute(config, **kwargs):
        executed.append(kwargs)
        return torch.full_like(kwargs["hidden_states"], 7)
    monkeypatch.setattr(runtime, "select_aiter_moe_config", select)
    monkeypatch.setattr(runtime, "execute_aiter_moe", execute)
    assert runtime.apply_grouped_w4a16_moe(**call) is (solution is not None)
    assert selected[0].block_size == 128
    assert selected[0].use_shuffle is False
    if solution is not None:
        assert executed[0]["w1"] is call["w1"]
        if solution == "moe_c":
            assert torch.all(executed[0]["w1_zp"] == 0x88)
            assert torch.all(executed[0]["w2_zp"] == 0x88)
        else:
            assert executed[0]["w1_zp"] is None
            assert executed[0]["w2_zp"] is None
        assert executed[0]["block_shape"] == [0, 128]
        assert torch.all(call["output"] == 7)
    else:
        assert not executed


@pytest.mark.parametrize("unsupported", ["triton", "zp", "ep", "router", "lora", "metadata", "int8"])
def test_unsupported_semantics_keep_original_wna16(monkeypatch, unsupported):
    call = _case()
    experts = call["experts"]
    if unsupported == "triton":
        experts.moe_config.moe_backend = "triton"
    elif unsupported == "zp":
        experts.quant_config.w1_zp = torch.ones(1)
    elif unsupported == "ep":
        call["global_num_experts"] = 4
    elif unsupported == "router":
        call["apply_router_weight_on_input"] = True
    elif unsupported == "lora":
        experts._lora_context = object()
    elif unsupported == "metadata":
        call["expert_tokens_meta"] = object()
    elif unsupported == "int8":
        experts.quant_config.use_int4_w4a16 = False
    def unexpected(*args, **kwargs):
        pytest.fail("Unsupported semantics must not reach AITER selector")
    monkeypatch.setattr(runtime, "select_aiter_moe_config", unexpected)
    assert runtime.apply_grouped_w4a16_moe(**call) is False


def test_moe_c_synthesizes_symmetric_gptq_zero_points(monkeypatch):
    call = _case()
    monkeypatch.setitem(sys.modules, "aiter.moe", SimpleNamespace(
        MoeQuantType=SimpleNamespace(W4A16="w4a16")))
    monkeypatch.setattr(
        runtime, "select_aiter_moe_config",
        lambda problem, cache_owner: SimpleNamespace(
            solution_type="moe_c", need_shuffle=False,
        ),
    )
    captured = {}
    monkeypatch.setattr(
        runtime, "execute_aiter_moe",
        lambda config, **kwargs: captured.update(kwargs) or kwargs["hidden_states"],
    )
    assert runtime.apply_grouped_w4a16_moe(**call) is True
    assert captured["w1_zp"].shape == (2, 128, 2)
    assert captured["w2_zp"].shape == (2, 128, 1)
    assert torch.all(captured["w1_zp"] == 0x88)
    assert torch.all(captured["w2_zp"] == 0x88)


def test_legacy_moe_c_group128_scale_shuffle_falls_back(monkeypatch):
    call = _case()
    monkeypatch.setitem(sys.modules, "aiter.moe", SimpleNamespace(
        MoeQuantType=SimpleNamespace(W4A16="w4a16")))
    monkeypatch.setattr(
        runtime, "select_aiter_moe_config",
        lambda problem, cache_owner: SimpleNamespace(
            solution_type="moe_c", need_shuffle_scale=True,
        ),
    )
    monkeypatch.setattr(
        runtime, "prepare_aiter_moe_scales",
        lambda *args, **kwargs: pytest.fail("legacy scale shuffle was reached"),
    )
    assert runtime.apply_grouped_w4a16_moe(**call) is False


@pytest.mark.parametrize("w1_zp,w2_zp", [(0x77, 0x66), (0x88, 0x88)])
def test_explicit_grouped_zero_points_fall_back_before_aiter_selection(
    monkeypatch, w1_zp, w2_zp,
):
    call = _case()
    quant = call["experts"].quant_config
    quant.w1_zp = torch.full((2, 128, 2), w1_zp, dtype=torch.uint8)
    quant.w2_zp = torch.full((2, 128, 1), w2_zp, dtype=torch.uint8)
    monkeypatch.setattr(
        runtime, "select_aiter_moe_config",
        lambda *args, **kwargs: pytest.fail("explicit zero points reached AITER"),
    )
    assert runtime.apply_grouped_w4a16_moe(**call) is False


@pytest.mark.parametrize("source", ["quant", "moe"])
def test_clamped_swiglu_falls_back_before_aiter_selection(monkeypatch, source):
    from vllm.model_executor.layers.fused_moe.activation import (
        ApplyMoEActivationConfig,
    )

    call = _case()
    experts = call["experts"]
    quant = experts.quant_config
    moe = experts.moe_config
    quant.gemm1_clamp_limit = 1.0 if source == "quant" else None
    quant.gemm1_alpha = None
    quant.gemm1_beta = None
    moe.swiglu_limit = 1.0 if source == "moe" else None
    moe.swiglu_alpha = None
    moe.swiglu_beta = None
    moe.activation_situ_beta = None
    moe.activation_situ_linear_beta = None
    experts.activation_config = ApplyMoEActivationConfig.from_configs(moe, quant)
    assert experts.activation_config.clamp_limit == 1.0
    monkeypatch.setattr(
        runtime, "select_aiter_moe_config",
        lambda *args, **kwargs: pytest.fail("clamped SwiGLU reached AITER"),
    )
    assert runtime.apply_grouped_w4a16_moe(**call) is False


@pytest.mark.parametrize("field,value", [("alpha", 1.2), ("beta", 0.2)])
def test_modified_silu_falls_back_before_aiter_selection(monkeypatch, field, value):
    from vllm.model_executor.layers.fused_moe.activation import (
        ApplyMoEActivationConfig,
    )

    call = _case()
    call["experts"].activation_config = ApplyMoEActivationConfig(**{field: value})
    monkeypatch.setattr(
        runtime, "select_aiter_moe_config",
        lambda *args, **kwargs: pytest.fail("modified SiLU reached AITER"),
    )
    assert runtime.apply_grouped_w4a16_moe(**call) is False
