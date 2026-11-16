# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from unittest.mock import Mock

import pytest
import torch

from tests.runtime_patch.test_lightop_w16a16_moe import _config, _target_module
from vllm_hcu.model_executor.layers.fused_moe.experts.deepep_bf16_moe import (
    DeepEPAutoBF16Experts,
    DeepEPBF16ContiguousExperts,
)
from vllm_hcu.patch.worker.op_opt.moe import patch_unquantized_oracle


def _auto_config():
    config = _config(backend="deep_gemm")
    config.moe_parallel_config.use_deepep_auto_kernels = True
    return config


def test_explicit_deepgemm_selects_auto_bf16_and_preserves_ht_weights(monkeypatch):
    target, calls = _target_module()
    target.logger = Mock()
    monkeypatch.setattr(
        DeepEPBF16ContiguousExperts,
        "_supports_current_device",
        staticmethod(lambda: True),
    )
    patch_unquantized_oracle.apply_to_module(target)
    config = _auto_config()
    backend, experts = target.select_unquantized_moe_backend(config)
    assert backend.name == "HCU_DEEPEP_BF16"
    assert experts is DeepEPAutoBF16Experts
    assert target.map_unquantized_backend("deep_gemm") is backend
    w13, w2 = torch.empty(4, 1024, 2048), torch.empty(4, 2048, 512)
    converted = target.convert_to_unquantized_kernel_format(backend, config, w13, w2)
    assert converted[0] is w13 and converted[1] is w2
    assert not calls  # Explicit BF16 request never delegates to AITER or Triton.


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"in_dtype": torch.float16}, "bfloat16"),
        ({"has_bias": True}, "bias"),
        ({"hidden_dim": 3000}, "divisible"),
        ({"intermediate_size_per_partition": 1000}, "divisible"),
        ({"is_lora_enabled": True}, "LoRA"),
        ({"apply_router_weight_on_input": True}, "router weighting"),
    ],
)
def test_unsupported_bf16_request_fails_before_loading(monkeypatch, change, reason):
    target, _ = _target_module()
    target.logger = Mock()
    monkeypatch.setattr(
        DeepEPBF16ContiguousExperts,
        "_supports_current_device",
        staticmethod(lambda: True),
    )
    patch_unquantized_oracle.apply_to_module(target)
    config = _auto_config()
    for key, value in change.items():
        setattr(config, key, value)
    with pytest.raises(ValueError, match=reason):
        target.select_unquantized_moe_backend(config)


def test_bf16_rejects_fixed_all2all_instead_of_building_auto(monkeypatch):
    config = _auto_config()
    config.moe_parallel_config.use_deepep_auto_kernels = False
    monkeypatch.setattr(
        DeepEPBF16ContiguousExperts,
        "_supports_current_device",
        staticmethod(lambda: True),
    )
    supported, reason = DeepEPAutoBF16Experts.is_supported_config(
        DeepEPAutoBF16Experts, config, None, None, None
    )
    assert not supported
    assert "parallel config" in reason


def test_bf16_does_not_request_deferred_quantization():
    experts = object.__new__(DeepEPBF16ContiguousExperts)
    assert experts.expects_unquantized_inputs is False


@pytest.mark.parametrize("entry,draft", [
    ("reload_weights", False),
    ("finish_weight_update", False),
    ("finish_weight_update", True),
])
@pytest.mark.parametrize("replace", [False, True])
def test_public_weight_update_refreshes_bf16_packed_copies_in_place(
    monkeypatch, entry, draft, replace,
):
    """Exercise the submitted Worker method without constructing a device runner."""
    import ast
    from pathlib import Path
    import sys
    from types import ModuleType, SimpleNamespace
    from vllm_hcu.model_executor.layers.fused_moe.experts import deepep_bf16_moe as impl

    monkeypatch.setattr(impl, "_deepgemm_op", lambda name: lambda x: x.transpose(-1, -2).contiguous())
    def model(value):
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(torch.full((2, 4, 3), value), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.full((2, 3, 2), value), requires_grad=False)
        packed = object.__new__(impl.DeepEPBF16MaskedExperts)
        packed._packed_w13 = packed._packed_w2 = None
        expert = object.__new__(impl.DeepEPAutoBF16Experts)
        expert._fixed_use_low_latency = None
        expert.ll_experts = packed
        layer.quant_method = SimpleNamespace(moe_kernel=SimpleNamespace(fused_experts=expert))
        expert.process_weights_after_loading(layer)
        return layer, packed
    target, target_packed = model(1.)
    draft_model, draft_packed = model(2.)
    chosen, packed = (draft_model, draft_packed) if draft else (target, target_packed)
    pointers = packed._packed_w13.data_ptr(), packed._packed_w2.data_ptr()
    def update(self, *args, **kwargs):
        with torch.no_grad():
            for name in ("w13_weight", "w2_weight"):
                weight = getattr(chosen, name)
                if replace:
                    setattr(chosen, name, torch.nn.Parameter(torch.full_like(weight, 7.), requires_grad=False))
                else:
                    weight.fill_(7.)
    base = type("Worker", (), {entry: update})
    source = Path(__file__).resolve().parents[2] / "vllm_hcu/v1/worker.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == "HcuGPUWorker")
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == entry]
    namespace = {"Worker": base}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    # Keep unrelated refresh providers out of this CPU lifecycle contract.
    for name, function in [
        ("vllm_hcu.patch.worker.core_fix.patch_deepseek_v4_attention", "refresh_compressor_nt_weights"),
        ("vllm_hcu.model_executor.layers.fused_moe.glm53_router_gemv", "refresh_glm53_router_gates"),
        ("vllm_hcu.model_executor.layers.kimi_k3_mixed_precision", "refresh_kimi_kda_fused_qkv"),
        ("vllm_hcu.patch.worker.core_fix.patch_deepseek_v4_rocm_wo_a_layout", "refresh_wo_a_bf16_caches"),
        ("vllm_hcu.model_executor.layers.quantization.compressed_tensors_w4a8_runtime", "refresh_hcu_w4a8_scale_caches"),
    ]:
        module = ModuleType(name)
        setattr(module, function, lambda *args: None)
        monkeypatch.setitem(sys.modules, name, module)
    module = ModuleType("vllm_hcu.model_executor.layers.quantization.grouped_w4a16_moe_runtime")
    module.snapshot_grouped_w4a16_scale_owners = lambda model: {}
    module.refresh_grouped_w4a16_scale_caches = lambda *args: None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    worker = namespace["HcuGPUWorker"]()
    worker.model_runner = SimpleNamespace(model=target, get_draft_model=lambda: draft_model)
    worker._weight_update_is_draft = draft
    getattr(worker, entry)()
    assert (packed._packed_w13.data_ptr(), packed._packed_w2.data_ptr()) == pointers
    torch.testing.assert_close(packed._packed_w13, chosen.w13_weight.transpose(-1, -2))
    torch.testing.assert_close(packed._packed_w2, chosen.w2_weight.transpose(-1, -2))
    untouched = target_packed if draft else draft_packed
    assert torch.all(untouched._packed_w13 == (1. if draft else 2.))


@pytest.mark.parametrize("mode", ["NONE", "FULL", "PIECEWISE", "FULL_AND_PIECEWISE", "FULL_DECODE_ONLY"])
@pytest.mark.parametrize("opt_in", [False, True])
def test_fixed_ht_graph_policy_precedes_experimental_override(monkeypatch, mode, opt_in):
    import ast
    import enum
    import os
    from pathlib import Path
    from types import SimpleNamespace
    modes = enum.Enum("Mode", "NONE FULL PIECEWISE FULL_AND_PIECEWISE FULL_DECODE_ONLY")
    path = Path(__file__).resolve().parents[2] / "vllm_hcu/patch/platform/core_fix/patch_compilation_config.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_normalize_deepep_auto_cudagraph_mode")
    namespace = {"os": os, "HcuFeatureConfig": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    monkeypatch.setenv("VLLM_HCU_DEEPEP_AUTO_ALLOW_CUDAGRAPH", "1" if opt_in else "0")
    result, reason = namespace[node.name](modes[mode], feature_config=SimpleNamespace(deepep_auto=True),
        fixed_use_low_latency=False, parallel_size=4, modes=tuple(modes))
    assert result is modes.NONE
    # Explicit dynamic HT/LL remains available; the fixed producer is distinct.
    if opt_in:
        result, reason = namespace[node.name](modes[mode], feature_config=SimpleNamespace(deepep_auto=True),
            fixed_use_low_latency=None, parallel_size=4, modes=tuple(modes))
        assert result is modes[mode] and reason == "explicit"


def test_online_eplb_is_refused_before_delegating_to_expert_selection(monkeypatch):
    monkeypatch.setattr(DeepEPBF16ContiguousExperts, "is_supported_config",
                        staticmethod(lambda *args: (True, None)))
    config = _auto_config()
    config.moe_parallel_config.enable_eplb = True
    supported, reason = DeepEPAutoBF16Experts.is_supported_config(
        DeepEPAutoBF16Experts, config, None, None, None)
    assert not supported and "EPLB" in reason


@pytest.mark.parametrize("hidden", [512, 768, 1024, 2048, 2560, 3072])
@pytest.mark.parametrize("custom,lightop", [(False, False), (False, True), (True, False), (True, True)])
def test_bf16_admission_matches_selected_ht_gather(monkeypatch, hidden, custom, lightop):
    from vllm_hcu.model_executor.layers.fused_moe.experts import deepep_bf16_moe as impl

    monkeypatch.setattr(impl.current_platform, "is_rocm", lambda: True)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", str(int(custom)))
    monkeypatch.setenv("VLLM_HCU_USE_LIGHTOP_EP_SCATTER", str(int(lightop)))
    # Isolate this admission boundary from unrelated host/target selection ABI.
    monkeypatch.setattr(DeepEPBF16ContiguousExperts, "is_supported_config",
                        staticmethod(lambda *args: (True, None)))
    config = _auto_config()
    config.hidden_dim = hidden
    supported, reason = DeepEPAutoBF16Experts.is_supported_config(
        DeepEPAutoBF16Experts, config, None, None, None)
    expected = (custom and lightop) or hidden % min(hidden, 1024) == 0
    assert supported is expected
    if not expected:
        assert "ep_gather" in reason


def test_bf16_masked_activation_preserves_expert_counts_and_padding(monkeypatch):
    from types import SimpleNamespace
    from vllm_hcu.model_executor.layers.fused_moe.experts import deepep_bf16_moe as impl
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    expert = object.__new__(impl.DeepEPBF16MaskedExperts)
    expert.num_dispatchers = 1
    e, m, n, k = 3, 8, 8, 4
    counts = torch.tensor([0, 1, 3], dtype=torch.int32)
    x = torch.arange(e * m * k, dtype=torch.float32).view(e, m, k) / 100
    w1 = torch.arange(e * n * k, dtype=torch.float32).view(e, n, k) / 100
    w2 = torch.arange(e * k * (n // 2), dtype=torch.float32).view(e, k, n // 2) / 100
    expert._packed_w13, expert._packed_w2 = w1, w2
    workspace = torch.full((e, m, n + n // 2), -123.)
    output = torch.full((e, m, k), -456.)
    calls = []

    def gemm(a, weight, out, actual_counts, expected_m, *, config):
        assert actual_counts is counts
        assert config == {"MODE": 2000}
        calls.append("gemm")
        for i, count in enumerate(counts.tolist()):
            out[i, :count].copy_(a[i, :count] @ weight[i].T)

    def activate(activation, out, inp, *, valid_token_counts):
        assert activation is MoEActivation.SILU
        assert valid_token_counts is counts
        assert inp.shape == (e, m, n) and out.shape == (e, m, n // 2)
        assert inp.is_contiguous() and out.is_contiguous()
        calls.append("activation")
        for i, count in enumerate(counts.tolist()):
            gate, up = inp[i, :count].chunk(2, dim=-1)
            out[i, :count].copy_(torch.nn.functional.silu(gate) * up)

    monkeypatch.setattr(impl, "_deepgemm_op", lambda _: gemm)
    expert.activation = activate
    expert.apply(output, x, w1, w2, torch.ones(4, 1), torch.zeros(4, 1, dtype=torch.int64),
                 MoEActivation.SILU, e, None, None, None, torch.empty(0), workspace,
                 SimpleNamespace(expert_num_tokens=counts), False)
    assert calls == ["gemm", "activation", "gemm"]
    activated = workspace.view(-1)[e * m * n:].view(e, m, n // 2)
    for i, count in enumerate(counts.tolist()):
        gate, up = (x[i, :count] @ w1[i].T).chunk(2, dim=-1)
        expected = (torch.nn.functional.silu(gate) * up) @ w2[i].T
        torch.testing.assert_close(output[i, :count], expected)
        assert torch.all(output[i, count:] == -456.)
        assert torch.all(activated[i, count:] == -123.)
