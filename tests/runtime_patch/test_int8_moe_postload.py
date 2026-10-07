# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.quantization import (
    compressed_tensors_moe_runtime,
)
from vllm_hcu.patch.worker.op_opt import (
    patch_compressed_tensors_moe_w8a8_int8,
)


def _int8_moe_module(backend: str, events: list[str], postload=None):
    class CompressedTensorsW8A8Int8MoEMethod:
        def __init__(self):
            self.int8_backend = SimpleNamespace(value=backend)
            self.moe = object()
            self.moe_quant_config = None

        def process_weights_after_loading(self, layer):
            events.append("official")
            self.moe_quant_config = object()
            if postload is not None:
                postload(layer)

    module = ModuleType(patch_compressed_tensors_moe_w8a8_int8.TARGET_MODULE)
    module.CompressedTensorsW8A8Int8MoEMethod = (
        CompressedTensorsW8A8Int8MoEMethod
    )
    return module


def test_int8_aiter_weights_are_installed_during_postload(monkeypatch):
    events = []
    module = _int8_moe_module("aiter", events)

    def prewarm(layer, moe, quant_config):
        events.append("aiter")
        assert layer == "layer"
        assert moe is method.moe
        assert quant_config is method.moe_quant_config

    monkeypatch.setattr(
        compressed_tensors_moe_runtime,
        "prewarm_aiter_quantized_moe",
        prewarm,
    )
    assert patch_compressed_tensors_moe_w8a8_int8.apply_to_module(module) is True
    assert patch_compressed_tensors_moe_w8a8_int8.apply_to_module(module) is False

    method = module.CompressedTensorsW8A8Int8MoEMethod()
    method.process_weights_after_loading("layer")
    assert events == ["official", "aiter"]


def test_int8_non_aiter_backend_preserves_official_postload(monkeypatch):
    events = []
    module = _int8_moe_module("triton", events)
    monkeypatch.setattr(
        compressed_tensors_moe_runtime,
        "prewarm_aiter_quantized_moe",
        lambda *_args: events.append("aiter"),
    )
    patch_compressed_tensors_moe_w8a8_int8.apply_to_module(module)

    method = module.CompressedTensorsW8A8Int8MoEMethod()
    method.process_weights_after_loading("layer")
    assert events == ["official"]


@pytest.mark.parametrize(
    "enabled,backend,replacement,preserved",
    [
        (True, "HCU_DEEPGEMM", "view", True),
        (True, "HCU_DEEPGEMM", "clone", False),
        (True, "HCU_DEEPGEMM", "subset", False),
        (True, "HCU_DEEPGEMM", "unmarked", False),
        (False, "HCU_DEEPGEMM", "view", False),
        (True, "triton", "view", False),
    ],
)
def test_postload_keeps_uva_only_for_same_storage(
    monkeypatch, enabled, backend, replacement, preserved,
):
    monkeypatch.setenv("VLLM_HCU_REGISTERED_CPU_OFFLOAD", str(int(enabled)))
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(
        torch.arange(32, dtype=torch.int8).view(4, 8), requires_grad=False,
    )
    if replacement != "unmarked":
        layer.w13_weight._vllm_is_uva_offloaded = True

    def postload(layer):
        data = layer.w13_weight.data
        if replacement == "clone":
            data = data.clone()
        elif replacement == "subset":
            data = data[:2]
        layer.w13_weight = torch.nn.Parameter(data.reshape(-1), requires_grad=False)

    module = _int8_moe_module(backend, [], postload)
    patch_compressed_tensors_moe_w8a8_int8.apply_to_module(module)
    method = module.CompressedTensorsW8A8Int8MoEMethod()
    method.process_weights_after_loading(layer)
    assert getattr(layer.w13_weight, "_vllm_is_uva_offloaded", False) is preserved


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU")
@pytest.mark.parametrize("layout", ["contiguous", "masked"])
def test_deepgemm_postload_retains_uva_through_loading_context(monkeypatch, layout):
    import deepgemm
    from vllm.model_executor.model_loader.utils import device_loading_context
    from vllm.model_executor.offloader.uva import UVAOffloader
    from vllm.model_executor.utils import replace_parameter
    from vllm.platforms import current_platform
    from vllm_hcu.patch.worker.core_fix import patch_hcu_uva_offloader

    current_platform.import_kernels()
    monkeypatch.setenv("VLLM_HCU_REGISTERED_CPU_OFFLOAD", "1")
    patch_hcu_uva_offloader.apply()
    experts, out_features, hidden, tokens = 2, 4608, 5120, 8
    layer = torch.nn.Module()
    torch.manual_seed(42)
    weight = torch.randint(
        -32, 32, (experts, out_features, hidden), device="cuda", dtype=torch.int8,
    )
    reference_weight = weight.clone()
    layer.w13_weight = torch.nn.Parameter(weight, requires_grad=False)
    offloader = UVAOffloader(weight.numel(), {"w13_weight"})
    offloader._maybe_offload_to_cpu(layer)
    registered_pointer = layer.w13_weight.data_ptr()

    def postload(layer):
        packed = getattr(deepgemm, f"marlin_i8_{layout}_weight")(layer.w13_weight)
        replace_parameter(layer, "w13_weight", packed)

    target = _int8_moe_module("HCU_DEEPGEMM", [], postload)
    patch_compressed_tensors_moe_w8a8_int8.apply_to_module(target)
    method = target.CompressedTensorsW8A8Int8MoEMethod()
    before = torch.cuda.memory.host_memory_stats()["active_bytes.allocated"]
    with device_loading_context(layer, torch.device("cuda:0")):
        method.process_weights_after_loading(layer)
    after = torch.cuda.memory.host_memory_stats()["active_bytes.allocated"]
    assert after == before
    assert layer.w13_weight.data_ptr() == registered_pointer
    assert layer.w13_weight._vllm_is_uva_offloaded

    shape = (256, hidden) if layout == "contiguous" else (experts, tokens, hidden)
    activation = torch.randint(-32, 32, shape, device="cuda", dtype=torch.int8)
    a_scale = torch.full(shape[:-1] + (1,), 0.01, device="cuda")
    w_scale = torch.full((experts, out_features, 1), 0.01, device="cuda")
    output = torch.empty(
        shape[:-1] + (out_features,), device="cuda", dtype=torch.bfloat16,
    )
    if layout == "contiguous":
        deepgemm.m_grouped_i8_gemm_nt_contiguous(
            (activation, a_scale), (layer.w13_weight, w_scale), output,
            torch.zeros(shape[0], device="cuda", dtype=torch.int32),
        )
        expected = (activation.float() * a_scale) @ (
            reference_weight[0].float() * w_scale[0]
        ).T
    else:
        deepgemm.m_grouped_i8_gemm_nt_masked(
            (activation, a_scale), (layer.w13_weight, w_scale), output,
            torch.full((experts,), tokens, device="cuda", dtype=torch.int32), tokens,
        )
        expected = (activation.float() * a_scale) @ (
            reference_weight.float() * w_scale
        ).transpose(-1, -2)
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=1e-2)
