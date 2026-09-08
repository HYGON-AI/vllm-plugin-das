# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from types import ModuleType, SimpleNamespace

from vllm_hcu.model_executor.layers.quantization import (
    compressed_tensors_moe_runtime,
)
from vllm_hcu.patch.worker.op_opt import (
    patch_compressed_tensors_moe_w8a8_int8,
)


def _int8_moe_module(backend: str, events: list[str]):
    class CompressedTensorsW8A8Int8MoEMethod:
        def __init__(self):
            self.int8_backend = SimpleNamespace(value=backend)
            self.moe = object()
            self.moe_quant_config = None

        def process_weights_after_loading(self, layer):
            events.append("official")
            self.moe_quant_config = object()

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
