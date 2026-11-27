# SPDX-License-Identifier: Apache-2.0
"""Real Core CT selector contracts; no model loading or device execution.

Load the complete experimental module while bypassing legacy package startup.
Only package namespaces are isolated; Core classes and target matching are real.
"""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)

ROOT = Path(__file__).resolve().parents[3]
PREFIX = "model.layers.1.mlp.experts"


@pytest.fixture
def selector(monkeypatch):
    for name in (
        "vllm_hcu",
        "vllm_hcu.models",
        "vllm_hcu.models.hy_v4_flash",
        "vllm_hcu.models.hy_v4_flash.nvidia",
    ):
        package = ModuleType(name)
        package.__path__ = [str(ROOT / name.replace(".", "/"))]
        monkeypatch.setitem(sys.modules, name, package)
    # Execute the real runtime module and restore its previous identity after
    # each case, including when earlier seam tests installed a test module.
    runtime_name = "vllm_hcu.models.hy_v4_flash.runtime"
    runtime_spec = importlib.util.spec_from_file_location(
        runtime_name, ROOT / "vllm_hcu/models/hy_v4_flash/runtime.py"
    )
    runtime = importlib.util.module_from_spec(runtime_spec)
    runtime_spec.loader.exec_module(runtime)
    monkeypatch.setitem(sys.modules, runtime_name, runtime)
    path = ROOT / "vllm_hcu/models/hy_v4_flash/nvidia/w4a8.py"
    spec = importlib.util.spec_from_file_location("_flash_w4a8_selector_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configuration(target):
    def group(bits, targets):
        return {
            "targets": targets,
            "format": "pack-quantized" if bits == 4 else "int-quantized",
            "weights": {
                "num_bits": bits, "type": "int", "strategy": "channel",
                "symmetric": True, "dynamic": False,
            },
            "input_activations": {
                "num_bits": 8, "type": "int", "strategy": "token",
                "symmetric": True, "dynamic": True,
            },
        }
    return {
        "quant_method": "compressed-tensors", "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "experts": group(4, [target]),
            "dense": group(8, ["re:.*q_b_proj$"]),
        },
    }


def routed_layer():
    # Constructor requires distributed/device setup. This fixture exercises only
    # the actual RoutedExperts identity and metadata read by the CT selector.
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.ckpt_gate_proj_name = "gate_proj"
    layer.ckpt_up_proj_name = "up_proj"
    layer.ckpt_down_proj_name = "down_proj"
    layer.moe_config = object()
    return layer


@pytest.mark.parametrize("target", ["Linear", "RoutedExperts", "re:.*experts.*"])
def test_official_ct_target_forms_select_reference_without_source_mutation(selector, target):
    source = CompressedTensorsConfig.from_config(configuration(target))
    original_map = deepcopy(source.target_scheme_map)
    original_config = deepcopy(source.config)
    local = selector.HYV4FlashW4A8Config.for_moe(source)
    method = local.get_quant_method(routed_layer(), PREFIX)
    assert isinstance(method, selector.HYV4FlashChannelW4A8Method)
    assert source.target_scheme_map == original_map
    assert source.config == original_config
    # Repeat selection to cover the official helper's idempotent expansion.
    assert isinstance(local.get_quant_method(routed_layer(), PREFIX), type(method))


def test_explicit_routed_scheme_keeps_precedence_over_linear_fallback(selector):
    config = configuration("Linear")
    config["config_groups"]["dense"]["targets"] = ["RoutedExperts"]
    local = selector.HYV4FlashW4A8Config.for_moe(
        CompressedTensorsConfig.from_config(config)
    )
    with pytest.raises(ValueError, match="channel INT4 / token INT8 experts"):
        local.get_quant_method(routed_layer(), PREFIX)
    assert local.target_scheme_map["RoutedExperts"]["weights"].num_bits == 8


def test_official_ignore_remains_enforced(selector):
    config = configuration("Linear")
    config["ignore"] = [f"{PREFIX}.0.gate_proj"]
    local = selector.HYV4FlashW4A8Config.for_moe(
        CompressedTensorsConfig.from_config(config)
    )
    with pytest.raises(ValueError, match="identical schemes"):
        local.get_quant_method(routed_layer(), PREFIX)
