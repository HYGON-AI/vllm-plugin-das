# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Reject unsupported CT schemes before the existing HCU owner allocates storage."""
from copy import deepcopy
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from vllm_hcu.model_executor.layers.quantization import compressed_tensors_w4a8_runtime as runtime
from vllm_hcu.patch.worker.op_opt import patch_compressed_tensors_moe_w4a8_int8 as patch


@pytest.fixture
def admission(monkeypatch):
    weight = NS(num_bits=4, type="int", symmetric=True, dynamic=False,
                strategy="channel", group_size=-1, actorder=None)
    activation = NS(num_bits=8, type="int", symmetric=True, dynamic=True,
                    strategy="token", group_size=None)
    scheme = {"weights": weight, "input_activations": activation, "format": "pack-quantized"}
    schemes = {f"experts.{expert}.{projection}_proj": deepcopy(scheme)
               for expert in range(2) for projection in ("gate", "up", "down")}
    lookups, constructed = [], []
    def lookup(layer, name):
        lookups.append(name)
        return schemes[name]
    config = NS(get_scheme_dict=lookup)
    layer = NS(moe_config=NS(num_logical_experts=2, has_bias=False, activation="silu"),
               apply_router_weight_on_input=False)
    module = ModuleType(patch.TARGET_MODULE)
    def original(quant_config, layer, layer_name):
        raise NotImplementedError(patch._CPU_ONLY_ERROR)
    module.CompressedTensorsMoEMethod = type("Native", (), {"get_moe_method": staticmethod(original)})
    monkeypatch.setattr(patch, "_hcu_enabled", lambda moe: True)
    def make(*args):
        constructed.append(args)
        return "existing-owner"
    monkeypatch.setattr(runtime, "make_hcu_w4a8_method", make)
    monkeypatch.setitem(sys.modules, "vllm.logger", NS(init_logger=lambda name: NS(info_once=lambda *args: None)))
    patch.apply_to_module(module)
    return module.CompressedTensorsMoEMethod.get_moe_method, config, layer, schemes, lookups, constructed


def test_uniform_channel_schemes_keep_the_existing_owner(admission):
    get, config, layer, schemes, lookups, constructed = admission
    # Both supported spellings of channel group size remain accepted.
    schemes["experts.1.up_proj"]["weights"].group_size = None
    assert get(config, layer, "experts") == "existing-owner"
    assert len(constructed) == 1
    assert set(lookups) == set(schemes)


@pytest.mark.parametrize("expert", [0, 1])
@pytest.mark.parametrize("projection", ["gate", "up", "down"])
@pytest.mark.parametrize("bad", ["ignored", "bits", "format", "output"])
def test_every_projection_is_checked_before_storage(admission, expert, projection, bad):
    get, config, layer, schemes, lookups, constructed = admission
    name = f"experts.{expert}.{projection}_proj"
    if bad == "ignored":
        schemes[name] = None
    elif bad == "bits":
        schemes[name]["weights"].num_bits = 8
    elif bad == "format":
        schemes[name]["format"] = "dense-quantized"
    else:
        schemes[name]["output_activations"] = NS(num_bits=8)
    with pytest.raises((ValueError, NotImplementedError)):
        get(config, layer, "experts")
    assert constructed == []


def test_future_native_support_remains_first(admission):
    _, config, layer, schemes, lookups, constructed = admission
    module = ModuleType(patch.TARGET_MODULE)
    def native(quant_config, layer, layer_name):
        return "future-native-owner"
    module.CompressedTensorsMoEMethod = type("Native", (), {"get_moe_method": staticmethod(native)})
    patch.apply_to_module(module)
    assert module.CompressedTensorsMoEMethod.get_moe_method(config, layer, "experts") == "future-native-owner"
    assert lookups == [] and constructed == []
