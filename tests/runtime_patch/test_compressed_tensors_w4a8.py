# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)

from vllm_hcu.model_executor.layers.quantization import (
    compressed_tensors_w4a8 as adapter,
)


def recipe(target="re:.*experts.*", **weight_changes):
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "int4": {
                "targets": [target],
                "format": "pack-quantized",
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "strategy": "channel",
                    "symmetric": True,
                    **weight_changes,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "strategy": "token",
                    "dynamic": True,
                    "symmetric": True,
                },
            }
        },
        "ignore": [],
    }


def test_all_byte_patterns_decode_identically():
    # Eight uint4b8 nibbles per int32; verify signed values independently.
    byte = torch.arange(256, dtype=torch.int32)
    packed = (byte | (byte << 8) | (byte << 16) | (byte << 24)).reshape(-1, 1)
    converted = adapter.convert_packed_int4(packed).view(torch.uint8)
    high = (converted >> 4).to(torch.int16)
    low = (converted & 15).to(torch.int16)
    high = torch.where(high >= 8, high - 16, high)
    low = torch.where(low >= 8, low - 16, low)
    assert torch.equal(high[:, 0], (byte & 15).to(torch.int16) - 8)
    assert torch.equal(low[:, 0], (byte >> 4).to(torch.int16) - 8)


@pytest.mark.parametrize(
    "target", ["re:.*experts.*", "RoutedExperts", "re:^RoutedExperts$"]
)
@pytest.mark.parametrize("layer_index", [3, 60, 78])
def test_native_selection(monkeypatch, target, layer_index):
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = SimpleNamespace(num_logical_experts=2)
    config = CompressedTensorsConfig.from_config(recipe(target))
    marker = object()
    monkeypatch.setattr(adapter, "HcuCompressedW4A8", lambda *a: marker)
    assert (
        adapter.select_method(config, layer, f"model.layers.{layer_index}.mlp.experts")
        is marker
    )


def test_group_format_and_strategy_inheritance(monkeypatch):
    data = recipe(strategy=None, group_size=-1)
    del data["config_groups"]["int4"]["format"]
    config = CompressedTensorsConfig.from_config(data)
    layer = SimpleNamespace(moe_config=SimpleNamespace(num_logical_experts=1))
    marker = object()
    monkeypatch.setattr(adapter, "HcuCompressedW4A8", lambda *a: marker)
    assert adapter.select_method(config, layer, "model.layers.60.mlp.experts") is marker


def test_ignored_projection_cannot_fall_into_quantized_adapter():
    data = recipe()
    data["ignore"] = ["model.layers.3.mlp.experts.1.down_proj"]
    config = CompressedTensorsConfig.from_config(data)
    layer = SimpleNamespace(moe_config=SimpleNamespace(num_logical_experts=2))
    with pytest.raises(ValueError, match="every expert"):
        adapter.select_method(config, layer, "model.layers.3.mlp.experts")


@pytest.mark.parametrize(
    "weight_changes",
    [{"num_bits": 8}, {"strategy": "group", "group_size": 128}, {"symmetric": False}],
)
def test_unrelated_schemes_are_not_selected(weight_changes):
    config = CompressedTensorsConfig.from_config(recipe(**weight_changes))
    layer = SimpleNamespace(moe_config=SimpleNamespace(num_logical_experts=1))
    assert adapter.select_method(config, layer, "model.layers.3.mlp.experts") is None


def test_owner_local_checkpoint_names():
    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    model.experts.quant_method = adapter.HcuCompressedW4A8.__new__(
        adapter.HcuCompressedW4A8
    )
    model.other = torch.nn.Module()
    weights = [("model.experts.0.up_proj.weight_packed", torch.ones(1))]
    # Root wrapper is not part of this submodel's named-module namespace.
    assert next(adapter.checkpoint_weights(model, weights))[0] == weights[0][0]
    # Check real full prefixes, including another native packed owner.
    parent = torch.nn.Module()
    parent.model = model
    rows = [
        ("model.experts.0.up_proj.weight_packed", torch.ones(1)),
        ("model.experts.0.up_proj.weight_shape", torch.ones(2)),
        ("model.other.experts.0.up_proj.weight_packed", torch.ones(1)),
    ]
    assert [name for name, _ in adapter.checkpoint_weights(parent, rows)] == [
        "model.experts.0.up_proj.weight",
        "model.other.experts.0.up_proj.weight_packed",
    ]


def test_mtp_owner_before_loader_name_rewrite():
    parent = torch.nn.Module()
    parent.model = torch.nn.Module()
    parent.model.layers = torch.nn.ModuleDict({"78": torch.nn.Module()})
    block = parent.model.layers["78"]
    block.mtp_block = torch.nn.Module()
    block.mtp_block.mlp = torch.nn.Module()
    block.mtp_block.mlp.experts = torch.nn.Module()
    block.mtp_block.mlp.experts.quant_method = adapter.HcuCompressedW4A8.__new__(
        adapter.HcuCompressedW4A8
    )
    rows = [("model.layers.78.mlp.experts.0.up_proj.weight_packed", torch.ones(1))]
    assert next(adapter.checkpoint_weights(parent, rows))[0] == (
        "model.layers.78.mlp.experts.0.up_proj.weight"
    )


@pytest.mark.parametrize("backend,ep", [("aiter", 1), ("triton", 2)])
def test_unsupported_execution_fails_at_construction(backend, ep):
    moe = SimpleNamespace(
        moe_backend=backend,
        moe_parallel_config=SimpleNamespace(ep_size=ep),
        has_bias=False,
    )
    with pytest.raises(ValueError):
        adapter.HcuCompressedW4A8(None, moe)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_native_expert_loader_tp_shards(tp_size):
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    hidden, intermediate = 32, 32
    for rank in range(tp_size):
        moe = SimpleNamespace(
            moe_backend="triton",
            moe_parallel_config=SimpleNamespace(ep_size=1, tp_size=tp_size),
            has_bias=False,
            tp_rank=rank,
            is_act_and_mul=True,
        )
        method = adapter.HcuCompressedW4A8(None, moe)
        layer = RoutedExperts.__new__(RoutedExperts)
        torch.nn.Module.__init__(layer)
        layer.moe_config = moe
        layer.quant_config = None
        layer.quant_method = method
        layer.expert_map_manager = SimpleNamespace(
            map_global_to_local=lambda index: index
        )
        method.create_weights(
            layer,
            2,
            hidden,
            intermediate // tp_size,
            torch.bfloat16,
            weight_loader=layer.weight_loader,
        )
        for expert in range(2):
            for projection, shard in [("gate", "w1"), ("up", "w3"), ("down", "w2")]:
                packed = torch.arange(128, dtype=torch.int32).reshape(32, 4) + expert
                param = layer.w2_weight if shard == "w2" else layer.w13_weight
                param.weight_loader(param, packed, "weight", shard, expert)
                expected = adapter.convert_packed_int4(packed)
                if shard == "w2":
                    expected = expected[
                        :,
                        rank * (intermediate // tp_size // 2) : (rank + 1)
                        * (intermediate // tp_size // 2),
                    ]
                    actual = param[expert]
                else:
                    expected = expected[
                        rank * (intermediate // tp_size) : (rank + 1)
                        * (intermediate // tp_size)
                    ]
                    start = 0 if shard == "w1" else intermediate // tp_size
                    actual = param[expert, start : start + intermediate // tp_size]
                assert torch.equal(actual, expected), (rank, expert, projection)
                scale = (
                    layer.w2_weight_scale if shard == "w2" else layer.w13_weight_scale
                )
                checkpoint_scale = torch.arange(1, 33, dtype=torch.bfloat16).reshape(
                    32, 1
                )
                scale.weight_loader(
                    scale, checkpoint_scale, "weight_scale", shard, expert
                )
                expected_scale = checkpoint_scale.float() / 16
                if shard == "w2":
                    actual_scale = scale[expert]
                else:
                    expected_scale = expected_scale[
                        rank * (intermediate // tp_size) : (rank + 1)
                        * (intermediate // tp_size)
                    ]
                    actual_scale = scale[
                        expert, start : start + intermediate // tp_size
                    ]
                assert torch.equal(actual_scale, expected_scale)
