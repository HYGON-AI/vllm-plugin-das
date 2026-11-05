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


@pytest.mark.parametrize("backend,ep", [("unsupported", 1), ("triton", 2)])
def test_unsupported_execution_fails_at_construction(backend, ep):
    moe = SimpleNamespace(
        moe_backend=backend,
        moe_parallel_config=SimpleNamespace(ep_size=ep),
        has_bias=False,
    )
    with pytest.raises(ValueError):
        adapter.HcuCompressedW4A8(None, moe)


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
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
        quant_config = CompressedTensorsConfig.from_config(recipe())
        method = adapter.HcuCompressedW4A8(quant_config, moe)
        layer = RoutedExperts.__new__(RoutedExperts)
        torch.nn.Module.__init__(layer)
        layer.moe_config = moe
        layer.quant_config = quant_config
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


def test_output_activation_quantization_is_not_silently_dropped():
    data = recipe()
    data["config_groups"]["int4"]["output_activations"] = {
        "num_bits": 8,
        "type": "int",
        "strategy": "tensor",
        "dynamic": False,
        "symmetric": True,
    }
    config = CompressedTensorsConfig.from_config(data)
    layer = SimpleNamespace(moe_config=SimpleNamespace(num_logical_experts=2))
    with pytest.raises(ValueError, match="output activation"):
        adapter.select_method(config, layer, "model.layers.3.mlp.experts")


@pytest.mark.parametrize(
    "target",
    [
        r"re:^model\.decoder\.mlp\.experts\..*",
        "model.decoder.mlp.experts.0.up_proj",
    ],
)
def test_mtp_quant_alias_keeps_physical_rules_and_ignore(target):
    data = recipe(target)
    config = CompressedTensorsConfig.from_config(data)
    config._add_fused_moe_to_target_scheme_map()
    layer = torch.nn.Identity()
    physical = "model.layers.78.mlp.experts.0.up_proj"
    first = adapter.mtp_quant_config(config, "model.layers.78")
    second = adapter.mtp_quant_config(config, "model.layers.79")
    assert first is not config and second is not first
    assert first.get_scheme_dict(layer, physical)["weights"].num_bits == 4
    assert second.get_scheme_dict(layer, physical) is None
    assert config.get_scheme_dict(layer, physical) is None
    assert first.get_scheme_dict(layer, physical.replace("78", "780")) is None
    # A physical checkpoint rule takes priority over the canonical alias.
    explicit = {**config.target_scheme_map[target], "weights": object()}
    first.target_scheme_map = {**first.target_scheme_map, physical: explicit}
    assert first.get_scheme_dict(layer, physical) is explicit
    first.ignore = [physical]
    assert first.get_scheme_dict(layer, physical) is None
    # Canonical ignore entries still apply when selecting canonical rules.
    second.ignore = ["model.decoder.mlp.experts.0.up_proj"]
    assert second.get_scheme_dict(layer, physical.replace("78", "79")) is None
    assert adapter.mtp_quant_config(None, "model.layers.78") is None


def test_actual_mtp_and_decoder_constructors_keep_numeric_runtime_prefix():
    # Execute both production __init__ bodies. The installed v0.25 wheel cannot
    # import this v0.28 model (V32SparseAttnIndexer is missing), so stub only
    # the expensive attention/MoE/normalization leaves, not constructor logic.
    import ast
    from pathlib import Path

    class Leaf(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.prefix = kwargs.get("prefix")
            self.quant_config = kwargs.get("quant_config")

    root = Path(__file__).resolve().parents[2]
    env = {
        "nn": torch.nn,
        "torch": torch,
        "copy": __import__("copy").copy,
        "RMSNorm": Leaf,
        "SharedHead": Leaf,
        "DeepseekV2MLAAttention": Leaf,
        "DeepseekV2Attention": Leaf,
        "DeepseekAttention": Leaf,
        "DeepseekV2MoE": Leaf,
        "DeepseekV2MLP": Leaf,
        "current_platform": SimpleNamespace(device_type="cpu"),
        "henvs": SimpleNamespace(VLLM_HCU_DISABLE_DSA=False),
    }
    for file, name in [
        ("deepseek_v2.py", "DeepseekV2DecoderLayer"),
        ("deepseek_mtp.py", "DeepSeekMultiTokenPredictorLayer"),
    ]:
        tree = ast.parse((root / "vllm_hcu/models" / file).read_text())
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name
        )
        cls.body = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        ]
        code = "from __future__ import annotations\n" + ast.unparse(cls)
        exec(compile(code, file, "exec"), env)  # noqa: S102 - trusted repo constructors
    hf = SimpleNamespace(
        model_type="glm_moe_dsa",
        hidden_size=8,
        rms_norm_eps=1e-6,
        n_routed_experts=2,
        first_k_dense_replace=3,
        num_attention_heads=1,
    )
    quant = CompressedTensorsConfig.from_config(recipe())
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf)
        ),
        quant_config=quant,
        model_config=SimpleNamespace(use_mla=False),
        cache_config=None,
        parallel_config=None,
    )
    layer = env["DeepSeekMultiTokenPredictorLayer"](config, "model.layers.78")
    assert layer.mtp_block.layer_idx == 78
    assert layer.mtp_block.self_attn.prefix == "model.layers.78.self_attn"
    assert layer.mtp_block.mlp.prefix == "model.layers.78.mlp"
    assert layer.mtp_block.mlp.quant_config is not quant
    assert config.quant_config is quant


def test_aiter_gemm2_nibble_regroup(monkeypatch):
    from aiter.ops import shuffle

    seen = []
    monkeypatch.setattr(
        shuffle,
        "w4a8_moe_layout_shuffle_gemm2",
        lambda value: seen.append(value.clone()) or value,
    )
    # Logical nibbles 0..7 regroup into high [0..3], low [4..7].
    packed = torch.tensor([0x01, 0x23, 0x45, 0x67], dtype=torch.uint8).view(torch.int8)
    result = adapter.repack_aiter_gemm2(packed.reshape(1, 1, 4))
    assert torch.equal(
        result.view(torch.uint8).flatten(),
        torch.tensor([0x04, 0x15, 0x26, 0x37], dtype=torch.uint8),
    )
    assert len(seen) == 1


@pytest.mark.parametrize("backend", ["auto", "aiter"])
def test_pd_aiter_prefill_decode_layout_selection_is_local(monkeypatch, backend):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from vllm_hcu.model_executor.layers.quantization import (
        compressed_tensors_moe_runtime as runtime,
    )

    config = SimpleNamespace(
        quant_type="W4A8",
        solution_type="moe_c",
        need_shuffle=True,
        need_shuffle_scale=True,
        config={},
    )
    moe = SimpleNamespace(
        moe_backend=backend,
        moe_parallel_config=SimpleNamespace(ep_size=1),
        has_bias=False,
        activation=MoEActivation.SILU,
        num_experts=2,
    )
    method = adapter.HcuCompressedW4A8(None, moe)
    layer = torch.nn.Module()
    method.create_weights(
        layer, 2, 16, 8, torch.bfloat16, weight_loader=lambda *args: None
    )
    for param in layer.parameters():
        param.data.zero_()
    calls = []
    monkeypatch.setattr(
        adapter, "repack_aiter_gemm2", lambda w: calls.append(w.data_ptr()) or w.clone()
    )
    monkeypatch.setattr(runtime, "prewarm_aiter_w4a8_moe", lambda *a: config)
    method.process_weights_after_loading(layer)
    method.process_weights_after_loading(layer)
    assert len(calls) == 2
    assert config.need_shuffle and config.need_shuffle_scale
    selected = []

    def select(problem, **kwargs):
        selected.append(problem.M)
        return config

    monkeypatch.setattr(runtime, "select_aiter_moe_config", select)
    executions = []

    def execute(local, **kwargs):
        assert local is not config
        assert not local.need_shuffle and not local.need_shuffle_scale
        assert kwargs["w1"] is layer.w13_weight and kwargs["w2"] is layer.w2_weight
        executions.append(kwargs)
        return kwargs["hidden_states"]

    monkeypatch.setattr(runtime, "execute_aiter_moe", execute)
    # Alternate prefill and decode token counts without changing installed data.
    for batch in [32, 1, 7, 1]:
        x = torch.ones(batch, 16, dtype=torch.bfloat16)
        actual = runtime.apply_aiter_w4a8_moe(
            method,
            layer,
            x,
            torch.ones(batch, 2),
            torch.zeros(batch, 2, dtype=torch.int32),
        )
        assert actual is x
        assert config.need_shuffle and config.need_shuffle_scale
    assert selected == [32, 1, 7, 1] and len(executions) == 4
    assert len(calls) == 2
    with pytest.raises(RuntimeError, match="no solution"):
        method.prepare_aiter_w4a8_config(layer, None)
    with pytest.raises(RuntimeError, match="fresh model"):
        layer.w13_weight.weight_loader(
            layer.w13_weight, torch.zeros(1, dtype=torch.int32)
        )
    layer.w13_weight_scale = torch.nn.Parameter(
        layer.w13_weight_scale.detach().clone(), requires_grad=False
    )
    with pytest.raises(RuntimeError, match="weights/scales changed"):
        method.prepare_aiter_w4a8_config(layer, config)
