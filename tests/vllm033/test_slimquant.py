# SPDX-License-Identifier: Apache-2.0
"""Distinct format contracts; no official quantization method replacements."""

from __future__ import annotations

import copy
import os

import pytest


def checkpoint_config():
    return {
        "quant_method": "slimquant_w4a8",
        "ignore": ["re:.*lm_head$"],
        "mixed_precision": {
            "experts": {
                "format": "pack-int4-nibble",
                "scope": "per_channel",
                "weight_bits": 4,
                "modules": [
                    "model.layers.*.mlp.experts.gate_up_proj",
                    "model.layers.*.mlp.experts.down_proj",
                ],
            },
            "w8a8_include": {
                "format": "int8-per-channel",
                "weight_bits": 8,
                "activation_bits": 8,
                "activation_scheme": "dynamic_per_token",
                "modules": ["model.layers.*.self_attn.q_proj"],
            },
            "ngram_embedding": {
                "format": "int8",
                "weight_bits": 8,
                "scope": "per_channel",
                "modules": ["model.layers.*.ple.ple_embedding.ngram_embedding.shard_*"],
            },
        },
    }


def test_all_packed_bytes_decode_canonical_signed_int4_without_rescaling():
    import torch
    from hcu_vllm_das.slimquant import unpack_signed_int4

    packed = torch.arange(256, dtype=torch.int16).to(torch.int8).reshape(2, 128)
    result = unpack_signed_int4(packed)
    reference = []
    for byte in range(256):
        high, low = divmod(byte, 16)
        reference.extend(
            [high if high < 8 else high - 16, low if low < 8 else low - 16]
        )
    assert result.dtype == torch.int8 and tuple(result.shape) == (2, 256)
    assert result.flatten().tolist() == reference
    assert packed.flatten().tolist() == [b if b < 128 else b - 256 for b in range(256)]


@pytest.mark.parametrize("group", ["experts", "w8a8_include", "ngram_embedding"])
def test_parser_rejects_missing_mixed_precision_group(group):
    from hcu_vllm_das.slimquant import SlimQuantConfig

    config = checkpoint_config()
    config["mixed_precision"].pop(group)
    with pytest.raises(ValueError, match="SlimQuant requires declared"):
        SlimQuantConfig.from_config(config)


def test_parser_maps_all_declared_module_lists_without_mutating_checkpoint():
    from hcu_vllm_das.slimquant import SlimQuantConfig
    from vllm.model_executor.models.utils import WeightsMapper

    raw = checkpoint_config()
    original = copy.deepcopy(raw)
    config = SlimQuantConfig.from_config(raw)
    config.apply_vllm_mapper(
        WeightsMapper(orig_to_new_prefix={"model.": "language_model.model."})
    )
    assert config.get_name() == "slimquant_w4a8"
    assert all(
        name.startswith("language_model.model.")
        for names in (config.expert_modules, config.dense_modules, config.ngram_modules)
        for name in names
    )
    assert config.ignore == ["re:.*lm_head$"]
    assert raw == original


@pytest.fixture
def native_parallel(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch
    import hcu_vllm_das.slimquant as implementation
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    # The standalone native table fixture has one serialized shard. Supply only
    # that public model-metadata boundary; device/backend/kernel paths stay real.
    metadata = SimpleNamespace(model_config=SimpleNamespace(
        hf_text_config=SimpleNamespace(split_ngram_parts=1)))
    monkeypatch.setattr(implementation, "get_current_vllm_config_or_none", lambda: metadata)
    torch.cuda.set_device(0)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{tmp_path / 'dist-store'}",
            backend="nccl",
        )
        initialize_model_parallel(1, 1)
        init_workspace_manager(torch.device("cuda:0"))
        try:
            yield
        finally:
            reset_workspace_manager()
            cleanup_dist_env_and_memory()


native = pytest.mark.skipif(
    os.environ.get("HCU_SLIMQUANT_LIVE") != "1",
    reason="explicit live format validation",
)


@pytest.mark.hcu
@native
@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("activation", ["bfloat16", "float16"])
def test_public_int8_ple_method_preserves_scales_and_changed_input_graph(
    native_parallel,
    pinned,
    activation,
):
    import torch
    from torch.nn import functional as F
    from hcu_vllm_das.slimquant import SlimQuantConfig
    from vllm.models.qwen4_exp.common.ngram_embedding import (
        Qwen4ExpPLEDeviceEmbedding,
        Qwen4ExpPLEEmbeddingMethod,
        Qwen4ExpPLEPinnedHostEmbedding,
    )

    prefix = "model.layers.0.ple.ple_embedding.ngram_embedding"
    config = SlimQuantConfig.from_config(checkpoint_config())
    method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(config, prefix)
    cls = Qwen4ExpPLEPinnedHostEmbedding if pinned else Qwen4ExpPLEDeviceEmbedding
    dtype = getattr(torch, activation)
    with torch.device("cuda:0"):
        layer = cls(
            16,
            3,
            params_dtype=dtype,
            padding_size=8,
            prefix=prefix,
            embedding_method=method,
            num_ngram_heads=2,
            max_total_tokens=4,
        )
    weight = (torch.arange(48).reshape(16, 3) % 17 - 8).to(torch.int8)
    # FP32 scales deliberately exceed BF16 scale storage precision.
    scales = (torch.arange(16).reshape(16, 1).float() + 1) * 0.01234567
    assert (
        layer.weight.dtype == torch.int8 and layer.weight_scale.dtype == torch.float32
    )
    layer.weight_scale.weight_loader(layer.weight_scale, scales, checkpoint_start=0)
    layer.weight.weight_loader(layer.weight, weight, checkpoint_start=0)
    method.process_weights_after_loading(layer)
    ids = torch.tensor([[0, 15], [1, 2]], device="cuda:0")
    lookup = layer.sync_lookup if pinned else layer
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        lookup(ids)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = lookup(ids)
    address = output.data_ptr()
    for values in ([[0, 15], [1, 2]], [[3, 4], [15, 0]]):
        ids.copy_(torch.tensor(values, device="cuda:0"))
        graph.replay()
        reference = F.embedding(torch.tensor(values), weight.float() * scales).to(dtype)
        torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)
        assert output.data_ptr() == address


@pytest.mark.hcu
@native
def test_packed_experts_enter_official_loader_with_standard_scale_units(
    native_parallel,
):
    import torch
    from hcu_vllm_das.slimquant import SlimQuantConfig, SlimQuantInt8Experts
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory

    with torch.device("cuda:0"):
        runner = FusedMoEFactory(
            num_experts=4,
            top_k=2,
            hidden_size=256,
            intermediate_size=128,
            params_dtype=torch.bfloat16,
            prefix="model.layers.0.mlp.experts",
            quant_config=SlimQuantConfig.from_config(checkpoint_config()),
        )
    experts = runner.routed_experts
    assert isinstance(experts.quant_method, SlimQuantInt8Experts)
    weights = []
    references = {}
    for name, shape in (("gate_up_proj", (4, 256, 128)), ("down_proj", (4, 256, 64))):
        packed = (
            torch.arange(torch.tensor(shape).prod().item(), dtype=torch.int64)
            .reshape(shape)
            .to(torch.int8)
        )
        scales = (
            torch.arange(1, shape[0] * shape[1] + 1)
            .reshape(shape[0], shape[1], 1)
            .float()
            / 4096
        )
        weights.extend([(f"{name}.weight_scale", scales), (name, packed)])
        unsigned = packed.to(torch.int16) % 256
        high = unsigned // 16
        low = unsigned % 16
        references[name] = (
            torch.stack(
                (
                    torch.where(high < 8, high, high - 16),
                    torch.where(low < 8, low, low - 16),
                ),
                -1,
            )
            .flatten(-2)
            .to(torch.int8),
            scales,
        )
    loaded = set(experts.load_weights(weights))
    assert loaded == {"w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"}
    for attr, name in (("w13", "gate_up_proj"), ("w2", "down_proj")):
        weight, scale = references[name]
        torch.testing.assert_close(
            getattr(experts, f"{attr}_weight").cpu(), weight, rtol=0, atol=0
        )
        torch.testing.assert_close(
            getattr(experts, f"{attr}_weight_scale").cpu(), scale * 16, rtol=0, atol=0
        )


@pytest.mark.hcu
@native
def test_expert_scale_domain_matches_independent_moe_reference_and_graph(
    native_parallel,
):
    """The serialized scale is 1/16 of the canonical signed-INT4 scale."""
    import torch
    from torch.nn import functional as F
    from hcu_vllm_das.slimquant import SlimQuantConfig
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory

    with torch.device("cuda:0"):
        runner = FusedMoEFactory(
            num_experts=4,
            top_k=2,
            hidden_size=256,
            intermediate_size=128,
            params_dtype=torch.bfloat16,
            prefix="model.layers.0.mlp.experts",
            quant_config=SlimQuantConfig.from_config(checkpoint_config()),
        )
    layer = runner.routed_experts
    # Fixed bytes have known alternating values: 0x12 -> [1,2], 0x34 -> [3,4].
    gate = torch.full((4, 256, 128), 0x12, dtype=torch.int8)
    down = torch.full((4, 256, 64), 0x34, dtype=torch.int8)
    scales = torch.arange(1, 5, dtype=torch.float32)[:, None, None] / 4096
    gate_scales = scales.expand(4, 256, 1).contiguous()
    down_scales = gate_scales.clone()
    original = gate_scales.clone()
    list(
        layer.load_weights(
            [
                ("gate_up_proj", gate),
                ("gate_up_proj.weight_scale", gate_scales),
                ("down_proj", down),
                ("down_proj.weight_scale", down_scales),
            ]
        )
    )
    layer.quant_method.process_weights_after_loading(layer)
    torch.testing.assert_close(gate_scales, original, rtol=0, atol=0)
    torch.testing.assert_close(down_scales, original, rtol=0, atol=0)
    x = torch.linspace(0.125, 0.5, 256, device="cuda:0", dtype=torch.bfloat16)
    x = x.repeat(3, 1)
    ids = torch.tensor([[0, 1], [1, 2], [2, 3]], device="cuda:0", dtype=torch.int32)
    routes = torch.tensor([[0.25, 0.75]] * 3, device="cuda:0")

    def apply():
        return layer.quant_method.apply(layer, x, routes, ids, None, None)

    apply()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = apply()
    address = output.data_ptr()
    for multiplier in (1.0, 0.5):
        x.copy_(torch.linspace(0.125, 0.5, 256, device="cuda:0") * multiplier)
        graph.replay()
        reference = torch.zeros((3, 256), dtype=torch.float64)
        for row in range(3):
            for route, expert in zip((0.25, 0.75), ids[row].tolist()):
                # Independent effective scale: serialized (expert+1)/4096
                # maps to canonical weights (expert+1)/256.
                effective = (expert + 1) / 256
                first = torch.tensor([1.0, 2.0], dtype=torch.float64).repeat(128)
                gate_value = x[row].cpu().double().dot(first * effective)
                activated = (F.silu(gate_value) * gate_value).repeat(128)
                second = torch.tensor([3.0, 4.0], dtype=torch.float64).repeat(64)
                reference[row] += route * activated.dot(second * effective)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(
            output.cpu().double(), reference, rtol=0.01, atol=0.001
        )
        assert output.data_ptr() == address
