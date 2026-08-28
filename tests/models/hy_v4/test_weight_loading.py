# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers import logits_processor as logits_module
from vllm.model_executor.layers.quantization.kv_cache import KVCacheScaleParameter
from vllm.model_executor.layers import vocab_parallel_embedding as vocab_module
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm_hcu.models.hy_v4 import model as hy_v4_model
from vllm_hcu.models.hy_v4.model import (
    HYV4ForCausalLM,
    HYV4Model,
    _dequantize_indexer_channel_fp8,
    _normalize_hyv4_config,
    _rewrite_hyv4_weight_name,
    _slice_sink_for_tp,
    _try_load_fp8_indexer_projection,
    _try_load_fp8_router_gate,
)
from vllm_hcu.patch.platform.core_fix import patch_logits_processor_head_dtype


def test_excluded_quant_config_is_not_forwarded_to_lm_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeQuantConfig:
        packed_modules_mapping = None

        @staticmethod
        def is_layer_excluded(prefix: str) -> bool:
            return prefix == "lm_head"

    class FakeInnerModel(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.make_empty_intermediate_tensors = object()

    class FakeLMHead(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            captured.update(kwargs)

    monkeypatch.setattr(hy_v4_model, "HYV4Model", FakeInnerModel)
    monkeypatch.setattr(hy_v4_model, "ParallelLMHead", FakeLMHead)
    monkeypatch.setattr(
        hy_v4_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )

    config = SimpleNamespace(
        vocab_size=64,
        hidden_size=32,
        enable_lm_head_fp32=True,
        tie_word_embeddings=False,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=FakeQuantConfig(),
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0)
        ),
    )

    HYV4ForCausalLM(vllm_config=vllm_config)

    assert captured["prefix"] == "lm_head"
    assert "params_dtype" not in captured
    assert captured["quant_config"] is None


def test_compute_logits_keeps_hidden_state_in_model_dtype() -> None:
    captured: dict[str, torch.Tensor] = {}
    model = object.__new__(HYV4ForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(soft_logits_capping=False)
    model.enable_lm_head_fp32 = True
    model.lm_head = torch.nn.Identity()

    def logits_processor(lm_head, hidden_states):
        captured["hidden_states"] = hidden_states
        return hidden_states.float()

    model.logits_processor = logits_processor
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)

    actual = model.compute_logits(hidden_states)

    assert captured["hidden_states"] is hidden_states
    assert actual.dtype == torch.float32


def test_v0251_logits_processor_accumulates_unquantized_hy_v4_head_in_fp32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_logits_processor_head_dtype.apply_to_module(logits_module)
    monkeypatch.setattr(vocab_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        vocab_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    current = SimpleNamespace(
        model_config=SimpleNamespace(head_dtype=torch.float32),
    )
    with set_current_vllm_config(current):
        processor = LogitsProcessor(vocab_size=64)
    processor._gather_logits = lambda logits: logits
    head = ParallelLMHead(64, 16, params_dtype=torch.bfloat16)
    with torch.no_grad():
        head.weight.copy_(
            torch.arange(head.weight.numel(), dtype=torch.float32)
            .reshape_as(head.weight)
            .to(torch.bfloat16)
            / 128
        )
    hidden_states = torch.linspace(
        -1,
        1,
        32,
        dtype=torch.float32,
    ).reshape(2, 16).to(torch.bfloat16)

    actual = processor(head, hidden_states)

    assert head.weight.dtype == torch.bfloat16
    assert actual.dtype == torch.float32
    expected = torch.nn.functional.linear(
        hidden_states.float(),
        head.weight.float(),
    )
    torch.testing.assert_close(actual, expected)


def test_normalize_hyv4_config_populates_runtime_aliases() -> None:
    config = SimpleNamespace(
        routed_scaling_factor=2.827,
        n_routed_experts=256,
        moe_intermediate_size=2048,
        n_shared_experts=1,
        norm_topk_prob=True,
    )

    assert _normalize_hyv4_config(config) is config
    assert config.router_scaling_factor == 2.827
    assert config.num_experts == 256
    assert config.expert_hidden_dim == 2048
    assert config.num_shared_experts == 1
    assert config.route_norm is True


@pytest.mark.parametrize(
    ("checkpoint_name", "parameter_name"),
    [
        (
            "model.layers.1.mlp.gate.e_score_correction_bias",
            "model.layers.1.mlp.expert_bias",
        ),
        (
            "model.layers.1.mlp.router.gate.weight",
            "model.layers.1.mlp.gate.weight",
        ),
        (
            "model.layers.0.hc_attn_layer.hc_pre.hc_fn",
            "model.layers.0.hc_attn_layer.hc_pre.hc_fn.weight",
        ),
        (
            "model.hc_head.hc_head_fn",
            "model.hc_head.hc_head_fn.weight",
        ),
        (
            "model.hc_head.hc_head_fn.weight",
            "model.hc_head.hc_head_fn.weight",
        ),
    ],
)
def test_rewrite_hyv4_weight_name_is_exact_and_idempotent(
    checkpoint_name: str,
    parameter_name: str,
) -> None:
    assert _rewrite_hyv4_weight_name(checkpoint_name) == parameter_name
    assert _rewrite_hyv4_weight_name(parameter_name) == parameter_name


def test_load_weights_maps_router_correction_bias_before_unknown_bias_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # AutoWeightsLoader removes the outer ``model.`` prefix before invoking
    # HYV4Model.load_weights.
    parameter_name = "layers.15.mlp.expert_bias"
    checkpoint_name = "layers.15.mlp.gate.e_score_correction_bias"
    parameter = torch.nn.Parameter(torch.full((4,), float("nan")))
    loaded_weight = torch.tensor([0.25, -0.5, 0.75, -1.0])

    class MinimalModel:
        config = SimpleNamespace(
            tie_word_embeddings=False,
            num_experts=4,
            num_attention_heads=8,
        )

        @staticmethod
        def named_parameters():
            return [(parameter_name, parameter)]

        @staticmethod
        def get_expert_mapping():
            return []

    monkeypatch.setattr(
        hy_v4_model, "get_pp_missing_layer_names", lambda model: set()
    )
    monkeypatch.setattr(hy_v4_model, "compute_skip_topk_layers", lambda config: set())
    monkeypatch.setattr(
        hy_v4_model, "is_pp_missing_parameter", lambda name, model: False
    )
    monkeypatch.setattr(
        hy_v4_model, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(hy_v4_model, "get_tensor_model_parallel_rank", lambda: 0)

    loaded = HYV4Model.load_weights(
        MinimalModel(),
        [(checkpoint_name, loaded_weight)],
    )

    assert loaded == {parameter_name}
    torch.testing.assert_close(parameter, loaded_weight)


def test_outer_load_weights_checks_correction_biases_after_all_prefix_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter_name = "model.layers.15.mlp.expert_bias"
    checkpoint_name = "model.layers.15.mlp.gate.e_score_correction_bias"
    parameter = torch.nn.Parameter(torch.full((4,), float("nan")))
    loaded_weight = torch.tensor([0.25, -0.5, 0.75, -1.0])
    norm_name = "model.norm.weight"
    norm_parameter = torch.nn.Parameter(torch.full((4,), float("nan")))
    norm_weight = torch.tensor([1.0, 1.25, 1.5, 1.75])
    linear_name = "model.hc_fn.weight"
    linear_parameter = torch.nn.Parameter(torch.full((2, 2), float("nan")))
    linear_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    quant_projection_name = "model.q_proj.weight"
    quant_projection_parameter = torch.nn.Parameter(
        torch.full((2, 2), float("nan"))
    )
    quant_projection_weight = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    runtime_scale_name = "model.layers.15.self_attn.mla_attn.q_scale"
    runtime_scale_parameter = KVCacheScaleParameter()

    class CheckpointLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = linear_parameter

    class QuantizedProjection(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = quant_projection_parameter
            self.quant_method = SimpleNamespace(
                process_weights_after_loading=lambda module: None
            )

    class InnerModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.expert_bias = parameter
            self.norm_weight = norm_parameter
            self.hc_fn = CheckpointLinear()
            self.q_proj = QuantizedProjection()
            self.config = SimpleNamespace(
                tie_word_embeddings=False,
                num_experts=4,
                num_attention_heads=8,
            )

        def named_parameters(self, *args, **kwargs):
            del args, kwargs
            return iter(
                [
                    ("layers.15.mlp.expert_bias", self.expert_bias),
                        ("norm.weight", self.norm_weight),
                        ("hc_fn.weight", self.hc_fn.weight),
                        ("q_proj.weight", self.q_proj.weight),
                ]
            )

        @staticmethod
        def get_expert_mapping():
            return []

        def load_weights(self, weights):
            return HYV4Model.load_weights(self, weights)

    class OuterModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = InnerModel()
            self.config = SimpleNamespace(
                tie_word_embeddings=False,
                num_hidden_layers=78,
                num_nextn_predict_layers=1,
            )
            self.quant_config = None

        def named_parameters(self, *args, **kwargs):
            del args, kwargs
            return iter(
                [
                    (parameter_name, self.model.expert_bias),
                    (norm_name, self.model.norm_weight),
                    (linear_name, self.model.hc_fn.weight),
                    (quant_projection_name, self.model.q_proj.weight),
                    (runtime_scale_name, runtime_scale_parameter),
                ]
            )

    monkeypatch.setattr(
        hy_v4_model, "get_pp_missing_layer_names", lambda model: set()
    )
    monkeypatch.setattr(hy_v4_model, "compute_skip_topk_layers", lambda config: set())
    monkeypatch.setattr(
        hy_v4_model, "is_pp_missing_parameter", lambda name, model: False
    )
    monkeypatch.setattr(
        hy_v4_model, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(hy_v4_model, "get_tensor_model_parallel_rank", lambda: 0)

    model = OuterModel()
    loaded = hy_v4_model.HYV4ForCausalLM.load_weights(
        model,
        [
            (checkpoint_name, loaded_weight),
            (norm_name, norm_weight),
            (linear_name, linear_weight),
            (quant_projection_name, quant_projection_weight),
        ],
    )
    assert loaded == {
        parameter_name,
        norm_name,
        linear_name,
        quant_projection_name,
    }
    torch.testing.assert_close(parameter, loaded_weight)
    torch.testing.assert_close(norm_parameter, norm_weight)
    torch.testing.assert_close(linear_parameter, linear_weight)
    torch.testing.assert_close(
        quant_projection_parameter,
        quant_projection_weight,
    )

    with pytest.raises(RuntimeError, match=r"model\.layers\.15\.mlp\.expert_bias"):
        hy_v4_model.HYV4ForCausalLM.load_weights(model, [])

    with pytest.raises(RuntimeError, match=r"model\.norm\.weight"):
        hy_v4_model.HYV4ForCausalLM.load_weights(
            model,
            [(checkpoint_name, loaded_weight)],
        )

    with pytest.raises(RuntimeError, match=r"model\.hc_fn\.weight"):
        hy_v4_model.HYV4ForCausalLM.load_weights(
            model,
            [
                (checkpoint_name, loaded_weight),
                (norm_name, norm_weight),
                (quant_projection_name, quant_projection_weight),
            ],
        )

    with pytest.raises(RuntimeError, match=r"model\.q_proj\.weight"):
        hy_v4_model.HYV4ForCausalLM.load_weights(
            model,
            [
                (checkpoint_name, loaded_weight),
                (norm_name, norm_weight),
                (linear_name, linear_weight),
            ],
        )


def test_slice_sink_for_tp_uses_contiguous_attention_head_shards() -> None:
    sink = torch.arange(64, dtype=torch.float32)

    actual = _slice_sink_for_tp(sink, num_heads=64, tp_size=8, tp_rank=3)

    torch.testing.assert_close(actual, torch.arange(24, 32, dtype=torch.float32))


@pytest.mark.parametrize(
    ("num_heads", "tp_size", "tp_rank", "message"),
    [
        (63, 8, 0, "divisible"),
        (64, 8, 8, "rank"),
        (64, 0, 0, "positive"),
    ],
)
def test_slice_sink_for_tp_rejects_invalid_topology(
    num_heads: int,
    tp_size: int,
    tp_rank: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _slice_sink_for_tp(
            torch.arange(64),
            num_heads=num_heads,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )


def test_slice_sink_for_tp_rejects_checkpoint_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="64 attention heads"):
        _slice_sink_for_tp(
            torch.arange(63),
            num_heads=64,
            tp_size=8,
            tp_rank=0,
        )


def test_indexer_channel_fp8_dequantizes_each_output_row() -> None:
    weight = torch.tensor(
        [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]],
        dtype=torch.float8_e4m3fn,
    )
    scale = torch.tensor([[0.25], [0.5]], dtype=torch.float32)

    actual = _dequantize_indexer_channel_fp8(weight, scale)
    expected = weight.float().mul(scale).to(torch.bfloat16)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


def test_indexer_rank_one_fp8_scale_dequantizes_each_output_row() -> None:
    weight = torch.ones(2, 3, dtype=torch.float8_e4m3fn)
    scale = torch.tensor([0.5, 2.0], dtype=torch.float32)

    actual = _dequantize_indexer_channel_fp8(weight, scale)

    expected = torch.tensor(
        [[0.5, 0.5, 0.5], [2.0, 2.0, 2.0]],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(actual, expected)


def test_indexer_fp8_dequantizes_128_by_128_scale_blocks() -> None:
    weight = torch.ones(256, 256, dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]], dtype=torch.float32)

    actual = _dequantize_indexer_channel_fp8(weight, scale)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(
        actual[:128, :128],
        torch.ones(128, 128, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        actual[:128, 128:],
        torch.full((128, 128), 2.0, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        actual[128:, :128],
        torch.full((128, 128), 4.0, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        actual[128:, 128:],
        torch.full((128, 128), 8.0, dtype=torch.bfloat16),
    )


def test_indexer_fp8_reinterprets_ue8m0_128_by_128_scale_blocks() -> None:
    weight = torch.ones(256, 256, dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[127, 128], [126, 129]], dtype=torch.uint8)

    actual = _dequantize_indexer_channel_fp8(weight, scale)

    expected = torch.empty(256, 256, dtype=torch.bfloat16)
    expected[:128, :128] = 1.0
    expected[:128, 128:] = 2.0
    expected[128:, :128] = 0.5
    expected[128:, 128:] = 4.0
    torch.testing.assert_close(actual, expected)


def test_indexer_mxfp8_dequantizes_each_32_element_block() -> None:
    weight = torch.ones(2, 64, dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[127, 128], [126, 129]], dtype=torch.uint8)

    actual = _dequantize_indexer_channel_fp8(weight, scale)
    decoded_scale = torch.exp2(scale.float() - 127.0)
    expected = (
        weight.float().view(2, 2, 32) * decoded_scale.unsqueeze(-1)
    ).view_as(weight).to(torch.bfloat16)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


def test_indexer_fp8_rejects_non_divisible_block_scale() -> None:
    weight = torch.ones(2, 4, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="divide"):
        _dequantize_indexer_channel_fp8(weight, torch.ones(1, 3))


@pytest.mark.parametrize(
    ("projection", "shard_id"),
    [("wk", 0), ("weights_proj", 1)],
)
def test_fp8_indexer_projection_pair_loads_the_correct_fused_shard(
    projection: str,
    shard_id: int,
) -> None:
    calls: list[tuple[torch.Tensor, int]] = []

    class FakeParameter:
        @staticmethod
        def weight_loader(param, weight, loaded_shard_id) -> None:
            del param
            calls.append((weight, loaded_shard_id))

    prefix = "layers.45.self_attn.indexer"
    fused_name = f"{prefix}.wk_weights_proj.weight"
    params = {fused_name: FakeParameter()}
    pending: dict[str, dict[str, torch.Tensor]] = {}
    loaded: set[str] = set()
    weight = torch.ones(2, 4, dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[0.25], [0.5]])

    assert _try_load_fp8_indexer_projection(
        f"{prefix}.{projection}.weight_scale",
        scale,
        pending,
        params,
        loaded,
        set(),
    )
    assert _try_load_fp8_indexer_projection(
        f"{prefix}.{projection}.weight",
        weight,
        pending,
        params,
        loaded,
        set(),
    )

    assert pending == {}
    assert loaded == {fused_name}
    assert len(calls) == 1
    actual_weight, actual_shard_id = calls[0]
    assert actual_shard_id == shard_id
    torch.testing.assert_close(
        actual_weight,
        (weight.float() * scale).to(torch.bfloat16),
    )


@pytest.mark.parametrize(
    "checkpoint_prefix",
    ["layers.10.mlp.gate", "layers.10.mlp.router.gate"],
)
def test_fp8_router_gate_pair_dequantizes_to_fp32(
    checkpoint_prefix: str,
) -> None:
    loaded_tensors: list[torch.Tensor] = []

    class FakeParameter:
        @staticmethod
        def weight_loader(param, weight) -> None:
            del param
            loaded_tensors.append(weight)

    parameter_name = "layers.10.mlp.gate.weight"
    params = {parameter_name: FakeParameter()}
    pending: dict[str, dict[str, torch.Tensor]] = {}
    loaded: set[str] = set()
    weight = torch.ones(2, 4, dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[0.25], [0.5]])

    assert _try_load_fp8_router_gate(
        f"{checkpoint_prefix}.weight_scale",
        scale,
        pending,
        params,
        loaded,
        set(),
    )
    assert _try_load_fp8_router_gate(
        f"{checkpoint_prefix}.weight",
        weight,
        pending,
        params,
        loaded,
        set(),
    )

    assert pending == {}
    assert loaded == {parameter_name}
    assert len(loaded_tensors) == 1
    assert loaded_tensors[0].dtype == torch.float32
    torch.testing.assert_close(loaded_tensors[0], weight.float() * scale)
