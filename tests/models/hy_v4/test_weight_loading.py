# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.quantization.kv_cache import KVCacheScaleParameter

from vllm_hcu.models.hy_v4 import model as hy_v4_model
from vllm_hcu.models.hy_v4.model import (
    HYV4ForCausalLM, HYV4Model, _normalize_hyv4_config,
    _rewrite_hyv4_weight_name, _slice_sink_for_tp,
)


def test_target_auto_loader_uses_ignore_unexpected_prefixes():
    from vllm.model_executor.models.utils import AutoWeightsLoader

    parameters = inspect.signature(AutoWeightsLoader).parameters
    assert "ignore_unexpected_prefixes" in parameters
    assert "skip_prefixes" not in parameters

def _add_fake_moe_metadata(model: torch.nn.Module) -> None:
    model.expert_weights = []
    model.num_moe_layers = 1
    model.num_expert_groups = 1
    model.num_logical_experts = 4
    model.num_physical_experts = 4
    model.num_local_physical_experts = 4
    model.num_routed_experts = 4
    model.num_shared_experts = 1
    model.num_redundant_experts = 0
    model.moe_layers = []


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
            _add_fake_moe_metadata(self)

    class FakeLMHead(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            captured.update(kwargs)

    monkeypatch.setattr(hy_v4_model, "HYV4Model", FakeInnerModel)
    monkeypatch.setattr(hy_v4_model, "ParallelLMHead", FakeLMHead)
    monkeypatch.setattr(hy_v4_model, "LogitsProcessor", lambda *args: object())
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


def test_non_modelopt_quant_config_is_forwarded_to_lm_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeInnerModel(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.make_empty_intermediate_tensors = object()
            _add_fake_moe_metadata(self)

    class FakeLMHead(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            captured.update(kwargs)

    monkeypatch.setattr(hy_v4_model, "HYV4Model", FakeInnerModel)
    monkeypatch.setattr(hy_v4_model, "ParallelLMHead", FakeLMHead)
    monkeypatch.setattr(hy_v4_model, "LogitsProcessor", lambda *args: object())
    monkeypatch.setattr(
        hy_v4_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )

    quant_config = SimpleNamespace()
    config = SimpleNamespace(
        vocab_size=64,
        hidden_size=32,
        tie_word_embeddings=False,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=quant_config,
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0)
        ),
    )

    HYV4ForCausalLM(vllm_config=vllm_config)

    assert captured["quant_config"] is quant_config


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




@pytest.fixture
def checkpoint_model(monkeypatch):
    monkeypatch.setattr(hy_v4_model, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(hy_v4_model, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(hy_v4_model, "get_pp_missing_layer_names", lambda model: set())
    monkeypatch.setattr(hy_v4_model, "is_pp_missing_parameter", lambda name, model: False)
    inner = object.__new__(HYV4Model)
    torch.nn.Module.__init__(inner)
    inner.config = SimpleNamespace(tie_word_embeddings=False, num_experts=0,
                                   num_attention_heads=4)
    inner.num_redundant_experts = 0
    inner.quant_config = None
    inner.norm = torch.nn.Linear(2, 2, bias=False)
    inner.hc_fn = torch.nn.Linear(2, 2, bias=False)
    inner.get_expert_mapping = lambda: []
    outer = object.__new__(HYV4ForCausalLM)
    torch.nn.Module.__init__(outer)
    outer.config = SimpleNamespace(tie_word_embeddings=False,
                                   num_hidden_layers=2, num_nextn_predict_layers=1)
    outer.quant_config = None
    outer.model = inner
    outer.lm_head = torch.nn.Linear(2, 2, bias=False)
    return outer


@pytest.fixture
def hyv4_fp8_checkpoint(checkpoint_model):
    """Small real parameter tree with the target's three quantization owners."""
    model = checkpoint_model
    del model.model.norm
    del model.model.hc_fn
    layer = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([layer])
    layer.self_attn = torch.nn.Module()
    indexer = layer.self_attn.indexer = torch.nn.Module()
    indexer.wk_weights_proj = torch.nn.Linear(64, 6, bias=False, dtype=torch.bfloat16)

    def load_merged(param, value, shard_id):
        start, count = [(0, 4), (4, 2)][shard_id]
        assert value.shape == (count, 64)
        with torch.no_grad():
            param[start:start + count].copy_(value)

    indexer.wk_weights_proj.weight.weight_loader = load_merged
    indexer.wq_b = torch.nn.Module()
    indexer.wq_b.weight = torch.nn.Parameter(
        torch.ones(4, 64).to(torch.float8_e4m3fn), requires_grad=False)
    indexer.wq_b.weight_scale = torch.nn.Parameter(torch.ones(4, 1))
    layer.mlp = torch.nn.Module()
    layer.mlp.gate = torch.nn.Linear(64, 4, bias=False, dtype=torch.float32)
    layer.mlp.expert_bias = torch.nn.Parameter(torch.zeros(4))
    return model


def _hyv4_fp8_weights(model, scale, *, scale_name="weight_scale"):
    prefix = "model.layers.0."
    # Deliberately interleave the outer LM-head group between a weight and
    # its scale: upstream AutoWeightsLoader invokes the inner loader twice.
    return [
        (prefix + "self_attn.indexer.wk.weight", torch.ones(4, 64).to(torch.float8_e4m3fn)),
        ("lm_head.weight", torch.ones_like(model.lm_head.weight)),
        (prefix + "mlp.router.gate.weight", torch.full((4, 64), 2.0).to(torch.float8_e4m3fn)),
        (prefix + "self_attn.indexer.wk." + scale_name, scale),
        (prefix + "mlp.router.gate." + scale_name, scale),
        (prefix + "self_attn.indexer.weights_proj.weight_scale", torch.full((2, 1), 0.5)),
        (prefix + "self_attn.indexer.weights_proj.weight", torch.full((2, 64), 6.0).to(torch.float8_e4m3fn)),
        (prefix + "self_attn.indexer.wq_b.weight", torch.full((4, 64), 4.0).to(torch.float8_e4m3fn)),
        (prefix + "self_attn.indexer.wq_b.weight_scale", torch.full((4, 1), 0.25)),
        (prefix + "mlp.gate.e_score_correction_bias", torch.arange(4.0)),
        ("model.mtp_layers.0.mlp.router.gate.weight_scale", torch.tensor(float("nan"))),
    ]


@pytest.mark.parametrize("layout", ["channel", "channel_column", "block", "raw_ue8m0", "mxfp8", "typed_ue8m0"])
@pytest.mark.parametrize("scale_name", ["weight_scale", "weight_scale_inv"])
def test_hyv4_fp8_loads_local_indexer_router_scales_without_touching_target(
    hyv4_fp8_checkpoint, layout, scale_name,
):
    if layout.startswith("channel"):
        scale = torch.tensor([0.5, 1.0, 2.0, 4.0])
        expected = scale[:, None].expand(4, 64)
        if layout == "channel_column":
            scale = scale[:, None]
    else:
        scale = torch.tensor([[0.5, 1.0], [2.0, 4.0]])
        expected = torch.tensor([[0.5] * 32 + [1.0] * 32] * 2
                                + [[2.0] * 32 + [4.0] * 32] * 2)
        if layout in ("raw_ue8m0", "typed_ue8m0", "mxfp8"):
            scale = torch.tensor([[126, 127], [128, 129]], dtype=torch.uint8)
        if layout == "mxfp8":
            scale = scale.repeat_interleave(2, dim=0)
        elif layout == "typed_ue8m0":
            scale = scale.view(torch.float8_e8m0fnu)
    model = hyv4_fp8_checkpoint
    loaded = model.load_weights(iter(_hyv4_fp8_weights(model, scale, scale_name=scale_name)))
    layer = model.model.layers[0]
    actual = layer.self_attn.indexer.wk_weights_proj.weight
    torch.testing.assert_close(actual[:4].float(), expected)
    torch.testing.assert_close(actual[4:].float(), torch.full((2, 64), 3.0))
    torch.testing.assert_close(layer.mlp.gate.weight, expected * 2)
    torch.testing.assert_close(layer.mlp.expert_bias, torch.arange(4.0))
    assert torch.isfinite(actual).all() and torch.isfinite(layer.mlp.gate.weight).all()
    assert layer.self_attn.indexer.wq_b.weight.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(layer.self_attn.indexer.wq_b.weight.float(), torch.full((4, 64), 4.0))
    torch.testing.assert_close(layer.self_attn.indexer.wq_b.weight_scale, torch.full((4, 1), 0.25))
    assert loaded == set(dict(model.named_parameters()))
    assert not hasattr(model.model, "_checkpoint_accounting")


@pytest.mark.parametrize("missing", ["weight", "weight_scale"])
def test_hyv4_fp8_incomplete_pair_fails_and_does_not_survive_reload(hyv4_fp8_checkpoint, missing):
    model = hyv4_fp8_checkpoint
    weights = _hyv4_fp8_weights(model, torch.ones(4, 1))
    missing_name = "model.layers.0.self_attn.indexer.wk." + missing
    with pytest.raises(RuntimeError, match="Incomplete HY V4 FP8"):
        model.load_weights((name, value) for name, value in weights if name != missing_name)
    assert not hasattr(model.model, "_checkpoint_accounting")
    model.load_weights(iter(weights))


def test_hyv4_fp8_duplicate_scale_alias_fails(hyv4_fp8_checkpoint):
    model = hyv4_fp8_checkpoint
    weights = _hyv4_fp8_weights(model, torch.ones(4, 1))
    weights.insert(5, ("model.layers.0.self_attn.indexer.wk.weight_scale_inv", torch.ones(4, 1)))
    with pytest.raises(RuntimeError, match="Duplicate HY V4"):
        model.load_weights(iter(weights))


def test_hyv4_fp8_scales_before_weights_preserves_fp32_router(hyv4_fp8_checkpoint):
    model = hyv4_fp8_checkpoint
    weights = _hyv4_fp8_weights(model, torch.full((4, 1), 1.0001))
    model.load_weights(reversed(weights))
    # The router must not round via BF16 when decoding its FP32 parameter.
    torch.testing.assert_close(model.model.layers[0].mlp.gate.weight,
                               torch.full((4, 64), 2.0002), rtol=0, atol=0)


@pytest.mark.parametrize("bad_weight", [torch.full((4, 64), float("nan")),
                                      torch.full((4, 64), 448.0)])
def test_hyv4_fp8_rejects_nonfinite_dequantized_weight(hyv4_fp8_checkpoint, bad_weight):
    model = hyv4_fp8_checkpoint
    weights = _hyv4_fp8_weights(model, torch.full((4, 1), 1e38))
    weights[0] = (weights[0][0], bad_weight.to(torch.float8_e4m3fn))
    with pytest.raises(ValueError, match="HY V4 FP8 dequantized weight must be finite"):
        model.load_weights(iter(weights))


@pytest.mark.parametrize("scale", [torch.ones(3), torch.ones(3, 2), torch.ones(1, 1, 1),
                                 torch.full((4, 1), float("nan")),
                                 torch.full((4, 1), -1.0),
                                 torch.full((4, 1), 255, dtype=torch.uint8)])
def test_hyv4_fp8_rejects_malformed_or_nonfinite_scale(hyv4_fp8_checkpoint, scale):
    with pytest.raises(ValueError, match="HY V4 FP8"):
        hyv4_fp8_checkpoint.load_weights(iter(_hyv4_fp8_weights(hyv4_fp8_checkpoint, scale)))


def test_checkpoint_exact_accounting_across_interleaved_prefixes(checkpoint_model):
    weights = [
        ("model.norm.weight", torch.full((2, 2), 2.0)),
        ("lm_head.weight", torch.full((2, 2), 3.0)),
        ("model.hc_fn", torch.full((2, 2), 4.0)),
    ]
    loaded = checkpoint_model.load_weights(iter(weights))
    assert loaded == {"model.norm.weight", "model.hc_fn.weight", "lm_head.weight"}
    assert torch.equal(checkpoint_model.model.hc_fn.weight, torch.full((2, 2), 4.0))


def test_packed_w4a8_is_rejected_before_reading_weights(checkpoint_model):
    checkpoint_model.quant_config = SimpleNamespace(
        checkpoint_format="hy4-w4a8-custom-v1"
    )

    def unread_weights():
        pytest.fail("packed W4A8 checkpoint was read before rejection")
        yield "unused", torch.empty(0)

    with pytest.raises(NotImplementedError, match="packed W4A8"):
        checkpoint_model.load_weights(unread_weights())


@pytest.mark.parametrize("missing", ["model.norm.weight", "model.hc_fn.weight", "lm_head.weight"])
def test_checkpoint_missing_parameter_fails(checkpoint_model, missing):
    weights = [(name, torch.zeros_like(param))
               for name, param in checkpoint_model.named_parameters() if name != missing]
    with pytest.raises(RuntimeError, match="Missing HY V4 checkpoint"):
        checkpoint_model.load_weights(iter(weights))


@pytest.mark.parametrize("unexpected", ["model.unknown.weight", "model.unknown.bias", "unknown.bias"])
def test_checkpoint_unexpected_parameter_fails(checkpoint_model, unexpected):
    weights = [(name, torch.zeros_like(param)) for name, param in checkpoint_model.named_parameters()]
    weights.append((unexpected, torch.ones(2)))
    with pytest.raises((RuntimeError, ValueError), match="Unknown|no module|Unexpected"):
        checkpoint_model.load_weights(iter(weights))


@pytest.mark.parametrize("duplicate", ["model.hc_fn", "model.hc_fn.weight"])
def test_checkpoint_duplicate_alias_fails(checkpoint_model, duplicate):
    weights = [(name, torch.zeros_like(param)) for name, param in checkpoint_model.named_parameters()]
    weights.append((duplicate, torch.ones(2, 2)))
    with pytest.raises(RuntimeError, match="Duplicate HY V4 checkpoint"):
        checkpoint_model.load_weights(iter(weights))


def _add_packed_projection(checkpoint_model):
    projection = torch.nn.Linear(2, 4, bias=False)
    def loader(param, weight, shard=None):
        with torch.no_grad():
            if shard is None:
                param.copy_(weight)
            else:
                param[2 * shard:2 * shard + 2].copy_(weight)
    projection.weight.weight_loader = loader
    checkpoint_model.model.mlp = torch.nn.Module()
    checkpoint_model.model.mlp.gate_up_proj = projection
    return projection


def test_checkpoint_missing_packed_half_fails(checkpoint_model):
    _add_packed_projection(checkpoint_model)
    weights = [(name, torch.zeros_like(param)) for name, param in checkpoint_model.named_parameters()
               if name != "model.mlp.gate_up_proj.weight"]
    weights.append(("model.mlp.gate_proj.weight", torch.ones(2, 2)))
    with pytest.raises(RuntimeError, match="Missing HY V4 checkpoint"):
        checkpoint_model.load_weights(iter(weights))


def test_checkpoint_packed_halves_across_prefix_groups(checkpoint_model):
    projection = _add_packed_projection(checkpoint_model)
    weights = [(name, torch.zeros_like(param)) for name, param in checkpoint_model.named_parameters()
               if name not in {"model.mlp.gate_up_proj.weight", "lm_head.weight"}]
    weights.extend([
        ("model.mlp.gate_proj.weight", torch.ones(2, 2)),
        ("lm_head.weight", torch.zeros(2, 2)),
        ("model.mlp.up_proj.weight", torch.full((2, 2), 2.0)),
    ])
    loaded = checkpoint_model.load_weights(iter(weights))
    assert loaded == set(dict(checkpoint_model.named_parameters()))
    torch.testing.assert_close(projection.weight, torch.tensor([[1.,1.],[1.,1.],[2.,2.],[2.,2.]]))


def test_checkpoint_duplicate_packed_and_split_fails(checkpoint_model):
    _add_packed_projection(checkpoint_model)
    weights = [(name, torch.zeros_like(param)) for name, param in checkpoint_model.named_parameters()]
    weights.append(("model.mlp.gate_proj.weight", torch.ones(2, 2)))
    with pytest.raises(RuntimeError, match="Duplicate HY V4 checkpoint"):
        checkpoint_model.load_weights(iter(weights))


def _add_experts(checkpoint_model):
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    model = checkpoint_model.model
    model.config.num_experts = 2
    model.mlp = torch.nn.Module()
    model.mlp.experts = torch.nn.Module()
    owner = torch.nn.Module()
    model.mlp.experts.routed_experts = owner
    owner.w13_weight = torch.nn.Parameter(torch.empty(2, 4, 2))
    owner.w2_weight = torch.nn.Parameter(torch.empty(2, 2, 2))
    def loader(param, weight, name, shard_id, expert_id, return_success=False):
        with torch.no_grad():
            if shard_id == "w2":
                param[expert_id].copy_(weight)
            else:
                shard = 0 if shard_id == "w1" else 1
                param[expert_id, shard * 2:shard * 2 + 2].copy_(weight)
        return True
    owner.w13_weight.weight_loader = loader
    owner.w2_weight.weight_loader = loader
    model.get_expert_mapping = lambda: RoutedExperts.build_expert_params_mapping(
        "gate_proj", "down_proj", "up_proj", 2)
    return owner


def _expert_checkpoint(checkpoint_model):
    weights = [(name, torch.zeros_like(param)) for name, param in checkpoint_model.named_parameters()
               if ".experts." not in name]
    weights += [(f"model.mlp.experts.{expert}.{projection}.weight", torch.full((2, 2), float(expert + 1)))
                for expert in range(2) for projection in ("gate_proj", "up_proj", "down_proj")]
    return weights


def test_checkpoint_counts_every_expert_shard(checkpoint_model):
    owner = _add_experts(checkpoint_model)
    loaded = checkpoint_model.load_weights(iter(_expert_checkpoint(checkpoint_model)))
    assert loaded == set(dict(checkpoint_model.named_parameters()))
    torch.testing.assert_close(owner.w13_weight[0], torch.ones(4, 2))
    torch.testing.assert_close(owner.w13_weight[1], torch.full((4, 2), 2.0))


def test_checkpoint_missing_expert_shard_fails(checkpoint_model):
    _add_experts(checkpoint_model)
    weights = _expert_checkpoint(checkpoint_model)
    weights = [(name, weight) for name, weight in weights if name != "model.mlp.experts.1.up_proj.weight"]
    with pytest.raises(RuntimeError, match="Missing HY V4 checkpoint"):
        checkpoint_model.load_weights(iter(weights))


def test_checkpoint_fused_and_split_expert_duplicate_fails(checkpoint_model):
    _add_experts(checkpoint_model)
    weights = _expert_checkpoint(checkpoint_model)
    weights.append(("model.mlp.experts.gate_up_proj", torch.ones(2, 4, 2)))
    with pytest.raises(RuntimeError, match="Duplicate HY V4 checkpoint"):
        checkpoint_model.load_weights(iter(weights))


def test_checkpoint_extra_fused_expert_fails(checkpoint_model):
    _add_experts(checkpoint_model)
    weights = [(name, weight) for name, weight in _expert_checkpoint(checkpoint_model)
               if ".gate_proj." not in name and ".up_proj." not in name]
    weights.append(("model.mlp.experts.gate_up_proj", torch.ones(3, 4, 2)))
    with pytest.raises(ValueError, match="checkpoint experts"):
        checkpoint_model.load_weights(iter(weights))


@pytest.mark.parametrize("method,args", [
    ("set_eplb_state", (torch.empty(0), torch.empty(0), torch.empty(0))),
    ("update_physical_experts_metadata", (2, 2)),
])
def test_eplb_state_interface_remains_inert_until_task_7(checkpoint_model, method, args):
    model = checkpoint_model.model
    model.moe_layers = []
    model.expert_weights = []
    model.layers = []
    model.num_local_physical_experts = 2
    model.num_logical_experts = 2
    with pytest.raises(NotImplementedError, match="Task 7"):
        getattr(model, method)(*args)


def test_checkpoint_sink_loads_local_tp_slice(checkpoint_model, monkeypatch):
    monkeypatch.setattr(hy_v4_model, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(hy_v4_model, "get_tensor_model_parallel_rank", lambda: 1)
    attention = torch.nn.Module()
    attention.learnable_sink_param = torch.nn.Parameter(torch.empty(2))
    checkpoint_model.model.self_attn = attention
    weights = [(name, torch.zeros_like(param)) for name, param in checkpoint_model.named_parameters()
               if name != "model.self_attn.learnable_sink_param"]
    weights.append(("model.self_attn.learnable_sink_param", torch.tensor([1., 2., 3., 4.])))
    assert checkpoint_model.load_weights(iter(weights)) == set(dict(checkpoint_model.named_parameters()))
    torch.testing.assert_close(attention.learnable_sink_param, torch.tensor([3., 4.]))
