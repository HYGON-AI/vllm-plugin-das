# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest
from torch import nn

from vllm_hcu.models.hy_v4 import mtp as hy_v4_mtp


def test_shared_head_uses_backbone_lm_head_quant_prefix(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeHead(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            captured["args"] = args
            captured.update(kwargs)

    monkeypatch.setattr(hy_v4_mtp, "ParallelLMHead", FakeHead)
    quant_config = SimpleNamespace(is_layer_excluded=lambda prefix: False)
    config = SimpleNamespace(vocab_size=64, hidden_size=16)

    head = hy_v4_mtp.HYV4SharedHead(config, quant_config)

    assert isinstance(head.head, FakeHead)
    assert captured["quant_config"] is quant_config
    assert captured["prefix"] == "lm_head"


def test_shared_head_drops_excluded_quant_config_for_vocab_layout(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeHead(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            captured["args"] = args
            captured.update(kwargs)

    quant_config = SimpleNamespace(
        is_layer_excluded=lambda prefix: prefix == "lm_head"
    )
    config = SimpleNamespace(vocab_size=64, hidden_size=16)
    monkeypatch.setattr(hy_v4_mtp, "ParallelLMHead", FakeHead)

    hy_v4_mtp.HYV4SharedHead(config, quant_config)

    assert captured["prefix"] == "lm_head"
    assert captured["quant_config"] is None


def test_predictor_masks_position_zero_embedding_before_mtp_layer() -> None:
    captured: dict[str, torch.Tensor] = {}

    class CaptureLayer(nn.Module):
        def forward(
            self,
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
        ):
            captured["inputs_embeds"] = inputs_embeds
            return previous_hidden_states

    predictor = object.__new__(hy_v4_mtp.HYV4MultiTokenPredictor)
    nn.Module.__init__(predictor)
    predictor.mtp_start_layer_idx = 78
    predictor.num_mtp_layers = 1
    predictor.spec_step_idx = 0
    predictor.layers = nn.ModuleDict({"78": CaptureLayer()})
    input_ids = torch.tensor([1, 2])
    positions = torch.tensor([0, 4])
    hidden = torch.ones((2, 3), dtype=torch.bfloat16)
    embeds = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.bfloat16,
    )

    actual = predictor(input_ids, positions, hidden, embeds)

    torch.testing.assert_close(actual, hidden)
    torch.testing.assert_close(
        captured["inputs_embeds"],
        torch.tensor(
            [[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]],
            dtype=torch.bfloat16,
        ),
    )


def test_mtp_compute_logits_keeps_projection_input_dtype() -> None:
    captured: dict[str, torch.Tensor] = {}

    class SharedHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(4, 8, bias=False, dtype=torch.float32)

        def forward(self, hidden_states):
            return hidden_states

    predictor = object.__new__(hy_v4_mtp.HYV4MultiTokenPredictor)
    nn.Module.__init__(predictor)
    predictor.mtp_start_layer_idx = 78
    predictor.num_mtp_layers = 1
    predictor.spec_step_idx = 0
    predictor.layers = nn.ModuleDict(
        {"78": nn.ModuleDict({"shared_head": SharedHead()})}
    )

    def logits_processor(lm_head, projection_input):
        captured["projection_input"] = projection_input
        return projection_input.float()

    predictor.logits_processor = logits_processor
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)

    actual = predictor.compute_logits(hidden_states)

    assert captured["projection_input"] is hidden_states
    assert actual.dtype == torch.float32


def test_sparse_mtp_forward_requires_shared_target_topk_buffer() -> None:
    mtp = object.__new__(hy_v4_mtp.HYV4MTP)
    nn.Module.__init__(mtp)
    mtp.model = SimpleNamespace(
        requires_topk_indices_buffer=True,
        topk_indices_buffer=None,
    )

    with pytest.raises(RuntimeError, match="target model's top-k"):
        mtp(
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0]),
            hidden_states=torch.ones((1, 2)),
        )


def test_mtp_layer_config_extends_backbone_lists_and_disables_ihc() -> None:
    config = SimpleNamespace(
        enable_ihc=True,
        layer_types=["full_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse"],
    )

    actual = hy_v4_mtp._make_mtp_layer_config(config, layer_idx=2)

    assert actual is not config
    assert actual.enable_ihc is False
    assert actual.layer_types == [
        "full_attention",
        "deepseek_sparse_attention",
        "deepseek_sparse_attention",
    ]
    assert actual.mlp_layer_types == ["dense", "sparse", "sparse"]
    assert config.enable_ihc is True
    assert len(config.layer_types) == 2


def test_modelopt_mtp_exclusions_are_copied_remapped_and_keep_wildcards() -> None:
    quant_config = SimpleNamespace(
        exclude_modules=[
            "lm_head",
            "model.mtp_layers.0.eh_proj",
            "model.mtp_layers.0.self_attn.linear_gate*",
        ]
    )

    actual = hy_v4_mtp._remap_mtp_quant_exclusions(
        quant_config,
        mtp_start_layer_idx=78,
        num_mtp_layers=1,
    )

    assert actual is not quant_config
    assert quant_config.exclude_modules == [
        "lm_head",
        "model.mtp_layers.0.eh_proj",
        "model.mtp_layers.0.self_attn.linear_gate*",
    ]
    assert actual.exclude_modules == [
        "lm_head",
        "model.mtp_layers.0.eh_proj",
        "model.mtp_layers.0.self_attn.linear_gate*",
        "model.layers.78.eh_proj",
        "model.layers.78.self_attn.linear_gate*",
    ]


def test_mtp_checkpoint_names_map_wrapper_and_decoder_parameters() -> None:
    assert hy_v4_mtp._rewrite_mtp_weight_name(
        "model.mtp_layers.0.eh_proj.weight", 78
    ) == "model.layers.78.eh_proj.weight"
    assert hy_v4_mtp._rewrite_mtp_weight_name(
        "model.mtp_layers.0.self_attn.q_a_proj.weight", 78
    ) == "model.layers.78.mtp_block.self_attn.q_a_proj.weight"
    assert hy_v4_mtp._rewrite_mtp_weight_name(
        "model.mtp_layers.0.mlp.gate.e_score_correction_bias", 78
    ) == "model.layers.78.mtp_block.mlp.expert_bias"
    assert hy_v4_mtp._rewrite_mtp_weight_name(
        "model.layers.77.self_attn.q_a_proj.weight", 78
    ) is None


def test_fused_expert_scale_resolves_to_scale_parameter_not_weight() -> None:
    weight_name = "model.layers.78.mtp_block.mlp.experts.routed_experts.w13_weight"
    scale_name = weight_name + "_scale"
    params = {weight_name: object(), scale_name: object()}

    actual = hy_v4_mtp._resolve_fused_expert_param(
        weight_name,
        "_scale",
        params,
    )

    assert actual == scale_name


def test_blockwise_fp8_mtp_expert_scale_normalizes_legacy_name_and_ue8m0_bits(
) -> None:
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
    )
    quant_config.is_scale_e8m0 = True
    raw_scale = torch.tensor([0, 127, 128, 255], dtype=torch.uint8)

    name, actual = hy_v4_mtp._prepare_mtp_fp8_expert_scale(
        quant_config,
        "model.layers.78.mtp_block.mlp.experts.gate_up_proj.scale",
        raw_scale,
    )

    assert name.endswith("gate_up_proj.weight_scale_inv")
    assert actual.dtype == torch.float8_e8m0fnu
    torch.testing.assert_close(actual.view(torch.uint8), raw_scale)


def test_modelopt_mtp_expert_scale_keeps_name_and_dtype() -> None:
    quant_config = SimpleNamespace(
        weight_block_size=[128, 128],
        is_scale_e8m0=True,
    )
    raw_scale = torch.tensor([0, 127, 128, 255], dtype=torch.uint8)
    checkpoint_name = (
        "model.layers.78.mtp_block.mlp.experts.gate_up_proj.scale"
    )

    name, actual = hy_v4_mtp._prepare_mtp_fp8_expert_scale(
        quant_config,
        checkpoint_name,
        raw_scale,
    )

    assert name == checkpoint_name
    assert actual is raw_scale


def test_mtp_fp8_quant_config_preserves_static_activation_without_blocks() -> None:
    config = SimpleNamespace(
        mtp_quant_algo="FP8",
        quantization_config={"activation_scheme": "static"},
    )

    actual = hy_v4_mtp._create_mtp_quant_config(config)

    assert actual.activation_scheme == "static"


def test_mtp_blockwise_fp8_quant_config_forces_dynamic_activation() -> None:
    config = SimpleNamespace(
        mtp_quant_algo="FP8",
        quantization_config={
            "activation_scheme": "static",
            "weight_block_size": [128, 128],
        },
    )

    actual = hy_v4_mtp._create_mtp_quant_config(config)

    assert actual.activation_scheme == "dynamic"


def test_fused_expert_scale_loader_targets_scale_parameter() -> None:
    weight_name = "model.layers.78.mtp_block.mlp.experts.routed_experts.w13_weight"
    scale_name = weight_name + "_scale"
    calls: list[tuple[torch.Tensor, str, int]] = []
    scale_param = nn.Parameter(torch.empty(1))

    def weight_loader(
        param,
        loaded_weight,
        name,
        shard_id,
        expert_id,
        return_success,
    ):
        assert param is scale_param
        assert name == scale_name
        assert return_success is True
        calls.append((loaded_weight.clone(), shard_id, expert_id))
        return True

    scale_param.weight_loader = weight_loader
    mtp = object.__new__(hy_v4_mtp.HYV4MTP)
    nn.Module.__init__(mtp)
    loaded_params: set[str] = set()
    checkpoint_scale = torch.arange(16, dtype=torch.uint8).view(2, 4, 2)

    consumed = mtp._load_expert_weight(
        "model.layers.78.mtp_block.mlp.experts.gate_up_proj_scale",
        checkpoint_scale,
        {scale_name: scale_param},
        loaded_params,
        [],
        {("model.layers.78.mtp_block.mlp", "w13_weight"): weight_name},
        num_experts=2,
    )

    assert consumed is True
    assert loaded_params == {scale_name}
    assert [(shard, expert) for _, shard, expert in calls] == [
        ("w1", 0),
        ("w1", 1),
        ("w3", 0),
        ("w3", 1),
    ]
    torch.testing.assert_close(calls[0][0], checkpoint_scale[0, :2])
    torch.testing.assert_close(calls[2][0], checkpoint_scale[0, 2:])


def test_mtp_load_weights_rewrites_wrapper_weight_and_is_strict(monkeypatch) -> None:
    parameter_name = "model.layers.78.eh_proj.weight"
    parameter = nn.Parameter(torch.full((2, 4), float("nan")))

    def weight_loader(param, loaded_weight):
        with torch.no_grad():
            param.copy_(loaded_weight)

    parameter.weight_loader = weight_loader

    class MinimalMTP(hy_v4_mtp.HYV4MTP):
        def named_parameters(self, *args, **kwargs):
            del args, kwargs
            return iter([(parameter_name, parameter)])

    mtp = object.__new__(MinimalMTP)
    nn.Module.__init__(mtp)
    mtp.config = SimpleNamespace(
        num_hidden_layers=78,
        n_routed_experts=0,
        num_attention_heads=8,
    )
    mtp.quant_config = None
    loaded_weight = torch.arange(8, dtype=torch.float32).view(2, 4)
    monkeypatch.setattr(hy_v4_mtp, "fused_moe_make_expert_params_mapping", lambda *a, **k: [])
    monkeypatch.setattr(hy_v4_mtp, "get_pp_missing_layer_names", lambda model: set())
    monkeypatch.setattr(hy_v4_mtp, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(hy_v4_mtp, "get_tensor_model_parallel_rank", lambda: 0)

    loaded = mtp.load_weights(
        [("model.mtp_layers.0.eh_proj.weight", loaded_weight)]
    )

    assert loaded == {parameter_name}
    torch.testing.assert_close(parameter, loaded_weight)
    with pytest.raises(RuntimeError, match=r"model\.layers\.78\.eh_proj\.weight"):
        mtp.load_weights([])


def test_set_topk_buffer_updates_every_sparse_draft_consumer() -> None:
    old_buffer = torch.zeros((2, 2), dtype=torch.int32)
    new_buffer = torch.ones((3, 2), dtype=torch.int32)
    indexer_op = SimpleNamespace(topk_indices_buffer=old_buffer)
    indexer = SimpleNamespace(
        topk_indices_buffer=old_buffer,
        indexer_op=indexer_op,
    )
    attention_impl = SimpleNamespace(topk_indices_buffer=old_buffer)
    self_attn = SimpleNamespace(
        is_sparse=True,
        topk_indices_buffer=old_buffer,
        indexer=indexer,
        mla_attn=SimpleNamespace(impl=attention_impl),
    )
    layer = SimpleNamespace(mtp_block=SimpleNamespace(self_attn=self_attn))
    model = SimpleNamespace(
        topk_indices_buffer=old_buffer,
        layers={"78": layer},
    )
    mtp = object.__new__(hy_v4_mtp.HYV4MTP)
    nn.Module.__init__(mtp)
    mtp.model = model

    mtp.set_topk_indices_buffer(new_buffer)

    assert model.topk_indices_buffer is new_buffer
    assert self_attn.topk_indices_buffer is new_buffer
    assert indexer.topk_indices_buffer is new_buffer
    assert indexer_op.topk_indices_buffer is new_buffer
    assert attention_impl.topk_indices_buffer is new_buffer
