# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.models.hy_v4.model import (
    _dequantize_indexer_channel_fp8,
    _normalize_hyv4_config,
    _rewrite_hyv4_weight_name,
    _slice_sink_for_tp,
    _try_load_fp8_indexer_projection,
)


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


def test_indexer_channel_fp8_rejects_non_channel_scale() -> None:
    weight = torch.ones(2, 4, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="per-output-channel"):
        _dequantize_indexer_channel_fp8(weight, torch.ones(1, 4))


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
