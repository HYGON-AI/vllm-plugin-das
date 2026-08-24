# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.models.hy_v4.model import (
    _normalize_hyv4_config,
    _rewrite_hyv4_weight_name,
    _slice_sink_for_tp,
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
