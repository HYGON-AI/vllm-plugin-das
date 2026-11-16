# SPDX-License-Identifier: Apache-2.0
"""CPU guards for DSA geometry and the norm/RoPE kernel's pointer layout."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("span", [-2, 0, 31, 258, 512])
def test_invalid_rotary_span_rejected_at_construction(span):
    from vllm_hcu.model_executor.layers.step4_dsa import Step4DSAGeometry

    with pytest.raises(ValueError, match="rope span"):
        Step4DSAGeometry(rope_span=span)
    with pytest.raises(ValueError, match="rope span"):
        Step4DSAGeometry.from_config({"sparse_indexer_rope_dim": span})


@pytest.mark.parametrize("span", [2, 32, 256])
def test_partial_and_full_rotary_spans_remain_supported(span):
    from vllm_hcu.model_executor.layers.step4_dsa import Step4DSAGeometry

    geometry = Step4DSAGeometry.from_config(
        SimpleNamespace(sparse_indexer_rope_dim=span)
    )
    assert geometry.rope_span == span


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"proxy_dim": 0}, "proxy_dim must be positive"),
        ({"num_indexer_heads": 0}, "num_indexer_heads must be positive"),
        ({"num_provider_groups": 0}, "num_provider_groups must be positive"),
        ({"num_indexer_heads": 15}, "do not divide"),
        ({"topk": 0}, "topk must be positive"),
        ({"region_size": 16}, "region_block_size"),
    ],
)
def test_invalid_fixed_geometry_rejected(kwargs, message):
    from vllm_hcu.model_executor.layers.step4_dsa import Step4DSAGeometry

    with pytest.raises(ValueError, match=message):
        Step4DSAGeometry(**kwargs)


def _indexer(tp_size, tp_rank):
    from vllm_hcu.model_executor.layers.step4_dsa import (
        Step4DSAGeometry,
        Step4Indexer,
    )

    owner = torch.nn.Module()
    owner.attn = SimpleNamespace(layer_name=f"test.dsa.tp{tp_size}.rank{tp_rank}")
    state = SimpleNamespace(num_kv_groups=0, register_layer=lambda slot: None)
    indexer = Step4Indexer(
        owner,
        geometry=Step4DSAGeometry(),
        state=state,
        dsa_slot=0,
        num_kv_groups=max(1, 4 // max(tp_size, 1)),
        hidden_size=8,
        tp_rank=tp_rank,
        tp_size=tp_size,
        rms_norm_eps=1e-6,
        rope_theta=1e4,
        rope_scaling=None,
        max_position=16,
        params_dtype=torch.bfloat16,
        num_attention_heads=64,
    )
    return owner, indexer


def test_disabled_merged_projection_preserves_separate_weights(monkeypatch):
    from vllm_hcu.model_executor.layers import step4_dsa

    monkeypatch.setattr(step4_dsa, "_MERGED_PROJ", False)
    monkeypatch.setattr(step4_dsa, "_INDEXERS", {})
    owner, indexer = _indexer(1, 0)
    original = tuple(
        getattr(owner, name).weight
        for name in ("sparse_indexer_q", "sparse_indexer_k", "sparse_indexer_z")
    )

    assert indexer._merged_projection(torch.bfloat16) is None
    current = tuple(
        getattr(owner, name).weight
        for name in ("sparse_indexer_q", "sparse_indexer_k", "sparse_indexer_z")
    )
    assert all(after is before for after, before in zip(current, original))
    assert not hasattr(indexer, "_merged_w")


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_provider_groups_shard_or_replicate_without_lost_heads(monkeypatch, tp_size):
    from vllm_hcu.model_executor.layers import step4_dsa

    monkeypatch.setattr(step4_dsa, "_INDEXERS", {})
    full = torch.arange(16).view(16, 1).expand(16, 8).float()
    for rank in range(tp_size):
        owner, indexer = _indexer(tp_size, rank)
        local_groups = max(1, 4 // tp_size)
        first_group = rank * local_groups if tp_size <= 4 else rank // (tp_size // 4)
        weight = owner.sparse_indexer_w.weight
        weight.weight_loader(weight, full)
        expected = full[first_group * 4 : (first_group + local_groups) * 4]
        torch.testing.assert_close(weight, expected)
        assert indexer.num_heads == local_groups * 4


@pytest.mark.parametrize("tp_size, tp_rank", [(0, 0), (2, -1), (2, 2), (3, 0), (6, 5)])
def test_invalid_provider_group_partition_rejected(tp_size, tp_rank):
    with pytest.raises(ValueError, match="tensor-parallel rank|must divide evenly"):
        _indexer(tp_size, tp_rank)


def _inputs():
    # Fused projections are column slices, so row stride exceeds the head width.
    fused = torch.empty(3, 1024)
    return {
        "index_q": fused[:, :512],
        "index_k": fused[:, 512:768],
        "index_z": fused[:, 768:],
        "q_norm_weight": torch.empty(256),
        "k_norm_weight": torch.empty(256),
        "k_norm_bias": torch.empty(256),
        "cos": torch.empty(8, 16),
        "sin": torch.empty(8, 16),
        "positions": torch.arange(3),
        "head_dim": 256,
        "num_q_heads": 2,
        "num_k_heads": 1,
        "rotary_dim": 16,
        "contiguous_z": False,
    }


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"rotary_dim": 256}, "rotated span"),
        ({"rotary_dim": 0}, "rotated span"),
        ({"num_k_heads": 0}, "must be positive"),
        ({"index_q": torch.empty(3, 256)}, "index_q"),
        ({"index_k": torch.empty(2, 256)}, "index_k"),
        ({"index_z": torch.empty(3, 128)}, "index_z"),
        ({"index_k": torch.empty(3, 512)[:, ::2]}, "packed head columns"),
        ({"k_norm_bias": torch.empty(128)}, "k_norm_bias"),
        ({"q_norm_weight": torch.empty(512)[::2]}, "q_norm_weight"),
        ({"sin": torch.empty(8, 32)[:, :16]}, "cos/sin"),
        ({"cos": torch.empty(8, 8)}, "cos/sin"),
        ({"positions": torch.arange(2)}, "positions"),
        ({"positions": torch.arange(6)[::2]}, "positions"),
        ({"positions": torch.arange(3).float()}, "positions"),
    ],
)
def test_invalid_kernel_metadata_fails_before_launch(monkeypatch, changes, message):
    from vllm_hcu.model_executor.layers import step4_dsa_kernels as kernels

    def unexpected_launch(*args, **kwargs):
        pytest.fail("invalid metadata reached the GPU kernel")

    monkeypatch.setattr(kernels, "_norm_rope", unexpected_launch)
    inputs = _inputs()
    inputs.update(changes)
    with pytest.raises(ValueError, match=message):
        kernels.indexer_norm_rope(**inputs)


def test_fused_row_strides_and_full_rotation_are_accepted(monkeypatch):
    from vllm_hcu.model_executor.layers import step4_dsa_kernels as kernels

    calls = []

    def record(x, *args, **kwargs):
        calls.append(kwargs)
        return x.contiguous()

    monkeypatch.setattr(kernels, "_norm_rope", record)
    inputs = _inputs()
    inputs.update(cos=torch.empty(8, 128), sin=torch.empty(8, 128), rotary_dim=128)
    q, k, z = kernels.indexer_norm_rope(**inputs)
    assert q.is_contiguous() and k.is_contiguous()
    assert z is inputs["index_z"]
    assert len(calls) == 2
    assert all(call["rotary_dim"] == 128 for call in calls)
