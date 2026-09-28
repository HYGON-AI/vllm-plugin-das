# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType

import pytest
import torch

from vllm_hcu.platforms import envs as henvs


@pytest.mark.parametrize(
    ("feature", "master", "expected"),
    ((None, None, True), ("0", None, False), ("1", "0", False)),
)
def test_fused_qwen3_rms_rope_policy_obeys_default_opt_out_and_master(
    monkeypatch: pytest.MonkeyPatch,
    feature: str | None,
    master: str | None,
    expected: bool,
) -> None:
    for name, value in (
        ("VLLM_HCU_USE_FUSED_RMS_ROPE", feature),
        ("VLLM_HCU_USE_CUSTOM_OPS", master),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    policy = getattr(henvs, "fused_qwen3_rms_rope_enabled", None)
    assert callable(policy), "Qwen3 fused RMS+RoPE policy is not implemented"
    assert policy() is expected


def _load_rms_rope_module():
    spec = importlib.util.find_spec("vllm_hcu.ops.rms_rope")
    assert spec is not None, "Qwen3 fused RMS+RoPE operator is not implemented"
    return importlib.import_module("vllm_hcu.ops.rms_rope")


def test_fused_rms_rope_custom_op_declares_query_and_key_mutable() -> None:
    _load_rms_rope_module()

    schema = str(torch.ops.vllm.hcu_fused_rms_rotary_embedding.default._schema)
    assert "Tensor(a1!) query" in schema
    assert "Tensor(a2!) key" in schema
    assert schema.endswith("-> (Tensor, Tensor)")


def test_fused_rms_rope_fake_preserves_query_and_key_shapes() -> None:
    rms_rope = _load_rms_rope_module()
    query = torch.empty((3, 32, 128), device="meta", dtype=torch.bfloat16)
    key = torch.empty((3, 8, 128), device="meta", dtype=torch.bfloat16)

    out_q, out_k = rms_rope._hcu_rms_rotary_embedding_fake(
        torch.empty(3, device="meta", dtype=torch.long),
        query,
        key,
        128,
        torch.empty((32, 128), device="meta", dtype=torch.bfloat16),
        True,
        torch.empty(128, device="meta", dtype=torch.bfloat16),
        torch.empty(128, device="meta", dtype=torch.bfloat16),
        1e-6,
    )

    assert out_q is query
    assert out_k is key


def test_fused_rms_rope_calls_public_lightop_attention_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rms_rope = _load_rms_rope_module()
    calls: list[tuple[object, ...]] = []
    attention = ModuleType("lightop.attention")

    def rms_rotary_embedding_fuse(*args):
        calls.append(args)
        return args[1], args[2]

    attention.rms_rotary_embedding_fuse = rms_rotary_embedding_fuse
    lightop = ModuleType("lightop")
    lightop.attention = attention
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.attention", attention)

    positions = torch.tensor([0, 1])
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 2, 8)
    cache = torch.randn(16, 8)
    weight_q = torch.randn(8)
    weight_k = torch.randn(8)

    result = rms_rope._hcu_rms_rotary_embedding_impl(
        positions,
        query,
        key,
        8,
        cache,
        True,
        weight_q,
        weight_k,
        1e-6,
    )

    assert result == (query, key)
    assert calls == [
        (
            positions,
            query,
            key,
            8,
            cache,
            True,
            weight_q,
            weight_k,
            None,
            None,
            1e-6,
        )
    ]
