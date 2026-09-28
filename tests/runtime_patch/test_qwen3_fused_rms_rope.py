# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType, SimpleNamespace

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
    assert schema.endswith("-> ()")


def test_fused_rms_rope_fake_returns_none_for_inplace_contract() -> None:
    rms_rope = _load_rms_rope_module()
    query = torch.empty((3, 32, 128), device="meta", dtype=torch.bfloat16)
    key = torch.empty((3, 8, 128), device="meta", dtype=torch.bfloat16)

    result = rms_rope._hcu_rms_rotary_embedding_fake(
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

    assert result is None


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

    assert result is None
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


def _load_qwen3_patch_module():
    spec = importlib.util.find_spec(
        "vllm_hcu.patch.worker.op_opt.patch_qwen3_attention"
    )
    assert spec is not None, "Qwen3 attention patch is not implemented"
    return importlib.import_module(
        "vllm_hcu.patch.worker.op_opt.patch_qwen3_attention"
    )


def _make_qwen3_target(*, forward=None):
    target = ModuleType("vllm.model_executor.models.qwen3")

    class Qwen3Attention:
        def forward(self, positions, hidden_states):
            self.original_calls += 1
            return hidden_states - 3

    if forward is not None:
        Qwen3Attention.forward = forward
    target.Qwen3Attention = Qwen3Attention
    return target, Qwen3Attention


def _make_attention_instance(attention_class):
    instance = attention_class()
    instance.original_calls = 0
    instance.qkv_calls = 0
    instance.q_size = 8
    instance.kv_size = 4
    instance.head_dim = 2
    instance.dual_chunk_attention_config = None
    qkv = torch.arange(32, dtype=torch.float32).reshape(2, 16)

    def qkv_proj(hidden_states):
        instance.qkv_calls += 1
        return qkv.clone(), None

    instance.qkv_proj = qkv_proj
    instance.q_norm = SimpleNamespace(
        weight=torch.tensor([1.0, 2.0]), variance_epsilon=1e-6
    )
    instance.k_norm = SimpleNamespace(
        weight=torch.tensor([3.0, 4.0]), variance_epsilon=1e-6
    )
    cache = torch.arange(64, dtype=torch.float32).reshape(32, 2)
    instance.rotary_emb = SimpleNamespace(
        head_size=2,
        rotary_dim=2,
        is_neox_style=True,
        cos_sin_cache=cache,
        _match_cos_sin_cache_dtype=lambda query: cache,
    )
    instance.attn = lambda query, key, value: torch.cat(
        (query, key, value), dim=-1
    )
    instance.o_proj = lambda hidden: (hidden + 1, None)
    return instance, qkv, cache


def test_qwen3_patch_fuses_supported_text_attention_before_output_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = _load_qwen3_patch_module()
    target, attention_class = _make_qwen3_target()
    instance, qkv, cache = _make_attention_instance(attention_class)
    fused_calls: list[tuple[object, ...]] = []

    def fused(*args):
        fused_calls.append(args)
        return args[1] + 10, args[2] + 20

    monkeypatch.delenv("VLLM_HCU_USE_FUSED_RMS_ROPE", raising=False)
    monkeypatch.delenv("VLLM_HCU_USE_CUSTOM_OPS", raising=False)
    monkeypatch.setattr(patch, "fused_rms_rotary_embedding", fused)
    assert patch.apply_to_module(target) is True

    positions = torch.tensor([0, 1])
    output = instance.forward(positions, torch.ones((2, 4)))

    query, key, value = qkv.split([8, 4, 4], dim=-1)
    expected = torch.cat((query + 10, key + 20, value), dim=-1) + 1
    assert torch.equal(output, expected)
    assert instance.original_calls == 0
    assert instance.qkv_calls == 1
    assert len(fused_calls) == 1
    call = fused_calls[0]
    assert call[0] is positions
    assert torch.equal(call[1], query)
    assert torch.equal(call[2], key)
    assert call[1].is_contiguous()
    assert call[2].is_contiguous()
    assert call[3:] == (
        2,
        cache,
        True,
        instance.q_norm.weight,
        instance.k_norm.weight,
        1e-6,
    )


@pytest.mark.parametrize(
    "unsupported",
    (
        "disabled",
        "mrope",
        "dual_chunk",
        "missing_cache",
        "malformed_cache",
        "stateful_rope",
        "partial_rope",
        "rank3_hidden",
    ),
)
def test_qwen3_patch_delegates_unsupported_inputs_before_qkv_projection(
    monkeypatch: pytest.MonkeyPatch,
    unsupported: str,
) -> None:
    patch = _load_qwen3_patch_module()
    target, attention_class = _make_qwen3_target()
    instance, _, _ = _make_attention_instance(attention_class)
    positions = torch.tensor([0, 1])
    hidden_states = torch.ones((2, 4))
    monkeypatch.delenv("VLLM_HCU_USE_CUSTOM_OPS", raising=False)
    monkeypatch.setenv(
        "VLLM_HCU_USE_FUSED_RMS_ROPE", "0" if unsupported == "disabled" else "1"
    )
    if unsupported == "mrope":
        positions = positions.repeat(3, 1)
    elif unsupported == "dual_chunk":
        instance.dual_chunk_attention_config = {"chunk_size": 8}
    elif unsupported == "missing_cache":
        del instance.rotary_emb.cos_sin_cache
    elif unsupported == "malformed_cache":
        instance.rotary_emb.cos_sin_cache = torch.empty((1, 32, 2))
    elif unsupported == "stateful_rope":
        instance.rotary_emb.update_cache = True
    elif unsupported == "partial_rope":
        instance.rotary_emb.rotary_dim = 1
    elif unsupported == "rank3_hidden":
        hidden_states = hidden_states.unsqueeze(0)

    assert patch.apply_to_module(target) is True
    output = instance.forward(positions, hidden_states)

    assert torch.equal(output, hidden_states - 3)
    assert instance.original_calls == 1
    assert instance.qkv_calls == 0


def test_qwen3_patch_is_idempotent_and_preserves_one_original() -> None:
    patch = _load_qwen3_patch_module()
    target, attention_class = _make_qwen3_target()
    original = attention_class.forward

    assert patch.apply_to_module(target) is True
    assert patch.apply_to_module(target) is False
    assert attention_class._vllm_hcu_original_forward is original


def test_qwen3_patch_registers_custom_ops_before_installing_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = _load_qwen3_patch_module()
    target, _ = _make_qwen3_target()
    calls: list[str] = []
    monkeypatch.delenv("VLLM_HCU_USE_FUSED_RMS_ROPE", raising=False)
    monkeypatch.delenv("VLLM_HCU_USE_CUSTOM_OPS", raising=False)
    monkeypatch.setattr(
        patch, "_register_fused_op_owner", lambda: calls.append("registered")
    )

    assert patch.apply_to_module(target) is True
    assert calls == ["registered"]


def test_qwen3_patch_rejects_forward_signature_drift() -> None:
    patch = _load_qwen3_patch_module()

    def incompatible(self, hidden_states):
        return hidden_states

    target, _ = _make_qwen3_target(forward=incompatible)
    with pytest.raises(RuntimeError, match="incompatible signature"):
        patch.apply_to_module(target)
