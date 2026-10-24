# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_hcu.models.hy_v4.attention import HYV4MLAAttention, Indexer


class _RecordingLinear:
    def __init__(self, output: torch.Tensor) -> None:
        self.output = output
        self.pairs: list[tuple[torch.Tensor, torch.Tensor] | None] = []

    def __call__(self, value: torch.Tensor):
        self.pairs.append(None)
        return self.output.clone(), None

    def _forward_with_hcu_quanted(
        self,
        value: torch.Tensor,
        pair: tuple[torch.Tensor, torch.Tensor] | None,
    ):
        self.pairs.append(pair)
        return self.output.clone(), None


class _FusedNorm:
    def __init__(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        self.quantized = quantized
        self.scale = scale
        self.calls: list[tuple[torch.dtype, bool]] = []

    def __call__(
        self,
        value: torch.Tensor,
        *,
        quant_dtype: torch.dtype,
        update_input: bool,
    ):
        self.calls.append((quant_dtype, update_input))
        return self.quantized, self.scale, None


def _attention_layer(*, fused_q_lora: bool):
    hidden = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    input_q = torch.ones_like(hidden, dtype=torch.int8)
    input_scale = torch.ones(2, 1, dtype=torch.float32)
    input_pair = (input_q, input_scale)
    q_c = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    q_c_q = torch.ones_like(q_c, dtype=torch.int8)
    q_c_scale = torch.full((2, 1), 0.25, dtype=torch.float32)
    q_a = _RecordingLinear(q_c)
    q_b = _RecordingLinear(torch.ones(2, 2, dtype=torch.bfloat16))
    kv_a = _RecordingLinear(torch.ones(2, 2, dtype=torch.bfloat16))
    indexer_pairs: list[tuple[torch.Tensor, torch.Tensor] | None] = []

    def indexer_and_attn(
        hidden_states,
        normalized_q_c,
        positions,
        q,
        kv_c_normed,
        k_pe,
        out,
        q_dcp_replicated=None,
        q_c_and_scale_quanted=None,
    ):
        indexer_pairs.append(q_c_and_scale_quanted)
        out.fill_(2)

    layer = SimpleNamespace(
        q_lora_rank=3,
        q_a_proj=q_a,
        q_a_layernorm=(
            _FusedNorm(q_c_q, q_c_scale) if fused_q_lora else torch.nn.Identity()
        ),
        q_b_proj=q_b,
        q_proj=None,
        kv_a_proj_with_mqa=kv_a,
        kv_lora_rank=1,
        qk_rope_head_dim=1,
        kv_a_layernorm=torch.nn.Identity(),
        num_local_heads=1,
        qk_head_dim=2,
        qk_nope_head_dim=1,
        v_head_dim=2,
        dcp_q_replicate=False,
        rotary_emb=lambda positions, q_pe, k_pe: (q_pe, k_pe),
        _indexer_and_attn=indexer_and_attn,
        gated_mla=False,
        linear_gate=None,
        gate_a2a_stream=None,
        o_proj=_RecordingLinear(torch.full((2, 4), 3, dtype=torch.bfloat16)),
        enable_fuse_q_lora_rmsnorm_quant=fused_q_lora,
        input_quant_dtype=torch.int8,
    )
    return (
        layer,
        hidden,
        input_pair,
        (q_c_q, q_c_scale),
        q_a,
        q_b,
        kv_a,
        indexer_pairs,
    )


def test_attention_reuses_input_quantization_for_q_a_and_kv_a() -> None:
    layer, hidden, input_pair, _, q_a, _, kv_a, _ = _attention_layer(
        fused_q_lora=True
    )

    HYV4MLAAttention.forward(
        layer,
        positions=torch.arange(2),
        hidden_states=hidden,
        x_and_scale_quanted=input_pair,
    )

    assert q_a.pairs == [input_pair]
    assert kv_a.pairs == [input_pair]


def test_attention_reuses_fused_q_lora_quantization_for_q_b_and_indexer() -> None:
    layer, hidden, input_pair, q_c_pair, _, q_b, _, indexer_pairs = (
        _attention_layer(fused_q_lora=True)
    )

    HYV4MLAAttention.forward(
        layer,
        positions=torch.arange(2),
        hidden_states=hidden,
        x_and_scale_quanted=input_pair,
    )

    assert q_b.pairs == [q_c_pair]
    assert indexer_pairs == [q_c_pair]
    assert layer.q_a_layernorm.calls == [(torch.int8, True)]


def test_attention_disabled_path_does_not_supply_quantized_pairs() -> None:
    layer, hidden, _, _, q_a, q_b, kv_a, indexer_pairs = _attention_layer(
        fused_q_lora=False
    )

    HYV4MLAAttention.forward(
        layer,
        positions=torch.arange(2),
        hidden_states=hidden,
    )

    assert q_a.pairs == [None]
    assert q_b.pairs == [None]
    assert kv_a.pairs == [None]
    assert indexer_pairs == [None]


def test_attention_indexer_reuses_q_lora_quantized_pair(monkeypatch) -> None:
    q_output = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    wq_b = _RecordingLinear(q_output)
    q_c = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    q_c_pair = (
        torch.ones_like(q_c, dtype=torch.int8),
        torch.ones(2, 1, dtype=torch.float32),
    )
    layer = SimpleNamespace(
        wq_b=wq_b,
        n_head=2,
        head_dim=2,
        rope_dim=1,
        wk_weights_proj=_RecordingLinear(torch.ones(2, 4, dtype=torch.bfloat16)),
        k_norm=torch.nn.Identity(),
        quant_block_size=2,
        scale_fmt=None,
        softmax_scale=1.0,
    )
    monkeypatch.setattr(
        "vllm_hcu.models.hy_v4.attention.per_token_group_quant_fp8",
        lambda value, *args, **kwargs: (
            value.to(torch.float8_e4m3fn),
            torch.ones(value.shape[0], 1),
        ),
    )

    Indexer.prepare_inputs(
        layer,
        hidden_states=torch.ones(2, 4, dtype=torch.bfloat16),
        qr=q_c,
        positions=torch.arange(2),
        rotary_emb=lambda positions, q_pe, k_pe: (q_pe, k_pe),
        qr_and_scale_quanted=q_c_pair,
    )

    assert wq_b.pairs == [q_c_pair]
