# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_hcu.models.hy_v4.attention import HYV4MLAAttention, Indexer
from vllm_hcu.models.hy_v4.model import HYV4DecoderLayer
from vllm_hcu.models.hy_v4.moe import HYV4FeedForward, HYV4MoEFused


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


def _attention_layer(*, reuse_quant: bool):
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
        q_a_layernorm=torch.nn.Identity(),
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
        enable_reuse_int8_quant=reuse_quant,
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


def test_attention_reuses_input_quantization_for_q_a_and_kv_a(monkeypatch) -> None:
    layer, hidden, input_pair, q_c_pair, q_a, _, kv_a, _ = _attention_layer(
        reuse_quant=True
    )
    monkeypatch.setattr(
        "vllm_hcu.models.hy_v4.attention._quantize_int8",
        lambda value: q_c_pair,
    )

    HYV4MLAAttention.forward(
        layer,
        positions=torch.arange(2),
        hidden_states=hidden,
        x_and_scale_quanted=input_pair,
    )

    assert q_a.pairs == [input_pair]
    assert kv_a.pairs == [input_pair]


def test_attention_quantizes_normalized_q_lora_once_for_q_b_and_indexer(
    monkeypatch,
) -> None:
    layer, hidden, input_pair, q_c_pair, _, q_b, _, indexer_pairs = (
        _attention_layer(reuse_quant=True)
    )
    monkeypatch.setattr(
        "vllm_hcu.models.hy_v4.attention._quantize_int8",
        lambda value: q_c_pair,
    )

    HYV4MLAAttention.forward(
        layer,
        positions=torch.arange(2),
        hidden_states=hidden,
        x_and_scale_quanted=input_pair,
    )

    assert q_b.pairs == [q_c_pair]
    assert indexer_pairs == [q_c_pair]


def test_attention_disabled_path_does_not_supply_quantized_pairs() -> None:
    layer, hidden, _, _, q_a, q_b, kv_a, indexer_pairs = _attention_layer(
        reuse_quant=False
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


class _DecoderBlock:
    def __init__(self, increment: float) -> None:
        self.increment = increment
        self.calls: list[tuple[torch.Tensor, object]] = []

    def __call__(self, *args, **kwargs):
        value = kwargs.get("hidden_states", args[0] if args else None)
        pair = kwargs.get("x_and_scale_quanted")
        self.calls.append((value.clone(), pair))
        return value + self.increment


class _IdentityHC:
    def prepare_input(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def pre(self, value: torch.Tensor):
        return value.clone(), torch.ones(value.shape[0], 1), value.clone()

    def post(self, value, residual, post_gates):
        return value


def test_decoder_ihc_keeps_norm_outputs_unquantized() -> None:
    attention = _DecoderBlock(1)
    moe = _DecoderBlock(2)
    layer = SimpleNamespace(
        hc_attn_layer=_IdentityHC(),
        hc_mlp_layer=_IdentityHC(),
        input_layernorm=lambda value: value + 3,
        post_attention_layernorm=lambda value: value + 4,
        self_attn=attention,
        mlp=moe,
    )
    hidden = torch.zeros(2, 4, dtype=torch.bfloat16)

    actual, residual = HYV4DecoderLayer._forward_ihc(
        layer,
        positions=torch.arange(2),
        hidden_states=hidden,
    )

    assert residual is None
    assert attention.calls[0][1] is None
    assert moe.calls[0][1] is None
    torch.testing.assert_close(attention.calls[0][0], torch.full_like(hidden, 3))
    torch.testing.assert_close(moe.calls[0][0], torch.full_like(hidden, 8))
    torch.testing.assert_close(actual, torch.full_like(hidden, 10))


def test_decoder_ihc_disabled_path_preserves_unquantized_calls() -> None:
    attention = _DecoderBlock(1)
    moe = _DecoderBlock(2)
    layer = SimpleNamespace(
        hc_attn_layer=_IdentityHC(),
        hc_mlp_layer=_IdentityHC(),
        input_layernorm=torch.nn.Identity(),
        post_attention_layernorm=torch.nn.Identity(),
        self_attn=attention,
        mlp=moe,
    )

    HYV4DecoderLayer._forward_ihc(
        layer,
        positions=torch.arange(2),
        hidden_states=torch.zeros(2, 4, dtype=torch.bfloat16),
    )

    assert attention.calls[0][1] is None
    assert moe.calls[0][1] is None


class _RecordingExperts:
    def __init__(self) -> None:
        self.calls: list[dict[str, torch.Tensor]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["hidden_states"] + 1


class _RouterGate:
    def __call__(self, value: torch.Tensor):
        return torch.ones(value.shape[0], 2), None


def test_moe_forwards_prequantized_input_to_routed_and_shared_experts(
    monkeypatch,
) -> None:
    experts = _RecordingExperts()
    layer = SimpleNamespace(
        gate=_RouterGate(),
        experts=experts,
        prefix="model.layers.1.mlp",
    )
    hidden = torch.zeros(4, 3)
    pair = (torch.ones(4, 3, dtype=torch.int8), torch.ones(4, 1))
    monkeypatch.setattr("vllm_hcu.models.hy_v4.moe._hyv4_moe_chunk_size", lambda: None)

    actual = HYV4MoEFused.forward(layer, hidden, x_and_scale_quanted=pair)

    assert experts.calls[0]["quanted_hidden_states"].data_ptr() == pair[0].data_ptr()
    assert experts.calls[0]["scale"].data_ptr() == pair[1].data_ptr()
    torch.testing.assert_close(actual, hidden + 1)


def test_moe_quantizes_input_once_when_pair_is_not_supplied(monkeypatch) -> None:
    experts = _RecordingExperts()
    hidden = torch.zeros(4, 3)
    pair = (torch.ones(4, 3, dtype=torch.int8), torch.ones(4, 1))
    layer = SimpleNamespace(
        gate=_RouterGate(),
        experts=experts,
        prefix="model.layers.1.mlp",
        enable_reuse_int8_quant=True,
    )
    calls = []
    monkeypatch.setattr(
        "vllm_hcu.models.hy_v4.moe._quantize_int8",
        lambda value: calls.append(value) or pair,
    )
    monkeypatch.setattr("vllm_hcu.models.hy_v4.moe._hyv4_moe_chunk_size", lambda: None)

    HYV4MoEFused.forward(layer, hidden)

    assert calls == [hidden]
    assert experts.calls[0]["quanted_hidden_states"].data_ptr() == pair[0].data_ptr()
    assert experts.calls[0]["scale"].data_ptr() == pair[1].data_ptr()


def test_moe_chunking_slices_hidden_quantized_input_and_scale_together(
    monkeypatch,
) -> None:
    experts = _RecordingExperts()
    layer = SimpleNamespace(
        gate=_RouterGate(),
        experts=experts,
        prefix="model.layers.78.mlp",
    )
    hidden = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    quantized = torch.arange(15, dtype=torch.int8).reshape(5, 3)
    scale = torch.arange(5, dtype=torch.float32).reshape(5, 1)
    monkeypatch.setattr("vllm_hcu.models.hy_v4.moe._hyv4_moe_chunk_size", lambda: 2)
    monkeypatch.setattr(
        "vllm_hcu.models.hy_v4.moe.linear_gate_pcp_shard_enabled", lambda: True
    )
    monkeypatch.setattr(
        "vllm_hcu.models.hy_v4.moe.in_replicated_mtp_batch", lambda: True
    )

    HYV4MoEFused.forward(
        layer,
        hidden,
        x_and_scale_quanted=(quantized, scale),
    )

    assert [call["hidden_states"].shape[0] for call in experts.calls] == [2, 2, 1]
    torch.testing.assert_close(
        torch.cat([call["quanted_hidden_states"] for call in experts.calls]),
        quantized,
    )
    torch.testing.assert_close(
        torch.cat([call["scale"] for call in experts.calls]),
        scale,
    )


class _FusedAct:
    def __init__(self, quantized: torch.Tensor, scale: torch.Tensor) -> None:
        self.quantized = quantized
        self.scale = scale

    def __call__(self, value: torch.Tensor, *, quant_dtype: torch.dtype):
        assert quant_dtype is torch.int8
        return self.quantized, self.scale


def test_feed_forward_reuses_input_and_fuses_down_projection_quantization() -> None:
    gate_up = _RecordingLinear(torch.ones(2, 8))
    down = _RecordingLinear(torch.full((2, 3), 7.0))
    input_pair = (torch.ones(2, 3, dtype=torch.int8), torch.ones(2, 1))
    activated_pair = (torch.ones(2, 4, dtype=torch.int8), torch.ones(2, 1))
    layer = SimpleNamespace(
        gate_up_proj=gate_up,
        down_proj=down,
        act_fn=_FusedAct(*activated_pair),
        enable_fuse_silu_mul_quant=True,
        quant_dtype=torch.int8,
    )

    actual = HYV4FeedForward.forward(
        layer,
        torch.zeros(2, 3),
        x_and_scale_quanted=input_pair,
    )

    assert gate_up.pairs == [input_pair]
    assert down.pairs == [activated_pair]
    torch.testing.assert_close(actual, torch.full((2, 3), 7.0))
