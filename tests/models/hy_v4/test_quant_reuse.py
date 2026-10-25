# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import (
    SlimQuantW4A8Int8AiterMoEMethod,
    SlimQuantW4A8Int8LinearMethod,
)
from vllm_hcu.models.hy_v4.attention import HYV4MLAAttention, Indexer
from vllm_hcu.models.hy_v4.moe import HYV4FeedForward, HYV4MoEFused


class _QuantMethod:
    def __init__(self, supports_quanted_inputs: bool) -> None:
        self._supports_quanted_inputs = supports_quanted_inputs

    def supports_quanted_inputs(self) -> bool:
        return self._supports_quanted_inputs


class _RecordingLinear:
    def __init__(
        self,
        output: torch.Tensor,
        *,
        supports_quanted_inputs: bool = True,
    ) -> None:
        self.output = output
        self.pairs: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        self.quant_method = _QuantMethod(supports_quanted_inputs)
        self.weight = torch.empty(1, dtype=torch.int8)

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


class _PublicOnlyLinear:
    """LoRA-like wrapper which intentionally exposes only public forward."""

    def __init__(self, output: torch.Tensor) -> None:
        self.output = output
        self.calls: list[torch.Tensor] = []

    def __call__(self, value: torch.Tensor):
        self.calls.append(value)
        return self.output.clone(), None


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


class _RecordingExperts:
    def __init__(
        self,
        *,
        pcp_size: int = 1,
        naive_dispatch: bool = False,
        routed_dtype: torch.dtype = torch.int8,
        supports_quanted_inputs: bool = True,
    ) -> None:
        self.calls: list[dict[str, torch.Tensor]] = []
        self.moe_config = SimpleNamespace(pcp_size=pcp_size)
        self.do_naive_dispatch_combine = naive_dispatch
        self.routed_experts = SimpleNamespace(
            w13_weight=torch.empty(1, dtype=routed_dtype)
        )
        self._shared_experts = SimpleNamespace(
            _layer_supports_x_and_scale_quanted=lambda: True
        )
        self._supports_quanted_inputs = supports_quanted_inputs

    def _quant_method_supports_quanted_inputs(self) -> bool:
        return self._supports_quanted_inputs

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["hidden_states"] + 1


class _WrappedExperts:
    """FusedMoEWithLoRA-like wrapper around the actual runner."""

    def __init__(self, runner: _RecordingExperts) -> None:
        self.runner = runner
        self.base_layer = runner
        self.moe_config = runner.moe_config

    def __call__(self, **kwargs):
        return self.base_layer(**kwargs)


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
        shared_experts=SimpleNamespace(
            gate_up_proj=_RecordingLinear(
                torch.zeros_like(hidden),
                supports_quanted_inputs=True,
            )
        ),
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


def test_moe_skips_auto_quantization_when_shared_gate_up_rejects_pair(
    monkeypatch,
) -> None:
    experts = _RecordingExperts()
    hidden = torch.zeros(4, 3)
    pair = (torch.ones(4, 3, dtype=torch.int8), torch.ones(4, 1))
    layer = SimpleNamespace(
        gate=_RouterGate(),
        experts=experts,
        shared_experts=SimpleNamespace(
            gate_up_proj=_RecordingLinear(
                torch.zeros_like(hidden),
                supports_quanted_inputs=False,
            )
        ),
        prefix="model.layers.1.mlp",
        enable_reuse_int8_quant=True,
    )
    calls = []
    monkeypatch.setattr(
        "vllm_hcu.models.hy_v4.moe._quantize_int8",
        lambda value: calls.append(value) or pair,
    )
    monkeypatch.setattr(
        "vllm_hcu.models.hy_v4.moe._hyv4_moe_chunk_size",
        lambda: None,
    )

    HYV4MoEFused.forward(layer, hidden)

    assert calls == []
    assert "quanted_hidden_states" not in experts.calls[0]
    assert "scale" not in experts.calls[0]


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


@torch.no_grad()
@pytest.mark.parametrize(
    ("pcp_size", "naive_dispatch", "chunk_size", "supply_pair"),
    [(8, False, None, False), (1, True, 2, True)],
)
def test_moe_drops_prequantized_auxiliary_input_for_dispatch_topologies(
    monkeypatch,
    pcp_size: int,
    naive_dispatch: bool,
    chunk_size: int | None,
    supply_pair: bool,
) -> None:
    experts = _RecordingExperts(
        pcp_size=pcp_size,
        naive_dispatch=naive_dispatch,
    )
    layer = SimpleNamespace(
        gate=_RouterGate(),
        experts=experts,
        prefix="model.layers.78.mlp",
        enable_reuse_int8_quant=True,
    )
    hidden = torch.zeros(5, 3)
    pair = (torch.ones(5, 3, dtype=torch.int8), torch.ones(5, 1))
    quantize_calls = []
    monkeypatch.setattr(
        "vllm_hcu.models.hy_v4.moe._quantize_int8",
        lambda value: quantize_calls.append(value) or pair,
    )
    monkeypatch.setattr(
        "vllm_hcu.models.hy_v4.moe._hyv4_moe_chunk_size",
        lambda: chunk_size,
    )
    monkeypatch.setattr(
        "vllm_hcu.models.hy_v4.moe.linear_gate_pcp_shard_enabled", lambda: True
    )
    monkeypatch.setattr(
        "vllm_hcu.models.hy_v4.moe.in_replicated_mtp_batch", lambda: True
    )

    HYV4MoEFused.forward(
        layer,
        hidden,
        x_and_scale_quanted=pair if supply_pair else None,
    )

    assert quantize_calls == []
    assert all("quanted_hidden_states" not in call for call in experts.calls)
    assert all("scale" not in call for call in experts.calls)


def test_moe_checks_naive_dispatch_on_lora_wrapped_runner(monkeypatch) -> None:
    runner = _RecordingExperts(naive_dispatch=True)
    experts = _WrappedExperts(runner)
    layer = SimpleNamespace(
        gate=_RouterGate(),
        experts=experts,
        prefix="model.layers.1.mlp",
        enable_reuse_int8_quant=True,
    )
    hidden = torch.zeros(4, 3)
    pair = (torch.ones(4, 3, dtype=torch.int8), torch.ones(4, 1))
    monkeypatch.setattr("vllm_hcu.models.hy_v4.moe._hyv4_moe_chunk_size", lambda: None)

    HYV4MoEFused.forward(layer, hidden, x_and_scale_quanted=pair)

    assert "quanted_hidden_states" not in runner.calls[0]
    assert "scale" not in runner.calls[0]


def test_moe_rejects_int8_pair_for_non_int8_routed_experts(monkeypatch) -> None:
    experts = _RecordingExperts(routed_dtype=torch.float8_e4m3fn)
    layer = SimpleNamespace(
        gate=_RouterGate(),
        experts=experts,
        prefix="model.layers.1.mlp",
        enable_reuse_int8_quant=True,
    )
    hidden = torch.zeros(4, 3)
    pair = (torch.ones(4, 3, dtype=torch.int8), torch.ones(4, 1))
    monkeypatch.setattr("vllm_hcu.models.hy_v4.moe._hyv4_moe_chunk_size", lambda: None)

    HYV4MoEFused.forward(layer, hidden, x_and_scale_quanted=pair)

    assert "quanted_hidden_states" not in experts.calls[0]
    assert "scale" not in experts.calls[0]


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


def test_feed_forward_uses_float_activation_when_down_rejects_quanted_input() -> None:
    gate_up = _RecordingLinear(torch.ones(2, 8))
    down = _RecordingLinear(
        torch.full((2, 3), 7.0),
        supports_quanted_inputs=False,
    )
    fused_act = _FusedAct(
        torch.ones(2, 4, dtype=torch.int8),
        torch.ones(2, 1),
    )
    float_act = lambda value: value[..., :4]
    layer = SimpleNamespace(
        gate_up_proj=gate_up,
        down_proj=down,
        act_fn=fused_act,
        float_act_fn=float_act,
        enable_fuse_silu_mul_quant=True,
        quant_dtype=torch.int8,
    )

    actual = HYV4FeedForward.forward(layer, torch.zeros(2, 3))

    assert down.pairs == [None]
    torch.testing.assert_close(actual, torch.full((2, 3), 7.0))


def test_feed_forward_preserves_public_forward_for_lora_wrapper() -> None:
    gate_up = _PublicOnlyLinear(torch.ones(2, 8))
    down = _RecordingLinear(
        torch.full((2, 3), 7.0),
        supports_quanted_inputs=False,
    )
    layer = SimpleNamespace(
        gate_up_proj=gate_up,
        down_proj=down,
        act_fn=lambda value: value[..., :4],
        enable_fuse_silu_mul_quant=False,
        quant_dtype=torch.int8,
    )

    input_pair = (torch.ones(2, 3, dtype=torch.int8), torch.ones(2, 1))
    actual = HYV4FeedForward.forward(
        layer,
        torch.zeros(2, 3),
        x_and_scale_quanted=input_pair,
    )

    assert gate_up.calls
    torch.testing.assert_close(actual, torch.full((2, 3), 7.0))


def test_attention_preserves_public_forward_for_lora_wrapper(monkeypatch) -> None:
    layer, hidden, _, _, _, _, _, _ = _attention_layer(reuse_quant=True)
    q_b = _PublicOnlyLinear(torch.ones(2, 2, dtype=torch.bfloat16))
    layer.q_b_proj = q_b
    monkeypatch.setattr(
        "vllm_hcu.models.hy_v4.attention._quantize_int8",
        lambda value: (
            torch.ones_like(value, dtype=torch.int8),
            torch.ones(*value.shape[:-1], 1),
        ),
    )

    HYV4MLAAttention.forward(
        layer,
        positions=torch.arange(2),
        hidden_states=hidden,
    )

    assert q_b.calls


def test_slimquant_linear_forwards_prequantized_pair_to_hcu_runtime(
    monkeypatch,
) -> None:
    pair = (torch.ones(2, 3, dtype=torch.int8), torch.full((2, 1), 0.25))
    calls = []

    class _Scheme:
        pass

    method = SlimQuantW4A8Int8LinearMethod.__new__(SlimQuantW4A8Int8LinearMethod)
    layer = SimpleNamespace(
        scheme=_Scheme(),
        weight=torch.ones(4, 3, dtype=torch.int8),
        weight_scale=torch.ones(4, 1),
        params_dtype=torch.bfloat16,
        input_scale=None,
        input_zero_point=None,
        azp_adj=None,
    )
    value = torch.zeros(2, 3)
    monkeypatch.setattr(
        method,
        "supports_quanted_inputs",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm_hcu.model_executor.layers.quantization.slimquant_w4a8.apply_int8_linear",
        lambda **kwargs: calls.append(kwargs) or torch.ones(2, 4),
    )

    method.apply(layer, value, x_and_scale_quanted=pair)

    assert calls[0]["input"] is value
    assert calls[0]["x_and_scale_quanted"] is pair


def test_slimquant_deepep_moe_rejects_prequantized_dispatch_input(
    monkeypatch,
) -> None:
    pair = (torch.ones(2, 3, dtype=torch.int8), torch.full((2, 1), 0.25))
    kernel_calls = []

    class _Kernel:
        def apply(self, **kwargs):
            kernel_calls.append(kwargs)
            return torch.ones(2, 3)

    method = SlimQuantW4A8Int8AiterMoEMethod.__new__(
        SlimQuantW4A8Int8AiterMoEMethod
    )
    method.moe_kernel = _Kernel()
    method.moe = SimpleNamespace(num_experts=2, activation="silu")
    layer = SimpleNamespace(
        w13_weight=torch.ones(2, 4, 3, dtype=torch.int8),
        w2_weight=torch.ones(2, 3, 2, dtype=torch.int8),
        global_num_experts=2,
        expert_map=None,
        activation="silu",
        apply_router_weight_on_input=False,
    )
    monkeypatch.setattr(
        "vllm_hcu.model_executor.layers.fused_moe.aiter_moe_dispatch.resolve_aiter_expert_maps",
        lambda expert_map, global_num_experts: (None, None),
    )

    assert not method.supports_quanted_inputs()
    with pytest.raises(RuntimeError, match="BF16 DeepEP dispatch"):
        method.apply(
            layer=layer,
            x=torch.zeros(2, 3),
            topk_weights=torch.ones(2, 1),
            topk_ids=torch.zeros(2, 1, dtype=torch.int64),
            shared_experts=None,
            shared_experts_input=None,
            i_q=pair[0],
            i_s=pair[1],
        )

    assert kernel_calls == []


def _load_moe_runner_contract():
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExpertsOrder,
    )

    path = (
        Path(__file__).resolve().parents[3]
        / "vllm_hcu/model_executor/layers/fused_moe/moe_runner.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MoERunner"
    )
    method_names = {
        "_quant_method",
        "_maybe_apply_shared_experts",
        "_quant_method_supports_quanted_inputs",
        "_quant_method_supports_shared_quanted_inputs",
        "_apply_quant_method",
    }
    methods = [
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    runner_class = ast.ClassDef(
        name="MoERunnerContract",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias("annotations")],
                level=0,
            ),
            runner_class,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "inspect": inspect,
        "torch": torch,
        "SharedExpertsOrder": SharedExpertsOrder,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["MoERunnerContract"]


def _runner_for_quant_method(quant_method, forward_modular):
    runner_class = _load_moe_runner_contract()
    runner = object.__new__(runner_class)

    class _SharedExperts:
        output = None

        def __call__(self, *args, **kwargs):
            del args, kwargs

    runner.routed_experts = SimpleNamespace(
        quant_method=quant_method,
        forward_modular=forward_modular,
    )
    runner._shared_experts = _SharedExperts()
    runner.router = SimpleNamespace(
        select_experts=lambda **_: pytest.fail(
            "preselected routing must bypass the router"
        )
    )
    return runner


def test_moe_runner_passes_shared_quanted_pair_without_quantizing_routed_input():
    calls = []

    class _SharedOnlyQuantMethod:
        is_monolithic = False

        def supports_quanted_inputs(self):
            return False

        def supports_shared_quanted_inputs(self):
            return True

        def apply(self, **kwargs):
            calls.append(kwargs)
            return kwargs["x"] + 1

    runner = _runner_for_quant_method(
        _SharedOnlyQuantMethod(),
        lambda **_: pytest.fail("shared-only support must bypass public fallback"),
    )
    hidden = torch.ones(2, 3, dtype=torch.bfloat16)
    pair = (torch.ones(2, 3, dtype=torch.int8), torch.ones(2, 1))

    _, output = runner._apply_quant_method(
        hidden,
        None,
        hidden,
        quanted_hidden_states=pair[0],
        scale=pair[1],
        topk_weights=torch.ones(2, 1),
        topk_ids=torch.zeros(2, 1, dtype=torch.int32),
    )

    assert calls[0]["x"] is hidden
    shared_pair = calls[0]["shared_experts_x_and_scale_quanted"]
    assert shared_pair[0] is pair[0]
    assert shared_pair[1] is pair[1]
    assert "i_q" not in calls[0] and "i_s" not in calls[0]
    torch.testing.assert_close(output, hidden + 1)


def test_moe_runner_shared_quanted_unsupported_uses_public_fallback():
    fallback_calls = []

    class _UnsupportedQuantMethod:
        is_monolithic = False

        def supports_quanted_inputs(self):
            return False

        def apply(self, **_):
            pytest.fail("unsupported method must use public forward_modular")

    def forward_modular(**kwargs):
        fallback_calls.append(kwargs)
        return kwargs["x"] + 2

    runner = _runner_for_quant_method(_UnsupportedQuantMethod(), forward_modular)
    hidden = torch.ones(2, 3, dtype=torch.bfloat16)
    pair = (torch.ones(2, 3, dtype=torch.int8), torch.ones(2, 1))

    _, output = runner._apply_quant_method(
        hidden,
        None,
        hidden,
        quanted_hidden_states=pair[0],
        scale=pair[1],
        topk_weights=torch.ones(2, 1),
        topk_ids=torch.zeros(2, 1, dtype=torch.int32),
    )

    assert fallback_calls[0]["x"] is hidden
    torch.testing.assert_close(output, hidden + 2)


@pytest.mark.parametrize("missing", ["quanted_hidden_states", "scale"])
def test_moe_runner_rejects_incomplete_quanted_input_pair(missing):
    class _QuantMethod:
        is_monolithic = False

    runner = _runner_for_quant_method(
        _QuantMethod(),
        lambda **_: pytest.fail("incomplete pair must fail before dispatch"),
    )
    hidden = torch.ones(2, 3, dtype=torch.bfloat16)
    kwargs = {
        "quanted_hidden_states": torch.ones(2, 3, dtype=torch.int8),
        "scale": torch.ones(2, 1),
        "topk_weights": torch.ones(2, 1),
        "topk_ids": torch.zeros(2, 1, dtype=torch.int32),
    }
    kwargs[missing] = None

    with pytest.raises(
        ValueError,
        match="requires quanted_hidden_states and scale",
    ):
        runner._apply_quant_method(hidden, None, hidden, **kwargs)


def test_slimquant_deepep_moe_forwards_shared_quanted_input_with_bf16_dispatch(
    monkeypatch,
) -> None:
    pair = (torch.ones(2, 3, dtype=torch.int8), torch.full((2, 1), 0.25))
    kernel_calls = []

    class _Kernel:
        def apply(self, **kwargs):
            kernel_calls.append(kwargs)
            return torch.ones(2, 3)

    method = SlimQuantW4A8Int8AiterMoEMethod.__new__(
        SlimQuantW4A8Int8AiterMoEMethod
    )
    method.moe_kernel = _Kernel()
    method.moe = SimpleNamespace(num_experts=2, activation="silu")
    layer = SimpleNamespace(
        w13_weight=torch.ones(2, 4, 3, dtype=torch.int8),
        w2_weight=torch.ones(2, 3, 2, dtype=torch.int8),
        global_num_experts=2,
        expert_map=None,
        activation="silu",
        apply_router_weight_on_input=False,
    )
    monkeypatch.setattr(
        "vllm_hcu.model_executor.layers.fused_moe.aiter_moe_dispatch.resolve_aiter_expert_maps",
        lambda expert_map, global_num_experts: (None, None),
    )
    hidden = torch.zeros(2, 3, dtype=torch.bfloat16)

    assert method.supports_shared_quanted_inputs()
    method.apply(
        layer=layer,
        x=hidden,
        topk_weights=torch.ones(2, 1),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
        shared_experts=object(),
        shared_experts_input=hidden,
        shared_experts_x_and_scale_quanted=pair,
    )

    assert kernel_calls[0]["hidden_states"] is hidden
    assert kernel_calls[0]["shared_experts_x_and_scale_quanted"] is pair
    assert "quanted_hidden_states" not in kernel_calls[0]
    assert "scale" not in kernel_calls[0]


def test_slimquant_deepep_moe_rejects_incomplete_shared_quanted_input(
    monkeypatch,
) -> None:
    method = SlimQuantW4A8Int8AiterMoEMethod.__new__(
        SlimQuantW4A8Int8AiterMoEMethod
    )
    method.moe_kernel = SimpleNamespace(
        apply=lambda **_: pytest.fail("invalid pair must fail before kernel")
    )
    method.moe = SimpleNamespace(num_experts=2, activation="silu")
    layer = SimpleNamespace(
        w13_weight=torch.ones(2, 4, 3, dtype=torch.int8),
        w2_weight=torch.ones(2, 3, 2, dtype=torch.int8),
        global_num_experts=2,
        expert_map=None,
        activation="silu",
        apply_router_weight_on_input=False,
    )
    monkeypatch.setattr(
        "vllm_hcu.model_executor.layers.fused_moe.aiter_moe_dispatch.resolve_aiter_expert_maps",
        lambda expert_map, global_num_experts: (None, None),
    )

    with pytest.raises(ValueError, match="both quantized input and scale"):
        method.apply(
            layer=layer,
            x=torch.zeros(2, 3, dtype=torch.bfloat16),
            topk_weights=torch.ones(2, 1),
            topk_ids=torch.zeros(2, 1, dtype=torch.int64),
            shared_experts=object(),
            shared_experts_input=torch.zeros(2, 3, dtype=torch.bfloat16),
            shared_experts_x_and_scale_quanted=(torch.ones(2, 3, dtype=torch.int8), None),
        )
