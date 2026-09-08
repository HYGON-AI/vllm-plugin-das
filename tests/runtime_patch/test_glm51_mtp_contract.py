# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU behavior contracts for the GLM-5.1 sparse MTP runtime."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def mtp_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load deepseek_mtp with only its HCU-owned dependencies stubbed."""

    deepseek_v2 = ModuleType("vllm_hcu.models.deepseek_v2")

    class DeepseekV2DecoderLayer(nn.Module):
        pass

    class DeepseekV2MixtureOfExperts:
        pass

    deepseek_v2.DeepseekV2DecoderLayer = DeepseekV2DecoderLayer
    deepseek_v2.DeepseekV2MixtureOfExperts = DeepseekV2MixtureOfExperts
    deepseek_v2.DeepseekV2MoE = type("DeepseekV2MoE", (), {})
    deepseek_v2._try_load_quantized_indexer_wk = lambda *args, **kwargs: False
    deepseek_v2.get_spec_layer_idx_from_weight_name = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, deepseek_v2.__name__, deepseek_v2)

    lightly_cp_utils = ModuleType("vllm_hcu.v1.attention.lightly_cp_utils")
    lightly_cp_utils.lightly_cp_inputs_splitting = lambda *args: args[:4]
    monkeypatch.setitem(sys.modules, lightly_cp_utils.__name__, lightly_cp_utils)

    module_name = "vllm_hcu.models._test_glm51_deepseek_mtp"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO / "vllm_hcu/models/deepseek_mtp.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _uninitialized(module_type: type[nn.Module]) -> nn.Module:
    instance = object.__new__(module_type)
    nn.Module.__init__(instance)
    return instance


def test_main_v32_mtp_architecture_uses_hcu_mtp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_hcu.models as models

    registrations: dict[str, str] = {}
    monkeypatch.setattr(
        models.ModelRegistry,
        "register_model",
        lambda architecture, implementation: registrations.__setitem__(
            architecture, implementation
        ),
    )

    models.register_model()

    assert registrations["DeepseekV32MTPModel"] == (
        "vllm_hcu.models.deepseek_mtp:DeepSeekMTP"
    )


def test_mtp_layer_returns_pre_norm_logits_and_post_norm_recycle_states(
    mtp_module: ModuleType,
) -> None:
    class FuseInputs(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            midpoint = value.shape[-1] // 2
            return value[..., :midpoint] + value[..., midpoint:]

    class MTPBlock(nn.Module):
        def forward(self, **kwargs):
            hidden_states = kwargs["hidden_states"]
            return hidden_states * 2, hidden_states * 3

    class FinalNorm(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + 7

    layer = _uninitialized(mtp_module.DeepSeekMultiTokenPredictorLayer)
    layer.enorm = nn.Identity()
    layer.hnorm = nn.Identity()
    layer.eh_proj = FuseInputs()
    layer.mtp_block = MTPBlock()
    layer.shared_head = FinalNorm()

    positions = torch.tensor([0, 1])
    previous = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    inputs = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

    logits_hidden, recycle_hidden = layer(
        torch.tensor([1, 2]),
        positions,
        previous,
        inputs,
    )

    # Position zero masks inputs, then the block returns 2*x with 3*x residual.
    expected_logits_hidden = torch.tensor([[5.0, 10.0], [165.0, 220.0]])
    torch.testing.assert_close(logits_hidden, expected_logits_hidden)
    torch.testing.assert_close(recycle_hidden, expected_logits_hidden + 7)


def test_mtp_sparse_index_controls_share_and_compact_all_layers(
    mtp_module: ModuleType,
) -> None:
    class MLAAttention(nn.Module):
        def __init__(self, values: torch.Tensor) -> None:
            super().__init__()
            self.skip_topk = False
            self.topk_indices_buffer = values.clone()

    class SelfAttention(nn.Module):
        def __init__(self, mla_attn: nn.Module) -> None:
            super().__init__()
            self.mla_attn = mla_attn

    class MTPBlock(nn.Module):
        def __init__(self, mla_attn: nn.Module) -> None:
            super().__init__()
            self.self_attn = SelfAttention(mla_attn)

    class Layer(nn.Module):
        def __init__(self, mla_attn: nn.Module | None = None) -> None:
            super().__init__()
            if mla_attn is not None:
                self.mtp_block = MTPBlock(mla_attn)

    first = MLAAttention(torch.arange(10).view(5, 2))
    second = MLAAttention(torch.arange(20, 30).view(5, 2))
    predictor = _uninitialized(mtp_module.DeepSeekMultiTokenPredictor)
    predictor.layers = nn.ModuleDict(
        {"0": Layer(first), "1": Layer(), "2": Layer(second)}
    )

    originals = [first.topk_indices_buffer.clone(), second.topk_indices_buffer.clone()]
    predictor.set_skip_topk(True)
    predictor.compact_topk_indices(torch.tensor([3, 1]))

    assert first.skip_topk is True
    assert second.skip_topk is True
    torch.testing.assert_close(first.topk_indices_buffer[:2], originals[0][[3, 1]])
    torch.testing.assert_close(second.topk_indices_buffer[:2], originals[1][[3, 1]])


def test_mtp_lightly_cp_gathers_both_hidden_state_contracts(
    mtp_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Layer(nn.Module):
        def forward(
            self,
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            spec_step_idx,
        ):
            del input_ids, positions, inputs_embeds, spec_step_idx
            return previous_hidden_states + 10, previous_hidden_states + 20

    context = SimpleNamespace(
        enable_lightly_cp=True,
        gather_indexes_tensor=torch.tensor([2, 0]),
    )
    monkeypatch.setattr(mtp_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(
        mtp_module,
        "lightly_cp_inputs_splitting",
        lambda hidden, positions, residual, inputs, tp_size, tp_rank: (
            hidden,
            positions,
            residual,
            inputs,
        ),
    )
    monkeypatch.setattr(
        mtp_module,
        "tensor_model_parallel_all_gather",
        lambda value, dim: torch.cat((value, value + 100), dim=dim),
    )

    predictor = _uninitialized(mtp_module.DeepSeekMultiTokenPredictor)
    predictor.mtp_start_layer_idx = 4
    predictor.num_mtp_layers = 1
    predictor.tp_size = 2
    predictor.tp_rank = 0
    predictor.layers = nn.ModuleDict({"4": Layer()})

    logits_hidden, recycle_hidden = predictor(
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        torch.tensor([[1.0], [2.0]]),
        torch.zeros(2, 1),
    )

    torch.testing.assert_close(logits_hidden, torch.tensor([[111.0], [11.0]]))
    torch.testing.assert_close(recycle_hidden, torch.tensor([[121.0], [21.0]]))
