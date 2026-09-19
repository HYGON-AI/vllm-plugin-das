"""Module-level contract for Kimi-K3 MLA output gating on vLLM 0.25.1."""

import inspect

import torch
from torch import nn

from vllm.model_executor.layers.mla import MLAModules
from vllm_hcu.models.kimi_k3.amd.linear import (
    _KimiMLAOutputGateCompat,
    _apply_kimi_mla_output_gate,
)


class _Projection(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, hidden_states):
        return hidden_states @ self.weight.t(), None


def test_old_mla_api_uses_hcu_compat_wrapper():
    assert "g_proj" not in inspect.signature(MLAModules).parameters
    assert issubclass(_KimiMLAOutputGateCompat, nn.Module)


def test_kimi_mla_output_gate_is_before_o_proj():
    hidden_states = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    attn_out = torch.tensor([[2.0, 4.0]], dtype=torch.float32)
    g_proj = _Projection(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    expected = attn_out * torch.sigmoid(hidden_states)
    actual = _apply_kimi_mla_output_gate(attn_out, hidden_states, g_proj)
    torch.testing.assert_close(actual, expected)


def test_output_gate_preserves_attention_width():
    hidden_states = torch.randn(3, 4)
    attn_out = torch.randn(3, 6)
    g_proj = _Projection(torch.randn(6, 4))
    gated = _apply_kimi_mla_output_gate(attn_out, hidden_states, g_proj)
    assert gated.shape == attn_out.shape
    assert torch.isfinite(gated).all()
