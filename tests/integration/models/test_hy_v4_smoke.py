"""CPU integration of the HYV4 decoder boundary; accelerator generation is a later gate."""
from types import SimpleNamespace
import torch
from vllm_hcu.models.hy_v4.model import HYV4DecoderLayer
from vllm_hcu.models.hy_v4.hc import HYV4HCLayer

class Attention(torch.nn.Module):
    def forward(self, positions, hidden_states):
        return hidden_states * 2

def test_hyv4_ihc_decoder_composes_attention_and_mlp():
    config = SimpleNamespace(enable_ihc=True, hidden_size=2, hc_mult=4,
                             hc_magnitude=2.0, hc_eps=1e-6, rms_norm_eps=1e-5)
    layer = object.__new__(HYV4DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.enable_ihc = True
    layer.hc_attn_layer = HYV4HCLayer(config, 0)
    layer.hc_mlp_layer = HYV4HCLayer(config, 0)
    layer.input_layernorm = torch.nn.Identity()
    layer.post_attention_layernorm = torch.nn.Identity()
    layer.self_attn = Attention()
    layer.mlp = torch.nn.Identity()
    with torch.no_grad():
        layer.hc_attn_layer.hc_pre.hc_fn.weight.zero_()
        layer.hc_mlp_layer.hc_pre.hc_fn.weight.zero_()
    x = torch.tensor([[1.0, -2.0]])
    actual, residual = layer(torch.tensor([0]), x, None)
    # Every pre gate is 1/4 + eps, every post gate is 1 + eps.
    after_attn = x * (1 + 2 * (1 + 4e-6) * (1 + 1e-6))
    expected = after_attn * (1 + (1 + 4e-6) * (1 + 1e-6))
    torch.testing.assert_close(actual, expected.unsqueeze(1).expand(-1, 4, -1))
    assert residual is None
