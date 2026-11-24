"""Weight-only CT must not reintroduce deleted KV scale placeholders on IPC."""
from torch import nn
from vllm.model_executor.layers.attention import Attention
from hcu_vllm_das.quantization.minimax31_draft import MiniMax31DraftInt8Config


def test_unquantized_kv_uses_ordinary_attention_scales():
    config=object.__new__(MiniMax31DraftInt8Config)
    config.kv_cache_scheme=None
    layer=object.__new__(Attention);nn.Module.__init__(layer)
    assert config.get_quant_method(layer,'language_model.model.dspark.layers.0.self_attn.attn') is None


def test_core_shares_target_embedding_when_draft_has_no_placeholder(monkeypatch):
    from types import SimpleNamespace
    from vllm.v1.worker.gpu.spec_decode.eagle import utils
    monkeypatch.setattr(utils,'get_pp_group',lambda:SimpleNamespace(world_size=1))
    draft=nn.Module();draft.has_own_embed_tokens=False
    inner=nn.Module();inner.embed_tokens=None
    target=nn.Module();target.embed_tokens=nn.Embedding(16,8)
    utils.maybe_share_target_embed(draft,inner,target)
    assert inner.embed_tokens is target.embed_tokens
    assert utils._should_share(draft,'has_own_lm_head',None,nn.Linear(8,16))
