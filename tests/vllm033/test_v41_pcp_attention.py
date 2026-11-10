"""Only HCU bridge contracts; PCP scheduling and Engram stay in Core."""
from types import SimpleNamespace as NS
import inspect

import pytest
import torch
from hcu_vllm_das import v41_pcp as bridge
from hcu_vllm_das.models import v41_pcp as model
from vllm.models.deepseek_v41.amd import model as native, rocm, vl_model
from vllm.models.deepseek_v41.attention import DeepseekV4Attention


def config():
    return NS(model_config=NS(enforce_eager=True, hf_config=NS(model_type="deepseek_v41")),
        parallel_config=NS(tensor_parallel_size=1, data_parallel_size=1,
            pipeline_parallel_size=1, prefill_context_parallel_size=8,
            decode_context_parallel_size=1, enable_expert_parallel=True,
            enable_eplb=False, enable_dbo=False), use_v2_model_runner=True,
        kernel_config=NS(moe_backend="deep_gemm_mega_moe", monokernel=False),
        speculative_config=None, lora_config=None, kv_transfer_config=None,
        cache_config=NS(swa_bounded_replay=False),
        attention_config=NS(backend=None, indexer_sparse_logits=False,
                            resolve_indexer_kv_dtype=lambda _: "fp8"))


def test_only_target_combination_is_admitted():
    cfg = config()
    assert bridge.validate_config(cfg)
    for owner, field, value in [(cfg, "speculative_config", object()),
        (cfg.parallel_config, "prefill_context_parallel_size", 1),
        (cfg.parallel_config, "enable_eplb", True),
        (cfg.kernel_config, "moe_backend", "deep_gemm")]:
        previous = getattr(owner, field)
        setattr(owner, field, value)
        with pytest.raises(ValueError):
            bridge.validate_config(cfg)
        setattr(owner, field, previous)


def test_native_constructors_preparation_and_engram_are_reused():
    for owned, core in [(model._DECODER_INIT, native.DeepseekV4DecoderLayer.__init__),
        (model._BACKBONE_INIT, native.DeepseekV4Model.__init__),
        (model._OUTER_INIT, vl_model.DeepseekV41ForCausalLM.__init__)]:
        assert owned.__code__ is core.__code__
        assert inspect.signature(owned) == inspect.signature(core)
    assert model._DECODER_INIT.__globals__["Engram"] is native.Engram
    assert bridge.HcuV41PcpAttention._prepare_and_attn is DeepseekV4Attention._prepare_and_attn
    assert bridge.HcuV41PcpAttention.forward is rocm.DeepseekV41ROCMAiterMLAAttention.forward
    assert bridge.HcuV41PcpAttention._alloc_attn_out is rocm.DeepseekV41ROCMAiterMLAAttention._alloc_attn_out
    assert bridge.HcuV41PcpBackend.get_builder_cls() is rocm.DeepseekV41ROCMAiterMLASparseMetadataBuilder
    assert bridge.HcuV41PcpBackend.supports_pcp()


@pytest.mark.parametrize("actual", [0, 2, 5])
def test_only_actual_local_queries_enter_native_attention(monkeypatch, actual):
    obj = object.__new__(bridge.HcuV41PcpAttention)
    torch.nn.Module.__init__(obj)
    obj.swa_cache_layer = NS(prefix="swa")
    monkeypatch.setattr(bridge, "get_forward_context", lambda: NS(
        attn_metadata={"swa": NS(num_decode_tokens=actual, num_prefill_tokens=0)}))
    calls = []
    def core(self, q, kv, positions, output):
        calls.append(q.shape[0])
        output.copy_(q * 2)
    monkeypatch.setattr(rocm.DeepseekV41ROCMAiterMLAAttention, "forward_mqa", core)
    q = torch.ones(5, 2, 4)
    out = torch.full_like(q, 999)
    obj.forward_mqa(q, torch.ones(5, 4), torch.arange(5), out)
    assert calls == ([actual] if actual else [])
    assert (out[:actual] == 2).all() and (out[actual:] == 0).all()


def test_native_rocm_serial_forward_and_bf16_allocation(monkeypatch):
    obj = object.__new__(bridge.HcuV41PcpAttention)
    torch.nn.Module.__init__(obj)
    obj.indexer = object()
    obj._prepare_and_attn_fn = object()
    obj.aux_stream_list = None
    obj.padded_heads, obj.head_dim = 2, 4
    monkeypatch.setattr(rocm, "get_forward_context", lambda: NS(attn_metadata={}))
    monkeypatch.setattr(rocm, "current_platform", NS(
        enable_multi_stream_overlap=lambda streams, metadata: streams is not None))
    calls = []
    def core(self, positions, hidden, scaling=None):
        calls.append(self.aux_stream_list)
        return hidden
    monkeypatch.setattr(DeepseekV4Attention, "forward", core)
    hidden = torch.empty(5, 8, dtype=torch.bfloat16)
    assert obj.forward(torch.arange(5), hidden) is hidden
    assert calls == [None]
    out = obj._alloc_attn_out(5, hidden)
    assert out.shape == (5, 2, 4) and out.dtype == torch.bfloat16
