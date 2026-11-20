"""Real official class/seam wiring; CPU only, not device acceptance."""
from types import SimpleNamespace as NS

import pytest
import torch

from hcu_adapter.kernels import pcp_fork
from hcu_vllm_das import v41_pcp as binding
from hcu_vllm_das.models import v41_pcp as model
from test_pcp_merge_contracts import config


@pytest.fixture
def split(monkeypatch):
    monkeypatch.setenv('VLLM_HCU_DSV41_PCP_ATTENTION', '1')
    monkeypatch.setenv('HCU_VLLM_PCP_IMPL', 'native')
    monkeypatch.setenv('HCU_VLLM_FP4_INDEXER_IMPL', 'stock')
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    cfg = config()
    cfg.additional_config = {'hcu_layer_split': {'enabled': True}}
    cfg.model_config.hf_config.kv_source_layer_ids = [2, 8, 14, 20]
    return cfg


def test_actual_model_selector_uses_plugin_subclass_only_when_enabled(split, monkeypatch):
    from hcu_vllm_das.v41_pcp_layer_split import HcuV41PcpAttention
    assert model.select_attention(split) is HcuV41PcpAttention
    from hcu_vllm_das.v41_pcp_layer_split import LayerSplitAttentionMixin
    assert HcuV41PcpAttention.__bases__ == (LayerSplitAttentionMixin, pcp_fork.build_classes()['attention'])
    assert HcuV41PcpAttention.forward is pcp_fork.build_classes()['attention'].forward
    split.additional_config = {}
    assert model.select_attention(split) is pcp_fork.build_classes()['attention']


@pytest.mark.parametrize('name,value', [
    ('HCU_VLLM_PCP_IMPL', 'stock'), ('HCU_DISABLE_OPS', 'vllm_pcp'),
    ('VLLM_HCU_DSV41_PCP_ATTENTION', '0'),
])
def test_incompatible_policy_cannot_silently_select_unsharded_model(split, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match='Layer Split'):
        model.select_attention(split)


def test_fp4_policy_cannot_bypass_disabled_pcp(split, monkeypatch):
    from hcu_vllm_das.adapter_api import adapter_module

    fp4 = adapter_module('fp4_indexer')
    monkeypatch.setattr(fp4, 'selected', lambda: True)
    monkeypatch.setenv('HCU_DISABLE_OPS', 'vllm_pcp')
    with pytest.raises(ValueError, match='native PCP'):
        model.select_attention(split)


def test_layer_split_requires_real_pcp_and_source_layers(split):
    split.parallel_config.prefill_context_parallel_size = 1
    with pytest.raises(ValueError, match='PCP>1'):
        model.select_attention(split)


def test_transfer_hook_keeps_official_guards_and_requires_owner_protocol(split, monkeypatch):
    binding.bind_adapter_classes()
    split.kv_transfer_config = NS(kv_connector='HcuLayerSplitMooncakeConnector',
        kv_connector_module_path='hcu_vllm_das.mooncake_layer_split', kv_role='kv_producer')
    assert binding.validate_config(split)
    assert pcp_fork.validate_config(split)  # Also called inside base constructors.
    split.speculative_config = NS(method='eagle')
    with pytest.raises(ValueError, match='supports DSpark speculative decoding only'):
        pcp_fork.validate_config(split)
    split.speculative_config = None
    split.kv_transfer_config.kv_role = 'kv_consumer'
    with pytest.raises(ValueError, match='producer'):
        pcp_fork.validate_config(split)
    split.kv_transfer_config.kv_role = 'kv_producer'
    split.additional_config = {}
    with pytest.raises(ValueError, match='replicated-cache'):
        pcp_fork.validate_config(split)


def test_empty_rank_collectives_precede_official_query_filter(split, monkeypatch):
    from hcu_vllm_das import v41_pcp_layer_split as impl
    from vllm.models.deepseek_v41.amd import rocm
    events = []
    obj = object.__new__(impl.HcuV41PcpAttention)
    torch.nn.Module.__init__(obj)
    obj.swa_cache_layer = NS(prefix='swa')
    metadata = NS(attn_metadata={'swa': NS(num_decode_tokens=0, num_prefill_tokens=0)})
    monkeypatch.setattr(impl, 'get_forward_context', lambda: metadata)
    monkeypatch.setattr(binding, 'get_forward_context', lambda: metadata)
    monkeypatch.setattr(obj, '_broadcast_compressed_cache', lambda: events.append('main'))
    monkeypatch.setattr(obj, '_broadcast_swa_cache', lambda: events.append('swa'))
    monkeypatch.setattr(rocm.DeepseekV41ROCMAiterMLAAttention, 'forward_mqa',
                        lambda *args: pytest.fail('empty rank ran native attention'))
    q = torch.ones(2, 1, 4)
    output = torch.full_like(q, 999)
    obj.forward_mqa(q, q, torch.arange(2), output)
    assert events == ['main', 'swa']
    assert torch.count_nonzero(output) == 0


def test_new_profile_worker_inherits_cache_localization():
    from hcu_adapter.bridges.vllm_worker import Worker
    from hcu_vllm_das.dsv41.worker import HcuDSV41PcpWorker
    assert HcuDSV41PcpWorker.initialize_from_config is Worker.initialize_from_config


def test_replay_auto_and_official_deepgemm_are_admitted_but_megamoe_refuses(split, monkeypatch):
    from compressed_tensors.quantization import QuantizationStrategy
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from hcu_vllm_das.quantization.channel_fp8_moe import (
        create_channel_fp8_moe_method, HcuDSV41ChannelFP8MoEMethod)
    parallel = FusedMoEParallelConfig(tp_size=1, pcp_size=8, dp_size=1, ep_size=8,
        tp_rank=0, pcp_rank=0, dp_rank=0, ep_rank=0, sp_size=1,
        use_ep=True, all2all_backend='deepep_high_throughput', enable_eplb=False)
    moe = FusedMoEConfig(num_experts=384, experts_per_token=6, hidden_dim=5120,
        intermediate_size=2304, num_local_experts=48, num_logical_experts=384,
        activation=MoEActivation.SILU, device='cpu', routing_method=RoutingMethodType.Default,
        moe_parallel_config=parallel, in_dtype=torch.bfloat16, moe_backend='auto',
        max_num_tokens=32768)
    weights = NS(strategy=QuantizationStrategy.CHANNEL, num_bits=8,
                 type='float', dynamic=False, block_structure=None)
    inputs = NS(strategy=QuantizationStrategy.TOKEN, num_bits=8, type='float', dynamic=True)
    split.cache_config.swa_bounded_replay = True
    split.parallel_config.expert_placement_strategy = 'linear'
    monkeypatch.setattr('vllm.config.get_current_vllm_config', lambda: split)
    method = create_channel_fp8_moe_method(weights, inputs, moe)
    assert type(method) is HcuDSV41ChannelFP8MoEMethod
    assert method._backend_mode == 'ht' and method._config is split
    from hcu_vllm_das.moe import dsv41_megamoe
    monkeypatch.setattr(dsv41_megamoe, 'get_current_vllm_config', lambda: split)
    moe.moe_backend = 'deep_gemm'
    from hcu_vllm_das.moe.dsv41_deepgemm import HcuDSV41DeepGemmMethod
    assert type(create_channel_fp8_moe_method(weights, inputs, moe)) is HcuDSV41DeepGemmMethod
    moe.moe_backend = 'deep_gemm_mega_moe'
    with pytest.raises(ValueError, match='does not support SWA bounded replay'):
        create_channel_fp8_moe_method(weights, inputs, moe)
