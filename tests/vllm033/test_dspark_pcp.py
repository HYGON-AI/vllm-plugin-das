"""PCP DSpark admission, rollback rows and replicated draft cache placement."""
import copy
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from hcu_vllm_das import dsa_cache_layout as placement
from hcu_vllm_das import dsa_cache_pools as pools
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs
from test_swa_cache_layer_split import config, real_groups


def dspark_config():
    cfg = config()
    cfg.parallel_config.enable_expert_parallel = True
    cfg.parallel_config.all2all_backend = 'deepep_high_throughput'
    cfg.parallel_config.enable_dbo = cfg.parallel_config.enable_eplb = False
    cfg.kernel_config = NS(monokernel=False, enable_jit_warmup=False)
    cfg.model_config.hf_config.sliding_window = 128
    cfg.model_config.max_model_len = 16384
    cfg.model_config.original_max_model_len = 65536
    cfg.max_in_flight_tokens = 16384
    cfg.num_prefill_lookahead_tokens = 5
    cfg.cache_config.swa_bounded_replay = True
    cfg.scheduler_config.max_num_seqs = 128
    cfg.scheduler_config.max_num_batched_tokens = 32768
    cfg.speculative_config = NS(method='dspark', num_speculative_tokens=5,
        enable_adaptive_verification=False, num_speculative_tokens_per_batch_size=None,
        draft_parallel_config=NS(tensor_parallel_size=1),
        target_parallel_config=copy.copy(cfg.parallel_config),
        draft_model_config=NS(enforce_eager=True, hf_config=NS(
            model_type='deepseek_v41', sliding_window=128,
            layer_types=['sliding_attention'] * 3)))
    return cfg


def test_native_loader_does_not_inherit_target_layer_split(monkeypatch):
    from hcu_vllm_das import dspark_agrs, dspark_graph, layer_split_config

    cfg = dspark_config()
    expert_layout = {'num_redundant_experts': 32}
    cfg.additional_config = {
        layer_split_config.KEY: {'enabled': True},
        'hcu_static_eplb': expert_layout,
    }
    monkeypatch.setattr(dspark_agrs, '_owned_quant_kind', lambda _: 'ct_w4a8')
    target, draft = object(), object()

    def native_loader(model, loader_cfg):
        assert model is target and loader_cfg is not cfg
        assert loader_cfg.parallel_config is cfg.parallel_config
        assert loader_cfg.additional_config == {'hcu_static_eplb': expert_layout}
        draft_cfg = copy.copy(loader_cfg)
        draft_cfg.parallel_config = copy.copy(loader_cfg.parallel_config)
        draft_cfg.parallel_config.prefill_context_parallel_size = 1
        layer_split_config._validate_config(draft_cfg)
        return draft

    assert dspark_graph._load_with_scope(native_loader, target, cfg) is draft
    assert layer_split_config.layer_split_enabled(cfg)
    assert cfg.parallel_config.prefill_context_parallel_size == 8


def test_draft_swa_is_replicated_and_cannot_merge_with_target_scratch(monkeypatch):
    cfg = dspark_config()
    cfg.additional_config = {'hcu_layer_split': {'enabled': True}}
    source = next(spec for group in real_groups() for name, spec in placement._specs(group)
        if name.endswith('.swa_cache'))
    target_name, draft_name = 'model.layers.0.attn.swa_cache', 'model.layers.40.attn.swa_cache'
    specs = {target_name: source, draft_name: replace(source, is_draft=True)}
    mixed = KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs.from_specs(specs))
    with patch.object(pools.DefaultKVCacheConfigBuilder, 'get_kv_cache_groups', return_value=[mixed]):
        groups = pools.IndependentDSACacheBuilder().get_kv_cache_groups(cfg, specs)
    assert len(groups) == 2
    assert groups[0].layer_names == [target_name] and not groups[0].is_eagle_group
    assert groups[1].layer_names == [draft_name] and groups[1].is_eagle_group
    offsets, pitches, costs, demands = pools.pool_geometry(cfg, groups)
    for rank in range(8):
        assert draft_name in offsets[rank]
    assert costs[1] >= pitches[draft_name]
    assert placement._owner(draft_name, (2, 8, 14, 20), 8, is_draft=True) is None
    assert not torch.cuda.is_initialized()


def test_native_pool_admission_retains_speculative_slots_and_recycles_after_rejection(monkeypatch):
    from test_dsa_cache_pools import manager_for, request
    cfg = dspark_config()
    cfg.additional_config = {'hcu_layer_split': {'enabled': True}}
    groups = pools.merge_swa_groups(real_groups())
    swa = next(spec for group in groups for name, spec in placement._specs(group)
        if name.endswith('.swa_cache'))
    drafts = {f'model.layers.{i}.attn.swa_cache': replace(swa, is_draft=True, extra_retained_tokens=5)
              for i in range(40, 43)}
    groups.append(KVCacheGroupSpec(list(drafts), UniformTypeKVCacheSpecs.from_specs(drafts), is_eagle_group=True))
    plan = pools.materialize(cfg, groups, (128*2**20)//pools.unit_cost(cfg, groups), rank=0)
    manager = manager_for(plan, pcp=8)
    initial = [p.get_num_free_blocks() for p in manager.block_pool.pools]
    req = request('verification', 127)
    assert manager.allocate_slots(req, 127, num_lookahead_tokens=5) is not None
    req.num_computed_tokens = 127
    req.append_output_token_ids([17])
    for accepted in (0, 5, 2):
        req.spec_token_ids = [17] * 5
        assert manager.allocate_slots(req, 6, num_lookahead_tokens=5) is not None
        req.num_computed_tokens += accepted + 1
        req.append_output_token_ids([17] * (accepted + 1))
        manager.coordinator.new_step_starts()
    draft_pool = manager.block_pool.pools[-1]
    held = draft_pool.get_new_blocks(draft_pool.get_num_free_blocks())
    before = [p.get_num_free_blocks() for p in manager.block_pool.pools]
    assert manager.allocate_slots(request('draft_exhausted', 127), 127, num_lookahead_tokens=5) is None
    assert before == [p.get_num_free_blocks() for p in manager.block_pool.pools]
    draft_pool.free_blocks(held)
    manager.free(req)
    assert [p.get_num_free_blocks() for p in manager.block_pool.pools] == initial


@pytest.mark.parametrize("window,layer_types,expected", [
    (128, ["sliding_attention"], True),
    (256, ["sliding_attention"], False),
    (128, ["full_attention"], False),
    (0, ["sliding_attention"], False),
])
def test_replay_checks_the_draft_window_and_attention_kind(window, layer_types, expected):
    from hcu_vllm_das.models.v41_pcp import _draft_fits_replay

    cfg = dspark_config()
    hf = cfg.speculative_config.draft_model_config.hf_config
    hf.sliding_window, hf.layer_types = window, layer_types
    assert _draft_fits_replay(cfg, 128) is expected
