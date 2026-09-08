# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import ModuleType, SimpleNamespace

from vllm_hcu.patch.platform.framework_opt import patch_kv_cache_coordinator


def _coordinator_module() -> ModuleType:
    class MambaManager:
        def cache_blocks(self, request, num_tokens, retention_interval=None):
            self.original_cache_call = (
                request,
                num_tokens,
                retention_interval,
            )

    class KVCacheCoordinator:
        def __init__(
            self,
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size,
            pcp_world_size,
            scheduler_block_size,
            hash_block_size,
            metrics_collector=None,
            num_prefill_lookahead=0,
        ):
            del (
                max_model_len,
                max_in_flight_tokens,
                enable_caching,
                enable_kv_cache_events,
                dcp_world_size,
                pcp_world_size,
                scheduler_block_size,
                hash_block_size,
                metrics_collector,
            )
            self.num_prefill_lookahead = num_prefill_lookahead
            self.eagle_group_ids = {
                index
                for index, group in enumerate(kv_cache_config.kv_cache_groups)
                if group.is_eagle_group
            }
            if use_eagle and not self.eagle_group_ids:
                self.eagle_group_ids = set(
                    range(len(kv_cache_config.kv_cache_groups))
                )

    module = ModuleType(patch_kv_cache_coordinator.TARGET_MODULE)
    module.KVCacheCoordinator = KVCacheCoordinator
    module.MambaManager = MambaManager
    return module


def _config(*groups: tuple[list[str], bool]) -> SimpleNamespace:
    return SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=names, is_eagle_group=is_eagle)
            for names, is_eagle in groups
        ]
    )


def _construct(module: ModuleType, config: SimpleNamespace):
    return module.KVCacheCoordinator(
        config,
        4096,
        128,
        True,
        True,
        False,
        1,
        1,
        64,
        64,
        num_prefill_lookahead=3,
    )


def test_mtp_indexer_group_skips_only_unmarked_all_group_eagle_fallback():
    module = _coordinator_module()
    assert patch_kv_cache_coordinator.apply_to_module(module) is True
    assert patch_kv_cache_coordinator.apply_to_module(module) is False

    combined = _config(
        (["model.layers.0.self_attn.indexer", "model.layers.61.mtp"], False)
    )
    combined_coordinator = _construct(module, combined)
    assert combined_coordinator.eagle_group_ids == set()
    assert combined_coordinator.num_prefill_lookahead == 3

    generic = _config((["model.layers.0.self_attn"], False))
    assert _construct(module, generic).eagle_group_ids == {0}

    explicit = _config(
        (["model.layers.0.self_attn.indexer", "model.layers.61.nextn"], True)
    )
    assert _construct(module, explicit).eagle_group_ids == {0}


def test_mamba_sparse_cache_retains_materialized_eagle_replay_boundary():
    module = _coordinator_module()
    assert patch_kv_cache_coordinator.apply_to_module(module) is True

    null_block = SimpleNamespace(is_null=True, block_hash=None)
    replay_block = SimpleNamespace(
        is_null=False,
        block_hash=None,
        block_hash_num_tokens=None,
    )

    class BlockPool:
        def cache_full_blocks(self, **kwargs):
            assert kwargs["num_cached_blocks"] == 6
            assert kwargs["num_full_blocks"] == 7
            assert kwargs["block_size"] == 576
            kwargs["blocks"][6].block_hash = "group-0-replay"
            kwargs["blocks"][6].block_hash_num_tokens = 4032

    manager = module.MambaManager()
    manager.drop_eagle_checkpoint_block = True
    manager.mamba_cache_mode = "align"
    manager.cache_hit_alignment_tokens = 576
    manager.block_size = 576
    manager.kv_cache_group_id = 0
    manager.block_pool = BlockPool()
    manager.cached_blocks_this_step = set()
    manager.req_to_blocks = {"request": [null_block] * 6 + [replay_block]}
    request = SimpleNamespace(request_id="request", num_prompt_tokens=5050)

    manager.cache_blocks(request, 4032, retention_interval=0)

    assert manager.original_cache_call == (request, 4032, 0)
    assert replay_block.block_hash_num_tokens == 4032
    assert manager.cached_blocks_this_step == {"group-0-replay"}
