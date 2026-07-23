# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import ModuleType, SimpleNamespace

from vllm_hcu.patch.platform.framework_opt import patch_kv_cache_coordinator


def _coordinator_module() -> ModuleType:
    class KVCacheCoordinator:
        def __init__(
            self,
            kv_cache_config,
            max_model_len,
            max_num_batched_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size,
            pcp_world_size,
            scheduler_block_size,
            hash_block_size,
            metrics_collector=None,
        ):
            del (
                max_model_len,
                max_num_batched_tokens,
                enable_caching,
                enable_kv_cache_events,
                dcp_world_size,
                pcp_world_size,
                scheduler_block_size,
                hash_block_size,
                metrics_collector,
            )
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
    )


def test_mtp_indexer_group_skips_only_unmarked_all_group_eagle_fallback():
    module = _coordinator_module()
    assert patch_kv_cache_coordinator.apply_to_module(module) is True
    assert patch_kv_cache_coordinator.apply_to_module(module) is False

    combined = _config(
        (["model.layers.0.self_attn.indexer", "model.layers.61.mtp"], False)
    )
    assert _construct(module, combined).eagle_group_ids == set()

    generic = _config((["model.layers.0.self_attn"], False))
    assert _construct(module, generic).eagle_group_ids == {0}

    explicit = _config(
        (["model.layers.0.self_attn.indexer", "model.layers.61.nextn"], True)
    )
    assert _construct(module, explicit).eagle_group_ids == {0}
