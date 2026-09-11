# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import torch

from vllm_hcu.patch.platform.framework_opt import patch_kv_cache_coordinator
from vllm_hcu.patch.worker.framework_opt import patch_mamba_hybrid_model_state


def _coordinator_module() -> ModuleType:
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


def test_mamba_resume_seed_uses_hybrid_manager_block_size():
    class MambaHybridModelState:
        def __init__(self):
            self._align_mode = True
            self.cache_config = SimpleNamespace(block_size=64)
            self._mamba_spec = None
            self._mamba_state_idx_gpu = torch.full((4,), -99, dtype=torch.int32)
            self.state_idx_seen = None

        def add_request(self, req_index, new_req_data):
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1)
                // self.cache_config.block_size
            )

        def _get_mamba_group_info(self, kv_cache_config):
            del kv_cache_config
            self._mamba_spec = SimpleNamespace(block_size=832)
            return [0], self._mamba_spec

        def preprocess_state(
            self,
            input_batch,
            block_tables,
            kv_cache_config,
            num_computed_tokens,
        ):
            del input_batch, block_tables, kv_cache_config, num_computed_tokens
            self.state_idx_seen = self._mamba_state_idx_gpu.clone()

    module = ModuleType(patch_mamba_hybrid_model_state.TARGET_MODULE)
    module.MambaHybridModelState = MambaHybridModelState
    assert patch_mamba_hybrid_model_state.apply_to_module(module) is True
    assert patch_mamba_hybrid_model_state.apply_to_module(module) is False

    state = module.MambaHybridModelState()
    resumed = SimpleNamespace(num_computed_tokens=4992)
    earlier = SimpleNamespace(num_computed_tokens=1664)
    state.add_request(1, earlier)
    state.add_request(2, resumed)
    assert state._mamba_state_idx_gpu[2].item() == 77

    state.preprocess_state(None, (), object(), torch.tensor([4992]))
    assert state.state_idx_seen[1].item() == 1
    assert state.state_idx_seen[2].item() == 5

    state.add_request(3, resumed)
    assert state._mamba_state_idx_gpu[3].item() == 5

    non_aligned = module.MambaHybridModelState()
    non_aligned._align_mode = False
    non_aligned.add_request(0, resumed)
    assert non_aligned._mamba_state_idx_gpu[0].item() == 77
