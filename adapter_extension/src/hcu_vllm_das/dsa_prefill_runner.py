# SPDX-License-Identifier: Apache-2.0
"""Prepare LayerSplit page IDs before the model enters attention."""
from functools import cache

import torch

from .dsa_prefill_pages import (
    PageGroup,
    PrefillPages,
    fresh_prefill_page_ids,
    fresh_swa_prefill_page_ids,
    prefill_pages_scope,
)


def _matches(config):
    from .dsa_cache_layer_split import split_config

    return split_config(config) is not None


def _page_groups(cache_config, block_sizes):
    from vllm.v1.kv_cache_interface import get_kv_cache_spec_sliding_window

    from .dsa_cache_layer_split import _source_block_sizes, _transport_regions

    # The independent planner assigns a disjoint page range to each group,
    # and offsets/owners within a group can alias in a rank-dependent way.
    # Only use this shortcut with the registered independent layout.
    groups = []
    if not _transport_regions:
        return ()
    for slot, group in enumerate(cache_config.kv_cache_groups):
        names = tuple(name for name in group.layer_names if name in _source_block_sizes)
        if not names:
            continue
        swa = tuple(name for name in names if name.endswith('.swa_cache'))
        # Don't plan a mixed group: SWA would not have the same page extents.
        if (swa and swa != names) or any(name not in _transport_regions for name in names):
            return ()
        size = block_sizes[slot]
        count = _transport_regions[names[0]][2]
        if any(_source_block_sizes[name] != size or _transport_regions[name][2] != count
               for name in names):
            return ()
        window = get_kv_cache_spec_sliding_window(group.kv_cache_spec) if swa else None
        if swa and not window:
            continue
        groups.append(PageGroup(slot, size, count, names, window))
    return tuple(groups)


def _upload(ids, device):
    return torch.tensor(ids, dtype=torch.int64, pin_memory=device.type == 'cuda').to(
        device, non_blocking=True)


def _add_swa_pages(pages, groups, batch, model_state, device):
    if not groups:
        return
    if model_state is None or not hasattr(model_state, 'decoder_replay_layers'):
        return
    replay = model_state.decoder_replay_layers
    if replay is not None and not all(hasattr(replay, name) for name in
                                      ('window', 'first_swa_prefix', 'replay_batch')):
        return
    plan = fresh_swa_prefill_page_ids(
        batch, groups,
        replay_window=replay.window if replay is not None else None,
        keep_prompt_rows=(replay is not None and
                          model_state.vllm_config.aux_output_config.enable_return_routed_experts),
    )
    if plan is None:
        return
    pages.replay_layers = replay
    for group in groups:
        full, first_replay, later_replay = plan[group.slot]
        uploaded = {}
        for ids in (full, first_replay, later_replay):
            if ids is not None and ids not in uploaded:
                uploaded[ids] = _upload(ids, device)
        for prefix in group.prefixes:
            pages.by_prefix[prefix] = uploaded[full]
            if replay is not None:
                ids = first_replay if prefix == replay.first_swa_prefix else later_replay
                if ids is not None:
                    pages.swa_replay_by_prefix[prefix] = uploaded[ids]


class LayerSplitPrefillRunnerMixin:
    def execute_model(self, scheduler_output, intermediate_tensors=None, *args, **kwargs):
        pages = None
        # Core passes dummy_run as a keyword. Positional optional arguments
        # are conservatively left to the normal transport path.
        if (not args and not kwargs.get('dummy_run', False) and not kwargs.get('is_profile', False)
                and self.kv_cache_config is not None and self.pcp_manager is not None
                and self.block_tables is not None
                and not self.block_tables.redirect_writes_to_null_block):
            groups = getattr(self, '_hcu_layer_split_page_groups', None)
            if groups is None:
                groups = _page_groups(self.kv_cache_config, self.block_tables.block_sizes)
                self._hcu_layer_split_page_groups = groups
            main_groups = tuple(group for group in groups if group.sliding_window is None)
            swa_groups = tuple(group for group in groups if group.sliding_window is not None)
            plan = fresh_prefill_page_ids(scheduler_output, main_groups) if groups else None
            if plan is not None:
                by_prefix, group_by_prefix = {}, {}
                for group in main_groups:
                    ids = _upload(plan[group.slot], self.device)
                    for prefix in group.prefixes:
                        by_prefix[prefix] = ids
                        group_by_prefix[prefix] = group.slot
                pages = PrefillPages(by_prefix, group_by_prefix)
                _add_swa_pages(pages, swa_groups, scheduler_output,
                               getattr(self, 'model_state', None), self.device)
        # The native replay view is rebuilt from the same physical block
        # tables and retains a subset of the original queries. Its global
        # main/index reads use the full plan; SWA uses its narrower read extent.
        # The scope never reaches sample_tokens()/dSpark draft execution.
        with prefill_pages_scope(pages):
            return super().execute_model(scheduler_output, intermediate_tensors, *args, **kwargs)


@cache
def layer_split_prefill_runner_cls(base):
    class LayerSplitPrefillRunner(LayerSplitPrefillRunnerMixin, base):
        pass

    return LayerSplitPrefillRunner


def wrap_profile_runner(base, config, device):
    return (layer_split_prefill_runner_cls(base) if _matches(config) else base)(config, device)


def _needs_prefill_runner(config):
    from .dsv41.profile_worker import w4a8_profile_selected

    return _matches(config) or w4a8_profile_selected(config)


def _create(config, device):
    from .dsv41.profile_worker import w4a8_profile_selected

    if w4a8_profile_selected(config):
        from .dsv41.w4a8_profile import W4A8PcpProfileRunner

        return wrap_profile_runner(W4A8PcpProfileRunner, config, device)
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    return layer_split_prefill_runner_cls(GPUModelRunner)(config, device)


def install():
    from hcu_adapter.bridges.vllm_model_runner import register_model_runner_factory

    # Compose profiling and Layer Split in one factory, leaving the Adapter
    # Worker in place for its post-load hooks and KV-cache localization.
    register_model_runner_factory('das-hcu-v41-layer-split-prefill',
                                  matches=_needs_prefill_runner, create=_create)
