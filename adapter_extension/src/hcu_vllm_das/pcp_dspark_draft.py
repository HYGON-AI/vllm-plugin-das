# SPDX-License-Identifier: Apache-2.0
"""Keep replicated draft attention separate from its PCP-owned MoE weights."""
from __future__ import annotations

import copy

from .dspark_capacity import token_capacity
from .dsv41.profile_worker import HT_FIXED_CAPACITY_KEY
from .layer_split_config import KEY as LAYER_SPLIT_KEY

_KEY = '_hcu_dsv41_pcp_dspark_draft'


def draft_config(cfg):
    """Mark only the private native DSpark draft construction configuration.

    DFlashSpeculator sets the draft's attention PCP width to one. Its MoE
    still uses the existing PCP/EP group and its partitioned expert weights.
    Retain that attention setting and record the distinct MoE ownership.
    """
    spec = cfg.speculative_config
    target = spec.target_parallel_config
    parallel = cfg.parallel_config
    if (spec.method != 'dspark' or not cfg.model_config.enforce_eager
            or not spec.draft_model_config.enforce_eager
            or getattr(spec, 'enable_adaptive_verification', False)
            or (parallel.tensor_parallel_size, parallel.data_parallel_size,
                parallel.prefill_context_parallel_size,
                parallel.decode_context_parallel_size) != (1, 1, 1, 1)
            or (target.tensor_parallel_size, target.data_parallel_size,
                target.pipeline_parallel_size) != (1, 1, 1)
            or target.prefill_context_parallel_size <= 1
            or not target.enable_expert_parallel
            or parallel.all2all_backend != 'deepep_high_throughput'
            or target.all2all_backend != parallel.all2all_backend):
        raise ValueError('PCP DSpark draft requires eager TP1/DP1 and target PCP/EP HT')
    capacity = token_capacity(cfg)
    result = copy.copy(cfg)
    result.additional_config = dict(cfg.additional_config or {})
    # The target's calibrated physical slots do not describe draft experts.
    result.additional_config.pop('hcu_static_eplb', None)
    result.additional_config.pop(LAYER_SPLIT_KEY, None)
    previous = result.additional_config.get(_KEY)
    previous_fixed_ht = (
        previous.get('target_fixed_ht', False) if isinstance(previous, dict) else False
    )
    target_fixed_ht = result.additional_config.pop(HT_FIXED_CAPACITY_KEY, previous_fixed_ht)
    result.additional_config[_KEY] = {
        'pcp_size': target.prefill_context_parallel_size,
        'capacity': vars(capacity),
        'target_fixed_ht': target_fixed_ht is True,
    }
    return result


def needs_isolated_ht_buffer(cfg):
    extra = getattr(cfg, 'additional_config', None) or {}
    marker = extra.get(_KEY)
    return isinstance(marker, dict) and marker.get('target_fixed_ht') is True


def moe_pcp_size(cfg):
    """Return the admitted MoE width while retaining draft attention PCP1."""
    extra = getattr(cfg, 'additional_config', None)
    marker = extra.get(_KEY) if isinstance(extra, dict) else None
    if marker is None:
        return cfg.parallel_config.prefill_context_parallel_size
    expected = draft_config(cfg).additional_config[_KEY]
    if marker != expected:
        raise ValueError('PCP DSpark draft ownership/capacity changed after construction')
    return expected['pcp_size']
