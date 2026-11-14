# SPDX-License-Identifier: Apache-2.0
"""Select the rank-local memory-profile worker through the HCU platform seam."""
from __future__ import annotations

from ..environment import dsv41_profile_worker_enabled

WORKER = "hcu_vllm_das.dsv41.worker.HcuDSV41PcpWorker"


def local_token_bound(global_tokens, max_reqs, pcp_size):
    # Core gives a prefill rank two ceil(query_len / (2 * PCP)) chunks.
    # Two extra rows per request cover their rounding and replicated decodes.
    return min(global_tokens, (global_tokens + pcp_size - 1) // pcp_size + 2 * max_reqs)


def profile_token_capacity(config, pcp_size, default_capacity):
    """Size per-rank storage for the worker selected during config resolution."""
    if config.parallel_config.worker_cls != WORKER:
        return default_capacity
    scheduler = config.scheduler_config
    return local_token_bound(scheduler.max_num_batched_tokens, scheduler.max_num_seqs, pcp_size)


def _select(worker_cls: str, config) -> str:
    """Replace only Core's own default; a deliberate choice stays untouched."""
    if dsv41_profile_worker_enabled():
        from .worker import uses_pcp_channel_fp8

        if uses_pcp_channel_fp8(config):
            return WORKER
    return worker_cls


def install(platform) -> bool:
    platform.register_worker_selector(_select)
    return True


__all__ = ["WORKER", "install"]
