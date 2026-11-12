# SPDX-License-Identifier: Apache-2.0
"""Install the rank-local PCP memory-profile worker through Core's own hook.

Core resolves ``parallel_config.worker_cls`` as an opaque class name and offers
no worker registry; the platform's ``check_and_update_config`` is the only
supported place a platform may choose it. Wrap that classmethod so the switch
applies to the config object Core is about to build the workers from.
"""
from __future__ import annotations

from ..environment import dsv41_profile_worker_enabled

WORKER = "hcu_vllm_das.dsv41.worker.HcuDSV41PcpWorker"
# Core's own default, and the Adapter's post-load worker that HcuPlatform
# substitutes for it. Both are platform choices our subclass may replace; any
# other value is an explicit operator choice and is left alone.
_DEFAULT_WORKERS = frozenset({
    "vllm.v1.worker.gpu_worker.Worker",
    "hcu_adapter.bridges.vllm_worker.Worker",
})
_UNSET = "auto"
_installed: dict[int, object] = {}


def local_token_bound(global_tokens, max_reqs, pcp_size):
    # Core gives a prefill rank two ceil(query_len / (2 * PCP)) chunks.
    # Two extra rows per request cover their rounding and replicated decodes.
    return min(global_tokens, (global_tokens + pcp_size - 1) // pcp_size + 2 * max_reqs)


def _select(worker_cls: str, config) -> str:
    """Replace only Core's own default; a deliberate choice stays untouched."""
    from .worker import uses_pcp_channel_fp8

    if worker_cls != _UNSET and worker_cls not in _DEFAULT_WORKERS:
        return worker_cls
    return WORKER if uses_pcp_channel_fp8(config) else worker_cls


def install(platform) -> bool:
    """Wrap this platform class once; returns whether the hook is owned here.

    ``platform`` may be the class itself or Core's ``current_platform``
    instance. A stand-in platform (tests, foreign hosts) that cannot accept
    attributes leaves the switch unowned instead of failing activation.
    """
    owner = platform if isinstance(platform, type) else type(platform)
    if _installed.get(id(owner)) is owner:
        return bool(getattr(owner, "_hcu_dsv41_profile_worker", False))
    # Resolve through getattr so inherited hooks are wrapped too; a classmethod
    # read off the class is bound, so unwrap it to one plain function.
    bound = getattr(owner, "check_and_update_config", None)
    previous = getattr(bound, "__func__", None)
    if not callable(previous):
        return False

    def check_and_update_config(cls, vllm_config):
        previous(cls, vllm_config)
        # Core and HcuPlatform both finish choosing a worker inside this hook,
        # so only their defaults remain for _select to replace.
        parallel = vllm_config.parallel_config
        if dsv41_profile_worker_enabled():
            parallel.worker_cls = _select(parallel.worker_cls, vllm_config)

    check_and_update_config.__wrapped__ = previous
    try:
        setattr(owner, "check_and_update_config", classmethod(check_and_update_config))
        setattr(owner, "_hcu_dsv41_profile_worker", True)
    except (AttributeError, TypeError):
        return False
    _installed[id(owner)] = owner
    return True


__all__ = ["WORKER", "install"]
