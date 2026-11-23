# SPDX-License-Identifier: Apache-2.0
"""Route only KV backing allocation through the selected worker's Fabric pool.

The optional HCU integration wraps the existing allocation entry points; their
packing, sharing, dtype, zero initialization and return values are unchanged.
No torch factory or process-wide allocator is replaced. A ContextVar selects
the worker during initialization, so unrelated threads/calls use the original
allocation path even after the wrappers have been installed.
"""

import functools
import inspect
import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

logger = logging.getLogger(__name__)
_ACTIVE_SCOPE = ContextVar("mooncake_kv_allocation_scope", default=None)
_HOOK_LOCK = threading.Lock()
_MARKER = "_mooncake_kv_allocation_wrapper"


@dataclass
class _AllocationScope:
    runner: object
    get_pool: object
    use_v2_model_runner: bool
    allocation_calls: int = 0


def _install_hook(owner, name, *, use_v2_model_runner, bound_runner=False):
    """Wrap the MRV2 module function or an MRV1 runner's own callable."""
    original = getattr(owner, name, None)
    if not callable(original):
        raise RuntimeError(f"Fabric KV allocation entry point is missing: {name}")
    if getattr(original, _MARKER, False):
        return
    signature = inspect.signature(original)
    if bound_runner:
        supported = inspect.ismethod(original) and original.__self__ is owner
    else:
        supported = "device" in signature.parameters
    if not supported:
        raise RuntimeError(f"Unsupported Fabric KV allocation signature: {name}")

    @functools.wraps(original)
    def allocate(*args, **kwargs):
        scope = _ACTIVE_SCOPE.get()
        if scope is None or scope.use_v2_model_runner != use_v2_model_runner:
            return original(*args, **kwargs)
        # MRV1 hooks live on the runner instance, including the inherited
        # static uniform allocator. Another runner must not borrow this pool.
        if not use_v2_model_runner and scope.runner is not owner:
            return original(*args, **kwargs)
        if bound_runner:
            device = owner.device
        else:
            arguments = signature.bind(*args, **kwargs).arguments
            device = arguments["device"]
        # The saved entry points receive explicit cuda:N devices. Never let a
        # device mismatch silently allocate a transfer buffer outside the pool.
        if str(device) != str(scope.runner.device):
            raise RuntimeError(
                f"Fabric KV allocation device mismatch: {device} != "
                f"{scope.runner.device}"
            )
        with scope.get_pool().use():
            result = original(*args, **kwargs)
        scope.allocation_calls += 1
        return result

    setattr(allocate, _MARKER, True)
    setattr(owner, name, allocate)


def _install_allocation_hooks(runner, use_v2_model_runner):
    with _HOOK_LOCK:
        if use_v2_model_runner:
            from vllm.v1.worker.gpu import attn_utils

            # init_kv_cache resolves this function in attn_utils' globals;
            # the alias imported by gpu.model_runner continues to work.
            _install_hook(attn_utils, "_allocate_kv_cache",
                          use_v2_model_runner=True)
        else:
            _install_hook(
                runner, "_allocate_kv_cache_tensors",
                use_v2_model_runner=False, bound_runner=True,
            )
            # MRV1's alternate cross-layer layout owns one contiguous KV
            # backing. It must not bypass Fabric if selected by the connector.
            _install_hook(
                runner, "allocate_uniform_kv_caches", use_v2_model_runner=False,
            )


@contextmanager
def kv_cache_allocation_scope(runner, get_pool, *, use_v2_model_runner):
    """Select an owner without putting the entire initialization in MemPool."""
    _install_allocation_hooks(runner, use_v2_model_runner)
    scope = _AllocationScope(runner, get_pool, use_v2_model_runner)
    token = _ACTIVE_SCOPE.set(scope)
    try:
        yield
        if not scope.allocation_calls:
            raise RuntimeError(
                "Fabric KV allocation hook was not reached; refusing to use "
                "unverified KV storage. Check the installed model-runner version."
            )
        logger.info(
            "Mooncake Fabric KV backing allocation finished on %s: %d allocator "
            "calls; auxiliary initialization used the original allocator",
            runner.device, scope.allocation_calls,
        )
    finally:
        _ACTIVE_SCOPE.reset(token)
