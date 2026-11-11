# SPDX-License-Identifier: Apache-2.0
"""Plugin-side binding for the adapter-owned V4.1 PCP attention.

The PCP backends, indexer and attention classes moved to
`hcu_adapter.kernels.pcp_fork`: PCP preparation and its admission rules are
upstream-defined behaviour, so the adapter owns them. The plugin keeps only what
is genuinely private to it:

* the FP8 reference scorer (`v41_indexer.indexer`) that PCP scoring reuses;
* the Engram config validator for the private host-table adapter;
* `ModelRegistry` publication of the owned model class.

Those two callables are injected into the adapter at activation, because the
adapter may not import the plugin (direction rule: plugin -> adapter only).
"""
from __future__ import annotations

from hcu_adapter.kernels import pcp_fork

validate_config = pcp_fork.validate_config
is_v41 = pcp_fork.is_v41
select_attention = pcp_fork.select_attention
build_classes = pcp_fork.build_classes
PcpAttentionError = pcp_fork.PcpAttentionError

_bound = False


_CLASS_NAMES = {
    "HcuV41PcpAttention": "attention",
    "HcuV41PcpIndexer": "indexer",
    "HcuV41PcpSparseIndexer": "sparse_indexer",
    "HcuV41PcpBackend": "backend",
    "HcuV41PcpSwaBackend": "swa_backend",
}


def __getattr__(name):
    """Keep the historical public surface of this module resolvable.

    The PCP classes used to be defined here; they now live in the adapter and are
    built lazily. `reference` is this plugin's own FP8 scorer. The first access
    resolves and caches into the module namespace, so `patch.object(v41_pcp, ...)`
    keeps working: patching the module dict must win over a later lookup.
    """
    key = _CLASS_NAMES.get(name)
    if key is not None:
        bind_adapter_classes()
        value = build_classes()[key]
    elif name == "reference":
        from . import v41_indexer

        value = v41_indexer
    elif name == "get_forward_context":
        # Historically re-exported here; the scoring path is the only consumer.
        from vllm.forward_context import get_forward_context

        value = get_forward_context
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def bind_adapter_classes() -> None:
    """Inject the plugin-owned scorer and Engram validator into the adapter.

    Idempotent per process: both setters refuse a second, different owner so a
    mid-flight takeover cannot silently change scoring.
    """
    global _bound
    if _bound:
        return
    from . import v41_indexer as reference
    from .v41_engram import validate_engram_config

    pcp_fork.set_indexer_reference(reference.indexer)
    pcp_fork.set_engram_validator(validate_engram_config)
    # Resolve through this module so a patch on `v41_pcp.get_forward_context`
    # (the historical seam) still reaches the adapter-owned attention class.
    pcp_fork.set_forward_context_resolver(lambda: globals()["get_forward_context"]())
    _bound = True


def attention_class():
    """Return the adapter-owned PCP attention class for this process."""
    bind_adapter_classes()
    return build_classes()["attention"]
