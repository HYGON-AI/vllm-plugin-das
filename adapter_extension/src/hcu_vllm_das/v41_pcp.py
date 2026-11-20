# SPDX-License-Identifier: Apache-2.0
"""Plugin-side binding for the adapter-owned V4.1 PCP attention.

The PCP backends, indexer and attention classes moved to
`hcu_adapter.kernels.pcp_fork`: PCP preparation and its admission rules are
upstream-defined behaviour, so the adapter owns them. The plugin keeps only what
is genuinely private to it:

* the FP8 reference scorer (`v41_indexer.indexer`) that PCP scoring reuses;
* the Engram config validator for the private host-table adapter;
* Layer Split attention, cache placement and its transfer contract;
* `ModelRegistry` publication of the owned model class.

The scorer and validators are injected into the adapter at activation, because the
adapter may not import the plugin (direction rule: plugin -> adapter only).
"""
from __future__ import annotations

import sys

from hcu_adapter.kernels import pcp_fork
from .layer_split_config import layer_split_enabled as layer_split_selected

def validate_config(cfg):
    validate_layer_split_policy(cfg)
    if layer_split_selected(cfg):
        from .dsa_cache_layer_split import split_config
        if split_config(cfg) is None:
            raise ValueError('Layer Split requires PCP>1 and KV source layers')
    return pcp_fork.validate_config(cfg)


def validate_layer_split_policy(cfg):
    if not layer_split_selected(cfg):
        return
    from .environment import _flag
    if not _flag('VLLM_HCU_DSV41_PCP_ATTENTION', default=False):
        raise ValueError('Layer Split requires VLLM_HCU_DSV41_PCP_ATTENTION=1')
    if pcp_fork.selected_mode() != 'native':
        raise ValueError('Layer Split requires native PCP; stock/kill-switch cannot shard caches')


def _validate_transfer(cfg):
    # Keep the official rejection unless the extension supplies its own protocol.
    if not layer_split_selected(cfg):
        raise ValueError('KV transfer requires a separate replicated-cache contract')
    validate_layer_split_policy(cfg)
    from .dsa_cache_layer_split import validate_layer_split_transfer
    validate_layer_split_transfer(cfg)

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
    pcp_fork.set_kv_transfer_validator(_validate_transfer)
    from .inverse_rope_policy import selected_mode
    if selected_mode() == "hcu":
        from .inverse_rope_pcp import prepare_forward_mqa
        setter = getattr(pcp_fork, "set_forward_mqa_provider", None)
        if not callable(setter):
            raise RuntimeError("Upgrade adapter-das for the PCP forward_mqa provider")
        setter(prepare_forward_mqa)
    # Resolve through this module so a patch on `v41_pcp.get_forward_context`
    # (the historical seam) still reaches the adapter-owned attention class.
    # Attribute lookup must trigger __getattr__ on a cold module; direct globals
    # lookup only worked after a replay had already materialized this export.
    pcp_fork.set_forward_context_resolver(
        lambda: getattr(sys.modules[__name__], "get_forward_context")())
    # Older adapters keep the stock path on the same Plugin release.
    q_rope_selected = getattr(pcp_fork, "q_rope_selected", None)
    if q_rope_selected is not None and q_rope_selected():
        from .pcp_q_rope import initialize

        pcp_fork.set_q_rope_provider(initialize)
    elif hasattr(pcp_fork, "require_q_rope_healthy"):
        pcp_fork.set_q_rope_provider(None)
    _bound = True


def attention_class(cfg=None, *, dspark=False):
    """Return the adapter-owned PCP attention class for this process."""
    bind_adapter_classes()
    validate_layer_split_policy(cfg)
    if layer_split_selected(cfg):
        from .dsa_cache_layer_split import apply_kv_source_write_suppression
        from .v41_pcp_layer_split import HcuV41PcpAttention, dspark_attention_class
        apply_kv_source_write_suppression()
        return dspark_attention_class() if dspark else HcuV41PcpAttention
    if dspark:
        from .pcp_dspark import HcuV41PcpDSparkAttention
        return HcuV41PcpDSparkAttention
    return build_classes()["attention"]
