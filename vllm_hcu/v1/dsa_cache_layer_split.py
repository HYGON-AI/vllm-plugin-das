# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DSA KV/indexer cache layer split across Prefill-CP ranks.

Source of truth: SGLang PR #29421
(``LayerSplitDSATokenToKVPool`` / ``LayerSplitIndexKeyCache`` /
``get_layer_shard_range`` / ``get_layer_owner`` /
``get_glm_dsa_layer_split_effective_num_layers``).

SGLang contract mirrored here:
- Contiguous even layer ownership across Prefill-CP (PCP) ranks.
- Only the owning rank commits durable MLA latent / indexer K.
- Reads go through owner-broadcast into a remote/scratch buffer
  (``get_broadcastable_buffer`` / ``_get_broadcastable_kv_buffer``) with
  ``remote_layer_id`` dedup.
- MLA writes use ``remote_kv_updatable`` in-place patch when the remote
  already holds that layer (``set_mla_kv_buffer``); indexer writes always
  ``invalidate`` first.
- ``prefetch_kv_buffer`` / ``maybe_prefetch_next_full_attention_kv`` kick
  the next layer's broadcast after attention releases the remote buffer.
- Effective per-rank layer count for sizing: ``ceil(L / CP) + 1``.

vLLM/HCU binds caches through packed runner tensors rather than SGLang's
token pool. Packed ``block_stride`` shrinks each page-size family to
``effective_num_layers`` slots (owned + one remote scratch); workers remap
per-layer offsets so non-owned layers alias the scratch slot (SGLang
``remote_kv_buffer``). Broadcast/prefetch/invalidate APIs below match the
SGLang pool methods; the scratch alias is the remote buffer.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Flat contiguous broadcast scratch keyed by (pool, device, dtype). Sized to
# the largest cache seen; full-pool and sparse payloads take a prefix view.
_broadcast_scratch: dict[tuple[int, ...], torch.Tensor] = {}
# Worker-process cached runtime state; installed during initialize_kv_cache
# so forward-time gating/broadcast never depend on a current vllm config.
_RUNTIME_STATE: "LayerSplitState | None" = None
_RUNTIME_STATE_INSTALLED = False
# SGLang LayerSplitDSATokenToKVPool remote-buffer bookkeeping, one state per
# cache family (MLA latent vs indexer). See ``_FamilyRemoteState``.
_FAMILY_REMOTE: dict[str, "_FamilyRemoteState"] = {}
# layer_idx -> {family_kind: cache tensor} for next-layer prefetch.
_LAYER_CACHE_REGISTRY: dict[int, dict[str, torch.Tensor]] = {}
# SGLang ``layer_broadcast_comm`` + ``kv_broadcast_stream``: a dedicated
# NCCL communicator over the PCP ranks so prefetch broadcasts can run on a
# side stream and overlap with FFN/MoE compute. When unavailable (unit
# tests, missing pynccl) prefetch degrades to the sync broadcast.
_BCAST_COMM: Any | None = None
_BCAST_STREAM: Any | None = None
_BCAST_COMM_INIT_DONE = False
# Unique block ids for the current forward (global batch). When set, owner
# broadcasts only those rows via index_select / index_copy_ so traffic tracks
# context length instead of the full KV pool. Must be identical on every PCP
# rank (install from the global block table, not the rank-local partition).
_ACTIVE_BLOCK_IDS: torch.Tensor | None = None
# Host-side (min, max) of ``_ACTIVE_BLOCK_IDS``; avoids per-layer syncs.
_ACTIVE_BLOCK_ID_RANGE: tuple[int, int] | None = None
# Opt-out: VLLM_HCU_DSA_SPARSE_BROADCAST=0 forces the legacy full-pool path.
_SPARSE_BROADCAST_ENV = "VLLM_HCU_DSA_SPARSE_BROADCAST"


def get_layer_shard_range(
    rank: int, shard_size: int, total_layers: int
) -> tuple[int, int]:
    """Contiguous ``[start, end)`` local-layer range owned by ``rank``.

    Layers are split as evenly as possible; the first ``total_layers %
    shard_size`` ranks own one extra layer.
    """
    if shard_size <= 0:
        raise ValueError(f"shard_size must be > 0; got {shard_size}")
    if total_layers < 0:
        raise ValueError(f"total_layers must be >= 0; got {total_layers}")
    if not 0 <= rank < shard_size:
        raise ValueError(
            f"rank must be in [0, {shard_size}); got {rank}"
        )
    base = total_layers // shard_size
    rem = total_layers % shard_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def get_layer_owner(
    local_layer_idx: int, shard_size: int, total_layers: int
) -> int:
    """CP rank that owns ``local_layer_idx`` under the contiguous split."""
    if not 0 <= local_layer_idx < total_layers:
        raise ValueError(
            f"local_layer_idx={local_layer_idx} out of range for "
            f"total_layers={total_layers}"
        )
    for rank in range(shard_size):
        start, end = get_layer_shard_range(rank, shard_size, total_layers)
        if start <= local_layer_idx < end:
            return rank
    raise ValueError(
        f"Invalid local_layer_idx={local_layer_idx} for "
        f"shard_size={shard_size}, total_layers={total_layers}"
    )


def effective_num_layers(num_layers: int, shard_size: int) -> int:
    """Per-rank owned layers plus one remote scratch slot for sizing.

    Matches SGLang ``get_glm_dsa_layer_split_effective_num_layers``:
    ``ceil(num_layers / shard_size) + 1``.
    """
    if shard_size <= 1:
        return num_layers
    owned_upper = (num_layers + shard_size - 1) // shard_size
    return max(1, owned_upper + 1)


def log_layer_shard_plan(
    *,
    total_layers: int,
    shard_size: int,
    shard_rank: int,
    start_layer: int = 0,
) -> None:
    partitions = []
    for rank in range(shard_size):
        st, ed = get_layer_shard_range(rank, shard_size, total_layers)
        partitions.append(f"r{rank}:[{st},{ed})")
    my_start, my_end = get_layer_shard_range(
        shard_rank, shard_size, total_layers
    )
    logger.warning(
        "DSA cache layer shard plan: layer_num=%d, shard_size=%d, "
        "rank=%d, local=[%d,%d), global=[%d,%d), partitions=%s",
        total_layers,
        shard_size,
        shard_rank,
        my_start,
        my_end,
        start_layer + my_start,
        start_layer + my_end,
        "; ".join(partitions),
    )


def is_dsa_cache_layer_split_requested(vllm_config: Any | None = None) -> bool:
    """Whether ``--enable-dsa-cache-layer-split`` (HCU sidecar) is set.

    Does not validate PCP/DSA prerequisites; call
    ``validate_dsa_cache_layer_split`` before enabling runtime paths.
    """
    if vllm_config is None:
        return False
    try:
        from vllm_hcu.patch.config import get_hcu_config

        return bool(get_hcu_config(vllm_config).enable_dsa_cache_layer_split)
    except Exception:
        return False


_DSA_LAYER_SPLIT_ARCHITECTURES = frozenset(
    {
        # These are the DSA model families implemented by the HCU model
        # registry.  The architecture check is intentionally paired with
        # the config capability check below, as in SGLang's is_deepseek_dsa.
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "GlmMoeDsaForCausalLM",
        "HYV4ForCausalLM",
    }
)


def _dsa_config_attr(config: Any, name: str) -> Any:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _dsa_hf_config(model_config: Any) -> Any | None:
    """Return the HF config that owns the DSA indexer fields."""
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        hf_config = getattr(model_config, "hf_text_config", None)
    return hf_config


# Architectures whose HCU runtime honors ``VLLM_HCU_DISABLE_DSA`` (the
# DeepSeek V2 decoder / MTP set ``is_v32 = ... and not VLLM_HCU_DISABLE_DSA``).
# HYV4 always builds its indexer, so the opt-out must not disable it here.
_DSA_ENV_OPT_OUT_ARCHITECTURES = frozenset(
    {
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "GlmMoeDsaForCausalLM",
    }
)


def _dsa_disabled_by_environment(architecture: str | None = None) -> bool:
    """Match the model runtime's environment-controlled DSA opt-out."""
    if (
        architecture is not None
        and architecture not in _DSA_ENV_OPT_OUT_ARCHITECTURES
    ):
        return False
    return os.environ.get("VLLM_HCU_DISABLE_DSA", "False").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def is_hcu_dsa_mla_model(model_config: Any | None) -> bool:
    """Whether ``model_config`` has the DSA MLA capability for HCU.

    SGLang's predicate is deliberately more specific than ``use_mla``: a
    supported DSA architecture must expose the sparse-indexer ``index_topk``
    field.  Keep the same fail-closed rule here so enabling PCP alone cannot
    route an ordinary MLA/GQA cache through DSA layer remapping.
    """
    if model_config is None:
        return False

    architectures = getattr(model_config, "architectures", None) or ()
    if not architectures or architectures[0] not in _DSA_LAYER_SPLIT_ARCHITECTURES:
        return False
    if _dsa_disabled_by_environment(architectures[0]):
        return False
    if getattr(model_config, "use_mla", None) is not True:
        return False

    hf_config = _dsa_hf_config(model_config)
    return hf_config is not None and _dsa_config_attr(hf_config, "index_topk") is not None


def _dsa_capability_error(model_config: Any | None) -> str:
    """Describe the first missing DSA capability for validation errors."""
    if model_config is None:
        return "model_config is missing"

    architectures = getattr(model_config, "architectures", None) or ()
    architecture = architectures[0] if architectures else None
    if architecture not in _DSA_LAYER_SPLIT_ARCHITECTURES:
        return (
            "architecture must be one of "
            f"{sorted(_DSA_LAYER_SPLIT_ARCHITECTURES)} (got {architecture!r})"
        )
    if _dsa_disabled_by_environment(architecture):
        return "VLLM_HCU_DISABLE_DSA is enabled"
    if getattr(model_config, "use_mla", None) is not True:
        return f"use_mla must be True (got {getattr(model_config, 'use_mla', None)!r})"
    hf_config = _dsa_hf_config(model_config)
    if hf_config is None or _dsa_config_attr(hf_config, "index_topk") is None:
        return "hf_config.index_topk is missing"
    return "the HCU DSA runtime capability is unavailable"


def _require_pd_prefill_worker(vllm_config: Any) -> None:
    """Only a PD-disaggregated prefill worker may split DSA cache layers.

    Mirrors SGLang ``model_hook`` (``disaggregation_mode == "prefill"``): a
    worker that also decodes reads every layer's cache on every step, so the
    owner broadcast would run per layer per decode step over the whole layer
    pool. The decode worker receives full per-layer cache via KV transfer.
    """
    kv_transfer = getattr(vllm_config, "kv_transfer_config", None)
    kv_role = getattr(kv_transfer, "kv_role", None)
    if kv_role != "kv_producer":
        raise ValueError(
            "enable_dsa_cache_layer_split is only supported on a "
            "PD-disaggregated prefill worker (--kv-transfer-config with "
            f"kv_role='kv_producer'); got kv_role={kv_role!r}. Workers that "
            "also run decode require full local DSA cache; drop "
            "--enable-dsa-cache-layer-split."
        )


def validate_dsa_cache_layer_split(vllm_config: Any) -> None:
    """Raise if the flag is on but SGLang prerequisites fail.

    Mirrors SGLang ``server_args`` / ``model_hook`` checks for
    ``--enable-dsa-cache-layer-split``: PD prefill worker, Prefill-CP > 1,
    no pipeline parallel, and an HCU-supported DSA MLA capability
    (architecture + ``index_topk`` + MLA runtime).
    """
    if not is_dsa_cache_layer_split_requested(vllm_config):
        return
    parallel = getattr(vllm_config, "parallel_config", None)
    if parallel is None:
        raise ValueError(
            "enable_dsa_cache_layer_split requires a parallel_config"
        )
    pcp = int(getattr(parallel, "prefill_context_parallel_size", 1) or 1)
    pp = int(getattr(parallel, "pipeline_parallel_size", 1) or 1)
    if pcp <= 1:
        raise ValueError(
            "enable_dsa_cache_layer_split requires "
            "prefill_context_parallel_size > 1 "
            f"(got {pcp})."
        )
    if pp > 1:
        raise ValueError(
            "enable_dsa_cache_layer_split does not support "
            f"pipeline_parallel_size > 1 (got {pp})."
        )
    _require_pd_prefill_worker(vllm_config)
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        raise ValueError(
            "enable_dsa_cache_layer_split requires a model_config"
        )
    if not is_hcu_dsa_mla_model(model_config):
        raise ValueError(
            "enable_dsa_cache_layer_split requires an HCU-supported DSA MLA "
            f"capability ({_dsa_capability_error(model_config)})."
        )
    total_layers = _total_layers_from_config(vllm_config)
    if total_layers is not None and pcp > total_layers:
        raise ValueError(
            "enable_dsa_cache_layer_split requires "
            "prefill_context_parallel_size <= num_hidden_layers "
            f"(got pcp={pcp}, num_hidden_layers={total_layers}); "
            "PCP ranks with an empty owned range cannot register Mooncake "
            "KV tensors."
        )
    _require_unsplittable_manager_block(vllm_config)


def _require_unsplittable_manager_block(vllm_config: Any) -> None:
    """Packed layer-split stores one cross-layer stride per manager block.

    Upstream ``create_kv_cache_views`` subdivides a manager block into kernel
    pages only when ``block_stride`` equals one dense page. The packed stride
    is larger than a page, so a manager block of 128 tokens cannot become two
    FLASHMLA sparse kernel blocks of 64. Fail here instead of at allocation.
    """
    cache_config = getattr(vllm_config, "cache_config", None)
    block_size = getattr(cache_config, "block_size", None)
    if block_size is None:
        return
    block_size = int(block_size)
    kernel_page = 64
    if block_size > kernel_page and block_size % kernel_page == 0:
        raise ValueError(
            "enable_dsa_cache_layer_split packs every layer into one block "
            "stride, so a manager block cannot be split into kernel blocks. "
            f"Use --block-size {kernel_page} (got {block_size})."
        )


def parse_layer_index(layer_name: str) -> int:
    """Extract the transformer layer index from a module prefix."""
    from vllm.model_executor.models.utils import extract_layer_index

    return int(extract_layer_index(layer_name))


@dataclass(frozen=True)
class LayerSplitState:
    """Runtime PCP ownership for DSA cache layer split."""

    shard_rank: int
    shard_size: int
    total_layers: int
    start_layer: int = 0

    def local_index(self, layer_idx: int) -> int:
        return int(layer_idx) - int(self.start_layer)

    def covers(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` participates in the backbone shard plan.

        MTP / speculative layers with ``layer_idx >= total_layers`` stay
        fully replicated (every rank commits and no owner broadcast).
        """
        local = self.local_index(layer_idx)
        return 0 <= local < self.total_layers

    def is_owned(self, layer_idx: int) -> bool:
        if not self.covers(layer_idx):
            return True
        start, end = get_layer_shard_range(
            self.shard_rank, self.shard_size, self.total_layers
        )
        local = self.local_index(layer_idx)
        return start <= local < end

    def owner_of(self, layer_idx: int) -> int:
        if not self.covers(layer_idx):
            return self.shard_rank
        return get_layer_owner(
            self.local_index(layer_idx), self.shard_size, self.total_layers
        )


def _resolve_vllm_config(vllm_config: Any | None) -> Any | None:
    if vllm_config is not None:
        return vllm_config
    try:
        from vllm.config import get_current_vllm_config_or_none

        return get_current_vllm_config_or_none()
    except Exception:
        return None


def _total_layers_from_config(vllm_config: Any) -> int | None:
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return None
    hf = getattr(model_config, "hf_config", None)
    if hf is None:
        hf = getattr(model_config, "hf_text_config", None)
    if hf is None:
        return None
    total = getattr(hf, "num_hidden_layers", None)
    if total is None:
        return None
    return int(total)


def get_runtime_layer_split_state(
    vllm_config: Any | None = None,
    *,
    shard_rank: int | None = None,
    shard_size: int | None = None,
    total_layers: int | None = None,
) -> LayerSplitState | None:
    """Return active layer-split state, or ``None`` when the feature is off.

    ``shard_rank`` / ``shard_size`` / ``total_layers`` may be injected for
    unit tests that do not initialize distributed process groups.
    """
    # Forward-time callers pass nothing. There, the ambient vllm config is
    # usually gone (``get_current_vllm_config_or_none`` is only set during
    # init) and the worker env is sanitized, so resolving from scratch would
    # silently disable gating + broadcast. ``install_runtime_layer_split_state``
    # caches the state during KV-cache init; prefer it for bare calls.
    if (
        vllm_config is None
        and shard_rank is None
        and shard_size is None
        and total_layers is None
        and _RUNTIME_STATE_INSTALLED
    ):
        return _RUNTIME_STATE

    cfg = _resolve_vllm_config(vllm_config)
    if not is_dsa_cache_layer_split_requested(cfg):
        return None

    if shard_size is None:
        if cfg is None:
            return None
        parallel = getattr(cfg, "parallel_config", None)
        shard_size = int(
            getattr(parallel, "prefill_context_parallel_size", 1) or 1
        )
    shard_size = int(shard_size)
    if shard_size <= 1:
        return None

    if total_layers is None:
        if cfg is None:
            return None
        total_layers = _total_layers_from_config(cfg)
        if total_layers is None:
            return None
    total_layers = int(total_layers)

    if shard_rank is None:
        try:
            from vllm.distributed.parallel_state import get_pcp_group

            shard_rank = int(get_pcp_group().rank_in_group)
        except Exception:
            # Before PCP init (or in unit tests), fall back to local rank env.
            for key in (
                "PCP_RANK",
                "RANK",
                "LOCAL_RANK",
                "OMPI_COMM_WORLD_LOCAL_RANK",
            ):
                raw = os.environ.get(key)
                if raw is not None and raw != "":
                    shard_rank = int(raw)
                    break
            if shard_rank is None:
                shard_rank = 0
    shard_rank = int(shard_rank)

    return LayerSplitState(
        shard_rank=shard_rank,
        shard_size=shard_size,
        total_layers=total_layers,
    )


def _init_layer_broadcast_comm(state: LayerSplitState | None) -> None:
    """SGLang ``_init_layer_broadcast_comm`` + ``kv_broadcast_stream``.

    A dedicated PyNccl communicator over the PCP ranks lets prefetch
    broadcasts run on a side stream and overlap with FFN/MoE compute.
    Collective: every PCP rank must reach this together (all do, from
    ``install_runtime_layer_split_state`` during KV-cache init). Falls back
    to ``None`` (sync prefetch) when pynccl or the group is unavailable.
    """
    global _BCAST_COMM, _BCAST_STREAM, _BCAST_COMM_INIT_DONE
    if _BCAST_COMM_INIT_DONE:
        return
    _BCAST_COMM_INIT_DONE = True
    if state is None or state.shard_size <= 1:
        return
    try:
        from vllm.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )
        from vllm.distributed.parallel_state import get_pcp_group

        pcp_group = get_pcp_group()
        comm = PyNcclCommunicator(
            group=pcp_group.cpu_group,
            device=pcp_group.device,
        )
        if getattr(comm, "disabled", True):
            logger.warning(
                "DSA layer-split broadcast communicator disabled; "
                "prefetch stays synchronous"
            )
            return
        _BCAST_COMM = comm
        _BCAST_STREAM = torch.cuda.Stream(device=pcp_group.device)
        logger.info(
            "DSA layer-split dedicated broadcast NCCL communicator ready: "
            "rank=%d world_size=%d",
            int(pcp_group.rank_in_group),
            int(pcp_group.world_size),
        )
    except Exception:
        logger.warning(
            "DSA layer-split broadcast communicator init failed; "
            "prefetch stays synchronous",
            exc_info=True,
        )


def install_runtime_layer_split_state(
    vllm_config: Any | None,
) -> LayerSplitState | None:
    """Resolve and cache layer-split state for forward-time bare lookups.

    Must run in the worker after the PCP group exists (KV-cache init), so
    ``shard_rank`` comes from the real group rather than an env fallback.
    """
    global _RUNTIME_STATE, _RUNTIME_STATE_INSTALLED
    _RUNTIME_STATE_INSTALLED = False
    clear_active_dsa_broadcast_block_ids()
    _FAMILY_REMOTE.clear()
    _LAYER_CACHE_REGISTRY.clear()
    state = get_runtime_layer_split_state(vllm_config)
    _RUNTIME_STATE = state
    _RUNTIME_STATE_INSTALLED = True
    _init_layer_broadcast_comm(state)
    return state


def reset_runtime_layer_split_state_for_tests() -> None:
    """Drop the cached runtime state (unit tests only)."""
    global _RUNTIME_STATE, _RUNTIME_STATE_INSTALLED
    global _BCAST_COMM, _BCAST_STREAM, _BCAST_COMM_INIT_DONE
    _RUNTIME_STATE = None
    _RUNTIME_STATE_INSTALLED = False
    clear_active_dsa_broadcast_block_ids()
    _FAMILY_REMOTE.clear()
    _LAYER_CACHE_REGISTRY.clear()
    _BCAST_COMM = None
    _BCAST_STREAM = None
    _BCAST_COMM_INIT_DONE = False


def should_commit_durable_dsa_cache(
    layer_name: str,
    vllm_config: Any | None = None,
    *,
    shard_rank: int | None = None,
    shard_size: int | None = None,
    total_layers: int | None = None,
) -> bool:
    """Whether this PCP rank should gather+write durable cache for ``layer``.

    When the feature is off every rank commits (legacy PCP behavior). When
    on, only the owning rank commits for backbone layers.
    """
    state = get_runtime_layer_split_state(
        vllm_config,
        shard_rank=shard_rank,
        shard_size=shard_size,
        total_layers=total_layers,
    )
    if state is None:
        return True
    return state.is_owned(parse_layer_index(layer_name))


@dataclass
class _FamilyRemoteState:
    """SGLang ``remote_kv_layer_id`` / pending-prefetch bookkeeping per family.

    One instance covers the MLA latent family or the indexer family. vLLM's
    packed scratch slot (or per-layer view under full alloc) is the broadcast
    destination — the SGLang ``remote_kv_buffer`` / ``remote_buffer``.
    """

    remote_layer_id: int | None = None
    remote_block_ids: torch.Tensor | None = None
    remote_full: bool = False
    pending_remote_layer_id: int | None = None
    pending_remote_block_ids: torch.Tensor | None = None
    pending_remote_full: bool = False
    pending_remote_broadcast: bool = False
    broadcast_stream: Any | None = None


def _family_kind(layer_name: str) -> str:
    name = layer_name.lower()
    if "indexer" in name or "index_k" in name:
        return "indexer"
    return "mla"


def _family_shape(cache: torch.Tensor) -> tuple[int, ...]:
    """Drop size-1 dims so ``[B, 1, N, C]`` and ``squeeze(1)`` match.

    The allocator registers the 4D tensor. MLA binds the squeezed 3D view
    of that same cache. A key that keeps the singleton dimension splits
    prefetch and the forward onto two broadcast states.
    """
    dims = tuple(int(dim) for dim in cache.shape if int(dim) != 1)
    return dims or tuple(int(dim) for dim in cache.shape)


def _family_key(cache: torch.Tensor, layer_name: str) -> str:
    # Kind separates MLA vs indexer. Normalized shape keeps every layer of
    # one packed family on the same broadcast state, including the 4D
    # registration view and the 3D view used by the forward.
    return (
        f"{_family_kind(layer_name)}|"
        f"{cache.device.type}:{cache.device.index}|"
        f"{cache.dtype}|{_family_shape(cache)}"
    )


def _family_state(cache: torch.Tensor, layer_name: str) -> _FamilyRemoteState:
    key = _family_key(cache, layer_name)
    st = _FAMILY_REMOTE.get(key)
    if st is None:
        st = _FamilyRemoteState()
        _FAMILY_REMOTE[key] = st
    return st


def _invalidate_remote_broadcast_state(
    active_ids: torch.Tensor | None,
) -> None:
    """Retain synchronized blocks while the active sparse set changes."""
    for family in tuple(_FAMILY_REMOTE.values()):
        if family.pending_remote_broadcast:
            _finalize_pending_broadcast(family, set_remote_layer_id=True)
        if active_ids is None:
            if not family.remote_full:
                family.remote_block_ids = None
            continue
        if family.remote_full or family.remote_block_ids is None:
            continue
        known = family.remote_block_ids
        if known.device != active_ids.device:
            active_ids_for_cache = active_ids.to(device=known.device)
        else:
            active_ids_for_cache = active_ids
        family.remote_block_ids = known[
            torch.isin(known, active_ids_for_cache)
        ]


def _scratch_for(
    cache: torch.Tensor,
    pool: str = "sync",
    *,
    numel: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    n = int(cache.numel() if numel is None else numel)
    key = (pool, cache.device.type, cache.device.index, cache.dtype)
    scratch = _broadcast_scratch.get(key)
    if scratch is None or scratch.numel() < n:
        # Size to the full pool so per-step sparse payloads never reallocate.
        capacity = max(n, int(cache.numel()))
        scratch = torch.empty(capacity, dtype=cache.dtype, device=cache.device)
        _broadcast_scratch[key] = scratch
    return scratch[:n].view(cache.shape if shape is None else shape)


def _sparse_broadcast_enabled() -> bool:
    raw = os.environ.get(_SPARSE_BROADCAST_ENV, "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


def set_active_dsa_broadcast_block_ids(
    block_ids: torch.Tensor | None,
) -> None:
    """Install unique block ids for the next owner-broadcasts in this forward.

    ``block_ids`` must be a 1-D int tensor, identical on every PCP rank.
    Pass ``None`` / empty to fall back to the full-pool broadcast. When the
    set changes, already synchronized rows are retained and only newly active
    rows are sent; removing rows does not require a collective.

    Dedup stays layer-granular when switching layers (SGLang
    ``remote_layer_id``). Within one layer, the remote scratch tracks active
    block ids so an expanding active set does not resend existing rows.
    """
    global _ACTIVE_BLOCK_IDS, _ACTIVE_BLOCK_ID_RANGE
    if block_ids is None or not isinstance(block_ids, torch.Tensor):
        ids = None
    elif block_ids.numel() == 0:
        ids = None
    else:
        ids = torch.unique(
            block_ids.reshape(-1).to(dtype=torch.long), sorted=True
        )

    previous = _ACTIVE_BLOCK_IDS
    changed = (previous is None) != (ids is None)
    if not changed and previous is not None and ids is not None:
        try:
            changed = not torch.equal(previous, ids)
        except RuntimeError:
            changed = True
    if changed:
        _invalidate_remote_broadcast_state(ids)
    _ACTIVE_BLOCK_IDS = ids
    # Resolve the id range once per forward. ``ids`` is sorted, so the range
    # check in ``_active_block_ids_for`` needs no per-layer device sync.
    _ACTIVE_BLOCK_ID_RANGE = (
        None if ids is None else (int(ids[0].item()), int(ids[-1].item()))
    )


def set_active_dsa_broadcast_block_ids_from_tables(
    block_tables: Any,
) -> None:
    """Derive active block ids from global ``[num_reqs, max_blocks]`` tables.

    Over-including pad / stale ids only wastes bandwidth; under-including
    would leave non-owners with missing KV. Trailing zeros are therefore
    kept when present (block 0 is a valid id).
    """
    if not _sparse_broadcast_enabled():
        set_active_dsa_broadcast_block_ids(None)
        return
    tables: list[torch.Tensor] = []
    if isinstance(block_tables, torch.Tensor):
        tables = [block_tables]
    elif isinstance(block_tables, (tuple, list)):
        tables = [t for t in block_tables if isinstance(t, torch.Tensor)]
    if not tables:
        set_active_dsa_broadcast_block_ids(None)
        return
    # Unified packed pool: MLA / indexer share one block-id space. Union
    # across groups so a group with a longer prefix cannot drop ids.
    flat = torch.cat([t.reshape(-1).to(dtype=torch.long) for t in tables], dim=0)
    set_active_dsa_broadcast_block_ids(flat)


def clear_active_dsa_broadcast_block_ids() -> None:
    """Drop the per-forward block-id set (tests / end of step)."""
    set_active_dsa_broadcast_block_ids(None)


def _active_block_ids_for(cache: torch.Tensor) -> torch.Tensor | None:
    if not _sparse_broadcast_enabled():
        return None
    ids = _ACTIVE_BLOCK_IDS
    id_range = _ACTIVE_BLOCK_ID_RANGE
    if ids is None or id_range is None or ids.numel() == 0:
        return None
    if cache.ndim < 1 or cache.shape[0] <= 0:
        return None
    # An invalid/sentinel block id must never become an index_select/index_copy
    # operand.  Falling back to the full pool is conservative and keeps every
    # PCP rank in the same collective path.
    if id_range[0] < 0 or id_range[1] >= int(cache.shape[0]):
        return None
    # Skip when the selection would not shrink the payload.
    if int(ids.numel()) >= int(cache.shape[0]):
        return None
    if ids.device != cache.device:
        ids = ids.to(device=cache.device)
    return ids


def _register_layer_cache(cache: torch.Tensor, layer_name: str) -> None:
    """Remember cache tensors so next-layer prefetch can find them."""
    try:
        layer_idx = parse_layer_index(layer_name)
    except Exception:
        return
    kind = _family_kind(layer_name)
    slot = _LAYER_CACHE_REGISTRY.setdefault(layer_idx, {})
    slot[kind] = cache


def register_dsa_cache_tensors(caches: Any) -> None:
    """Register all bound attention caches for asynchronous prefetch.

    The SGLang pool has every layer's cache available when it is constructed.
    vLLM binds caches one layer at a time during runner initialization, so a
    forward-only registration would discover the current layer too late for
    next-layer prefetch.  Register the completed cache map once, after binding,
    while retaining the normal layer-name parsing and family filtering.
    """
    if not isinstance(caches, dict):
        return
    if _RUNTIME_STATE is None:
        return
    for layer_name, cache in caches.items():
        if isinstance(layer_name, str) and isinstance(cache, torch.Tensor):
            _register_layer_cache(cache, layer_name)


def _broadcast_selected_blocks(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    owner: int,
    my_rank: int,
    broadcast_fn: Any,
    *,
    pool: str,
) -> None:
    """Owner ``index_select`` → contiguous broadcast → non-owner ``index_copy_``."""
    rest_shape = tuple(cache.shape[1:])
    row_numel = cache[0].numel() if cache.shape[0] > 0 else int(cache.numel())
    n_rows = int(block_ids.numel())
    payload = _scratch_for(
        cache,
        pool=pool,
        numel=n_rows * row_numel,
        shape=(n_rows, *rest_shape),
    )
    if my_rank == owner:
        # index_select gathers rows even when ``cache`` is a strided packed view.
        payload.copy_(cache.index_select(0, block_ids))
    broadcast_fn(payload)
    if my_rank != owner:
        cache.index_copy_(0, block_ids, payload)


def _prepare_remote_selection(
    cache: torch.Tensor,
    layer_idx: int,
    family: _FamilyRemoteState,
) -> tuple[torch.Tensor | None, bool, bool]:
    """Return (ids, full, needs_collective) for an owner synchronization."""
    if family.remote_layer_id != layer_idx:
        family.remote_block_ids = None
        family.remote_full = False
    block_ids = _active_block_ids_for(cache)
    if block_ids is None:
        return None, True, not family.remote_full
    if family.remote_full:
        return block_ids, False, False
    known = family.remote_block_ids
    if known is None or known.numel() == 0:
        needed = block_ids
    else:
        if known.device != block_ids.device:
            known_for_cache = known.to(device=block_ids.device)
        else:
            known_for_cache = known
        needed = block_ids[~torch.isin(block_ids, known_for_cache)]
    return needed, False, bool(needed.numel())


def _record_remote_selection(
    family: _FamilyRemoteState,
    layer_idx: int,
    block_ids: torch.Tensor | None,
    full: bool,
) -> None:
    """Record the rows known to be present in the remote scratch view."""
    if family.remote_layer_id != layer_idx:
        family.remote_block_ids = None
        family.remote_full = False
    family.remote_layer_id = layer_idx
    if full:
        family.remote_block_ids = None
        family.remote_full = True
        return
    family.remote_full = False
    if block_ids is None or block_ids.numel() == 0:
        return
    if family.remote_block_ids is None:
        family.remote_block_ids = block_ids.detach().clone()
    else:
        known = family.remote_block_ids
        if known.device != block_ids.device:
            block_ids = block_ids.to(device=known.device)
        family.remote_block_ids = torch.unique(
            torch.cat((known, block_ids.detach())), sorted=True
        )


def _broadcast_tensor_from_owner(
    cache: torch.Tensor,
    layer_idx: int,
    owner: int,
    pcp_group: Any,
    family: _FamilyRemoteState,
) -> None:
    """SGLang ``LayerSplitDSATokenToKVPool._broadcast_tensor_from_owner``.

    Prefers a sparse payload of the active batch's block ids when installed;
    otherwise broadcasts the full cache (profile / warmup / opt-out).
    Non-contiguous packed views bounce through a contiguous scratch.
    """
    my_rank = int(pcp_group.rank_in_group)
    block_ids, full, needs_collective = _prepare_remote_selection(
        cache, layer_idx, family
    )
    if not needs_collective:
        _record_remote_selection(family, layer_idx, block_ids, full)
        return
    if block_ids is not None:
        _broadcast_selected_blocks(
            cache,
            block_ids,
            owner,
            my_rank,
            lambda payload: pcp_group.broadcast(payload, src=owner),
            pool="sync",
        )
    elif cache.is_contiguous():
        pcp_group.broadcast(cache, src=owner)
    else:
        scratch = _scratch_for(cache)
        if my_rank == owner:
            scratch.copy_(cache)
        pcp_group.broadcast(scratch, src=owner)
        if my_rank != owner:
            cache.copy_(scratch)
    _record_remote_selection(family, layer_idx, block_ids, full)


def _broadcast_tensor_from_owner_async(
    cache: torch.Tensor,
    layer_idx: int,
    owner: int,
    my_rank: int,
    fam: _FamilyRemoteState,
) -> bool:
    """SGLang ``prefetch_kv_buffer`` fast path on the dedicated comm/stream.

    Enqueues the owner broadcast on ``_BCAST_STREAM`` so it overlaps with
    FFN/MoE on the current stream. Returns False when the dedicated
    communicator is unavailable (caller falls back to the sync broadcast).
    All copies (scratch bounce included) run on the side stream; readers
    synchronize via ``_finalize_pending_broadcast``.
    """
    if _BCAST_COMM is None or _BCAST_STREAM is None:
        return False
    # The side stream must not overtake work already queued on the current
    # stream (the just-finished attention that still reads the scratch slot,
    # and the owner's durable write for this layer).
    _BCAST_STREAM.wait_stream(torch.cuda.current_stream())
    block_ids, full, needs_collective = _prepare_remote_selection(
        cache, layer_idx, fam
    )
    if not needs_collective:
        _record_remote_selection(fam, layer_idx, block_ids, full)
        return False

    def _nccl_broadcast(tensor: torch.Tensor) -> None:
        _BCAST_COMM.broadcast(tensor, src=owner, stream=_BCAST_STREAM)

    with torch.cuda.stream(_BCAST_STREAM):
        if block_ids is not None:
            _broadcast_selected_blocks(
                cache,
                block_ids,
                owner,
                my_rank,
                _nccl_broadcast,
                pool="prefetch",
            )
        elif cache.is_contiguous():
            _nccl_broadcast(cache)
        else:
            scratch = _scratch_for(cache, pool="prefetch")
            if my_rank == owner:
                scratch.copy_(cache)
            _nccl_broadcast(scratch)
            if my_rank != owner:
                cache.copy_(scratch)
    fam.broadcast_stream = _BCAST_STREAM
    fam.pending_remote_layer_id = layer_idx
    fam.pending_remote_block_ids = (
        None if block_ids is None else block_ids.detach().clone()
    )
    fam.pending_remote_full = full
    fam.pending_remote_broadcast = True
    return True


def _finalize_pending_broadcast(
    fam: _FamilyRemoteState, *, set_remote_layer_id: bool = True
) -> None:
    """SGLang ``_finalize_pending_kv_broadcast``."""
    if not fam.pending_remote_broadcast:
        return
    if fam.broadcast_stream is not None:
        try:
            torch.cuda.current_stream().wait_stream(fam.broadcast_stream)
        except Exception:
            pass
    fam.pending_remote_broadcast = False
    if set_remote_layer_id and fam.pending_remote_layer_id is not None:
        _record_remote_selection(
            fam,
            fam.pending_remote_layer_id,
            fam.pending_remote_block_ids,
            fam.pending_remote_full,
        )
    fam.pending_remote_layer_id = None
    fam.pending_remote_block_ids = None
    fam.pending_remote_full = False


def invalidate_dsa_cache_broadcast(
    cache: torch.Tensor,
    layer_name: str,
) -> None:
    """SGLang indexer ``invalidate`` / ``invalidate_index_buffer_for_layer``.

    Must run on **every** PCP rank before a durable indexer gather/write.
    Owner-only invalidation lets non-owners skip the next NCCL broadcast
    and deadlocks the group.
    """
    if not isinstance(cache, torch.Tensor) or cache.numel() == 0:
        return
    try:
        layer_idx = parse_layer_index(layer_name)
    except Exception:
        return
    fam = _family_state(cache, layer_name)
    if fam.pending_remote_layer_id == layer_idx:
        _finalize_pending_broadcast(fam, set_remote_layer_id=False)
    if fam.remote_layer_id == layer_idx:
        fam.remote_layer_id = None
        fam.remote_block_ids = None
        fam.remote_full = False


def begin_mla_dsa_cache_write(
    cache: torch.Tensor,
    layer_name: str,
) -> bool:
    """SGLang ``set_mla_kv_buffer`` remote_kv_updatable probe.

    Returns True when the remote/scratch already holds ``layer_name`` so the
    write can patch it in place (every rank) and skip the post-write
    NCCL broadcast. Finalizes any pending prefetch for this layer first.
    """
    if not isinstance(cache, torch.Tensor) or cache.numel() == 0:
        return False
    try:
        layer_idx = parse_layer_index(layer_name)
    except Exception:
        return False
    if get_runtime_layer_split_state() is None:
        return False
    fam = _family_state(cache, layer_name)
    if fam.pending_remote_layer_id == layer_idx:
        _finalize_pending_broadcast(fam, set_remote_layer_id=True)
    _, _, needs_collective = _prepare_remote_selection(cache, layer_idx, fam)
    return fam.remote_layer_id == layer_idx and not needs_collective


def broadcast_dsa_cache_from_owner(
    cache: torch.Tensor,
    layer_name: str,
    vllm_config: Any | None = None,
    *,
    shard_rank: int | None = None,
    shard_size: int | None = None,
    total_layers: int | None = None,
    pcp_group: Any | None = None,
) -> torch.Tensor:
    """SGLang ``get_broadcastable_buffer`` / ``_get_broadcastable_kv_buffer``.

    Ensures ``cache`` holds the owning rank's durable bytes for ``layer_name``.
    A matching ``remote_layer_id`` skips the collective only when all currently
    active sparse rows are already synchronized.

    Call sites must invoke this on **every** PCP rank the same number of
    times per layer forward. Replicated MTP is a hard no-op.
    """
    if not isinstance(cache, torch.Tensor) or cache.numel() == 0:
        return cache

    try:
        from vllm_hcu.model_executor.layers.attention.pcp import (
            in_replicated_mtp_batch,
        )

        if in_replicated_mtp_batch():
            return cache
    except Exception:
        pass

    state = get_runtime_layer_split_state(
        vllm_config,
        shard_rank=shard_rank,
        shard_size=shard_size,
        total_layers=total_layers,
    )
    if state is None:
        return cache

    layer_idx = parse_layer_index(layer_name)
    if not state.covers(layer_idx):
        return cache

    _register_layer_cache(cache, layer_name)
    fam = _family_state(cache, layer_name)

    # Finalize any in-flight prefetch before reading (SGLang).
    if fam.pending_remote_broadcast:
        _finalize_pending_broadcast(
            fam,
            set_remote_layer_id=fam.pending_remote_layer_id == layer_idx,
        )

    if fam.remote_layer_id == layer_idx and fam.remote_full:
        return cache

    owner = state.owner_of(layer_idx)
    if pcp_group is None:
        from vllm.distributed.parallel_state import get_pcp_group

        pcp_group = get_pcp_group()

    _broadcast_tensor_from_owner(cache, layer_idx, owner, pcp_group, fam)
    return cache


def prefetch_dsa_cache_from_owner(
    cache: torch.Tensor,
    layer_name: str,
    vllm_config: Any | None = None,
    *,
    shard_rank: int | None = None,
    shard_size: int | None = None,
    total_layers: int | None = None,
    pcp_group: Any | None = None,
) -> None:
    """SGLang ``prefetch_kv_buffer``: kick owner-broadcast for ``layer_name``.

    With the dedicated layer-broadcast communicator (see
    ``_init_layer_broadcast_comm``), the broadcast is enqueued on a side
    stream and overlaps with FFN/MoE; readers finalize it via
    ``broadcast_dsa_cache_from_owner`` / ``begin_mla_dsa_cache_write``.
    Without it, this follows SGLang's sync fallback: broadcast immediately
    and set ``remote_layer_id``.

    Collective: every PCP rank must call this the same number of times per
    layer forward (wired after ``self.attn`` in the decoder layer).
    """
    if not isinstance(cache, torch.Tensor) or cache.numel() == 0:
        return

    try:
        from vllm_hcu.model_executor.layers.attention.pcp import (
            in_replicated_mtp_batch,
        )

        if in_replicated_mtp_batch():
            return
    except Exception:
        pass

    state = get_runtime_layer_split_state(
        vllm_config,
        shard_rank=shard_rank,
        shard_size=shard_size,
        total_layers=total_layers,
    )
    if state is None:
        return

    layer_idx = parse_layer_index(layer_name)
    if not state.covers(layer_idx):
        return

    _register_layer_cache(cache, layer_name)
    fam = _family_state(cache, layer_name)
    if fam.remote_layer_id == layer_idx:
        return
    if fam.pending_remote_broadcast:
        if fam.pending_remote_layer_id == layer_idx:
            return
        _finalize_pending_broadcast(fam, set_remote_layer_id=False)

    owner = state.owner_of(layer_idx)
    if _broadcast_tensor_from_owner_async(
        cache, layer_idx, owner, state.shard_rank, fam
    ):
        return

    # Sync fallback (== SGLang when layer_broadcast_comm is None).
    broadcast_dsa_cache_from_owner(
        cache,
        layer_name,
        vllm_config,
        shard_rank=shard_rank,
        shard_size=shard_size,
        total_layers=total_layers,
        pcp_group=pcp_group,
    )


def _prefetch_registered_layer(layer_idx: int, kind: str) -> None:
    slot = _LAYER_CACHE_REGISTRY.get(int(layer_idx))
    if not slot:
        return
    cache = slot.get(kind)
    if cache is None:
        return
    if kind == "indexer":
        layer_name = f"model.layers.{layer_idx}.self_attn.indexer"
    else:
        layer_name = f"model.layers.{layer_idx}.self_attn.attn"
    prefetch_dsa_cache_from_owner(cache, layer_name)


def maybe_prefetch_same_layer_mla_cache(layer_idx: int) -> None:
    """SGLang indexer ``get_k_and_scale`` → ``pool.prefetch_kv_buffer(layer_id)``."""
    state = get_runtime_layer_split_state()
    if state is None or not state.covers(int(layer_idx)):
        return
    _prefetch_registered_layer(int(layer_idx), "mla")


def maybe_prefetch_next_dsa_cache(
    current_layer_idx: int,
    *,
    kind: str = "mla",
) -> None:
    """SGLang ``maybe_prefetch_next_full_attention_kv``.

    Prefetch the next registered backbone layer's cache for ``kind`` after
    the current layer's attention has released the remote buffer.
    """
    state = get_runtime_layer_split_state()
    if state is None:
        return
    next_idx = int(current_layer_idx) + 1
    if not state.covers(next_idx):
        return
    _prefetch_registered_layer(next_idx, kind)


def clear_broadcast_scratch_for_tests() -> None:
    """Drop cached broadcast scratch / remote-layer state (unit tests only)."""
    _broadcast_scratch.clear()
    _FAMILY_REMOTE.clear()
    _LAYER_CACHE_REGISTRY.clear()
    clear_active_dsa_broadcast_block_ids()


def maybe_log_runtime_layer_shard_plan(
    vllm_config: Any | None = None,
) -> bool:
    """Log the shard plan when layer-split is active. Returns True if logged."""
    try:
        state = get_runtime_layer_split_state(vllm_config)
    except Exception:
        logger.debug(
            "DSA cache layer-split plan lookup failed",
            exc_info=True,
        )
        return False
    if state is None:
        return False
    log_layer_shard_plan(
        total_layers=state.total_layers,
        shard_size=state.shard_size,
        shard_rank=state.shard_rank,
    )
    return True


def max_owned_count_for_layers(
    layer_indices: set[int],
    total_layers: int,
    shard_size: int,
) -> int:
    """Max owned-backbone count in ``layer_indices`` across CP ranks.

    Mirrors SGLang indexer sizing: intersect each rank's contiguous owned
    range with the family's backbone layer set, then take the max (+ scratch
    is applied by the caller).
    """
    if not layer_indices or shard_size <= 1:
        return len(layer_indices)
    max_owned = 0
    for rank in range(shard_size):
        start, end = get_layer_shard_range(rank, shard_size, total_layers)
        owned = sum(1 for idx in layer_indices if start <= idx < end)
        if owned > max_owned:
            max_owned = owned
    return max_owned


def phys_slots_for_layer_family(
    layer_indices: list[int],
    total_layers: int,
    shard_size: int,
) -> int:
    """Physical packed slots for one page-size family (owned + scratch + MTP).

    Backbone layers (``0 <= idx < total_layers``) shrink to
    ``max_owned + 1``; layers at ``idx >= total_layers`` (MTP / draft) stay
    fully replicated after the scratch slot.
    """
    backbone = {idx for idx in layer_indices if 0 <= idx < total_layers}
    n_extra = sum(1 for idx in layer_indices if idx >= total_layers)
    if not backbone or shard_size <= 1:
        return len(layer_indices)
    return max_owned_count_for_layers(backbone, total_layers, shard_size) + 1 + n_extra


def local_packed_slot_for_layer(
    state: LayerSplitState, layer_idx: int
) -> int:
    """Rank-local packed slot for a full-backbone family (owned or scratch)."""
    if not state.covers(layer_idx):
        raise ValueError(
            f"layer_idx={layer_idx} is outside backbone "
            f"[0, {state.total_layers})"
        )
    n_slots = effective_num_layers(state.total_layers, state.shard_size)
    scratch = n_slots - 1
    if not state.is_owned(layer_idx):
        return scratch
    start, _ = get_layer_shard_range(
        state.shard_rank, state.shard_size, state.total_layers
    )
    return state.local_index(layer_idx) - start


def local_packed_slot_for_family_layer(
    state: LayerSplitState,
    layer_idx: int,
    family_layer_indices: list[int],
) -> int:
    """Rank-local slot within a (possibly sparse) page-size family.

    Layout: ``[dense owned 0 .. max_owned) | scratch | extras...]`` where
    extras are MTP / draft layers with ``idx >= total_layers``. Scratch is
    reserved at ``max_owned`` so ``block_stride`` stays rank-uniform.
    """
    backbone = sorted(
        {idx for idx in family_layer_indices if 0 <= idx < state.total_layers}
    )
    extras = sorted(
        {idx for idx in family_layer_indices if idx >= state.total_layers}
    )
    if not backbone or state.shard_size <= 1:
        ordered = list(family_layer_indices)
        try:
            return ordered.index(layer_idx)
        except ValueError as exc:
            raise ValueError(
                f"layer_idx={layer_idx} not in family {family_layer_indices}"
            ) from exc

    max_owned = max_owned_count_for_layers(
        set(backbone), state.total_layers, state.shard_size
    )
    scratch = max_owned
    if layer_idx >= state.total_layers:
        try:
            return scratch + 1 + extras.index(layer_idx)
        except ValueError as exc:
            raise ValueError(
                f"extra layer_idx={layer_idx} not in family extras {extras}"
            ) from exc
    if layer_idx not in backbone:
        raise ValueError(
            f"layer_idx={layer_idx} not in family backbone {backbone}"
        )
    if not state.is_owned(layer_idx):
        return scratch
    start, end = get_layer_shard_range(
        state.shard_rank, state.shard_size, state.total_layers
    )
    owned_in_family = [idx for idx in backbone if start <= idx < end]
    return owned_in_family.index(layer_idx)


def _layer_page_sizes(kv_cache_config: Any) -> dict[str, int]:
    layer_ps: dict[str, int] = {}
    for group in getattr(kv_cache_config, "kv_cache_groups", []) or []:
        spec = group.kv_cache_spec
        per_layer = getattr(spec, "kv_cache_specs", None)
        # UniformType: prefer the full per-layer map (may be a superset of
        # group.layer_names depending on how groups were assembled).
        if per_layer:
            for layer_name, layer_spec in per_layer.items():
                layer_ps[str(layer_name)] = int(layer_spec.page_size_bytes)
        for layer_name in getattr(group, "layer_names", []) or []:
            if layer_name in layer_ps:
                continue
            layer_ps[str(layer_name)] = int(spec.page_size_bytes)
    return layer_ps


def _infer_page_size_from_packed(
    packed: list[Any], layer_name: str
) -> int | None:
    """Best-effort page size from packed tensor offset cadence."""
    matches = [
        t for t in packed if layer_name in _tensor_layer_names(t)
    ]
    if not matches:
        return None
    # Same-family tensors share a page size; recover it from distinct offsets
    # within the family that also contains this layer (via nearby tensors).
    stride = int(matches[0].block_stride)
    family = [t for t in packed if int(getattr(t, "block_stride", 0) or 0) == stride]
    offsets = sorted({int(t.offset) for t in family})
    if len(offsets) < 2:
        return None
    # GCD of consecutive gaps is the page size (families are packed densely).
    gaps = [b - a for a, b in zip(offsets, offsets[1:]) if b > a]
    if not gaps:
        return None
    g = gaps[0]
    for x in gaps[1:]:
        while x:
            g, x = x, g % x
    return int(g) if g > 0 else None


def _tensor_layer_names(tensor: Any) -> list[str]:
    names = getattr(tensor, "shared_by", None)
    if names is None:
        names = getattr(tensor, "layers", None)
    return list(names or [])


def _primary_layer_name(layer_names: Any) -> str:
    if not layer_names:
        raise ValueError("packed KV tensor has empty layers/shared_by")
    return layer_names[0]


def _kv_tensor_field_names(tensor_cls: type) -> set[str]:
    fields = getattr(tensor_cls, "__dataclass_fields__", None)
    if fields is not None:
        return set(fields)
    # Fallback for simple test doubles.
    try:
        import inspect

        return set(inspect.signature(tensor_cls).parameters)
    except (TypeError, ValueError):
        return set()


@dataclass
class _PackedKVCacheTensor:
    """Descriptor for packed HCU caches on the modern shared-by ABI.

    Some v0.28 builds expose only ``size`` and ``shared_by`` on the upstream
    descriptor.  The HCU runner needs explicit byte offsets for block-outer
    packing, so keep those fields on an HCU-owned descriptor rather than
    silently dropping the packing plan.
    """

    size: int
    shared_by: list[str]
    block_stride: int
    offset: int = 0
    layer_stride: int = 0
    # All descriptors belonging to one allocation carry the same identity.
    # Numeric size/stride alone is not sufficient: two independent pools can
    # legitimately have identical dimensions.
    backing_id: Any | None = None

    @property
    def layers(self) -> list[str]:
        """Compatibility alias used by older HCU planner adapters."""
        return self.shared_by


def _set_packed_backing_id(tensor: Any, backing_id: Any | None) -> Any:
    """Attach an allocation identity when an upstream descriptor permits it."""
    if backing_id is None:
        return tensor
    try:
        setattr(tensor, "backing_id", backing_id)
    except (AttributeError, TypeError) as exc:
        # Frozen/slots-based legacy descriptors cannot carry the metadata.  Do
        # not silently fall back to numeric-size aliasing.
        raise ValueError(
            "Packed KV descriptor cannot preserve backing_id metadata"
        ) from exc
    if packed_kv_backing_id(tensor) != backing_id:
        raise ValueError(
            "Packed KV descriptor did not retain backing_id metadata"
        )
    return tensor


def packed_kv_backing_id(tensor: Any) -> Any | None:
    """Return the explicit packed-allocation identity, if one is present."""
    value = getattr(tensor, "backing_id", None)
    if value is None:
        value = getattr(tensor, "_vllm_hcu_packed_backing_id", None)
    return value


def make_packed_kv_tensor(
    tensor_cls: type,
    *,
    size: int,
    layer_names: list[str],
    page_size: int,
    offset: int,
    block_stride: int,
    backing_id: Any | None = None,
) -> Any:
    """Build a packed ``KVCacheTensor`` for V1 (shared_by) or V2 (layers).

    vLLM 0.25.1 V2 bind path uses::

        offset + l * layer_stride + b * block_stride

    With one layer per tensor (``l == 0``), ``layer_stride`` is unused for
    addressing but still required by the dataclass. Legacy HCU runners use
    ``shared_by`` + ``offset`` + ``block_stride`` instead. ``backing_id`` is
    an opaque identity shared by descriptors that alias one allocation.
    """
    names = list(layer_names)
    if not names:
        raise ValueError("packed KV tensor requires at least one layer name")
    fields = _kv_tensor_field_names(tensor_cls)
    kwargs: dict[str, Any] = {"size": int(size)}
    if {"layers", "layer_stride", "block_stride", "offset"}.issubset(
        fields
    ):
        kwargs["layers"] = names
        kwargs["layer_stride"] = int(page_size)
        kwargs["block_stride"] = int(block_stride)
        kwargs["offset"] = int(offset)
        return _set_packed_backing_id(tensor_cls(**kwargs), backing_id)
    if {"shared_by", "block_stride", "offset"}.issubset(fields):
        kwargs["shared_by"] = names
        if "layer_stride" in fields:
            kwargs["layer_stride"] = int(page_size)
        kwargs["offset"] = int(offset)
        kwargs["block_stride"] = int(block_stride)
        return _set_packed_backing_id(tensor_cls(**kwargs), backing_id)
    if "shared_by" in fields:
        # Modern upstream descriptors omit the packing fields.  Use the HCU
        # descriptor so the runner can still allocate one packed backing pool.
        return _PackedKVCacheTensor(
            size=int(size),
            shared_by=names,
            layer_stride=int(page_size),
            block_stride=int(block_stride),
            offset=int(offset),
            backing_id=backing_id,
        )
    if "layers" in fields:
        return _PackedKVCacheTensor(
            size=int(size),
            shared_by=names,
            layer_stride=int(page_size),
            block_stride=int(block_stride),
            offset=int(offset),
            backing_id=backing_id,
        )
    # Test doubles / permissive constructors.
    try:
        kwargs.update(
            shared_by=names,
            layers=names,
            layer_stride=int(page_size),
            offset=int(offset),
            block_stride=int(block_stride),
        )
        return _set_packed_backing_id(tensor_cls(**kwargs), backing_id)
    except TypeError:
        return _PackedKVCacheTensor(
            size=int(size),
            shared_by=names,
            layer_stride=int(page_size),
            block_stride=int(block_stride),
            offset=int(offset),
            backing_id=backing_id,
        )


def align_packed_block_stride(payload_bytes: int, kv_cache_groups: Any) -> int:
    """Pad a packed block pitch up to a whole number of every layer's rows.

    BF16 MLA reads the packed buffer as ``[blocks, tokens, width]`` and
    requires ``stride(0)`` to be a multiple of the row. Summing page sizes
    can leave a remainder. The pad is a trailing gap inside ``block_stride``;
    layer offsets stay on the unpadded payload. Planner and worker remap
    must apply the same pad or startup rejects the layout.
    """
    align = 1
    for group in kv_cache_groups or []:
        spec = getattr(group, "kv_cache_spec", None)
        per_layer = getattr(spec, "kv_cache_specs", None)
        specs = (
            list(per_layer.values())
            if isinstance(per_layer, dict) and per_layer
            else [spec]
        )
        for layer_spec in specs:
            if layer_spec is None:
                continue
            page = int(getattr(layer_spec, "page_size_bytes", 0) or 0)
            block = int(getattr(layer_spec, "block_size", 0) or 0)
            if page > 0 and block > 0 and page % block == 0:
                align = math.lcm(align, page // block)
    if align > 1 and payload_bytes % align:
        payload_bytes = (payload_bytes + align - 1) // align * align
    return int(payload_bytes)


def remap_packed_kv_tensor_offsets(
    kv_cache_config: Any,
    vllm_config: Any | None = None,
) -> bool:
    """Remap packed layer offsets to owned slots + scratch (worker-local).

    Shared ``KVCacheConfig`` is rank-agnostic; each worker applies this before
    allocate so non-owned layers alias the family scratch slot (SGLang
    ``remote_kv_buffer``). Emits V2 ``layers``/``layer_stride`` when the
    installed ``KVCacheTensor`` requires them.
    Returns True when offsets were rewritten.
    """
    state = get_runtime_layer_split_state(vllm_config)
    if state is None:
        return False
    if getattr(kv_cache_config, "_vllm_hcu_dsa_remap_applied", False):
        return False

    tensors = list(getattr(kv_cache_config, "kv_cache_tensors", []) or [])
    packed = [t for t in tensors if int(getattr(t, "block_stride", 0) or 0) > 0]
    if not packed:
        logger.warning("DSA layer-split packed remap skipped: no block_stride tensors")
        return False

    layer_ps = _layer_page_sizes(kv_cache_config)
    buckets: dict[int, list[str]] = {}
    seen: set[str] = set()
    missing_ps: list[str] = []
    for tensor in packed:
        for layer_name in _tensor_layer_names(tensor):
            if layer_name in seen:
                continue
            seen.add(layer_name)
            page_size = layer_ps.get(layer_name)
            if page_size is None:
                page_size = _infer_page_size_from_packed(packed, layer_name)
            if page_size is None:
                missing_ps.append(str(layer_name))
                continue
            buckets.setdefault(int(page_size), []).append(layer_name)

    # Provisional offsets clamp every layer past the scratch slot onto one
    # slot, so an unremapped layer would overwrite owned cache. Fail closed.
    if missing_ps:
        raise RuntimeError(
            "DSA layer-split packed remap could not resolve page sizes for "
            f"{len(missing_ps)} packed layers (example={missing_ps[:3]}); "
            "refusing to run with aliased owned-layer cache offsets."
        )

    # Stable page-size order so planner/remap agree on block layout.
    page_sizes = sorted(buckets)
    family_indices: dict[int, list[int]] = {}
    phys_slots: dict[int, int] = {}
    for page_size in page_sizes:
        layers = buckets[page_size]
        indices = [parse_layer_index(name) for name in layers]
        family_indices[page_size] = indices
        phys_slots[page_size] = phys_slots_for_layer_family(
            indices, state.total_layers, state.shard_size
        )

    # Match the planner no-shrink fallback: when owned+scratch is not smaller
    # than the native family layout (phys_slots >= original layer count), the
    # shared KV config kept the full stride. Remapping would invent a larger
    # stride and fail the block_stride check (e.g. 1 layer / PCP2).
    owned_scratch_bytes = sum(ps * phys_slots[ps] for ps in page_sizes)
    full_bytes = sum(ps * len(buckets[ps]) for ps in page_sizes)
    if owned_scratch_bytes >= full_bytes:
        logger.warning(
            "DSA layer-split packed remap skipped: no-shrink layout "
            "(owned+scratch=%s >= full=%s) phys_slots=%s bucket_lens=%s",
            owned_scratch_bytes,
            full_bytes,
            phys_slots,
            {ps: len(buckets[ps]) for ps in page_sizes},
        )
        return False

    new_stride = align_packed_block_stride(
        owned_scratch_bytes,
        getattr(kv_cache_config, "kv_cache_groups", None),
    )
    old_stride = int(packed[0].block_stride)
    if old_stride != new_stride:
        raise RuntimeError(
            f"DSA layer-split packed remap: block_stride={old_stride} != "
            f"expected owned+scratch stride={new_stride} "
            f"phys_slots={phys_slots} "
            f"bucket_lens={ {ps: len(buckets[ps]) for ps in page_sizes} }; "
            "the shared KV planner and the worker remap disagree on the "
            "family layout, so owned layers would alias one slot."
        )

    layer_offset: dict[str, int] = {}
    layer_page: dict[str, int] = {}
    base = 0
    for page_size in page_sizes:
        layers = buckets[page_size]
        indices = family_indices[page_size]
        n_phys = phys_slots[page_size]
        for layer_name, layer_idx in zip(layers, indices):
            slot = local_packed_slot_for_family_layer(state, layer_idx, indices)
            if not 0 <= slot < n_phys:
                raise RuntimeError(
                    f"DSA layer-split slot {slot} out of range for "
                    f"page_size={page_size} n_phys={n_phys} layer={layer_name}"
                )
            layer_offset[layer_name] = base + slot * page_size
            layer_page[layer_name] = page_size
        base += n_phys * page_size

    tensor_cls = type(packed[0])
    new_tensors = []
    for tensor in tensors:
        if int(getattr(tensor, "block_stride", 0) or 0) <= 0:
            new_tensors.append(tensor)
            continue
        names = _tensor_layer_names(tensor)
        primary = _primary_layer_name(names)
        page_size = layer_page.get(primary)
        if page_size is None:
            page_size = int(
                getattr(tensor, "layer_stride", 0)
                or layer_ps.get(primary, 0)
                or 0
            )
        if page_size <= 0:
            raise RuntimeError(
                f"DSA layer-split remap missing page_size for {primary}"
            )
        new_tensors.append(
            make_packed_kv_tensor(
                tensor_cls,
                size=int(tensor.size),
                layer_names=names,
                page_size=page_size,
                offset=layer_offset.get(primary, int(tensor.offset)),
                block_stride=new_stride,
                backing_id=packed_kv_backing_id(tensor),
            )
        )

    kv_cache_config.kv_cache_tensors = new_tensors
    try:
        kv_cache_config._vllm_hcu_dsa_remap_applied = True
    except Exception:
        logger.debug("could not tag kv_cache_config as remapped")
    owned_start, owned_end = get_layer_shard_range(
        state.shard_rank, state.shard_size, state.total_layers
    )
    logger.warning(
        "DSA layer-split packed KV remap: rank=%d owned=[%d,%d) "
        "phys_slots=%s block_stride=%d api=%s",
        state.shard_rank,
        owned_start,
        owned_end,
        phys_slots,
        new_stride,
        sorted(_kv_tensor_field_names(tensor_cls)) or "unknown",
    )
    return True


__all__ = [
    "LayerSplitState",
    "begin_mla_dsa_cache_write",
    "broadcast_dsa_cache_from_owner",
    "clear_active_dsa_broadcast_block_ids",
    "clear_broadcast_scratch_for_tests",
    "effective_num_layers",
    "get_layer_owner",
    "get_layer_shard_range",
    "get_runtime_layer_split_state",
    "invalidate_dsa_cache_broadcast",
    "is_dsa_cache_layer_split_requested",
    "is_hcu_dsa_mla_model",
    "local_packed_slot_for_family_layer",
    "local_packed_slot_for_layer",
    "log_layer_shard_plan",
    "make_packed_kv_tensor",
    "max_owned_count_for_layers",
    "maybe_log_runtime_layer_shard_plan",
    "maybe_prefetch_next_dsa_cache",
    "maybe_prefetch_same_layer_mla_cache",
    "parse_layer_index",
    "phys_slots_for_layer_family",
    "packed_kv_backing_id",
    "prefetch_dsa_cache_from_owner",
    "remap_packed_kv_tensor_offsets",
    "set_active_dsa_broadcast_block_ids",
    "set_active_dsa_broadcast_block_ids_from_tables",
    "should_commit_durable_dsa_cache",
    "validate_dsa_cache_layer_split",
]
