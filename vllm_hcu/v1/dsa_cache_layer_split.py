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

import os
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Contiguous scratch used when broadcasting non-contiguous packed views.
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
    {"HYV4ForCausalLM", "GlmMoeDsaForCausalLM"}
)


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
    no pipeline parallel, and a DSA MLA architecture (HYV4 / GLM-DSA).
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
        return
    architectures = getattr(model_config, "architectures", None) or ()
    arch_set = set(architectures)
    if not arch_set.intersection(_DSA_LAYER_SPLIT_ARCHITECTURES):
        raise ValueError(
            "enable_dsa_cache_layer_split is only supported for DSA MLA "
            f"models ({sorted(_DSA_LAYER_SPLIT_ARCHITECTURES)}); "
            f"got architectures={list(architectures)!r}."
        )
    use_mla = getattr(model_config, "use_mla", None)
    if use_mla is False:
        raise ValueError(
            "enable_dsa_cache_layer_split requires MLA "
            f"(use_mla=True); got use_mla={use_mla!r}."
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
    pending_remote_layer_id: int | None = None
    pending_remote_broadcast: bool = False
    broadcast_stream: Any | None = None


def _family_kind(layer_name: str) -> str:
    name = layer_name.lower()
    if "indexer" in name or "index_k" in name:
        return "indexer"
    return "mla"


def _family_key(cache: torch.Tensor, layer_name: str) -> str:
    # Shape/dtype identify the packed family; kind separates MLA vs indexer
    # when shapes happen to collide.
    return (
        f"{_family_kind(layer_name)}|"
        f"{cache.device.type}:{cache.device.index}|"
        f"{cache.dtype}|{tuple(cache.shape)}"
    )


def _family_state(cache: torch.Tensor, layer_name: str) -> _FamilyRemoteState:
    key = _family_key(cache, layer_name)
    st = _FAMILY_REMOTE.get(key)
    if st is None:
        st = _FamilyRemoteState()
        _FAMILY_REMOTE[key] = st
    return st


def _scratch_for(cache: torch.Tensor, pool: str = "sync") -> torch.Tensor:
    key = (pool, cache.device.type, cache.device.index, cache.dtype, cache.numel())
    scratch = _broadcast_scratch.get(key)
    if (
        scratch is None
        or scratch.device != cache.device
        or scratch.dtype != cache.dtype
        or scratch.numel() != cache.numel()
    ):
        scratch = torch.empty(cache.numel(), dtype=cache.dtype, device=cache.device)
        _broadcast_scratch[key] = scratch
    return scratch.view(cache.shape)


def _register_layer_cache(cache: torch.Tensor, layer_name: str) -> None:
    """Remember cache tensors so next-layer prefetch can find them."""
    try:
        layer_idx = parse_layer_index(layer_name)
    except Exception:
        return
    kind = _family_kind(layer_name)
    slot = _LAYER_CACHE_REGISTRY.setdefault(layer_idx, {})
    slot[kind] = cache


def _broadcast_tensor_from_owner(
    cache: torch.Tensor,
    layer_idx: int,
    owner: int,
    pcp_group: Any,
) -> None:
    """SGLang ``LayerSplitDSATokenToKVPool._broadcast_tensor_from_owner``.

    Broadcasts ``cache`` from ``owner``. Non-contiguous packed views bounce
    through a contiguous scratch (NCCL requirement).
    """
    if cache.is_contiguous():
        pcp_group.broadcast(cache, src=owner)
        return
    scratch = _scratch_for(cache)
    if int(pcp_group.rank_in_group) == owner:
        scratch.copy_(cache)
    pcp_group.broadcast(scratch, src=owner)
    if int(pcp_group.rank_in_group) != owner:
        cache.copy_(scratch)


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
    with torch.cuda.stream(_BCAST_STREAM):
        if cache.is_contiguous():
            _BCAST_COMM.broadcast(cache, src=owner, stream=_BCAST_STREAM)
        else:
            scratch = _scratch_for(cache, pool="prefetch")
            if my_rank == owner:
                scratch.copy_(cache)
            _BCAST_COMM.broadcast(scratch, src=owner, stream=_BCAST_STREAM)
            if my_rank != owner:
                cache.copy_(scratch)
    fam.broadcast_stream = _BCAST_STREAM
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
        fam.remote_layer_id = fam.pending_remote_layer_id
    fam.pending_remote_layer_id = None


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
    return fam.remote_layer_id == layer_idx


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

    Ensures ``cache`` holds the owning rank's durable bytes for ``layer_name``,
    skipping NCCL when ``remote_layer_id`` already matches (SGLang dedup).

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

    if fam.remote_layer_id == layer_idx:
        return cache

    owner = state.owner_of(layer_idx)
    if pcp_group is None:
        from vllm.distributed.parallel_state import get_pcp_group

        pcp_group = get_pcp_group()

    _broadcast_tensor_from_owner(cache, layer_idx, owner, pcp_group)
    fam.remote_layer_id = layer_idx
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
        fam.pending_remote_layer_id = layer_idx
        fam.pending_remote_broadcast = True
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
    names = getattr(tensor, "layers", None)
    if names is None:
        names = getattr(tensor, "shared_by", None)
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


def make_packed_kv_tensor(
    tensor_cls: type,
    *,
    size: int,
    layer_names: list[str],
    page_size: int,
    offset: int,
    block_stride: int,
) -> Any:
    """Build a packed ``KVCacheTensor`` for V1 (shared_by) or V2 (layers).

    vLLM 0.25.1 V2 bind path uses::

        offset + l * layer_stride + b * block_stride

    With one layer per tensor (``l == 0``), ``layer_stride`` is unused for
    addressing but still required by the dataclass. Legacy HCU runners use
    ``shared_by`` + ``offset`` + ``block_stride`` instead.
    """
    names = list(layer_names)
    if not names:
        raise ValueError("packed KV tensor requires at least one layer name")
    fields = _kv_tensor_field_names(tensor_cls)
    kwargs: dict[str, Any] = {"size": int(size)}
    if "layers" in fields:
        kwargs["layers"] = names
        kwargs["layer_stride"] = int(page_size)
        kwargs["block_stride"] = int(block_stride)
        kwargs["offset"] = int(offset)
    elif "shared_by" in fields:
        kwargs["shared_by"] = names
        if "offset" in fields:
            kwargs["offset"] = int(offset)
        if "block_stride" in fields:
            kwargs["block_stride"] = int(block_stride)
        if "layer_stride" in fields:
            kwargs["layer_stride"] = int(page_size)
    else:
        # Test doubles / permissive constructors.
        kwargs.update(
            shared_by=names,
            layers=names,
            layer_stride=int(page_size),
            offset=int(offset),
            block_stride=int(block_stride),
        )
    return tensor_cls(**kwargs)


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

    if all(phys_slots[ps] == len(buckets[ps]) for ps in page_sizes):
        logger.warning(
            "DSA layer-split packed remap skipped: families already full-size "
            "phys_slots=%s",
            phys_slots,
        )
        return False

    new_stride = sum(ps * phys_slots[ps] for ps in page_sizes)
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
    "clear_broadcast_scratch_for_tests",
    "effective_num_layers",
    "get_layer_owner",
    "get_layer_shard_range",
    "get_runtime_layer_split_state",
    "invalidate_dsa_cache_broadcast",
    "is_dsa_cache_layer_split_requested",
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
    "prefetch_dsa_cache_from_owner",
    "remap_packed_kv_tensor_offsets",
    "should_commit_durable_dsa_cache",
    "validate_dsa_cache_layer_split",
]
