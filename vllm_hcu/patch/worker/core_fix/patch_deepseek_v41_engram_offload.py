# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Offload and PCP-shard DeepSeek V4.1 Engram tables on HCU.

The table layout and lookup kernel remain the community head-sharded design.
For PCP>1, HCU extends the shard count to TP*PCP and gathers PCP token rows
around the existing lookup and TP head collective.
"""

from __future__ import annotations

import functools
import mmap
import weakref
from types import ModuleType

import torch

from vllm.logger import init_logger
from vllm.model_executor.utils import set_weight_attrs

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

logger = init_logger(__name__)

TARGET_MODULE = "vllm.models.deepseek_v41.common.engram"
PATCH_ID = "worker.core_fix.deepseek_v41.engram_offload"
TARGETS = (f"{TARGET_MODULE}.Engram._create_embedding",)
_CLASS_MARKER = "_vllm_hcu_dsv41_engram_offload_storage"
_CREATE_MARKER = "_vllm_hcu_dsv41_engram_offload_create_embedding"
_ENGRAM_METHOD_MARKER = "_vllm_hcu_dsv41_engram_pcp_methods"


def _engram_config():
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    return None if config is None else getattr(config, "engram_config", None)


def _cpu_offload_enabled() -> bool:
    config = _engram_config()
    return bool(config is not None and getattr(config, "cpu_offload", False))


def _require_supported_config() -> None:
    config = _engram_config()
    if config is None:
        raise RuntimeError(
            "DeepSeek V4.1 Engram CPU offload requires EngramConfig, but no "
            "active engram_config was found"
        )
    if getattr(config, "dp_shared_memory", False):
        raise ValueError(
            "HCU DeepSeek V4.1 Engram CPU offload does not yet support "
            "dp_shared_memory=true; use dp_shared_memory=false"
        )
    if getattr(config, "embedding_across_dp", False):
        raise ValueError(
            "HCU DeepSeek V4.1 Engram PCP sharding does not support "
            "embedding_across_dp=true; use embedding_across_dp=false"
        )


def _pcp_group():
    from vllm.distributed.parallel_state import get_pcp_group

    try:
        return get_pcp_group()
    except AssertionError:
        # Model construction tests and PCP=1 startup can precede group setup.
        return None


def _pcp_size() -> int:
    group = _pcp_group()
    return 1 if group is None else int(group.world_size)


def _pcp_rank() -> int:
    group = _pcp_group()
    return 0 if group is None else int(group.rank_in_group)


def _pcp_shard_info(
    tp_size: int, tp_rank: int, pcp_size: int, pcp_rank: int
) -> tuple[int, int]:
    """Return the community-compatible TP-major Engram shard coordinates."""
    return tp_size * pcp_size, tp_rank * pcp_size + pcp_rank


def _gather_padded_hash_ids(hash_ids: torch.Tensor):
    """Return rank-major PCP hashes and the local equal-sized token slot."""
    group = _pcp_group()
    if group is None or int(group.world_size) == 1:
        return hash_ids, int(hash_ids.shape[0])

    # Callers hand in a layer slice of [tokens, layers, heads], which is not
    # contiguous and cannot feed all_gather_into_tensor directly.
    hash_ids = hash_ids.contiguous()
    count = torch.tensor(
        [hash_ids.shape[0]], dtype=torch.int64, device=hash_ids.device
    )
    counts = group.all_gather(count, dim=0)
    slot = int(counts.max().item())
    if hash_ids.shape[0] < slot:
        padding = hash_ids.new_full(
            (slot - hash_ids.shape[0], *hash_ids.shape[1:]), -1
        )
        hash_ids = torch.cat((hash_ids, padding), dim=0)
    return group.all_gather(hash_ids, dim=0), slot


def _install_pcp_engram_methods(engram_class) -> None:
    """Install PCP token staging around the community Engram implementation."""
    if getattr(engram_class, _ENGRAM_METHOD_MARKER, False):
        return

    original_init_staging = require_callable(
        engram_class, "_init_staging", f"{TARGET_MODULE}.Engram._init_staging"
    )
    original_prepare = require_callable(
        engram_class,
        "prepare_embeddings",
        f"{TARGET_MODULE}.Engram.prepare_embeddings",
    )
    original_embed = require_callable(
        engram_class, "embed", f"{TARGET_MODULE}.Engram.embed"
    )

    @functools.wraps(original_init_staging)
    def hcu_init_staging(self, max_tokens, head_dim):
        pcp_size = _pcp_size()
        if pcp_size == 1 or not getattr(
            self.embed_tokens, _CLASS_MARKER, False
        ):
            return original_init_staging(self, max_tokens, head_dim)
        return original_init_staging(self, max_tokens * pcp_size, head_dim)

    @functools.wraps(original_prepare)
    def hcu_prepare_embeddings(self, hash_ids):
        pcp_size = _pcp_size()
        if pcp_size == 1 or not getattr(
            self.embed_tokens, _CLASS_MARKER, False
        ):
            return original_prepare(self, hash_ids)

        gathered, slot = _gather_padded_hash_ids(hash_ids)
        rows = self.staged_rows[: gathered.shape[0]]
        if rows.shape[0] != gathered.shape[0]:
            raise RuntimeError(
                "HCU PCP Engram staging buffer is too small: "
                f"need {gathered.shape[0]}, have {rows.shape[0]}"
            )
        self.embed_tokens.lookup(gathered, rows)
        self._hcu_pcp_local_tokens = int(hash_ids.shape[0])
        self._hcu_pcp_slot = slot

    @functools.wraps(original_embed)
    def hcu_embed(self, hash_ids):
        pcp_size = _pcp_size()
        if pcp_size == 1 or not getattr(
            self.embed_tokens, _CLASS_MARKER, False
        ):
            return original_embed(self, hash_ids)

        group = _pcp_group()
        assert group is not None
        local_tokens = int(
            getattr(self, "_hcu_pcp_local_tokens", hash_ids.shape[0])
        )
        slot = int(getattr(self, "_hcu_pcp_slot", local_tokens))
        staged = self._ready_rows(slot * pcp_size)
        pcp_rows = group.all_gather(staged, dim=1)
        start = int(group.rank_in_group) * slot
        rows = pcp_rows[start : start + local_tokens]
        if self.embed_tokens.tp_size > 1:
            from vllm.distributed import tensor_model_parallel_all_gather

            rows = tensor_model_parallel_all_gather(rows, dim=1)
        return rows[:, : self.embed_tokens.n_hash_cols]

    setattr(engram_class, "_init_staging", hcu_init_staging)
    setattr(engram_class, "prepare_embeddings", hcu_prepare_embeddings)
    setattr(engram_class, "embed", hcu_embed)
    setattr(engram_class, _ENGRAM_METHOD_MARKER, True)


def _is_pin_memory_available() -> bool:
    try:
        from vllm.utils.platform_utils import is_pin_memory_available

        return bool(is_pin_memory_available())
    except (ImportError, AttributeError):
        try:
            probe = torch.empty(1, device="cpu", pin_memory=True)
            del probe
            return True
        except (RuntimeError, TypeError):
            return False


def _is_uva_available() -> bool:
    return hasattr(torch.ops._C, "get_cuda_view_from_cpu_tensor")


def _accelerator_view_from_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.ops._C.get_cuda_view_from_cpu_tensor(tensor)


def _host_register(pointer: int, num_bytes: int) -> None:
    """Register an existing host allocation as pinned device-visible memory."""
    result = torch.cuda.cudart().cudaHostRegister(pointer, num_bytes, 0)
    value = getattr(result, "value", result)
    if value != 0:
        raise RuntimeError(f"cudaHostRegister failed: {result}")


def _host_unregister(pointer: int) -> None:
    try:
        result = torch.cuda.cudart().cudaHostUnregister(pointer)
        value = getattr(result, "value", result)
        if value != 0:
            logger.warning("HCU Engram cudaHostUnregister failed: %s", result)
    except Exception as exc:  # pragma: no cover - best effort teardown
        logger.warning("HCU Engram cudaHostUnregister raised: %r", exc)


def _register_host_tensor(tensor: torch.Tensor, owner: object) -> torch.Tensor:
    """Pin an anonymous host tensor for zero-copy lookup.

    Large ``hipHostMalloc`` requests are unreliable on HCU (observed
    ``hipErrorOutOfMemory`` for 12-47 GiB blocks), so the offload path
    allocates anonymous host memory and registers it with
    ``cudaHostRegister`` instead.  The finalizer is anchored to *owner* (the
    embedding module) because ``nn.Parameter`` shares storage with, but does
    not keep alive, the tensor returned here.
    """
    pointer = tensor.data_ptr()
    num_bytes = tensor.numel() * tensor.element_size()
    _host_register(pointer, num_bytes)
    try:
        if not tensor.is_pinned():
            raise RuntimeError(
                "CUDA did not recognize the registered Engram storage"
            )
    except BaseException:
        # Roll back so a failed check cannot leave a stale registration.
        _host_unregister(pointer)
        raise
    finalizer = weakref.finalize(owner, _host_unregister, pointer)
    finalizer.atexit = False  # type: ignore[misc]
    return tensor


def _allocate_registered_host_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, owner: object
) -> torch.Tensor:
    """Allocate anonymous pages before registering them with the HCU driver.

    HCU's large ``pin_memory=True`` allocation path is unreliable.  Keeping
    the mmap object on the embedding owner is required: ``torch.frombuffer``
    does not make the Python mmap lifetime an explicit part of the module's
    storage contract, while the registered pages must remain alive until the
    owner is destroyed.
    """
    numel = 1
    for dim in shape:
        numel *= dim
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    mapping = mmap.mmap(
        -1,
        nbytes,
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    raw = torch.frombuffer(mapping, dtype=dtype, count=numel).view(shape)
    try:
        registered = _register_host_tensor(raw, owner)
        mappings = getattr(owner, "_hcu_engram_mappings", None)
        if mappings is None:
            mappings = []
            setattr(owner, "_hcu_engram_mappings", mappings)
        mappings.append(mapping)
        return registered
    except BaseException:
        mapping.close()
        raise


def _make_offload_embedding_class(base: type) -> type:
    if getattr(base, _CLASS_MARKER, False):
        return base

    class HcuDeepseekV41ParallelEngramEmbedding(base):
        """Community head shards with HCU PCP-aware host storage."""

        _vllm_hcu_dsv41_engram_offload_storage = True

        def __init__(self, *args, **kwargs):
            self._hcu_uva_views = None
            self._hcu_uva_view_src = None
            super().__init__(*args, **kwargs)
            # Match the official offload implementation for dummy-weight mode.
            set_weight_attrs(self.weight, {"dummy_weight_value": 1.0})
            set_weight_attrs(self.weight_scale_inv, {"dummy_weight_value": 127})
            offloaded_bytes = (
                self.weight.numel() * self.weight.element_size()
                + self.weight_scale_inv.numel()
                * self.weight_scale_inv.element_size()
            )
            logger.info(
                "HCU DeepSeek V4.1 Engram CPU offload active: %.2f GiB per "
                "rank (weight=%s, weight_scale_inv=%s, shard=%s, "
                "dp_shared_memory=false)",
                offloaded_bytes / 1024**3,
                tuple(self.weight.shape),
                tuple(self.weight_scale_inv.shape),
                "TPxPCP" if _pcp_size() > 1 else "TP",
            )

        def _get_shard_info(self):
            pcp_size = _pcp_size()
            if pcp_size == 1:
                return super()._get_shard_info()
            # Match the community EDP TP-major head order.  PCP ranks with a
            # fixed TP rank own adjacent head shards; the PCP collective then
            # restores those heads before the TP collective completes order.
            from vllm.distributed import get_tensor_model_parallel_rank

            return _pcp_shard_info(
                self.tp_size,
                get_tensor_model_parallel_rank(),
                pcp_size,
                _pcp_rank(),
            )

        def _allocate_weights(self):
            if not _is_pin_memory_available():
                raise RuntimeError(
                    "HCU DeepSeek V4.1 Engram CPU offload requires CPU pinned "
                    "memory, but pinned allocation is unavailable"
                )
            requested = (
                self.part_num_embeddings
                * (self.dim + self.dim // self.block_size)
                / 1024**3
            )
            try:
                weight = _allocate_registered_host_tensor(
                    (self.part_num_embeddings, self.dim),
                    torch.float8_e4m3fn,
                    self,
                )
                scales = _allocate_registered_host_tensor(
                    (self.part_num_embeddings, self.dim // self.block_size),
                    torch.uint8,
                    self,
                )
            except (RuntimeError, TypeError) as exc:
                raise RuntimeError(
                    "HCU DeepSeek V4.1 Engram CPU offload failed to allocate "
                    f"{requested:.2f} GiB of registered host memory for this "
                    "rank"
                ) from exc
            return weight, scales

        def _storage(self):
            weight = self.weight.data
            scales = self.weight_scale_inv.data
            if weight.device.type != "cpu" or scales.device.type != "cpu":
                raise RuntimeError(
                    "HCU DeepSeek V4.1 Engram offloaded parameters were moved "
                    "off CPU after loading"
                )
            if not weight.is_pinned() or not scales.is_pinned():
                raise RuntimeError(
                    "HCU DeepSeek V4.1 Engram offloaded parameters must remain "
                    "in pinned CPU memory"
                )
            src = (weight.data_ptr(), scales.data_ptr())
            if self._hcu_uva_view_src != src:
                if not _is_uva_available():
                    raise RuntimeError(
                        "HCU DeepSeek V4.1 Engram CPU offload requires "
                        "torch.ops._C.get_cuda_view_from_cpu_tensor"
                    )
                self._hcu_uva_views = (
                    _accelerator_view_from_cpu_tensor(weight),
                    _accelerator_view_from_cpu_tensor(scales),
                )
                self._hcu_uva_view_src = src
            assert self._hcu_uva_views is not None
            return self._hcu_uva_views

    HcuDeepseekV41ParallelEngramEmbedding.__name__ = (
        "HcuDeepseekV41ParallelEngramEmbedding"
    )
    HcuDeepseekV41ParallelEngramEmbedding.__qualname__ = (
        "HcuDeepseekV41ParallelEngramEmbedding"
    )
    setattr(HcuDeepseekV41ParallelEngramEmbedding, _CLASS_MARKER, True)
    return HcuDeepseekV41ParallelEngramEmbedding


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    engram_class = require_class(target, "Engram", f"{TARGET_MODULE}.Engram")
    embedding_base = require_class(
        target,
        "ParallelEngramEmbedding",
        f"{TARGET_MODULE}.ParallelEngramEmbedding",
    )
    original = require_callable(engram_class, "_create_embedding", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("self", "layout", "layer_hash_index"),
    )
    if getattr(original, _CREATE_MARKER, False):
        return False

    offload_embedding = _make_offload_embedding_class(embedding_base)
    _install_pcp_engram_methods(engram_class)

    @functools.wraps(original)
    def hcu_create_embedding(self, layout, layer_hash_index):
        if not _cpu_offload_enabled():
            return original(self, layout, layer_hash_index)
        _require_supported_config()
        if not _is_pin_memory_available():
            raise RuntimeError(
                "HCU DeepSeek V4.1 Engram CPU offload requires CPU pinned "
                "memory, but pinned allocation is unavailable"
            )
        if not _is_uva_available():
            raise RuntimeError(
                "HCU DeepSeek V4.1 Engram CPU offload requires the HCU UVA "
                "operator torch.ops._C.get_cuda_view_from_cpu_tensor, but it "
                "is unavailable"
            )
        return offload_embedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            tuple(
                size
                for order in layout.primes[layer_hash_index]
                for size in order
            ),
        )

    setattr(hcu_create_embedding, _CREATE_MARKER, True)
    setattr(engram_class, "_vllm_hcu_original_create_embedding", original)
    setattr(engram_class, "_create_embedding", hcu_create_embedding)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
