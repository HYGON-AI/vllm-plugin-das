# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Offload and parallel-shard DeepSeek V4.1 Engram tables on HCU.

The table layout and lookup kernel remain the community head-sharded design.
For PCP>1, HCU extends the shard count to TP*PCP and gathers PCP token rows
around the existing lookup and TP head collective.  For the first HCU DP
implementation, ``embedding_across_dp=true`` extends the shard count to
TP*EDP and gathers DP token rows using the community Engram DP group.
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
_DP_SIZE_MARKER = "_vllm_hcu_dsv41_engram_dp_size"


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
    if getattr(config, "embedding_across_dp", False) and not getattr(
        config, "cpu_offload", False
    ):
        raise ValueError(
            "HCU DeepSeek V4.1 Engram embedding_across_dp requires "
            "cpu_offload=true"
        )
    if getattr(config, "embedding_across_dp", False) and _pcp_size() > 1:
        raise ValueError(
            "HCU DeepSeek V4.1 Engram DP sharding does not yet support "
            "embedding_across_dp=true together with PCP>1"
        )
    if getattr(config, "embedding_across_dp", False):
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        parallel = getattr(vllm_config, "parallel_config", None)
        group = _engram_dp_group()
        if (
            parallel is None
            or int(parallel.tensor_parallel_size) != 1
            or int(parallel.pipeline_parallel_size) != 1
            or int(parallel.prefill_context_parallel_size) != 1
            or int(parallel.data_parallel_size) <= 1
            or group is None
            or int(group.world_size) <= 1
        ):
            raise ValueError(
                "HCU DeepSeek V4.1 Engram embedding_across_dp currently "
                "requires TP=PCP=PP=1, DP>1, and an initialized Engram DP group"
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


def _engram_dp_group():
    from vllm.distributed.parallel_state import get_engram_dp_group

    return get_engram_dp_group()


def _engram_dp_size() -> int:
    config = _engram_config()
    if config is None or not getattr(config, "embedding_across_dp", False):
        return 1
    group = _engram_dp_group()
    return 1 if group is None else int(group.world_size)


def _engram_dp_rank() -> int:
    group = _engram_dp_group()
    return 0 if group is None else int(group.rank_in_group)


def _engram_instance_dp_size(engram) -> int:
    """Return the DP shard count captured when this Engram was built."""
    embedding = getattr(engram, "embed_tokens", None)
    captured = getattr(embedding, _DP_SIZE_MARKER, None)
    return int(captured) if captured is not None else _engram_dp_size()


def _require_engram_dp_group(dp_size: int):
    """Return the Engram DP group, failing closed on a stale shard count.

    The staged row window is indexed as `slot * group.world_size` with a
    `dp_rank * slot` offset, so a group that no longer matches the shard
    count captured at build time would silently read another rank rows.
    """
    group = _engram_dp_group()
    world = 1 if group is None else int(group.world_size)
    if world != dp_size:
        raise RuntimeError(
            "HCU Engram DP sharding requires the runtime DP group to "
            "match the shard count captured at build time: "
            f"group={world}, shards={dp_size}"
        )
    return group


def _engram_dp_shard_info(
    tp_size: int, tp_rank: int, dp_size: int, dp_rank: int
) -> tuple[int, int]:
    """Return TP-major shard coordinates for the community EDP layout."""
    return tp_size * dp_size, tp_rank * dp_size + dp_rank


def _dp_token_slot(group) -> int:
    from vllm.forward_context import get_forward_context

    metadata = get_forward_context().dp_metadata
    if metadata is None:
        raise RuntimeError(
            "HCU Engram DP sharding requires ForwardContext.dp_metadata"
        )
    counts = metadata.num_tokens_across_dp_cpu
    if counts.numel() != int(group.world_size):
        raise RuntimeError(
            "HCU Engram DP sharding currently requires the node-local Engram "
            "DP group to match the model DP group: "
            f"metadata={counts.numel()}, group={group.world_size}"
        )
    return int(counts.max().item())


def _gather_padded_dp_hash_ids(hash_ids: torch.Tensor):
    """Gather DP replica hashes into a common token slot."""
    group = _engram_dp_group()
    if group is None or int(group.world_size) == 1:
        return hash_ids, int(hash_ids.shape[0])
    hash_ids = hash_ids.contiguous()
    slot = _dp_token_slot(group)
    if hash_ids.shape[0] > slot:
        raise RuntimeError(
            "HCU Engram DP hash batch exceeds the synchronized token slot: "
            f"tokens={hash_ids.shape[0]}, slot={slot}"
        )
    if hash_ids.shape[0] < slot:
        padding = hash_ids.new_full(
            (slot - hash_ids.shape[0], *hash_ids.shape[1:]), -1
        )
        hash_ids = torch.cat((hash_ids, padding), dim=0)
    return group.all_gather(hash_ids, dim=0), slot


def _gather_dp_engram_rows(
    staged: torch.Tensor, local_tokens: int, slot: int, dp_rank: int
) -> torch.Tensor:
    """Gather rank-local head rows and keep this DP replica's tokens."""
    group = _engram_dp_group()
    assert group is not None
    from vllm.models.deepseek_v41.common.engram import _engram_select_rows

    gathered = group.all_gather(staged, dim=0)
    local_heads, dim = staged.shape[1:]
    rows = staged.new_empty((local_tokens, group.world_size * local_heads, dim))
    _engram_select_rows(
        gathered,
        rows,
        staged.shape[0],
        dp_rank * slot,
        local_heads * dim,
    )
    return rows


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
    original_forward = require_callable(
        engram_class, "forward", f"{TARGET_MODULE}.Engram.forward"
    )

    @functools.wraps(original_init_staging)
    def hcu_init_staging(self, max_tokens, head_dim):
        pcp_size = _pcp_size()
        dp_size = _engram_instance_dp_size(self)
        if max(pcp_size, dp_size) == 1 or not getattr(
            self.embed_tokens, _CLASS_MARKER, False
        ):
            return original_init_staging(self, max_tokens, head_dim)
        return original_init_staging(
            self, max_tokens * max(pcp_size, dp_size), head_dim
        )

    @functools.wraps(original_prepare)
    def hcu_prepare_embeddings(self, hash_ids):
        pcp_size = _pcp_size()
        dp_size = _engram_instance_dp_size(self)
        if (pcp_size == 1 and dp_size == 1) or not getattr(
            self.embed_tokens, _CLASS_MARKER, False
        ):
            return original_prepare(self, hash_ids)

        if pcp_size > 1 and dp_size > 1:
            raise RuntimeError("HCU Engram does not yet support PCP+DP sharding")
        if dp_size > 1:
            _require_engram_dp_group(dp_size)
            gathered, slot = _gather_padded_dp_hash_ids(hash_ids)
        else:
            gathered, slot = _gather_padded_hash_ids(hash_ids)
        rows = self.staged_rows[: gathered.shape[0]]
        if rows.shape[0] != gathered.shape[0]:
            raise RuntimeError(
                "HCU PCP Engram staging buffer is too small: "
                f"need {gathered.shape[0]}, have {rows.shape[0]}"
            )
        self.embed_tokens.lookup(gathered, rows)
        if dp_size > 1:
            self._hcu_dp_local_tokens = int(hash_ids.shape[0])
            self._hcu_dp_slot = slot
        else:
            self._hcu_pcp_local_tokens = int(hash_ids.shape[0])
            self._hcu_pcp_slot = slot

    @functools.wraps(original_embed)
    def hcu_embed(self, hash_ids):
        pcp_size = _pcp_size()
        dp_size = _engram_instance_dp_size(self)
        if (pcp_size == 1 and dp_size == 1) or not getattr(
            self.embed_tokens, _CLASS_MARKER, False
        ):
            return original_embed(self, hash_ids)

        if pcp_size > 1 and dp_size > 1:
            raise RuntimeError("HCU Engram does not yet support PCP+DP sharding")

        if dp_size > 1:
            _require_engram_dp_group(dp_size)
            local_tokens = int(
                getattr(self, "_hcu_dp_local_tokens", hash_ids.shape[0])
            )
            slot = int(getattr(self, "_hcu_dp_slot", local_tokens))
            staged = self._ready_rows(slot * dp_size)
            rows = _gather_dp_engram_rows(
                staged, local_tokens, slot, _engram_dp_rank()
            )
            if self.embed_tokens.tp_size > 1:
                from vllm.distributed import tensor_model_parallel_all_gather

                rows = tensor_model_parallel_all_gather(rows, dim=1)
            return rows[:, : self.embed_tokens.n_hash_cols]

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

    @functools.wraps(original_forward)
    def hcu_forward(self, hidden_states, hash_ids, token_mask=None):
        if _engram_instance_dp_size(self) <= 1:
            return original_forward(self, hidden_states, hash_ids, token_mask)

        # DP warmup can leave a replica with no local token rows after the
        # EDP gather.  The upstream method calls WKV before checking the empty
        # output, while HCU channelwise FP8 scaled-mm rejects M=0.
        if hidden_states.shape[0] == 0:
            return torch.empty_like(hidden_states)
        embedded = self.embed(hash_ids)
        if (
            embedded.shape[0] == 0
            or embedded.shape[1] == 0
        ):
            return torch.zeros_like(hidden_states)

        from vllm.models.deepseek_v41.common.engram import (
            _fused_engram_post_wkv_kernel,
        )
        from vllm.triton_utils import triton

        kv = self.wkv(embedded.flatten(-2))
        num_kv_tokens = hash_ids.shape[0]
        if token_mask is not None and token_mask.shape != (num_kv_tokens,):
            raise ValueError("Engram token mask shape does not match hash ids")
        num_tokens = hidden_states.shape[0]
        if self.use_sequence_parallel:
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = (num_kv_tokens + tp_size - 1) // tp_size
            if hidden_states.shape[0] != shard_size:
                raise RuntimeError(
                    "HCU DP Engram sequence-parallel token shape mismatch: "
                    f"hidden={hidden_states.shape[0]}, shard={shard_size}"
                )
            start = min(tp_rank * shard_size, num_kv_tokens)
            num_kv_tokens = min(shard_size, num_kv_tokens - start)
            if token_mask is not None:
                token_mask = token_mask[start : start + num_kv_tokens]

        _, hc_mult, dim = hidden_states.shape
        if kv.ndim != 2 or kv.shape[1] != (hc_mult + 1) * dim:
            raise RuntimeError("HCU DP Engram WKV output shape mismatch")
        output = torch.empty_like(hidden_states)
        block_size = triton.next_power_of_2(dim)
        num_warps = 8 if block_size >= 2048 else 4
        mask = token_mask if token_mask is not None else hidden_states
        _fused_engram_post_wkv_kernel[(num_tokens * hc_mult,)](
            hidden_states,
            kv,
            self.q_weight,
            self.k_weight,
            mask,
            output,
            num_kv_tokens,
            hidden_states.stride(0),
            hidden_states.stride(1),
            hidden_states.stride(2),
            kv.stride(0),
            kv.stride(1),
            self.q_weight.stride(0),
            self.q_weight.stride(1),
            self.k_weight.stride(0),
            self.k_weight.stride(1),
            token_mask.stride(0) if token_mask is not None else 0,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            self.eps,
            self.clamp_value,
            DIM=dim,
            HC_MULT=hc_mult,
            BLOCK_SIZE=block_size,
            HAS_MASK=token_mask is not None,
            num_warps=num_warps,
        )
        return output

    setattr(engram_class, "_init_staging", hcu_init_staging)
    setattr(engram_class, "prepare_embeddings", hcu_prepare_embeddings)
    setattr(engram_class, "embed", hcu_embed)
    setattr(engram_class, "forward", hcu_forward)
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
            setattr(self, _DP_SIZE_MARKER, max(1, _engram_dp_size()))
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
                (
                    "TPxPCP"
                    if _pcp_size() > 1
                    else "TPxEDP"
                    if _engram_dp_size() > 1
                    else "TP"
                ),
            )

        def _get_shard_info(self):
            pcp_size = _pcp_size()
            dp_size = _engram_dp_size()
            if dp_size > 1:
                if pcp_size > 1:
                    raise ValueError(
                        "HCU Engram does not yet support PCP+DP sharding"
                    )
                from vllm.distributed import get_tensor_model_parallel_rank

                return _engram_dp_shard_info(
                    self.tp_size,
                    get_tensor_model_parallel_rank(),
                    dp_size,
                    _engram_dp_rank(),
                )
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
