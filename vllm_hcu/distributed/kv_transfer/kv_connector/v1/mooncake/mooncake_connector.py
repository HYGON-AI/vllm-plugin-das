# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# HCU-owned v0.21 Mooncake connector; migrated from 56 audited legacy segments.
import asyncio
import logging
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import httpx
import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio

import vllm_hcu.platforms.envs as henvs
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    TransferTopology,
    get_current_attn_backend,
    get_current_attn_backends,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.stats import (
    MooncakeKVConnectorStats,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_global_first_rank,
    is_local_first_rank,
)
from vllm.distributed.utils import get_pp_indices
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
from vllm.v1.request import RequestStatus
from vllm.v1.worker.utils import select_common_block_size

logger = init_logger(__name__)

_CMPL_UUID_RE = re.compile(
    r"cmpl-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)


def ttft_trace_enabled() -> bool:
    return henvs.VLLM_HCU_MOONCAKE_TTFT_TRACE


def transfer_id_from_req(
    req_id: str | None,
    kv_params: dict[str, Any] | None = None,
) -> str | None:
    if kv_params and kv_params.get("transfer_id"):
        return str(kv_params["transfer_id"])
    if not req_id:
        return None
    match = _CMPL_UUID_RE.match(req_id)
    if match:
        return f"xfer-{match.group(1)}"
    return None


def log_ttft_event(
    event: str,
    *,
    transfer_id: str | None = None,
    req_id: str | None = None,
    kv_params: dict[str, Any] | None = None,
) -> None:
    if not ttft_trace_enabled():
        return
    tid = transfer_id if transfer_id and transfer_id != "-" else None
    if tid is None:
        tid = transfer_id_from_req(req_id, kv_params)
    logger.debug(
        "Mooncake TTFT_EVENT event=%s ts=%.6f transfer_id=%s req_id=%s",
        event,
        time.time(),
        tid or "-",
        req_id or "-",
    )


try:
    from mooncake.engine import TransferEngine
except ImportError:
    logger.warning(
        "Please install mooncake by following the instructions at "
        "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
        "to run VLLM with MooncakeTransferEngine."
    )
    TransferEngine = None

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

ReqId = str  # Internal scheduler request ID
TransferId = str  # KV transfer coordination ID (shared by P/D)


def _parse_model_layer_index(layer_name: str) -> int:
    parts = layer_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            return int(parts[i + 1])
    raise ValueError(
        f"Cannot parse transformer layer index from KV cache layer name: "
        f"{layer_name}"
    )


def _cache_type_sort_key(layer_name: str) -> int:
    if layer_name.endswith(".indexer") or ".indexer." in layer_name:
        return 0
    return 1


@dataclass(frozen=True)
class TransferRegion:
    base_addr: int
    block_len: int
    kv_block_len: int


def _get_tp_ratio(local_tp_size: int, remote_tp_size: int) -> int:
    """Return TP ratio for heterogeneous KV transfer.

    Sign convention:
    - ``1``: homogeneous TP
    - ``> 1``: P_TP > D_TP (one local block maps into multiple remote slots)
    - ``< 0``: D_TP > P_TP (one remote block pulls from a local head slice)

    Used by region-based planning and per-token MHA copy helpers alike.
    """
    if local_tp_size >= remote_tp_size:
        assert local_tp_size % remote_tp_size == 0, (
            f"Local tensor parallel size {local_tp_size} is not divisible "
            f"by remote tensor parallel size {remote_tp_size}."
        )
        return local_tp_size // remote_tp_size

    assert remote_tp_size % local_tp_size == 0, (
        f"Remote tensor parallel size {remote_tp_size} is not divisible "
        f"by local tensor parallel size {local_tp_size}."
    )
    return -(remote_tp_size // local_tp_size)


def _get_head_split_ratio(tp_ratio: int) -> int:
    """Head-chunk count for heterogeneous TP (always positive)."""
    if tp_ratio > 1:
        return tp_ratio
    if tp_ratio < 0:
        return -tp_ratio
    return 1


def _expand_transfer_regions(
    base_addrs: list[int],
    block_lens: list[int],
    is_kv_layout_blocks_first: bool,
) -> list[TransferRegion]:
    """Expand registered KV tensors into the regions transferred by Mooncake."""
    assert len(base_addrs) == len(block_lens), (
        "Mooncake transfer regions require matching numbers of base addresses "
        f"and block lengths, got {len(base_addrs)} and {len(block_lens)}."
    )
    regions: list[TransferRegion] = []
    for base_addr, block_len in zip(base_addrs, block_lens):
        kv_block_len = block_len // 2 if is_kv_layout_blocks_first else block_len
        regions.append(
            TransferRegion(
                base_addr=base_addr,
                block_len=block_len,
                kv_block_len=kv_block_len,
            )
        )
        if is_kv_layout_blocks_first:
            regions.append(
                TransferRegion(
                    base_addr=base_addr + kv_block_len,
                    block_len=block_len,
                    kv_block_len=kv_block_len,
                )
            )
    return regions


def _compute_sender_transfer_plan(
    local_tp_rank: int,
    local_tp_size: int,
    remote_tp_rank: int,
    remote_tp_size: int,
    local_kv_block_len: int,
    remote_kv_block_len: int,
    producer_cache_replicated: bool,
) -> tuple[bool, int, int, int]:
    """Plan one producer-rank to one consumer-rank copy for heterogeneous TP."""
    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)

    if tp_ratio == 1:
        return True, 0, 0, local_kv_block_len

    if tp_ratio > 0:
        if producer_cache_replicated:
            return local_tp_rank % tp_ratio == 0, 0, 0, local_kv_block_len
        return (
            True,
            0,
            (local_tp_rank % tp_ratio) * local_kv_block_len,
            local_kv_block_len,
        )

    if producer_cache_replicated:
        return True, 0, 0, local_kv_block_len

    ratio_abs = -tp_ratio
    return (
        True,
        (remote_tp_rank % ratio_abs) * remote_kv_block_len,
        0,
        remote_kv_block_len,
    )


def _can_coalesce_block_transfers(
    local_region_block_len: int,
    remote_region_block_len: int,
    src_region_offset: int,
    dst_region_offset: int,
    transfer_len: int,
) -> bool:
    """Whether a contiguous block group can be emitted as one larger copy."""
    return (
        src_region_offset == 0
        and dst_region_offset == 0
        and transfer_len == local_region_block_len
        and transfer_len == remote_region_block_len
    )


def _validate_block_lens_match(
    local_block_lens: list[int],
    remote_block_lens: list[int],
) -> str | None:
    """Cross-validate per-layer block lengths between producer and consumer."""
    if len(local_block_lens) != len(remote_block_lens):
        return (
            "Mooncake block_lens count mismatch between producer and consumer: "
            f"local={len(local_block_lens)}, remote={len(remote_block_lens)}"
        )
    for idx, (local_len, remote_len) in enumerate(
        zip(local_block_lens, remote_block_lens)
    ):
        if local_len != remote_len:
            return (
                f"Mooncake block_len mismatch at cache entry {idx}: "
                f"local={local_len}, remote={remote_len}"
            )
    return None


def _validate_hetero_slot_size_bytes(
    local_slot_size_bytes: int,
    remote_slot_size_bytes: int,
    tp_ratio: int,
) -> str | None:
    """Validate per-token slot sizes for heterogeneous TP."""
    if tp_ratio > 1:
        expected_remote = local_slot_size_bytes * tp_ratio
        if remote_slot_size_bytes != expected_remote:
            return (
                "Mooncake hetero TP slot_size_bytes mismatch (P>D): "
                f"local={local_slot_size_bytes}, remote={remote_slot_size_bytes}, "
                f"expected_remote={expected_remote}, tp_ratio={tp_ratio}"
            )
    elif tp_ratio < 0:
        ratio_abs = -tp_ratio
        expected_remote = local_slot_size_bytes // ratio_abs
        if remote_slot_size_bytes != expected_remote:
            return (
                "Mooncake hetero TP slot_size_bytes mismatch (D>P): "
                f"local={local_slot_size_bytes}, remote={remote_slot_size_bytes}, "
                f"expected_remote={expected_remote}, tp_ratio={tp_ratio}"
            )
    elif local_slot_size_bytes != remote_slot_size_bytes:
        return (
            "Mooncake slot_size_bytes mismatch between producer and consumer: "
            f"local={local_slot_size_bytes}, remote={remote_slot_size_bytes}"
        )
    return None


def _validate_hetero_block_lens_match(
    local_block_lens: list[int],
    remote_block_lens: list[int],
    tp_ratio: int,
) -> str | None:
    """Cross-validate per-layer block lengths for heterogeneous TP."""
    if len(local_block_lens) != len(remote_block_lens):
        return (
            "Mooncake block_lens count mismatch between producer and consumer: "
            f"local={len(local_block_lens)}, remote={len(remote_block_lens)}"
        )
    for idx, (local_len, remote_len) in enumerate(
        zip(local_block_lens, remote_block_lens)
    ):
        if tp_ratio > 1:
            expected_remote = local_len * tp_ratio
            if remote_len != expected_remote:
                return (
                    f"Mooncake hetero TP block_len mismatch at cache entry {idx} "
                    f"(P>D): local={local_len}, remote={remote_len}, "
                    f"expected_remote={expected_remote}, tp_ratio={tp_ratio}"
                )
        elif tp_ratio < 0:
            ratio_abs = -tp_ratio
            expected_remote = local_len // ratio_abs
            if remote_len != expected_remote:
                return (
                    f"Mooncake hetero TP block_len mismatch at cache entry {idx} "
                    f"(D>P): local={local_len}, remote={remote_len}, "
                    f"expected_remote={expected_remote}, tp_ratio={tp_ratio}"
                )
        elif local_len != remote_len:
            return (
                f"Mooncake block_len mismatch at cache entry {idx}: "
                f"local={local_len}, remote={remote_len}"
            )
    return None


def _uses_mha_hetero_per_layer_transfers(
    *,
    split_k_and_v: bool,
    homogeneous_tp: bool,
    use_mla: bool,
    split_kv_cache_layout: str,
) -> bool:
    """Whether hetero TP uses per-layer 3-case MHA transfer helpers."""
    return (
        not use_mla
        and not homogeneous_tp
        and (split_k_and_v or split_kv_cache_layout != "HND")
    )


def _append_homogeneous_fa_layer_transfers(
    *,
    src_ptrs: list[int],
    dst_ptrs: list[int],
    lengths: list[int],
    local_layer_addr: int,
    layer_block_len: int,
    remote_layer_addr: int,
    layer_remote_block_len: int,
    group_local_block_ids: list[list[int]],
    group_remote_block_ids: list[list[int]],
) -> None:
    """Grouped contiguous transfer for homogeneous TP with FA split K/V."""
    for group_local_block_id, group_remote_block_id in zip(
        group_local_block_ids, group_remote_block_ids
    ):
        num_blocks = len(group_local_block_id)
        src_ptrs.append(
            local_layer_addr + group_local_block_id[0] * layer_block_len
        )
        dst_ptrs.append(
            remote_layer_addr + group_remote_block_id[0] * layer_remote_block_len
        )
        lengths.append(layer_remote_block_len * num_blocks)


def _append_mha_hetero_tp_layer_transfers(
    *,
    src_ptrs: list[int],
    dst_ptrs: list[int],
    lengths: list[int],
    local_layer_addr: int,
    layer_block_len: int,
    remote_layer_addr: int,
    layer_remote_block_len: int,
    group_local_block_ids: list[list[int]],
    group_remote_block_ids: list[list[int]],
    tp_ratio: int,
    split_ratio: int,
    split_kv: bool,
    split_kv_is_nhd: bool,
    head_rank: int,
    block_size: int,
    slot_size_bytes: int,
) -> None:
    """Per-layer MHA/MQA/GQA KV copy with heterogeneous TP.

    Callers must guarantee ``tp_ratio != 1`` (homogeneous TP uses other paths).

    3-case dispatch:
    1. P_TP > D_TP (``tp_ratio > 1``): per-token or block slice
    2. D_TP > P_TP, HND split K/V: block-level head chunk per block
    3. D_TP > P_TP, NHD split or combined K+V: per-token head slice
    """
    assert tp_ratio != 1, (
        "_append_mha_hetero_tp_layer_transfers requires heterogeneous TP"
    )
    rank_offset = 0
    if split_ratio > 1:
        rank_offset = (head_rank % split_ratio) * layer_remote_block_len

    for group_local_block_id, group_remote_block_id in zip(
        group_local_block_ids, group_remote_block_ids
    ):
        if tp_ratio > 1:
            # Case 1: P_TP > D_TP
            chunk_idx = head_rank % split_ratio
            if split_kv and not split_kv_is_nhd:
                for l_idx, r_idx in zip(group_local_block_id, group_remote_block_id):
                    src_ptrs.append(local_layer_addr + l_idx * layer_block_len)
                    dst_ptrs.append(
                        remote_layer_addr
                        + r_idx * layer_remote_block_len
                        + chunk_idx * layer_block_len
                    )
                    lengths.append(layer_block_len)
            else:
                pos_stride_p = slot_size_bytes
                pos_stride_d = pos_stride_p * split_ratio
                h_off = chunk_idx * pos_stride_p
                for l_idx, r_idx in zip(group_local_block_id, group_remote_block_id):
                    for p in range(block_size):
                        src_ptrs.append(
                            local_layer_addr + l_idx * layer_block_len + p * pos_stride_p
                        )
                        dst_ptrs.append(
                            remote_layer_addr
                            + r_idx * layer_remote_block_len
                            + p * pos_stride_d
                            + h_off
                        )
                        lengths.append(pos_stride_p)
                        if not split_kv:
                            src_ptrs.append(
                                local_layer_addr
                                + l_idx * layer_block_len
                                + layer_block_len // 2
                                + p * pos_stride_p
                            )
                            dst_ptrs.append(
                                remote_layer_addr
                                + r_idx * layer_remote_block_len
                                + layer_remote_block_len // 2
                                + p * pos_stride_d
                                + h_off
                            )
                            lengths.append(pos_stride_p)
        elif tp_ratio < 0 and split_kv and not split_kv_is_nhd:
            # Case 2: D_TP > P_TP with HND split K/V
            for l_idx, r_idx in zip(group_local_block_id, group_remote_block_id):
                src_ptrs.append(
                    local_layer_addr + l_idx * layer_block_len + rank_offset
                )
                dst_ptrs.append(
                    remote_layer_addr + r_idx * layer_remote_block_len
                )
                lengths.append(layer_remote_block_len)
        else:
            # Case 3: D_TP > P_TP with NHD split or combined K+V
            ratio_abs = -tp_ratio
            pos_stride_p = slot_size_bytes
            pos_stride_d = pos_stride_p // ratio_abs
            h_off_bytes = (head_rank % ratio_abs) * pos_stride_d
            for l_idx, r_idx in zip(group_local_block_id, group_remote_block_id):
                for p in range(block_size):
                    src_ptrs.append(
                        local_layer_addr
                        + l_idx * layer_block_len
                        + p * pos_stride_p
                        + h_off_bytes
                    )
                    dst_ptrs.append(
                        remote_layer_addr
                        + r_idx * layer_remote_block_len
                        + p * pos_stride_d
                    )
                    lengths.append(pos_stride_d)
                    if not split_kv:
                        src_ptrs.append(
                            local_layer_addr
                            + l_idx * layer_block_len
                            + layer_block_len // 2
                            + p * pos_stride_p
                            + h_off_bytes
                        )
                        dst_ptrs.append(
                            remote_layer_addr
                            + r_idx * layer_remote_block_len
                            + layer_remote_block_len // 2
                            + p * pos_stride_d
                        )
                        lengths.append(pos_stride_d)


def _validate_asymmetric_region_lengths(
    local_regions: list[TransferRegion],
    remote_regions: list[TransferRegion],
    local_tp_size: int,
    remote_tp_size: int,
    producer_cache_replicated: bool,
) -> str | None:
    """Validate transfer-region metadata for a fixed producer/consumer pair.

    This checks registered KV regions, not per-request block counts. A region
    corresponds to one registered KV tensor, or one K/V half after expansion
    for layouts that store K and V together.
    """
    if len(local_regions) != len(remote_regions):
        return (
            "Mooncake asymmetric TP requires matching KV region counts between "
            "producer and consumer."
        )

    if producer_cache_replicated:
        return None

    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)
    for idx, (local_region, remote_region) in enumerate(
        zip(local_regions, remote_regions)
    ):
        if tp_ratio == 1:
            if local_region.kv_block_len != remote_region.kv_block_len:
                return (
                    "Mooncake KV region length mismatch for homogeneous TP at "
                    f"region {idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}."
                )
        elif tp_ratio > 0:
            if remote_region.kv_block_len != local_region.kv_block_len * tp_ratio:
                return (
                    "Mooncake destination KV region length does not match the "
                    "producer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )
        else:
            ratio_abs = -tp_ratio
            if local_region.kv_block_len != remote_region.kv_block_len * ratio_abs:
                return (
                    "Mooncake source KV region length does not match the "
                    "consumer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )

    return None


def _get_tensor_dense_flag(tensor: torch.Tensor) -> bool | None:
    is_dense = getattr(tensor, "is_non_overlapping_and_dense", None)
    if callable(is_dense):
        return bool(is_dense())
    return None


class MooncakeXferMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict[ReqId, tuple[TransferId, list[list[int]]]]
    kv_caches_base_addr: list[int]
    block_lens: list[int]
    block_len: int = 0
    slot_size_bytes: int = 0
    # For asymmetric PP: pair P/D caches by global model layer index.
    src_layer_offset: int = 0
    model_layer_start: int = -1
    model_layer_end: int = -1
    # Head-chunk index for heterogeneous TP (D TP rank when D>P, P TP
    # chunk index when P>D). remote_tp_rank always carries the decoder TP
    # rank for pairing bookkeeping.
    xfer_head_rank: int = -1


class MooncakeXferResponseStatus(IntEnum):
    # Transfer finished
    FINISH = 0
    # Continue to receive
    CONTINUE = 1
    # Something wrong, see err_msg
    ERROR = 2


class MooncakeXferResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    status: MooncakeXferResponseStatus
    ok_reqs: list[ReqId] | None = None
    err_reqs: list[ReqId] | None = None
    err_msg: str | None = None


@dataclass
class PullReqMeta:
    d_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[list[int]]
    remote_engine_id: EngineId
    remote_bootstrap_addr: str
    # Set expire time to avoid infinitely sending requests.
    expire_time: float = float("inf")
    # Designed for one D pairing to multiple P
    pull_tasks_count: int = 0


@dataclass
class SendBlockMeta:
    p_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[list[int]]
    ready: asyncio.Event
    expire_time: float = float("inf")
    need_send: int = 0
    sent: int = 0
    sending: int = 0
    ttft_send_start_logged: bool = False


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        # Use (engine_id, dp_rank) to group reqs with same dp.
        # See comments in MooncakeBootstrapServer.
        self.reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]] = defaultdict(dict)
        self.reqs_to_send: dict[ReqId, tuple[TransferId, list[list[int]]]] = {}
        self.reqs_not_processed: set[TransferId] = set()

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[list[int]],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
    ):
        transfer_id = kv_transfer_params["transfer_id"]
        if load_remote_cache:
            remote_engine_id = kv_transfer_params["remote_engine_id"]
            self.reqs_to_recv[remote_engine_id][request_id] = PullReqMeta(
                d_req_id=request_id,
                local_block_ids=local_block_ids,
                remote_engine_id=remote_engine_id,
                remote_bootstrap_addr=kv_transfer_params["remote_bootstrap_addr"],
                transfer_id=transfer_id,
            )
        else:
            self.reqs_to_send[request_id] = (transfer_id, local_block_ids)


class MooncakeConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            assert kv_cache_config is not None, (
                "kv_cache_config is required for SCHEDULER role"
            )
            self.connector_scheduler: MooncakeConnectorScheduler | None = (
                MooncakeConnectorScheduler(vllm_config, self.engine_id, kv_cache_config)
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig):
        if vllm_config.model_config is None:
            # This fallback mostly exists for unit tests that instantiate the
            # connector without a fully populated model config.
            logger.warning_once(
                "Unable to detect current VLLM config. "
                "Fallback to default kv cache layout."
            )
            return None
        if vllm_config.model_config.use_mla:
            return None
        # HCU CUSTOM FA writes V into a contiguous NHD-style split buffer.
        # Forcing HND yields a permuted non-contiguous V view and corrupts
        # reshape_and_cache / varlen prefill on the producer.
        if henvs.VLLM_HCU_USE_CUSTOM_FLASH_ATTN:
            backend = get_current_attn_backend(vllm_config)
            if backend.__name__ == "HcuFlashAttentionBackend":
                logger.info_once(
                    "MooncakeConnector keeping default NHD KV cache layout for "
                    "HCU CUSTOM FA (HND breaks V cache writes)."
                )
                return None
        logger.info_once(
            "MooncakeConnector setting KV cache layout to HND for "
            "heterogeneous TP-safe KV transfer."
        )
        return "HND"

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        """MooncakeConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        pass

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Return worker-local transfer stats since the last call.

        Note the P/D asymmetry: because Mooncake is P-push (P calls
        batch_transfer_sync_write), P records successful transfer latency,
        bytes, and descriptor counts, while D only records failures
        (recv/ZMQ errors). Aggregated NIXL-style dashboards will find
        successful-transfer metrics on the P worker, not D.
        """
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return MooncakeKVConnectorStats(data=data or {})


class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size

        assert vllm_config.kv_transfer_config
        self.is_kv_producer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_producer"
        )
        self.is_kv_consumer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        )
        logger.info("Initializing Mooncake Transfer Engine Scheduler %s", engine_id)

        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and any(
                not isinstance(g.kv_cache_spec, FullAttentionSpec)
                for g in kv_cache_config.kv_cache_groups
            )
        )

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[list[int]]]] = {}
        self._reqs_need_send: dict[ReqId, tuple[Request, list[list[int]]]] = {}
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[TransferId] = set()

        # Compute sliding window block counts per KV cache group.
        sw_sizes_tokens: list[tuple[int, int]] = [
            (g.kv_cache_spec.sliding_window, g.kv_cache_spec.block_size)
            if isinstance(g.kv_cache_spec, SlidingWindowSpec)
            else (0, self.block_size)
            for g in kv_cache_config.kv_cache_groups
        ]
        # cdiv(n_tokens, block_size) gives blocks/window; add 1 to
        # conservatively account for boundary overlap.
        self.blocks_per_sw = [
            cdiv(n_tokens, block_size) + 1 if n_tokens else 0
            for n_tokens, block_size in sw_sizes_tokens
        ]

    def get_sw_clipped_blocks(
        self,
        block_ids: tuple[list[int], ...] | list[list[int]],
    ) -> list[list[int]]:
        """Clip per-group block IDs to sliding window size."""
        if len(block_ids) == 0 or not self._is_hma_required:
            return list(block_ids)
        return [
            blocks[-self.blocks_per_sw[i] :] if self.blocks_per_sw[i] > 0 else blocks
            for i, blocks in enumerate(block_ids)
        ]

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if not params:
            return 0, False

        if params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            assert not self.is_kv_producer
            token_ids = request.prompt_token_ids or []
            count = len(token_ids) - num_computed_tokens
            if count > 0:
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: "
            "req_id=%s num_external_tokens=%s, kv_transfer_params=%s",
            request.request_id,
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("do_remote_prefill"):
            assert not self.is_kv_producer
            if all(
                p in params
                for p in ("remote_engine_id", "remote_bootstrap_addr", "transfer_id")
            ):
                # If remote_blocks and num_external_tokens = 0, we have
                # a full prefix cache hit on the D worker. We need to call
                # send_notif in _read_blocks to free the memory on the P.
                unhashed_block_ids = (
                    blocks.get_unhashed_block_ids_all_groups()
                    if num_external_tokens > 0
                    else ()
                )
                local_block_ids = self.get_sw_clipped_blocks(unhashed_block_ids)
                logger.debug(
                    "Mooncake pull blocks for req %s: unhashed=%s clipped=%s",
                    request.request_id,
                    unhashed_block_ids,
                    local_block_ids,
                )
                # Get unhashed blocks to pull from remote.
                self._reqs_need_recv[request.request_id] = (request, local_block_ids)
                log_ttft_event(
                    "d_alloc",
                    transfer_id=params.get("transfer_id"),
                    req_id=request.request_id,
                )
            else:
                logger.warning(
                    "Got invalid KVTransferParams: %s. This "
                    "request will not utilize KVTransfer",
                    params,
                )
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

        elif params.get("do_remote_decode"):
            assert not self.is_kv_consumer
            if not params.get("transfer_id"):
                logger.warning("Missing transfer_id in kv_transfer_params from router!")
            else:
                # Add an empty list to worker to create event.
                self._reqs_need_send[request.request_id] = (request, [])
                log_ttft_event(
                    "p_alloc",
                    transfer_id=params.get("transfer_id"),
                    req_id=request.request_id,
                )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        # Loop through scheduled reqs and convert to PullReqMeta.
        if not self.is_kv_producer:
            for req_id, (req, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                )
            self._reqs_need_recv.clear()

        if not self.is_kv_consumer:
            for req_id, (req, block_ids) in self._reqs_need_send.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    load_remote_cache=False,
                )
            self._reqs_need_send.clear()
            meta.reqs_not_processed = self._reqs_not_processed
            self._reqs_not_processed = set()

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished, req_id=%s, request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params or not params.get("transfer_id"):
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            assert not self.is_kv_producer
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if not params.get("do_remote_decode"):
            return False, None

        assert not self.is_kv_consumer

        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            # Also include the case of a P/D Prefill request with immediate
            # block free (eg abort). Stop tracking this request.
            self._reqs_not_processed.add(params["transfer_id"])
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = any(len(group) > 0 for group in block_ids)

        if delay_free_blocks:
            self._reqs_need_send[request.request_id] = (
                request,
                self.get_sw_clipped_blocks(block_ids),
            )

        return delay_free_blocks, None


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        if TransferEngine is None:
            logger.error("Mooncake is not available")
            raise RuntimeError("Mooncake is not available")
        logger.info("Initializing Mooncake Transfer Engine worker %s", engine_id)

        self.vllm_config = vllm_config
        # Capture device BEFORE TransferEngine init — MNNVL's NVLink allocator
        # may change the current CUDA device during engine.initialize().
        self.device_id = torch.accelerator.current_device_index()
        current_platform.set_device(self.device_id)

        self.engine = TransferEngine()
        self.hostname = get_ip()

        assert (kv_transfer_config := vllm_config.kv_transfer_config)
        self.is_kv_producer: bool = kv_transfer_config.kv_role == "kv_producer"
        self.is_kv_consumer: bool = kv_transfer_config.kv_role == "kv_consumer"
        self.num_sender_workers = kv_transfer_config.kv_connector_extra_config.get(
            "num_workers", 10
        )
        # Create more tasks than workers to keep the thread pool saturated.
        # Tasks can await async events, so a surplus (2x is a robust heuristic)
        # prevents workers from idling.
        self.num_sender_tasks = self.num_sender_workers * 2
        protocol = kv_transfer_config.kv_connector_extra_config.get(  # type: ignore[union-attr]
            "mooncake_protocol", "rdma"
        )
        logger.info(
            "The Mooncake Transfer Engine is using %s as its protocol.", protocol
        )
        ret_value = self.engine.initialize(self.hostname, "P2PHANDSHAKE", protocol, "")
        if ret_value != 0:
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

        self.rpc_port = self.engine.get_rpc_port()

        logger.debug(
            "Mooncake Transfer Engine initialized at %s:%d",
            self.hostname,
            self.rpc_port,
        )

        self._remote_agents: dict[EngineId, dict[int, dict[int, str]]] = {}
        self._pending_bootstrap_queries: dict[str, asyncio.Event] = {}
        self.side_channel_port: int = 0  # we will bind it in register_kv_caches()
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_blocks = 0
        self.block_len_per_layer: list[int] = []
        self.seen_base_addresses: list[int] = []

        assert (parallel_config := vllm_config.parallel_config)
        dp_rank = parallel_config.data_parallel_index
        dp_local_rank = parallel_config.data_parallel_rank_local
        self.dp_rank = dp_local_rank if parallel_config.local_engines_only else dp_rank
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.pp_rank = get_pp_group().rank_in_group
        self.pp_size = pp_size

        self.kv_caches_base_addr: list[int] = []
        self.cache_entry_model_layer: list[int] = []
        self.cache_entry_layer_names: list[str] = []
        self.model_layer_to_cache_indices: dict[int, list[int]] = {}
        self.device_kv_caches: dict[str, torch.Tensor] = {}
        self.reqs_need_send: dict[TransferId, SendBlockMeta] = {}

        # For kv_both, we will act both prefiller and decoder.
        if not self.is_kv_consumer:
            # Background threads for sending kvcaches to D.
            # Each pool thread must be bound to the correct CUDA device
            # because CUDA device selection is thread-local.
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_sender_workers,
                thread_name_prefix="vllm-mooncake-sender",
                initializer=self._bind_sender_thread_device,
            )
            logger.debug(
                "Mooncake Prefiller: use %d workers to send kvcaches",
                self.num_sender_workers,
            )
            # An asyncio queue to buffer incoming requests for the sender
            self.sender_worker_queue = asyncio.Queue[tuple[bytes, bytes]]()
            self.sender_loop = asyncio.new_event_loop()
            # Background thread for processing new sending requests.
            self._sender_listener_t = threading.Thread(
                target=_async_loop, args=(self.sender_loop,), daemon=True
            )
            self._sender_listener_t.start()

            # Start bootstrap server on global rank 0.
            if should_launch_bootstrap_server(vllm_config):
                _, port = get_mooncake_bootstrap_addr(vllm_config)
                self.bootstrap_server = MooncakeBootstrapServer("0.0.0.0", port)
                self.bootstrap_server.start()

        if not self.is_kv_producer:
            self.receiver_loop = asyncio.new_event_loop()
            self._mooncake_receiver_t = threading.Thread(
                target=_async_loop, args=(self.receiver_loop,), daemon=True
            )
            self._mooncake_receiver_t.start()
            logger.debug("Mooncake Decoder: start receiver thread")

        self.finished_sending_reqs: set[ReqId] = set()
        self.finished_recving_reqs: set[ReqId] = set()

        self.xfer_stats = MooncakeKVConnectorStats()

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.kv_cache_config = kv_cache_config
        self.use_mla = self.model_config.use_mla
        self.block_len_per_layer: list[int] = []
        self._sync_block_size_with_kernel()

        # Get the attention backend from the first layer
        # NOTE (NickLucche) models with multiple backends are not supported yet
        backend = get_current_attn_backend(vllm_config)
        self.backend_name = backend.get_name()
        self.kv_cache_layout = get_kv_cache_layout()
        logger.debug("Detected attention backend %s", self.backend_name)
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

        if self.kv_cache_layout == "HND":
            self.split_kv_cache_layout = "HND"
        elif (
            backend.__name__ == "HcuFlashAttentionBackend"
            and henvs.VLLM_HCU_USE_CUSTOM_FLASH_ATTN
        ):
            self.split_kv_cache_layout = "HND"
        else:
            self.split_kv_cache_layout = "NHD"
        logger.debug(
            "Detected split KV layout %s from backend=%s "
            "VLLM_HCU_USE_CUSTOM_FLASH_ATTN=%s",
            self.split_kv_cache_layout,
            backend.__name__,
            henvs.VLLM_HCU_USE_CUSTOM_FLASH_ATTN,
        )

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.tp_size}
        self.transfer_topo = TransferTopology(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            is_mamba=False,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=[backend],
        )

        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
        self._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)

    def _sync_block_size_with_kernel(self) -> None:
        # When speculative decoding (e.g. Eagle) is enabled, the main model
        # and draft model may use different attention backends with different
        # physical block sizes. Pick the common (smallest) block size so that
        # KV-cache registration and transfer work correctly for both models.
        backends = get_current_attn_backends(self.vllm_config)
        kernel_block_size = select_common_block_size(self.block_size, backends)
        if self.block_size != kernel_block_size:
            logger.info_once(
                "User-specified logical block size (%s) does not match"
                " physical kernel block size (%s). Using the latter.",
                self.block_size,
                kernel_block_size,
            )
            assert self.block_size > kernel_block_size
            self.block_size = kernel_block_size

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Cleanup background threads on destruction."""
        self.async_zmq_ctx.term()
        if not self.is_kv_consumer:
            self._sender_executor.shutdown(wait=False)
            if self.sender_loop.is_running():
                self.sender_loop.call_soon_threadsafe(self.sender_loop.stop)
                self._sender_listener_t.join()
            if should_launch_bootstrap_server(self.vllm_config) and hasattr(
                self, "bootstrap_server"
            ):
                self.bootstrap_server.shutdown()
        if not self.is_kv_producer and self.receiver_loop.is_running():
            self.receiver_loop.call_soon_threadsafe(self.receiver_loop.stop)
            self._mooncake_receiver_t.join()

    async def register_worker_with_bootstrap(self):
        host, port = get_mooncake_bootstrap_addr(self.vllm_config)
        url = make_zmq_path("http", host, port) + "/register"
        worker_addr = make_zmq_path("tcp", self.hostname, self.side_channel_port)
        payload = RegisterWorkerPayload(
            engine_id=self.engine_id,
            dp_rank=self.dp_rank,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            addr=worker_addr,
        )
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload.model_dump())
                    response.raise_for_status()
                logger.debug("Successfully registered with bootstrap server at %s", url)
                break
            except httpx.ConnectError:
                # Bootstrap server not ready, wait for a while and retry.
                await asyncio.sleep(1)
            except Exception as e:
                err_msg = (
                    e.response.text if isinstance(e, httpx.HTTPStatusError) else str(e)
                )
                logger.error(
                    "Error registering %s with bootstrap server: %s", payload, err_msg
                )
                raise e

    async def _mooncake_sender_listener(self, ready_event: threading.Event):
        """
        Background thread that listens for Mooncake requests, dispatches them
        to a thread pool, and sends acknowledgments upon completion.
        """

        sock = self.async_zmq_ctx.socket(zmq.ROUTER)
        self.side_channel_port = sock.bind_to_random_port(f"tcp://{self.hostname}")
        logger.debug(
            "Mooncake sender starting listening on path: tcp://%s:%d",
            self.hostname,
            self.side_channel_port,
        )

        await self.register_worker_with_bootstrap()

        # Create async worker tasks that process items from the queue
        sender_tasks = [
            asyncio.create_task(self._sender_worker(sock))
            for _ in range(self.num_sender_tasks)
        ]

        ready_event.set()

        try:
            while True:
                identity, metadata_bytes = await sock.recv_multipart()
                await self.sender_worker_queue.put((identity, metadata_bytes))
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake sender thread.")
        except Exception as e:
            logger.error("Error in Mooncake sender thread: %s. Exiting thread.", str(e))
        finally:
            # Clean up worker tasks
            for task in sender_tasks:
                task.cancel()
            await asyncio.gather(*sender_tasks, return_exceptions=True)
            sock.close()

    async def _sender_worker(self, sock: zmq.asyncio.Socket):
        while True:
            try:
                identity, metadata_bytes = await self.sender_worker_queue.get()
                try:
                    metadata = self._xfer_meta_decoder.decode(metadata_bytes)
                    await self.send_kv_to_decode(identity, sock, metadata)
                except Exception as e:
                    logger.error("Error processing Mooncake xfer request: %s", e)
                    error_response = MooncakeXferResponse(
                        status=MooncakeXferResponseStatus.ERROR, err_msg=str(e)
                    )
                    await sock.send_multipart(
                        (identity, self._encoder.encode(error_response))
                    )
                finally:
                    self.sender_worker_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in _sender_worker: %s", e)

    async def send_kv_to_decode(
        self, identity: bytes, sock: zmq.asyncio.Socket, meta: MooncakeXferMetadata
    ):
        pending_reqs: dict[ReqId, SendBlockMeta] = {}
        remote_tp_ranks = self.transfer_topo.handshake_target_ranks(meta.remote_tp_size)
        homogeneous_tp = self.tp_size == meta.remote_tp_size
        if homogeneous_tp and meta.remote_tp_rank not in remote_tp_ranks:
            logger.warning(
                "Mooncake handshake: D remote_tp_rank %d not in expected pairing "
                "%s for P tp_rank %d (continuing with block_lens validation)",
                meta.remote_tp_rank,
                remote_tp_ranks,
                self.tp_rank,
            )
        xfer_tp_ratio = (
            _get_tp_ratio(self.tp_size, meta.remote_tp_size)
            if not homogeneous_tp
            else 1
        )
        slot_err: str | None = None
        if not homogeneous_tp and meta.block_lens:
            if meta.slot_size_bytes <= 0:
                slot_err = (
                    "Mooncake hetero TP handshake requires slot_size_bytes > 0 "
                    f"(got {meta.slot_size_bytes})"
                )
            else:
                slot_err = _validate_hetero_slot_size_bytes(
                    self.slot_size_bytes,
                    meta.slot_size_bytes,
                    tp_ratio=xfer_tp_ratio,
                )
        elif meta.block_lens and meta.slot_size_bytes > 0:
            slot_err = _validate_hetero_slot_size_bytes(
                self.slot_size_bytes, meta.slot_size_bytes, tp_ratio=1
            )
        if slot_err is not None:
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=slot_err,
            )
            await sock.send_multipart(
                (identity, self._encoder.encode(response))
            )
            return
        local_base_addrs, local_block_lens = self._resolve_sender_cache_addrs(meta)
        if meta.block_lens:
            if homogeneous_tp:
                block_lens_err = _validate_block_lens_match(
                    local_block_lens, meta.block_lens
                )
            else:
                block_lens_err = _validate_hetero_block_lens_match(
                    local_block_lens, meta.block_lens, tp_ratio=xfer_tp_ratio
                )
            if block_lens_err is not None:
                response = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.ERROR,
                    err_msg=block_lens_err,
                )
                await sock.send_multipart(
                    (identity, self._encoder.encode(response))
                )
                return
        local_regions = self._get_transfer_regions(
            local_base_addrs, local_block_lens
        )
        remote_regions = self._get_transfer_regions(
            meta.kv_caches_base_addr, meta.block_lens
        )
        if meta.block_lens:
            skip_region_validation = _uses_mha_hetero_per_layer_transfers(
                split_k_and_v=self.transfer_topo.split_k_and_v,
                homogeneous_tp=homogeneous_tp,
                use_mla=self.use_mla,
                split_kv_cache_layout=self.split_kv_cache_layout,
            )
            if not skip_region_validation:
                validation_err = _validate_asymmetric_region_lengths(
                    local_regions=local_regions,
                    remote_regions=remote_regions,
                    local_tp_size=self.tp_size,
                    remote_tp_size=meta.remote_tp_size,
                    producer_cache_replicated=self._producer_cache_is_replicated(),
                )
                if validation_err is not None:
                    response = MooncakeXferResponse(
                        status=MooncakeXferResponseStatus.ERROR,
                        err_msg=validation_err,
                    )
                    await sock.send_multipart(
                        (identity, self._encoder.encode(response))
                    )
                    return
        for d_req_id, (transfer_id, _) in meta.req_blocks.items():
            if transfer_id not in self.reqs_need_send:
                # This req is not enqueued in P side yet, create it here.
                self.reqs_need_send[transfer_id] = SendBlockMeta(
                    p_req_id="",
                    transfer_id=transfer_id,
                    local_block_ids=[],
                    ready=asyncio.Event(),
                )
            send_meta = self.reqs_need_send[transfer_id]
            pending_reqs[d_req_id] = send_meta

        async def wait_and_ret(
            d_req_id: ReqId, send_meta: SendBlockMeta
        ) -> tuple[ReqId, SendBlockMeta]:
            await send_meta.ready.wait()
            return d_req_id, send_meta

        wait_tasks = [
            asyncio.create_task(wait_and_ret(d_req_id, send_meta))
            for d_req_id, send_meta in pending_reqs.items()
        ]

        while wait_tasks:
            done, pending = await asyncio.wait(
                wait_tasks,
                timeout=envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                # Timeout, abort all pending requests.
                for task in wait_tasks:
                    task.cancel()
                logger.warning(
                    "Timeout waiting for P side ready: %s", list(pending_reqs)
                )
                response = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.FINISH,
                    err_reqs=list(pending_reqs),
                    err_msg="Timeout waiting for P side ready.",
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))
                break

            wait_tasks = list(pending)
            response_status = (
                MooncakeXferResponseStatus.CONTINUE
                if wait_tasks
                else MooncakeXferResponseStatus.FINISH
            )
            ready_reqs: list[tuple[ReqId, SendBlockMeta]] = []
            for task in done:
                d_req_id, send_meta = task.result()
                del pending_reqs[d_req_id]
                # Do we still in reqs_need_send (not expired)?
                if send_meta.transfer_id in self.reqs_need_send:
                    # Mark it sending to avoid expiration.
                    send_meta.sending += 1
                    if not send_meta.need_send:
                        self.resolve_need_send(send_meta, remote_tp_ranks)
                    ready_reqs.append((d_req_id, send_meta))
                else:
                    # Otherwise (expired, very unlikely), just forget it.
                    logger.warning(
                        "Request %s expired before sending on P side.", d_req_id
                    )

            (
                src_ptrs,
                dst_ptrs,
                lengths,
                err_reqs,
                err_msg,
            ) = await self._build_transfer_params(
                ready_reqs,
                meta,
                local_regions,
                remote_regions,
            )
            err_req_set = set(err_reqs)
            ok_ready_reqs = [
                (d_req_id, send_meta)
                for d_req_id, send_meta in ready_reqs
                if d_req_id not in err_req_set
            ]

            if src_ptrs:
                for _, send_meta in ok_ready_reqs:
                    if not send_meta.ttft_send_start_logged:
                        log_ttft_event(
                            "p_send_kv_start",
                            transfer_id=send_meta.transfer_id,
                            req_id=send_meta.p_req_id or None,
                        )
                        send_meta.ttft_send_start_logged = True
                remote_session = f"{meta.remote_hostname}:{meta.remote_port}"
                ret_value = await self.sender_loop.run_in_executor(
                    self._sender_executor,
                    self._send_blocks,
                    remote_session,
                    src_ptrs,
                    dst_ptrs,
                    lengths,
                )

                if ret_value != 0:
                    transfer_err_msg = f"Mooncake transfer engine returned {ret_value}"
                    err_msg = (
                        transfer_err_msg
                        if err_msg is None
                        else f"{err_msg}; {transfer_err_msg}"
                    )
                    err_reqs = list(err_reqs)
                    for d_req_id, _ in ok_ready_reqs:
                        err_reqs.append(d_req_id)
                        err_req_set.add(d_req_id)
                    ok_ready_reqs = []

            for d_req_id, send_meta in ready_reqs:
                send_meta.sending -= 1

                if d_req_id in err_req_set:
                    continue

                send_meta.sent += 1
                if send_meta.sent == send_meta.need_send:
                    log_ttft_event(
                        "p_send_kv_done",
                        transfer_id=send_meta.transfer_id,
                        req_id=send_meta.p_req_id or None,
                    )
                if (
                    send_meta.sent == send_meta.need_send
                    and self.reqs_need_send.pop(send_meta.transfer_id, None) is not None
                ):
                    self.finished_sending_reqs.add(send_meta.p_req_id)

            response = MooncakeXferResponse(
                status=response_status,
                ok_reqs=[d_req_id for d_req_id, _ in ok_ready_reqs] or None,
                err_reqs=err_reqs or None,
                err_msg=err_msg,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))

    def resolve_need_send(self, send_meta: SendBlockMeta, remote_tp_ranks: list[int]):
        # Prepare for heterogeneous TP (one P pairs to multiple D)
        send_meta.need_send = len(remote_tp_ranks)
        logger.debug(
            "Mooncake request %s will be served by %d consumer TP workers: %s",
            send_meta.transfer_id,
            send_meta.need_send,
            remote_tp_ranks,
        )

    async def _build_transfer_params(
        self,
        ready_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeXferMetadata,
        local_regions: list[TransferRegion],
        remote_regions: list[TransferRegion],
    ) -> tuple[list[int], list[int], list[int], list[ReqId], str | None]:
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        err_reqs: list[ReqId] = []
        err_msg: str | None = None
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        split_kv = self.transfer_topo.split_k_and_v
        homogeneous_tp = self.tp_size == agent_meta.remote_tp_size
        xfer_head_rank = (
            agent_meta.xfer_head_rank
            if agent_meta.xfer_head_rank >= 0
            else agent_meta.remote_tp_rank
        )
        homogeneous_fa_layer_pairs: list[tuple[int, int, int, int]] | None = None
        if split_kv and homogeneous_tp and not self.use_mla:
            # Layer pairing validated once in send_kv_to_decode via block_lens.
            local_base_addrs, local_block_lens = self._resolve_sender_cache_addrs(
                agent_meta
            )
            homogeneous_fa_layer_pairs = list(
                zip(
                    local_base_addrs,
                    local_block_lens,
                    agent_meta.kv_caches_base_addr,
                    agent_meta.block_lens,
                )
            )

        use_hetero_mha_per_layer = _uses_mha_hetero_per_layer_transfers(
            split_k_and_v=split_kv,
            homogeneous_tp=homogeneous_tp,
            use_mla=self.use_mla,
            split_kv_cache_layout=self.split_kv_cache_layout,
        )
        hetero_mha_layer_pairs: list[tuple[int, int, int, int]] | None = None
        hetero_mha_tp_ratio = 1
        hetero_mha_split_ratio = 1
        hetero_mha_split_kv_is_nhd = False
        hetero_setup_err: str | None = None
        if use_hetero_mha_per_layer:
            cache_kind = "FA split" if split_kv else "combined"
            try:
                resolved_layer_pairs = self._iter_fa_layer_cache_pairs(agent_meta)
            except RuntimeError as e:
                hetero_setup_err = str(e)
                resolved_layer_pairs = []
            if hetero_setup_err is None and not resolved_layer_pairs:
                hetero_setup_err = f"no {cache_kind} KV cache layers to transfer"
            if hetero_setup_err is None:
                hetero_mha_tp_ratio = _get_tp_ratio(
                    self.tp_size, agent_meta.remote_tp_size
                )
                hetero_mha_split_ratio = _get_head_split_ratio(hetero_mha_tp_ratio)
                hetero_mha_split_kv_is_nhd = self._split_kv_is_nhd_for_hetero_xfer()
                hetero_setup_err = self._validate_fa_hetero_tp_block_sizes(
                    tp_ratio=hetero_mha_tp_ratio,
                    split_kv=split_kv,
                    remote_block_len=resolved_layer_pairs[0][3],
                )
            if hetero_setup_err is None:
                hetero_mha_layer_pairs = resolved_layer_pairs
                layout_label = (
                    f"split={self.split_kv_cache_layout}"
                    f"/physical={self.kv_cache_layout or 'NHD'}"
                    if split_kv
                    else f"combined-{self.split_kv_cache_layout}"
                )
                logger.debug(
                    "xfer params hetero: tp_ratio=%d split_ratio=%d split_kv=%s "
                    "layout=%s block_len=%d remote_block_len=%d "
                    "slot_size_bytes=%d xfer_head_rank=%d remote_tp_rank=%d "
                    "src_layer_offset=%d model_layers=[%d,%d) num_layers=%d",
                    hetero_mha_tp_ratio,
                    hetero_mha_split_ratio,
                    split_kv,
                    layout_label,
                    self.block_len,
                    resolved_layer_pairs[0][3],
                    self.slot_size_bytes,
                    xfer_head_rank,
                    agent_meta.remote_tp_rank,
                    agent_meta.src_layer_offset,
                    agent_meta.model_layer_start,
                    agent_meta.model_layer_end,
                    len(resolved_layer_pairs),
                )
            else:
                logger.error(
                    "hetero MHA transfer setup failed: %s", hetero_setup_err
                )

        for d_req_id, send_meta in ready_reqs:
            _, remote_block_ids_per_group = agent_meta.req_blocks[d_req_id]

            if not remote_block_ids_per_group or all(
                len(g) == 0 for g in remote_block_ids_per_group
            ):
                continue

            # Per-group partial hit trimming, then flatten.
            # With HMA, groups share the same KV tensor but use different
            # block ranges.  We trim and concatenate so the coalescer and
            # address math see one flat block list — same as non-HMA, but
            # now including blocks from every group.
            local_block_ids: list[int] = []
            remote_block_ids: list[int] = []
            has_block_error = False
            if len(send_meta.local_block_ids) != len(remote_block_ids_per_group):
                logger.error(
                    "req %s: KV group count mismatch: local=%d, remote=%d",
                    d_req_id,
                    len(send_meta.local_block_ids),
                    len(remote_block_ids_per_group),
                )
                err_reqs.append(d_req_id)
                if err_msg is None:
                    err_msg = "KV group count mismatch"
                continue
            for local_group, remote_group in zip(
                send_meta.local_block_ids, remote_block_ids_per_group
            ):
                n_local = len(local_group)
                n_remote = len(remote_group)
                if n_local < n_remote:
                    logger.error(
                        "req %s: local blocks(%d) < remote blocks(%d) "
                        "in a KV cache group",
                        d_req_id,
                        n_local,
                        n_remote,
                    )
                    has_block_error = True
                    break
                if n_local > n_remote:
                    # Partial prefix cache hit: just read uncomputed blocks.
                    local_group = local_group[-n_remote:]
                local_block_ids.extend(local_group)
                remote_block_ids.extend(remote_group)

            if has_block_error:
                err_reqs.append(d_req_id)
                if err_msg is None:
                    err_msg = "P num blocks less than D"
                continue

            if not local_block_ids:
                continue

            # Group by indices
            group_local_block_ids, group_remote_block_ids = group_concurrent_contiguous(
                local_block_ids, remote_block_ids
            )

            if homogeneous_fa_layer_pairs is not None:
                for (
                    local_layer_addr,
                    layer_block_len,
                    remote_layer_addr,
                    layer_remote_block_len,
                ) in homogeneous_fa_layer_pairs:
                    _append_homogeneous_fa_layer_transfers(
                        src_ptrs=src_ptrs,
                        dst_ptrs=dst_ptrs,
                        lengths=lengths,
                        local_layer_addr=local_layer_addr,
                        layer_block_len=layer_block_len,
                        remote_layer_addr=remote_layer_addr,
                        layer_remote_block_len=layer_remote_block_len,
                        group_local_block_ids=group_local_block_ids,
                        group_remote_block_ids=group_remote_block_ids,
                    )
                logger.debug(
                    "Sending kv_caches for request %s (%d blocks) to %s",
                    d_req_id,
                    len(local_block_ids),
                    remote_session,
                )
                continue

            if use_hetero_mha_per_layer:
                if hetero_setup_err is not None:
                    logger.error("req %s: %s", d_req_id, hetero_setup_err)
                    err_reqs.append(d_req_id)
                    if err_msg is None:
                        err_msg = hetero_setup_err
                    continue
                assert hetero_mha_layer_pairs is not None
                for (
                    local_layer_addr,
                    layer_block_len,
                    remote_layer_addr,
                    layer_remote_block_len,
                ) in hetero_mha_layer_pairs:
                    _append_mha_hetero_tp_layer_transfers(
                        src_ptrs=src_ptrs,
                        dst_ptrs=dst_ptrs,
                        lengths=lengths,
                        local_layer_addr=local_layer_addr,
                        layer_block_len=layer_block_len,
                        remote_layer_addr=remote_layer_addr,
                        layer_remote_block_len=layer_remote_block_len,
                        group_local_block_ids=group_local_block_ids,
                        group_remote_block_ids=group_remote_block_ids,
                        tp_ratio=hetero_mha_tp_ratio,
                        split_ratio=hetero_mha_split_ratio,
                        split_kv=split_kv,
                        split_kv_is_nhd=hetero_mha_split_kv_is_nhd,
                        head_rank=xfer_head_rank,
                        block_size=self.block_size,
                        slot_size_bytes=self.slot_size_bytes,
                    )

                logger.debug(
                    "Sending MHA hetero TP kv_caches for request %s (%d blocks) "
                    "to %s",
                    d_req_id,
                    len(local_block_ids),
                    remote_session,
                )
                continue

            logger.debug(
                "xfer params req=%s: region path local_tp=%d remote_tp=%d "
                "use_mla=%s split_kv=%s blocks_first=%s xfer_head_rank=%d "
                "src_layer_offset=%d model_layers=[%d,%d) local_regions=%d "
                "remote_regions=%d",
                d_req_id,
                self.tp_size,
                agent_meta.remote_tp_size,
                self.use_mla,
                split_kv,
                self.transfer_topo.is_kv_layout_blocks_first,
                xfer_head_rank,
                agent_meta.src_layer_offset,
                agent_meta.model_layer_start,
                agent_meta.model_layer_end,
                len(local_regions),
                len(remote_regions),
            )

            if self.use_mla and agent_meta.model_layer_start >= 0:
                try:
                    layer_pairs = self._iter_fa_layer_cache_pairs(agent_meta)
                except RuntimeError as e:
                    logger.error("req %s: %s", d_req_id, e)
                    err_reqs.append(d_req_id)
                    if err_msg is None:
                        err_msg = str(e)
                    continue
                for (
                    local_layer_addr,
                    layer_block_len,
                    remote_layer_addr,
                    layer_remote_block_len,
                ) in layer_pairs:
                    should_transfer, src_region_offset, dst_region_offset, transfer_len = (
                        self._get_sender_transfer_plan(
                            local_kv_block_len=layer_block_len,
                            remote_kv_block_len=layer_remote_block_len,
                            remote_tp_rank=xfer_head_rank,
                            remote_tp_size=agent_meta.remote_tp_size,
                        )
                    )
                    if not should_transfer:
                        continue

                    assert src_region_offset + transfer_len <= layer_block_len, (
                        "Computed source transfer region exceeds local KV block size."
                    )
                    assert (
                        dst_region_offset + transfer_len <= layer_remote_block_len
                    ), (
                        "Computed destination transfer region exceeds remote KV "
                        "block size."
                    )
                    can_coalesce = _can_coalesce_block_transfers(
                        local_region_block_len=layer_block_len,
                        remote_region_block_len=layer_remote_block_len,
                        src_region_offset=src_region_offset,
                        dst_region_offset=dst_region_offset,
                        transfer_len=transfer_len,
                    )

                    for group_local_block_id, group_remote_block_id in zip(
                        group_local_block_ids, group_remote_block_ids
                    ):
                        if can_coalesce:
                            src_ptrs.append(
                                local_layer_addr
                                + group_local_block_id[0] * layer_block_len
                                + src_region_offset
                            )
                            dst_ptrs.append(
                                remote_layer_addr
                                + group_remote_block_id[0] * layer_remote_block_len
                                + dst_region_offset
                            )
                            lengths.append(transfer_len * len(group_local_block_id))
                        else:
                            for local_block_id, remote_block_id in zip(
                                group_local_block_id, group_remote_block_id
                            ):
                                src_ptrs.append(
                                    local_layer_addr
                                    + local_block_id * layer_block_len
                                    + src_region_offset
                                )
                                dst_ptrs.append(
                                    remote_layer_addr
                                    + remote_block_id * layer_remote_block_len
                                    + dst_region_offset
                                )
                                lengths.append(transfer_len)

                logger.debug(
                    "Sending MLA model-layer kv_caches for request %s (%d blocks) "
                    "to %s",
                    d_req_id,
                    len(local_block_ids),
                    remote_session,
                )
                continue

            if len(local_regions) != len(remote_regions):
                logger.error(
                    "req %s: KV region count mismatch: local=%d remote=%d",
                    d_req_id,
                    len(local_regions),
                    len(remote_regions),
                )
                err_reqs.append(d_req_id)
                if err_msg is None:
                    err_msg = "KV region count mismatch"
                continue

            for local_region, remote_region in zip(local_regions, remote_regions):
                should_transfer, src_region_offset, dst_region_offset, transfer_len = (
                    self._get_sender_transfer_plan(
                        local_kv_block_len=local_region.kv_block_len,
                        remote_kv_block_len=remote_region.kv_block_len,
                        remote_tp_rank=xfer_head_rank,
                        remote_tp_size=agent_meta.remote_tp_size,
                    )
                )
                if not should_transfer:
                    # Replicated KV cache: only one producer rank in the TP group
                    # needs to send the actual bytes for this paired decoder rank.
                    # TODO: Account for replicated producer KV in
                    # get_target_remote_ranks() so we can avoid sending
                    # unnecessary ZMQ requests and remove this branch.
                    continue

                assert src_region_offset + transfer_len <= local_region.kv_block_len, (
                    "Computed source transfer region exceeds local KV block size."
                )
                assert dst_region_offset + transfer_len <= remote_region.kv_block_len, (
                    "Computed destination transfer region exceeds remote KV block size."
                )
                # Collapse one contiguous block group into a single larger
                # transfer descriptor when the per-block copy is identical.
                can_coalesce = _can_coalesce_block_transfers(
                    local_region_block_len=local_region.block_len,
                    remote_region_block_len=remote_region.block_len,
                    src_region_offset=src_region_offset,
                    dst_region_offset=dst_region_offset,
                    transfer_len=transfer_len,
                )

                for group_local_block_id, group_remote_block_id in zip(
                    group_local_block_ids, group_remote_block_ids
                ):
                    if can_coalesce:
                        src_ptrs.append(
                            local_region.base_addr
                            + group_local_block_id[0] * local_region.block_len
                            + src_region_offset
                        )
                        dst_ptrs.append(
                            remote_region.base_addr
                            + group_remote_block_id[0] * remote_region.block_len
                            + dst_region_offset
                        )
                        lengths.append(transfer_len * len(group_local_block_id))
                    else:
                        for local_block_id, remote_block_id in zip(
                            group_local_block_id, group_remote_block_id
                        ):
                            src_ptrs.append(
                                local_region.base_addr
                                + local_block_id * local_region.block_len
                                + src_region_offset
                            )
                            dst_ptrs.append(
                                remote_region.base_addr
                                + remote_block_id * remote_region.block_len
                                + dst_region_offset
                            )
                            lengths.append(transfer_len)

                if local_region is local_regions[0]:
                    logger.debug(
                        "Mooncake transfer plan for request %s: local_tp=%d "
                        "remote_tp=%d remote_tp_rank=%d local_block_len=%d "
                        "remote_block_len=%d src_offset=%d dst_offset=%d "
                        "transfer_len=%d coalesce=%s",
                        d_req_id,
                        self.tp_size,
                        agent_meta.remote_tp_size,
                        xfer_head_rank,
                        local_region.block_len,
                        remote_region.block_len,
                        src_region_offset,
                        dst_region_offset,
                        transfer_len,
                        can_coalesce,
                    )

            logger.debug(
                "Sending kv_caches for request %s (%d blocks) to %s",
                d_req_id,
                len(local_block_ids),
                remote_session,
            )

        return src_ptrs, dst_ptrs, lengths, err_reqs, err_msg

    def _split_kv_is_nhd_for_hetero_xfer(self) -> bool:
        """True for per-token head slices; False for block-level chunks.
        Uses ``split_kv_cache_layout``, not ``kv_cache_layout``. HCU CUSTOM FA
        keeps NHD for attention but sets HND here (dense-packed split K/V).
        """
        return self.split_kv_cache_layout == "NHD"

    def _bind_sender_thread_device(self) -> None:
        """ThreadPoolExecutor initializer — binds each pool thread to the
        correct CUDA device.  CUDA device selection is thread-local, so
        without this, NVLink transfers fail for TP ranks > 0."""
        current_platform.set_device(self.device_id)

    def _send_blocks(
        self,
        remote_session: str,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
    ) -> int:
        total_bytes = sum(lengths)
        logger.debug(
            "Mooncake batch_transfer to %s: %d descriptors, %d bytes",
            remote_session,
            len(src_ptrs),
            total_bytes,
        )
        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )
        duration = time.perf_counter() - start_time
        if ret_value == 0:
            self.xfer_stats.record_transfer(
                duration_s=duration,
                total_bytes=sum(lengths),
                num_descs=len(src_ptrs),
            )
            logger.debug("Sending to %s done, took %s", remote_session, duration)
        else:
            self.xfer_stats.record_failed_transfer()
            logger.warning(
                "Sending to %s failed (ret=%s) after %s (%d descriptors, %d bytes)",
                remote_session,
                ret_value,
                duration,
                len(src_ptrs),
                sum(lengths),
            )
        return ret_value

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in mooncake."""

        logger.info("Registering KV_Caches. use_mla: %s", self.use_mla)

        kv_data_ptrs = []
        kv_data_lens = []
        seen_base_addresses = []
        base_addr_to_idx: dict[int, int] = {}
        self.block_len_per_layer = []
        self.slot_size_bytes_per_layer = []
        self.cache_entry_model_layer = []
        self.cache_entry_layer_names = []
        self.model_layer_to_cache_indices = {}

        split_k_and_v = self.transfer_topo.split_k_and_v
        tensor_size_bytes = None
        for layer_name in sorted(kv_caches.keys()):
            cache_or_caches = kv_caches[layer_name]
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
            model_layer = _parse_model_layer_index(layer_name)
            logger.debug(
                "registering layer %s with %d cache tensor(s)",
                layer_name,
                len(cache_list),
            )

            for cache in cache_list:
                self._log_debug_cache_registration(layer_name, cache)
                base_addr = cache.data_ptr()
                if base_addr in base_addr_to_idx:
                    cache_idx = base_addr_to_idx[base_addr]
                    layer_indices = self.model_layer_to_cache_indices.setdefault(
                        model_layer, []
                    )
                    if cache_idx not in layer_indices:
                        layer_indices.append(cache_idx)
                    continue

                seen_base_addresses.append(base_addr)
                cache_idx = len(seen_base_addresses) - 1
                base_addr_to_idx[base_addr] = cache_idx
                self.cache_entry_model_layer.append(model_layer)
                self.cache_entry_layer_names.append(layer_name)
                self.model_layer_to_cache_indices.setdefault(model_layer, []).append(
                    cache_idx
                )

                if tensor_size_bytes is None:
                    tensor_size_bytes = cache.nbytes
                    self.num_blocks = cache.shape[0]
                assert cache.shape[0] == self.num_blocks, (
                    "All kv cache tensors must have the same number of blocks"
                )
                curr_tensor_size_bytes = cache.nbytes

                if self.use_mla:
                    kernel_block_size = cache.shape[-2]
                else:
                    candidate_axes = (-3, -2, -1)
                    candidate_sizes = [cache.shape[ax] for ax in candidate_axes]
                    if self.block_size in candidate_sizes:
                        kernel_block_size = self.block_size
                    else:
                        raise AssertionError(
                            "Cannot infer KV cache block_size from tensor shape. "
                            f"expected block_size={self.block_size}, "
                            f"shape={tuple(cache.shape)}, "
                            f"candidates={candidate_sizes}."
                        )
                assert kernel_block_size > 0, (
                    f"Invalid KV cache physical block_size for layer {layer_name}: "
                    f"got {kernel_block_size} from shape {tuple(cache.shape)}"
                )
                assert self.block_size % kernel_block_size == 0, (
                    f"KV cache logical block_size must be a multiple of physical "
                    f"block_size for layer {layer_name}: logical={self.block_size}, "
                    f"physical={kernel_block_size}, shape={tuple(cache.shape)}"
                )

                if split_k_and_v and not self.use_mla:
                    # FA split K/V (HND): head chunks are dense-packed per block.
                    assert tensor_size_bytes == curr_tensor_size_bytes, (
                        "All kv cache tensors must have the same size"
                    )
                    block_len = curr_tensor_size_bytes // self.num_blocks
                    register_len = curr_tensor_size_bytes
                elif self.use_mla:
                    # MLA: stride(0) is the physical kernel-block byte stride.
                    # Mooncake transfer addresses are indexed by the cache's
                    # physical block IDs, so block_len and registered length
                    # must use the same physical stride.
                    block_len = cache.stride(0) * cache.element_size()
                    register_len = self.num_blocks * block_len
                else:
                    # Combined K+V: dense bytes per block (aligned with workspace).
                    assert tensor_size_bytes == curr_tensor_size_bytes, (
                        "All kv cache tensors must have the same size"
                    )
                    block_len = curr_tensor_size_bytes // self.num_blocks
                    register_len = curr_tensor_size_bytes

                self.block_len_per_layer.append(block_len)
                if self.use_mla:
                    self.slot_size_bytes_per_layer.append(
                        block_len // kernel_block_size
                    )
                kv_data_ptrs.append(base_addr)
                kv_data_lens.append(register_len)

        self.kv_caches_base_addr = seen_base_addresses
        self.seen_base_addresses = seen_base_addresses
        for layer_idx, cache_indices in self.model_layer_to_cache_indices.items():
            cache_indices.sort(
                key=lambda idx: _cache_type_sort_key(self.cache_entry_layer_names[idx])
            )

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        assert tensor_size_bytes is not None
        assert self.num_blocks != 0
        assert self.block_len_per_layer
        if self.use_mla:
            assert self.slot_size_bytes_per_layer
            self.block_len = self.block_len_per_layer[0]
            self.slot_size_bytes = self.slot_size_bytes_per_layer[0]
        else:
            self.block_len = self.block_len_per_layer[0]
            assert self.block_len % self.block_size == 0, (
                f"Invalid KV block layout: block_len={self.block_len} is not "
                f"divisible by block_size={self.block_size}."
            )
            per_token_bytes = self.block_len // self.block_size
            if split_k_and_v:
                self.slot_size_bytes = per_token_bytes
            else:
                assert per_token_bytes % 2 == 0, (
                    "Combined K+V layout expects even per-token bytes. "
                    f"got per_token_bytes={per_token_bytes}."
                )
                self.slot_size_bytes = per_token_bytes // 2
        self.device_kv_caches = kv_caches
        logger.info(
            "Mooncake KV registered: num_blocks=%d block_len=%d slot_size_bytes=%d "
            "split_k_and_v=%s blocks_first=%s cache_entries=%d model_layers=%d "
            "pp_rank=%d pp_size=%d",
            self.num_blocks,
            self.block_len,
            self.slot_size_bytes,
            split_k_and_v,
            self.transfer_topo.is_kv_layout_blocks_first,
            len(seen_base_addresses),
            len(self.model_layer_to_cache_indices),
            self.pp_rank,
            self.pp_size,
        )
        logger.debug(
            "registered num_blocks=%d block_lens=%s",
            self.num_blocks,
            self.block_len_per_layer,
        )

        # No need to launch server for D node.
        if self.is_kv_consumer:
            return

        ready_event = threading.Event()
        asyncio.run_coroutine_threadsafe(
            self._mooncake_sender_listener(ready_event), self.sender_loop
        )
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    async def fetch_finished_recving_reqs(self) -> set[ReqId]:
        finished_recving_reqs = self.finished_recving_reqs
        self.finished_recving_reqs = set()
        return finished_recving_reqs

    async def fetch_finished_sending_reqs(self) -> set[ReqId]:
        finished_sending_reqs = self.finished_sending_reqs
        self.finished_sending_reqs = set()

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()

        expired_transfer_id = []
        for transfer_id, send_meta in self.reqs_need_send.items():
            if (
                send_meta.p_req_id
                and send_meta.expire_time < now
                and send_meta.sending == 0
            ):
                logger.warning(
                    "Request %s timed out after %d seconds without "
                    "being sent. Freeing its blocks on the producer side.",
                    send_meta.p_req_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
                self.xfer_stats.record_kv_expired_req()
                finished_sending_reqs.add(send_meta.p_req_id)
                expired_transfer_id.append(transfer_id)

        for transfer_id in expired_transfer_id:
            del self.reqs_need_send[transfer_id]

        return finished_sending_reqs

    def get_finished(self) -> tuple[set[str] | None, set[str] | None]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        recv_fut = None
        send_fut = None
        if not self.is_kv_producer:
            recv_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_reqs(), self.receiver_loop
            )

        if not self.is_kv_consumer:
            send_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_sending_reqs(), self.sender_loop
            )

        finished_recving_reqs = recv_fut.result() if recv_fut else set()
        finished_sending_reqs = send_fut.result() if send_fut else set()

        if finished_sending_reqs or finished_recving_reqs:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                len(finished_sending_reqs),
                len(finished_recving_reqs),
            )

        return finished_sending_reqs or None, finished_recving_reqs or None

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Return transfer stats collected since the last call, or None
        if nothing has been recorded in this interval."""
        if self.xfer_stats.is_empty():
            return None
        return self.xfer_stats.clone_and_reset()

    async def receive_kv_from_single_worker(
        self,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
        addr_slice: tuple[int, int] | None = None,
        src_layer_offset: int = 0,
        chunk_idx: int | None = None,
        model_layer_start: int = -1,
        model_layer_end: int = -1,
    ):
        req_ids = set(pull_metas)
        base_addrs = self.kv_caches_base_addr
        block_lens = self.block_len_per_layer
        if model_layer_start >= 0:
            base_addrs = []
            block_lens = []
            for layer_idx in range(model_layer_start, model_layer_end):
                cache_indices = sorted(
                    self.model_layer_to_cache_indices.get(layer_idx, []),
                    key=lambda idx: _cache_type_sort_key(
                        self.cache_entry_layer_names[idx]
                    ),
                )
                for cache_idx in cache_indices:
                    base_addrs.append(self.kv_caches_base_addr[cache_idx])
                    block_lens.append(self.block_len_per_layer[cache_idx])
        elif addr_slice is not None:
            base_addrs = base_addrs[addr_slice[0]:addr_slice[1]]
            block_lens = block_lens[addr_slice[0]:addr_slice[1]]

        metadata = MooncakeXferMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            remote_tp_size=self.tp_size,
            remote_tp_rank=(
                chunk_idx if chunk_idx is not None else self.tp_rank
            ),
            req_blocks={
                req_id: (pull_meta.transfer_id, pull_meta.local_block_ids)
                for req_id, pull_meta in pull_metas.items()
            },
            kv_caches_base_addr=base_addrs,
            block_lens=block_lens,
            block_len=block_lens[0] if block_lens else 0,
            slot_size_bytes=self.slot_size_bytes,
            src_layer_offset=src_layer_offset if model_layer_start < 0 else 0,
            model_layer_start=model_layer_start,
            model_layer_end=model_layer_end,
            xfer_head_rank=(
                chunk_idx if chunk_idx is not None else self.tp_rank
            ),
        )

        encoded_data = self._encoder.encode(metadata)
        logger.debug(
            "Size of encoded MooncakeXferMetadata: %d bytes", len(encoded_data)
        )
        logger.debug(
            "Sending kv transfer request for %s on path: %s", req_ids, worker_addr
        )

        # Send query for the request.
        try:
            with make_zmq_socket(
                self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
            ) as sock:
                # If something goes wrong, let P wait timeout first (in asyncio.wait()).
                sock.setsockopt(
                    zmq.RCVTIMEO, (envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60) * 1000
                )
                await sock.send(encoded_data)
                while True:
                    ret_msg = await sock.recv()
                    response = self._xfer_resp_decoder.decode(ret_msg)
                    if response.status == MooncakeXferResponseStatus.ERROR:
                        err_reqs = response.err_reqs or list(pull_metas.keys())
                        self.process_pulling_result(
                            MooncakeXferResponse(
                                status=MooncakeXferResponseStatus.ERROR,
                                err_reqs=err_reqs,
                                err_msg=response.err_msg,
                            ),
                            pull_metas,
                        )
                        self.xfer_stats.record_failed_recv()
                        return
                    self.process_pulling_result(response, pull_metas)
                    if response.status == MooncakeXferResponseStatus.FINISH:
                        break
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
        except Exception as e:
            logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)
            self.process_pulling_result(
                MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.ERROR,
                    err_reqs=list(pull_metas.keys()),
                    err_msg=str(e),
                ),
                pull_metas,
            )
            self.xfer_stats.record_failed_recv()
            return

    def process_pulling_result(
        self,
        response: MooncakeXferResponse,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        if response.ok_reqs:
            logger.debug("pulling kv_caches for %s finished", response.ok_reqs)
        if response.err_reqs:
            logger.error(
                "pulling kv_caches for %s failed: %s",
                response.err_reqs,
                response.err_msg,
            )

        for req_id in (response.ok_reqs or []) + (response.err_reqs or []):
            pull_meta = pull_metas[req_id]
            # No race because we are in async loop.
            pull_meta.pull_tasks_count -= 1
            if pull_meta.pull_tasks_count == 0:
                self.finished_recving_reqs.add(pull_meta.d_req_id)
                log_ttft_event(
                    "d_kv_ready",
                    transfer_id=pull_meta.transfer_id,
                    req_id=pull_meta.d_req_id,
                )

    def _fail_pull_metas(
        self, pull_metas: dict[ReqId, PullReqMeta], err_msg: str
    ) -> None:
        # Bootstrap / topology errors happen before receive_kv sets task counts.
        for pull_meta in pull_metas.values():
            if pull_meta.pull_tasks_count <= 0:
                pull_meta.pull_tasks_count = 1
        self.process_pulling_result(
            MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_reqs=list(pull_metas.keys()),
                err_msg=err_msg,
            ),
            pull_metas,
        )

    async def _send_notification_only(
        self,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
        chunk_idx: int,
    ):
        """Send notification to remote worker without actual data transfer.

        Used for MLA case where KV cache is replicated and only one chunk needs
        to be transferred, but other ranks need to be notified.
        """
        req_ids = set(pull_metas)
        metadata = MooncakeXferMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            remote_tp_size=self.tp_size,
            remote_tp_rank=chunk_idx,
            req_blocks={
                req_id: (pull_meta.transfer_id, [])
                for req_id, pull_meta in pull_metas.items()
            },
            kv_caches_base_addr=[],
            block_lens=[],
            xfer_head_rank=chunk_idx,
        )
        encoded_data = self._encoder.encode(metadata)
        try:
            with make_zmq_socket(
                self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
            ) as sock:
                sock.setsockopt(
                    zmq.RCVTIMEO, (envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60) * 1000
                )
                await sock.send(encoded_data)
                while True:
                    ret_msg = await sock.recv()
                    response = self._xfer_resp_decoder.decode(ret_msg)
                    if response.status == MooncakeXferResponseStatus.ERROR:
                        err_reqs = response.err_reqs or list(pull_metas.keys())
                        self.process_pulling_result(
                            MooncakeXferResponse(
                                status=MooncakeXferResponseStatus.ERROR,
                                err_reqs=err_reqs,
                                err_msg=response.err_msg,
                            ),
                            pull_metas,
                        )
                        return
                    self.process_pulling_result(response, pull_metas)
                    if response.status == MooncakeXferResponseStatus.FINISH:
                        break
                logger.debug(
                    "Notification-only finished for %s on %s", req_ids, worker_addr
                )
        except Exception as e:
            logger.warning(
                "Failed to send notification-only for %s to %s: %s",
                req_ids,
                worker_addr,
                e,
            )
            self.process_pulling_result(
                MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.ERROR,
                    err_reqs=list(pull_metas.keys()),
                    err_msg=str(e),
                ),
                pull_metas,
            )

    async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr: str):
        url = remote_bootstrap_addr + "/query"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data: dict = response.json()
                for _, dp_entry in data.items():
                    remote_engine_id = dp_entry["engine_id"]
                    self._remote_agents[remote_engine_id] = {
                        int(tp_rank): {
                            int(pp_rank): worker_addr
                            for pp_rank, worker_addr in tp_entry.items()
                        }
                        for tp_rank, tp_entry in dp_entry["worker_addr"].items()
                    }
                    self._tp_size[remote_engine_id] = len(dp_entry["worker_addr"])
        except Exception as e:
            logger.error(
                "Failed to connect to bootstrap server %s: %s",
                remote_bootstrap_addr,
                e,
            )

        # Always notify others regardless of connection success or failure.
        self._pending_bootstrap_queries[remote_bootstrap_addr].set()
        del self._pending_bootstrap_queries[remote_bootstrap_addr]

    def receive_kv(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_tp_ranks = self.transfer_topo.handshake_target_ranks(
            self._tp_size[remote_engine_id]
        )

        p_pp_size = 1
        sample_tp = next(iter(self._remote_agents.get(remote_engine_id, {})), None)
        if sample_tp is None:
            self._fail_pull_metas(
                pull_metas,
                f"no remote TP topology found for engine {remote_engine_id}",
            )
            return
        p_pp_size = len(self._remote_agents[remote_engine_id][sample_tp])
        if p_pp_size <= 0:
            self._fail_pull_metas(
                pull_metas,
                f"invalid remote PP size {p_pp_size} for engine {remote_engine_id}",
            )
            return

        if p_pp_size == self.pp_size:
            count = len(remote_tp_ranks)
            if count != 1:
                logger.debug(
                    "Mooncake: TP_P > TP_D detected. D rank %d will read from "
                    "%d P ranks (remote_tp_ranks=%s)",
                    self.tp_rank,
                    count,
                    remote_tp_ranks,
                )
                for pull_meta in pull_metas.values():
                    pull_meta.pull_tasks_count = count
                for i, remote_tp_rank in enumerate(remote_tp_ranks):
                    worker_addr = self._remote_agents[remote_engine_id][
                        remote_tp_rank
                    ][self.pp_rank]
                    if self.use_mla and i > 0:
                        asyncio.create_task(
                            self._send_notification_only(worker_addr, pull_metas, i)
                        )
                        continue
                    asyncio.create_task(
                        self.receive_kv_from_single_worker(
                            worker_addr,
                            pull_metas,
                            chunk_idx=i,
                        )
                    )
                return
            for pull_meta in pull_metas.values():
                pull_meta.pull_tasks_count = count
            for remote_tp_rank in remote_tp_ranks:
                worker_addr = self._remote_agents[remote_engine_id][remote_tp_rank][
                    self.pp_rank
                ]
                asyncio.create_task(
                    self.receive_kv_from_single_worker(worker_addr, pull_metas)
                )
        else:
            total_model_layers = self.model_config.get_total_num_hidden_layers()
            d_model_start, d_model_end = get_pp_indices(
                total_model_layers, self.pp_rank, self.pp_size
            )

            tasks = 0
            for remote_tp_rank in remote_tp_ranks:
                tp_entry = self._remote_agents[remote_engine_id].get(remote_tp_rank, {})
                for pp_rank in range(p_pp_size):
                    worker_addr = tp_entry.get(pp_rank)
                    if worker_addr is None:
                        continue
                    p_model_start, p_model_end = get_pp_indices(
                        total_model_layers, pp_rank, p_pp_size
                    )
                    if d_model_start >= p_model_end or p_model_start >= d_model_end:
                        continue
                    tasks += 1

            if tasks == 0:
                self._fail_pull_metas(
                    pull_metas,
                    f"asymmetric PP has zero overlap tasks for engine "
                    f"{remote_engine_id}",
                )
                return

            for pull_meta in pull_metas.values():
                pull_meta.pull_tasks_count = tasks

            needs_chunk_idx = len(remote_tp_ranks) > 1
            for i, remote_tp_rank in enumerate(remote_tp_ranks):
                tp_entry = self._remote_agents[remote_engine_id].get(remote_tp_rank, {})
                for pp_rank in range(p_pp_size):
                    worker_addr = tp_entry.get(pp_rank)
                    if worker_addr is None:
                        continue
                    p_model_start, p_model_end = get_pp_indices(
                        total_model_layers, pp_rank, p_pp_size
                    )
                    if d_model_start >= p_model_end or p_model_start >= d_model_end:
                        continue
                    overlap_model_start = max(d_model_start, p_model_start)
                    overlap_model_end = min(d_model_end, p_model_end)
                    if self.use_mla and i > 0:
                        asyncio.create_task(
                            self._send_notification_only(worker_addr, pull_metas, i)
                        )
                        continue
                    asyncio.create_task(
                        self.receive_kv_from_single_worker(
                            worker_addr,
                            pull_metas,
                            model_layer_start=overlap_model_start,
                            model_layer_end=overlap_model_end,
                            chunk_idx=i if needs_chunk_idx else None,
                        )
                    )

    async def handle_new_engine_id(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_bootstrap_addr = next(iter(pull_metas.values())).remote_bootstrap_addr
        if remote_bootstrap_addr not in self._pending_bootstrap_queries:
            self._pending_bootstrap_queries[remote_bootstrap_addr] = asyncio.Event()
            await self._connect_to_prefiller_bootstrap(remote_bootstrap_addr)
        else:
            await self._pending_bootstrap_queries[remote_bootstrap_addr].wait()

        if remote_engine_id not in self._remote_agents:
            self._fail_pull_metas(
                pull_metas,
                f"engine_id {remote_engine_id} not found at {remote_bootstrap_addr}",
            )
            return

        self.receive_kv(remote_engine_id, pull_metas)

    async def _start_load_kv(
        self, reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]]
    ):
        for remote_engine_id, pull_metas in reqs_to_recv.items():
            if remote_engine_id not in self._remote_agents:
                asyncio.create_task(
                    self.handle_new_engine_id(remote_engine_id, pull_metas)
                )
            else:
                self.receive_kv(remote_engine_id, pull_metas)

    async def record_send_reqs(self, metadata: MooncakeConnectorMetadata):
        for p_req_id, (transfer_id, block_ids) in metadata.reqs_to_send.items():
            if block_ids:
                # Already gone through request_finished()
                send_meta = self.reqs_need_send[transfer_id]
                send_meta.p_req_id = p_req_id
                send_meta.local_block_ids = block_ids
                send_meta.expire_time = (
                    time.perf_counter() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                )
                send_meta.ready.set()
                log_ttft_event(
                    "p_ready",
                    transfer_id=transfer_id,
                    req_id=p_req_id,
                )
            else:
                # From update_state_after_alloc(),
                # but not reach request_finished() yet
                # This may be already created by send_kv_to_decode()
                # when D is sending MooncakeXferMetadata.
                if transfer_id not in self.reqs_need_send:
                    self.reqs_need_send[transfer_id] = SendBlockMeta(
                        p_req_id=p_req_id,
                        transfer_id=transfer_id,
                        local_block_ids=[],
                        ready=asyncio.Event(),
                    )
        for transfer_id in metadata.reqs_not_processed:
            send_meta = self.reqs_need_send.pop(transfer_id, None)
            if send_meta is None:
                logger.debug(
                    "Mooncake send request %s already removed or never registered",
                    transfer_id,
                )
                continue
            assert not send_meta.ready.is_set()

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        if not self.is_kv_producer and metadata.reqs_to_recv:
            asyncio.run_coroutine_threadsafe(
                self._start_load_kv(metadata.reqs_to_recv), self.receiver_loop
            )

        if not self.is_kv_consumer and (
            metadata.reqs_to_send or metadata.reqs_not_processed
        ):
            asyncio.run_coroutine_threadsafe(
                self.record_send_reqs(metadata), self.sender_loop
            )

    def _producer_cache_is_replicated(self) -> bool:
        return self.transfer_topo.local_replicates_kv_cache

    def _iter_fa_layer_cache_pairs(
        self, agent_meta: MooncakeXferMetadata
    ) -> list[tuple[int, int, int, int]]:
        """Pair local/remote FA split K/V cache entries for one transfer."""
        if agent_meta.model_layer_start >= 0:
            pairs: list[tuple[int, int, int, int]] = []
            remote_cursor = 0
            remote_addrs = agent_meta.kv_caches_base_addr
            remote_lens = agent_meta.block_lens
            for layer_idx in range(
                agent_meta.model_layer_start, agent_meta.model_layer_end
            ):
                cache_indices = sorted(
                    self.model_layer_to_cache_indices.get(layer_idx, []),
                    key=lambda idx: _cache_type_sort_key(
                        self.cache_entry_layer_names[idx]
                    ),
                )
                if not cache_indices:
                    logger.warning(
                        "P worker missing KV cache for model layer %d", layer_idx
                    )
                    continue
                for local_idx in cache_indices:
                    if remote_cursor >= len(remote_addrs):
                        raise RuntimeError(
                            "Remote KV cache entry count mismatch for model layer "
                            f"{layer_idx}: remote_cursor={remote_cursor} "
                            f"remote_entries={len(remote_addrs)}"
                        )
                    pairs.append(
                        (
                            self.kv_caches_base_addr[local_idx],
                            self.block_len_per_layer[local_idx],
                            remote_addrs[remote_cursor],
                            remote_lens[remote_cursor],
                        )
                    )
                    remote_cursor += 1
            if remote_cursor != len(remote_addrs):
                raise RuntimeError(
                    "Remote KV cache entry count mismatch after pairing: "
                    f"paired={remote_cursor} remote_entries={len(remote_addrs)} "
                    f"model_layers=[{agent_meta.model_layer_start},"
                    f"{agent_meta.model_layer_end})"
                )
            return pairs

        local_addrs, local_lens = self._resolve_sender_cache_addrs(agent_meta)
        remote_addrs = agent_meta.kv_caches_base_addr
        remote_lens = agent_meta.block_lens
        if len(local_addrs) != len(remote_addrs):
            raise RuntimeError(
                "FA KV cache entry count mismatch: "
                f"local={len(local_addrs)} remote={len(remote_addrs)}"
            )
        if len(local_lens) != len(remote_lens):
            raise RuntimeError(
                "FA KV block_len count mismatch: "
                f"local={len(local_lens)} remote={len(remote_lens)}"
            )
        return list(zip(local_addrs, local_lens, remote_addrs, remote_lens))

    def _validate_fa_hetero_tp_block_sizes(
        self,
        tp_ratio: int,
        split_kv: bool,
        remote_block_len: int,
    ) -> str | None:
        """Validate block sizes for heterogeneous TP transfers."""
        block_len = self.block_len
        if tp_ratio > 1:
            num_splits = tp_ratio
            if remote_block_len != block_len * num_splits:
                return (
                    f"P>D block_len mismatch: local={block_len}, "
                    f"remote={remote_block_len}, num_splits={num_splits}"
                )
            remote_block_size = block_len // self.slot_size_bytes
            if not split_kv:
                remote_block_size //= 2
        elif tp_ratio < 0:
            ratio_abs = -tp_ratio
            if remote_block_len != block_len // ratio_abs:
                return (
                    "Remote D worker KV layer cache has incompatible shape/dtype "
                    "for the current TP ratio."
                )
            per_token_remote = self.slot_size_bytes // ratio_abs
            if per_token_remote <= 0 or remote_block_len % per_token_remote != 0:
                return (
                    "Remote D worker KV per-token bytes are incompatible "
                    "for the current TP ratio."
                )
            remote_block_size = remote_block_len // per_token_remote
            if not split_kv:
                remote_block_size //= 2
        else:
            return None
        if self.block_size != remote_block_size:
            return "Remote P worker with different block size is not supported"
        return None

    def _resolve_sender_cache_addrs(
        self, agent_meta: MooncakeXferMetadata
    ) -> tuple[list[int], list[int]]:
        """Resolve local KV cache base addresses for a transfer request."""
        if agent_meta.model_layer_start >= 0:
            base_addrs: list[int] = []
            block_lens: list[int] = []
            for layer_idx in range(
                agent_meta.model_layer_start, agent_meta.model_layer_end
            ):
                cache_indices = sorted(
                    self.model_layer_to_cache_indices.get(layer_idx, []),
                    key=lambda idx: _cache_type_sort_key(
                        self.cache_entry_layer_names[idx]
                    ),
                )
                for cache_idx in cache_indices:
                    base_addrs.append(self.kv_caches_base_addr[cache_idx])
                    block_lens.append(self.block_len_per_layer[cache_idx])
            return base_addrs, block_lens
        local_base = self.kv_caches_base_addr
        local_lens = self.block_len_per_layer
        if agent_meta.src_layer_offset:
            off = agent_meta.src_layer_offset
            return local_base[off:], local_lens[off:]
        return local_base, local_lens

    def _get_transfer_regions(
        self, base_addrs: list[int], block_lens: list[int]
    ) -> list[TransferRegion]:
        return _expand_transfer_regions(
            base_addrs=base_addrs,
            block_lens=block_lens,
            is_kv_layout_blocks_first=self.transfer_topo.is_kv_layout_blocks_first,
        )

    def _get_sender_transfer_plan(
        self,
        local_kv_block_len: int,
        remote_kv_block_len: int,
        remote_tp_rank: int,
        remote_tp_size: int,
    ) -> tuple[bool, int, int, int]:
        return _compute_sender_transfer_plan(
            local_tp_rank=self.tp_rank,
            local_tp_size=self.tp_size,
            remote_tp_rank=remote_tp_rank,
            remote_tp_size=remote_tp_size,
            local_kv_block_len=local_kv_block_len,
            remote_kv_block_len=remote_kv_block_len,
            producer_cache_replicated=self._producer_cache_is_replicated(),
        )

    def _log_debug_cache_registration(
        self, layer_name: str, cache: torch.Tensor
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            "Mooncake register view layer=%s shape=%s stride=%s "
            "storage_offset=%d contiguous=%s dense=%s data_ptr=%d",
            layer_name,
            tuple(cache.shape),
            tuple(cache.stride()),
            cache.storage_offset(),
            cache.is_contiguous(),
            _get_tensor_dense_flag(cache),
            cache.data_ptr(),
        )


def group_concurrent_contiguous(
    src_indices: list[int], dst_indices: list[int]
) -> tuple[list[list[int]], list[list[int]]]:
    """Vectorised NumPy implementation."""
    if len(src_indices) == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


def get_mooncake_side_channel_port(vllm_config: VllmConfig) -> int:
    # This logic is now centralized
    return (
        envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
        + vllm_config.parallel_config.data_parallel_index
        * vllm_config.parallel_config.tensor_parallel_size
    )


def _async_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def should_launch_bootstrap_server(vllm_config: VllmConfig) -> bool:
    assert (parallel_config := vllm_config.parallel_config)
    # In hybrid or external LB mode,
    # each instance should have its own bootstrap server.
    #
    # In internal LB mode,
    # only the real global first rank need to launch the bootstrap server.
    # Hybrid/external LB: each local engine keeps its own bootstrap.
    # Multi-node prefill (Ray/internal LB): only global rank 0 on dp index 0.
    return (
        is_local_first_rank() if parallel_config.local_engines_only
        else is_global_first_rank()
    ) and (
        parallel_config.local_engines_only or parallel_config.data_parallel_index == 0
    )


def get_mooncake_bootstrap_addr(vllm_config: VllmConfig) -> tuple[str, int]:
    """
    Returns the address of the Mooncake bootstrap server.
    This is only used by prefillers to register workers.
    Decoders should get addr from kv_transfer_params.
    """
    assert (parallel_config := vllm_config.parallel_config)
    if parallel_config.local_engines_only:
        # In hybrid or external LB mode, connect to local server.
        host = "127.0.0.1"
    else:
        host = parallel_config.data_parallel_master_ip
    port = envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
    return (host, port)
