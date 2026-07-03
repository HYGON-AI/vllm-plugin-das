# SPDX-License-Identifier: Apache-2.0

"""
vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector
Support heterogeneous TP/PP
"""

PATCHES = [

# Import HCU platform env vars (VLLM_HCU_USE_CUSTOM_FLASH_ATTN, etc.)
(
'''import zmq.asyncio

from vllm import envs
from vllm.config import VllmConfig
''',
'''import zmq.asyncio

import vllm_hcu.platforms.envs as henvs
from vllm import envs
from vllm.config import VllmConfig
''',
),

# Import is_global_first_rank for bootstrap server launch logic
(
'''    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_local_first_rank,
)
''',
'''    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_global_first_rank,
    is_local_first_rank,
)
''',
),

# Import get_pp_indices for asymmetric PP layer overlap
(
'''    is_local_first_rank,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
''',
'''    is_local_first_rank,
)
from vllm.distributed.utils import get_pp_indices
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
''',
),

# Add helpers to parse model layer index and sort cache entries (indexer first)
(
'''

@dataclass(frozen=True)
class TransferRegion:
''',
'''

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
''',
),

# Update _get_tp_ratio docstring with P>D / D>P sign convention
(
'''
def _get_tp_ratio(local_tp_size: int, remote_tp_size: int) -> int:
    """Return the TP ratio used by heterogeneous TP transfer planning.

    Positive values mean one local rank maps into a larger remote KV region.
    Negative values mean one local rank must gather from multiple remote KV
    regions.
    """
    if local_tp_size >= remote_tp_size:
''',
'''
def _get_tp_ratio(local_tp_size: int, remote_tp_size: int) -> int:
    """Return TP ratio for heterogeneous KV transfer.

    Sign convention:
    - ``1``: homogeneous TP
    - ``> 1``: P_TP > D_TP (one local block maps into multiple remote slots)
    - ``< 0``: D_TP > P_TP (one remote block pulls from a local head slice)

    Used by region-based planning and per-token MHA copy helpers alike.
    """
    if local_tp_size >= remote_tp_size:
''',
),

# Add _get_head_split_ratio for heterogeneous TP head-chunk count
(
'''    )
    return -(remote_tp_size // local_tp_size)


''',
'''    )
    return -(remote_tp_size // local_tp_size)


def _get_head_split_ratio(tp_ratio: int) -> int:
    """Head-chunk count for heterogeneous TP (always positive)."""
    if tp_ratio > 1:
        return tp_ratio
    if tp_ratio < 0:
        return -tp_ratio
    return 1


''',
),

# Add hetero TP validation helpers and per-layer MHA transfer functions
(
'''

def _validate_asymmetric_region_lengths(
    local_regions: list[TransferRegion],
''',
'''

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
''',
),

# Extend MooncakeXferMetadata with block_len, slot_size_bytes, PP/TP fields
(
'''    kv_caches_base_addr: list[int]
    block_lens: list[int]


''',
'''    kv_caches_base_addr: list[int]
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


''',
),

# Keep NHD KV cache layout for HCU CUSTOM FA (HND breaks V cache writes)
(
'''        if vllm_config.model_config.use_mla:
            return None
        logger.info_once(
            "MooncakeConnector setting KV cache layout to HND for "
''',
'''        if vllm_config.model_config.use_mla:
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
''',
),

# Add debug log for scheduler pull block clipping
(
'''                )
                local_block_ids = self.get_sw_clipped_blocks(unhashed_block_ids)
                # Get unhashed blocks to pull from remote.
                self._reqs_need_recv[request.request_id] = (request, local_block_ids)
''',
'''                )
                local_block_ids = self.get_sw_clipped_blocks(unhashed_block_ids)
                logger.debug(
                    "Mooncake pull blocks for req %s: unhashed=%s clipped=%s",
                    request.request_id,
                    unhashed_block_ids,
                    local_block_ids,
                )
                # Get unhashed blocks to pull from remote.
                self._reqs_need_recv[request.request_id] = (request, local_block_ids)
''',
),

# Remove pipeline parallelism unsupported error in MooncakeConnectorWorker
(
'''        self.dp_rank = dp_local_rank if parallel_config.local_engines_only else dp_rank
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        if pp_size > 1:
            raise ValueError(
                "Mooncake Transfer Engine does not support pipeline parallelism yet."
            )
        self.pp_rank = get_pp_group().rank_in_group
''',
'''        self.dp_rank = dp_local_rank if parallel_config.local_engines_only else dp_rank
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.pp_rank = get_pp_group().rank_in_group
''',
),

# Track pp_size on worker for asymmetric PP
(
'''        self.pp_rank = get_pp_group().rank_in_group

        self.kv_caches_base_addr: list[int] = []
''',
'''        self.pp_rank = get_pp_group().rank_in_group
        self.pp_size = pp_size

        self.kv_caches_base_addr: list[int] = []
''',
),

# Add model-layer cache index tracking fields on worker
(
'''
        self.kv_caches_base_addr: list[int] = []
        self.device_kv_caches: dict[str, torch.Tensor] = {}
        self.reqs_need_send: dict[TransferId, SendBlockMeta] = {}
''',
'''
        self.kv_caches_base_addr: list[int] = []
        self.cache_entry_model_layer: list[int] = []
        self.cache_entry_layer_names: list[str] = []
        self.model_layer_to_cache_indices: dict[int, list[int]] = {}
        self.device_kv_caches: dict[str, torch.Tensor] = {}
        self.reqs_need_send: dict[TransferId, SendBlockMeta] = {}
''',
),

# Initialize block_len_per_layer list before cache registration
(
'''        self.kv_cache_config = kv_cache_config
        self.use_mla = self.model_config.use_mla
        self._sync_block_size_with_kernel()

''',
'''        self.kv_cache_config = kv_cache_config
        self.use_mla = self.model_config.use_mla
        self.block_len_per_layer: list[int] = []
        self._sync_block_size_with_kernel()

''',
),

# Detect split_kv_cache_layout (NHD vs HND) for HCU CUSTOM FA
(
'''        logger.debug("Detected attention backend %s", self.backend_name)
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.tp_size}
''',
'''        logger.debug("Detected attention backend %s", self.backend_name)
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
''',
),

# Handshake: homo TP rank warn, hetero TP force slot_size_bytes, block_lens validate
(
'''        pending_reqs: dict[ReqId, SendBlockMeta] = {}
        remote_tp_ranks = self.transfer_topo.handshake_target_ranks(meta.remote_tp_size)
        if meta.remote_tp_rank not in remote_tp_ranks:
            # This D worker does not pair with the P worker.
            msg = (
                "This D tp_rank "
                f"{meta.remote_tp_rank} is not paired with P tp_rank "
                f"{self.tp_rank}; expected one of {remote_tp_ranks}."
            )
            logger.error(msg)
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=msg,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        local_regions = self._get_transfer_regions(
''',
'''        pending_reqs: dict[ReqId, SendBlockMeta] = {}
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
''',
),

# Resolve sender cache addrs for local transfer regions in handshake
(
'''        local_regions = self._get_transfer_regions(
            self.kv_caches_base_addr, self.block_len_per_layer
        )
        remote_regions = self._get_transfer_regions(
''',
'''        local_regions = self._get_transfer_regions(
            local_base_addrs, local_block_lens
        )
        remote_regions = self._get_transfer_regions(
''',
),

# Skip region validation for MHA hetero per-layer; validate block_lens first
(
'''            meta.kv_caches_base_addr, meta.block_lens
        )
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
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        for d_req_id, (transfer_id, _) in meta.req_blocks.items():
            if transfer_id not in self.reqs_need_send:
''',
'''            meta.kv_caches_base_addr, meta.block_lens
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
''',
),

# Add hetero MHA / homogeneous FA layer transfer setup in _do_send_kv
(
'''        err_msg: str | None = None
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        for d_req_id, send_meta in ready_reqs:
''',
'''        err_msg: str | None = None
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
''',
),

# Add homogeneous FA, hetero MHA, and MLA per-layer KV send paths
(
'''            )

            for local_region, remote_region in zip(local_regions, remote_regions):
                should_transfer, src_region_offset, dst_region_offset, transfer_len = (
''',
'''            )

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
''',
),

# Use xfer_head_rank in sender transfer plan
(
'''                        local_kv_block_len=local_region.kv_block_len,
                        remote_kv_block_len=remote_region.kv_block_len,
                        remote_tp_rank=agent_meta.remote_tp_rank,
                        remote_tp_size=agent_meta.remote_tp_size,
                    )
''',
'''                        local_kv_block_len=local_region.kv_block_len,
                        remote_kv_block_len=remote_region.kv_block_len,
                        remote_tp_rank=xfer_head_rank,
                        remote_tp_size=agent_meta.remote_tp_size,
                    )
''',
),

# Use xfer_head_rank in receiver transfer plan
(
'''                        self.tp_size,
                        agent_meta.remote_tp_size,
                        agent_meta.remote_tp_rank,
                        local_region.block_len,
                        remote_region.block_len,
''',
'''                        self.tp_size,
                        agent_meta.remote_tp_size,
                        xfer_head_rank,
                        local_region.block_len,
                        remote_region.block_len,
''',
),

# Add _split_kv_is_nhd_for_hetero_xfer helper
(
'''        return src_ptrs, dst_ptrs, lengths, err_reqs, err_msg

    def _bind_sender_thread_device(self) -> None:
        """ThreadPoolExecutor initializer — binds each pool thread to the
''',
'''        return src_ptrs, dst_ptrs, lengths, err_reqs, err_msg

    def _split_kv_is_nhd_for_hetero_xfer(self) -> bool:
        """True for per-token head slices; False for block-level chunks.
        Uses ``split_kv_cache_layout``, not ``kv_cache_layout``. HCU CUSTOM FA
        keeps NHD for attention but sets HND here (dense-packed split K/V).
        """
        return self.split_kv_cache_layout == "NHD"

    def _bind_sender_thread_device(self) -> None:
        """ThreadPoolExecutor initializer — binds each pool thread to the
''',
),

# Add debug log for batch_transfer_sync_write byte count
(
'''        lengths: list[int],
    ) -> int:
        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
''',
'''        lengths: list[int],
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
''',
),

# Use base_addr_to_idx dict to deduplicate cache registration
(
'''        kv_data_lens = []
        seen_base_addresses = []
        self.block_len_per_layer = []
''',
'''        kv_data_lens = []
        seen_base_addresses = []
        base_addr_to_idx: dict[int, int] = {}
        self.block_len_per_layer = []
''',
),

# Track cache entry model layer metadata during registration
(
'''        self.block_len_per_layer = []

        split_k_and_v = self.transfer_topo.split_k_and_v
''',
'''        self.block_len_per_layer = []
        self.cache_entry_model_layer = []
        self.cache_entry_layer_names = []
        self.model_layer_to_cache_indices = {}

        split_k_and_v = self.transfer_topo.split_k_and_v
''',
),

# Iterate kv_caches in sorted order for deterministic registration
(
'''        split_k_and_v = self.transfer_topo.split_k_and_v
        tensor_size_bytes = None
        for layer_name, cache_or_caches in kv_caches.items():
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
''',
'''        split_k_and_v = self.transfer_topo.split_k_and_v
        tensor_size_bytes = None
        for layer_name in sorted(kv_caches.keys()):
            cache_or_caches = kv_caches[layer_name]
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
''',
),

# Parse model layer index per cache entry during registration
(
'''            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
            logger.debug(
                "registering layer %s with %d cache tensor(s)",
''',
'''            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
            model_layer = _parse_model_layer_index(layer_name)
            logger.debug(
                "registering layer %s with %d cache tensor(s)",
''',
),

# Map duplicate base_addr to existing cache index with model layer tracking
(
'''                self._log_debug_cache_registration(layer_name, cache)
                base_addr = cache.data_ptr()
                if base_addr in seen_base_addresses:
                    continue

''',
'''                self._log_debug_cache_registration(layer_name, cache)
                base_addr = cache.data_ptr()
                if base_addr in base_addr_to_idx:
                    cache_idx = base_addr_to_idx[base_addr]
                    layer_indices = self.model_layer_to_cache_indices.setdefault(
                        model_layer, []
                    )
                    if cache_idx not in layer_indices:
                        layer_indices.append(cache_idx)
                    continue

''',
),

# Record cache entry model layer and layer name on registration
(
'''
                seen_base_addresses.append(base_addr)

                if tensor_size_bytes is None:
''',
'''
                seen_base_addresses.append(base_addr)
                cache_idx = len(seen_base_addresses) - 1
                base_addr_to_idx[base_addr] = cache_idx
                self.cache_entry_model_layer.append(model_layer)
                self.cache_entry_layer_names.append(layer_name)
                self.model_layer_to_cache_indices.setdefault(model_layer, []).append(
                    cache_idx
                )

                if tensor_size_bytes is None:
''',
),

# Compute block_len by layout type (FA split, MLA stride, combined K+V)
(
'''                    "All kv cache tensors must have the same number of blocks"
                )

                # Use stride-based block length so RDMA reaches the last
                # block's padding (e.g. DeepseekV4 MLA alignment). stride(0)
                # reflects the actual byte distance between consecutive
                # blocks in GPU memory, which matches or exceeds the
                # shape-based size.
                block_len = cache.stride(0) * cache.element_size()

                self.block_len_per_layer.append(block_len)
''',
'''                    "All kv cache tensors must have the same number of blocks"
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
                assert self.block_size == kernel_block_size, (
                    f"KV cache block_size mismatch for layer {layer_name}: "
                    f"expected {self.block_size}, got {kernel_block_size} "
                    f"from shape {tuple(cache.shape)}"
                )

                if split_k_and_v and not self.use_mla:
                    # FA split K/V (HND): head chunks are dense-packed per block.
                    assert tensor_size_bytes == curr_tensor_size_bytes, (
                        "All kv cache tensors must have the same size"
                    )
                    block_len = curr_tensor_size_bytes // self.num_blocks
                    register_len = curr_tensor_size_bytes
                elif self.use_mla:
                    # MLA: stride-based block length so RDMA reaches block padding.
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
''',
),

# Use register_len for Mooncake memory registration length
(
'''                self.block_len_per_layer.append(block_len)
                kv_data_ptrs.append(base_addr)
                kv_data_lens.append(self.num_blocks * block_len)

        self.kv_caches_base_addr = seen_base_addresses
''',
'''                self.block_len_per_layer.append(block_len)
                kv_data_ptrs.append(base_addr)
                kv_data_lens.append(register_len)

        self.kv_caches_base_addr = seen_base_addresses
''',
),

# Sort model_layer_to_cache_indices by cache type (indexer first)
(
'''        self.kv_caches_base_addr = seen_base_addresses
        self.seen_base_addresses = seen_base_addresses

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
''',
'''        self.kv_caches_base_addr = seen_base_addresses
        self.seen_base_addresses = seen_base_addresses
        for layer_idx, cache_indices in self.model_layer_to_cache_indices.items():
            cache_indices.sort(
                key=lambda idx: _cache_type_sort_key(self.cache_entry_layer_names[idx])
            )

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
''',
),

# Compute block_len and slot_size_bytes after cache registration
(
'''        assert tensor_size_bytes is not None
        assert self.num_blocks != 0
        self.device_kv_caches = kv_caches
''',
'''        assert tensor_size_bytes is not None
        assert self.num_blocks != 0
        assert self.block_len_per_layer
        self.block_len = self.block_len_per_layer[0]
        assert self.block_len % self.block_size == 0, (
            f"Invalid KV block layout: block_len={self.block_len} is not "
            f"divisible by block_size={self.block_size}."
        )
        per_token_bytes = self.block_len // self.block_size
        if self.use_mla:
            self.slot_size_bytes = per_token_bytes
        elif split_k_and_v:
            self.slot_size_bytes = per_token_bytes
        else:
            assert per_token_bytes % 2 == 0, (
                "Combined K+V layout expects even per-token bytes. "
                f"got per_token_bytes={per_token_bytes}."
            )
            self.slot_size_bytes = per_token_bytes // 2
        self.device_kv_caches = kv_caches
''',
),

# Add info log summarizing registered KV cache layout
(
'''        self.device_kv_caches = kv_caches
        logger.debug(
            "registered num_blocks=%d block_lens=%s",
''',
'''        self.device_kv_caches = kv_caches
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
''',
),

# Extend receive_kv_from_single_worker for PP layer range and TP chunk params
(
'''        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        req_ids = set(pull_metas)
''',
'''        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
        addr_slice: tuple[int, int] | None = None,
        src_layer_offset: int = 0,
        chunk_idx: int | None = None,
        model_layer_start: int = -1,
        model_layer_end: int = -1,
    ):
        req_ids = set(pull_metas)
''',
),

# Resolve base_addrs/block_lens by model layer range or addr slice
(
'''    ):
        req_ids = set(pull_metas)
        metadata = MooncakeXferMetadata(
            remote_hostname=self.hostname,
''',
'''    ):
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
''',
),

# Set remote_tp_rank from chunk_idx in xfer metadata
(
'''            remote_port=self.rpc_port,
            remote_tp_size=self.tp_size,
            remote_tp_rank=self.tp_rank,
            req_blocks={
                req_id: (pull_meta.transfer_id, pull_meta.local_block_ids)
''',
'''            remote_port=self.rpc_port,
            remote_tp_size=self.tp_size,
            remote_tp_rank=(
                chunk_idx if chunk_idx is not None else self.tp_rank
            ),
            req_blocks={
                req_id: (pull_meta.transfer_id, pull_meta.local_block_ids)
''',
),

# Populate extended MooncakeXferMetadata fields on pull
(
'''                for req_id, pull_meta in pull_metas.items()
            },
            kv_caches_base_addr=self.kv_caches_base_addr,
            block_lens=self.block_len_per_layer,
        )

''',
'''                for req_id, pull_meta in pull_metas.items()
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

''',
),

# Propagate pull errors via process_pulling_result on xfer response error
(
'''                    response = self._xfer_resp_decoder.decode(ret_msg)
                    if response.status == MooncakeXferResponseStatus.ERROR:
                        logger.error(
                            "Error happens during transferring kvcache for %s: %s",
                            req_ids,
                            response.err_msg,
                        )
                        self.xfer_stats.record_failed_recv()
''',
'''                    response = self._xfer_resp_decoder.decode(ret_msg)
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
''',
),

# Propagate pull errors via process_pulling_result on xfer exception
(
'''        except Exception as e:
            logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)
            self.xfer_stats.record_failed_recv()
            return
''',
'''        except Exception as e:
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
''',
),

# Refactor process_pulling_result to handle ok/err reqs in one loop
(
'''        pull_metas: dict[ReqId, PullReqMeta],
    ):
        ok_reqs: list[ReqId] = response.ok_reqs or []

        for req_id in ok_reqs:
            pull_meta = pull_metas[req_id]
            # No race because we are in async loop.
''',
'''        pull_metas: dict[ReqId, PullReqMeta],
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
''',
),

# Add _fail_pull_metas and _send_notification_only for MLA replicated cache
(
'''                self.finished_recving_reqs.add(pull_meta.d_req_id)

        if ok_reqs:
            logger.debug("pulling kv_caches for %s finished", ok_reqs)

        if response.err_reqs:
            logger.error(
                "pulling kv_caches for %s failed: %s",
                response.err_reqs,
                response.err_msg,
            )

''',
'''                self.finished_recving_reqs.add(pull_meta.d_req_id)

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

''',
),

# Support heterogeneous TP and asymmetric PP in receive_kv_caches
(
'''            self._tp_size[remote_engine_id]
        )
        count = len(remote_tp_ranks)
        logger.debug(
            "Receiving Mooncake KV for engine %s from producer TP ranks %s",
            remote_engine_id,
            remote_tp_ranks,
        )
        for pull_meta in pull_metas.values():
            pull_meta.pull_tasks_count = count
        for remote_tp_rank in remote_tp_ranks:
            worker_addr = self._remote_agents[remote_engine_id][remote_tp_rank][0]
            asyncio.create_task(
                self.receive_kv_from_single_worker(worker_addr, pull_metas)
            )

    async def handle_new_engine_id(
''',
'''            self._tp_size[remote_engine_id]
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
''',
),

# Use _fail_pull_metas when remote engine_id not found
(
'''
        if remote_engine_id not in self._remote_agents:
            logger.error(
                "Failed to find remote engine_id %s from bootstrap server %s",
                remote_engine_id,
                remote_bootstrap_addr,
            )
            return
''',
'''
        if remote_engine_id not in self._remote_agents:
            self._fail_pull_metas(
                pull_metas,
                f"engine_id {remote_engine_id} not found at {remote_bootstrap_addr}",
            )
            return
''',
),

# Handle reqs_not_processed with pop(transfer_id, None)
(
'''                    )
        for transfer_id in metadata.reqs_not_processed:
            send_meta = self.reqs_need_send.pop(transfer_id)
            if send_meta:
                assert not send_meta.ready.is_set()

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
''',
'''                    )
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
''',
),

# Add _iter_fa_layer_cache_pairs, _validate_fa_hetero_tp_block_sizes, _resolve_sender_cache_addrs
(
'''    def _producer_cache_is_replicated(self) -> bool:
        return self.transfer_topo.local_replicates_kv_cache

    def _get_transfer_regions(
''',
'''    def _producer_cache_is_replicated(self) -> bool:
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
''',
),

# Update should_launch_bootstrap_server for hybrid/multi-node LB
(
'''    # In internal LB mode,
    # only the real global first rank need to launch the bootstrap server.
    return is_local_first_rank() and (
        parallel_config.local_engines_only or parallel_config.data_parallel_index == 0
    )
''',
'''    # In internal LB mode,
    # only the real global first rank need to launch the bootstrap server.
    # Hybrid/external LB: each local engine keeps its own bootstrap.
    # Multi-node prefill (Ray/internal LB): only global rank 0 on dp index 0.
    return (
        is_local_first_rank() if parallel_config.local_engines_only
        else is_global_first_rank()
    ) and (
        parallel_config.local_engines_only or parallel_config.data_parallel_index == 0
    )
''',
),

# Mooncake TTFT trace (6-segment model, VLLM_HCU_MOONCAKE_TTFT_TRACE=1)
(
'''import asyncio
import logging
import threading
''',
'''import asyncio
import logging
import re
import threading
''',
),

(
'''logger = init_logger(__name__)

try:
    from mooncake.engine import TransferEngine
''',
'''logger = init_logger(__name__)

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
''',
),

(
'''    sent: int = 0
    sending: int = 0


class MooncakeConnectorMetadata(KVConnectorMetadata):
''',
'''    sent: int = 0
    sending: int = 0
    ttft_send_start_logged: bool = False


class MooncakeConnectorMetadata(KVConnectorMetadata):
''',
),

(
'''                # Get unhashed blocks to pull from remote.
                self._reqs_need_recv[request.request_id] = (request, local_block_ids)
            else:
                logger.warning(
                    "Got invalid KVTransferParams: %s. This "
                    "request will not utilize KVTransfer",
                    params,
                )
''',
'''                # Get unhashed blocks to pull from remote.
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
''',
),

(
'''                # Add an empty list to worker to create event.
                self._reqs_need_send[request.request_id] = (request, [])

    def build_connector_meta(
''',
'''                # Add an empty list to worker to create event.
                self._reqs_need_send[request.request_id] = (request, [])
                log_ttft_event(
                    "p_alloc",
                    transfer_id=params.get("transfer_id"),
                    req_id=request.request_id,
                )

    def build_connector_meta(
''',
),

(
'''            if src_ptrs:
                remote_session = f"{meta.remote_hostname}:{meta.remote_port}"
                ret_value = await self.sender_loop.run_in_executor(
''',
'''            if src_ptrs:
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
''',
),

(
'''                send_meta.sent += 1
                if (
                    send_meta.sent == send_meta.need_send
                    and self.reqs_need_send.pop(send_meta.transfer_id, None) is not None
                ):
                    self.finished_sending_reqs.add(send_meta.p_req_id)

            response = MooncakeXferResponse(
''',
'''                send_meta.sent += 1
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
''',
),

(
'''            if pull_meta.pull_tasks_count == 0:
                self.finished_recving_reqs.add(pull_meta.d_req_id)

    def _fail_pull_metas(
''',
'''            if pull_meta.pull_tasks_count == 0:
                self.finished_recving_reqs.add(pull_meta.d_req_id)
                log_ttft_event(
                    "d_kv_ready",
                    transfer_id=pull_meta.transfer_id,
                    req_id=pull_meta.d_req_id,
                )

    def _fail_pull_metas(
''',
),

(
'''                send_meta.ready.set()
            else:
                # From update_state_after_alloc(),
                # but not reach request_finished() yet
                # This may be already created by send_kv_to_decode()
''',
'''                send_meta.ready.set()
                log_ttft_event(
                    "p_ready",
                    transfer_id=transfer_id,
                    req_id=p_req_id,
                )
            else:
                # From update_state_after_alloc(),
                # but not reach request_finished() yet
                # This may be already created by send_kv_to_decode()
''',
),

]
