# SPDX-License-Identifier: Apache-2.0

"""
vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector
Support heterogeneous TP/PP
"""

PATCHES = [

# Import HCU env flags for split KV layout decision
(
"""
import zmq
import zmq.asyncio

from vllm import envs
""",
"""
import zmq
import zmq.asyncio

import vllm_hcu.platforms.envs as henvs
from vllm import envs
""",
),

# Extend transfer metadata for heterogeneous TP/PP
(
"""
    remote_tp_rank: int
    req_blocks: dict[ReqId, tuple[TransferId, list[int]]]
    kv_caches_base_addr: list[int]


class MooncakeXferResponseStatus(IntEnum):
""",
"""
    remote_tp_rank: int
    req_blocks: dict[ReqId, tuple[TransferId, list[int]]]
    kv_caches_base_addr: list[int]
    block_len: int
    slot_size_bytes: int
    # For asymmetric PP (P_SIZE < D_SIZE): when D pulls from P,
    # this tells P which offset (in kv_caches_base_addr entries) to start
    # reading its local cache from.
    src_layer_offset: int = 0


class MooncakeXferResponseStatus(IntEnum):
""",
),

# Enable PP size tracking in Mooncake worker initialization
(
"""
        dp_local_rank = parallel_config.data_parallel_rank_local
        self.dp_rank = dp_local_rank if parallel_config.local_engines_only else dp_rank
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        if pp_size > 1:
            raise ValueError(
                "Mooncake Transfer Engine does not support pipeline parallelism yet."
            )
        self.pp_rank = get_pp_group().rank_in_group

        self.kv_caches_base_addr: list[int] = []
        self.device_kv_caches: dict[str, torch.Tensor] = {}
""",
"""
        dp_local_rank = parallel_config.data_parallel_rank_local
        self.dp_rank = dp_local_rank if parallel_config.local_engines_only else dp_rank
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.pp_rank = get_pp_group().rank_in_group
        self.pp_size = pp_size

        self.kv_caches_base_addr: list[int] = []
        self.device_kv_caches: dict[str, torch.Tensor] = {}
""",
),

# Infer split KV layout from backend and HCU custom flash-attn env
(
"""
        backend = get_current_attn_backend(vllm_config)
        self.backend_name = backend.get_name()
        self.kv_cache_layout = get_kv_cache_layout()
        logger.debug("Detected attention backend %s", self.backend_name)
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.tp_size}
""",
"""
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
""",
),

# Remove strict tp-rank guard to allow heterogeneous TP routing
(
"""
        pending_reqs: dict[ReqId, SendBlockMeta] = {}
        remote_tp_ranks = self.kv_topo.get_target_remote_ranks(meta.remote_tp_size)
        if self.tp_rank not in remote_tp_ranks:
            # This D worker does not pair with the P worker.
            msg = f"This P tp_rank {self.tp_rank} not in remote D target ranks {remote_tp_ranks}"  # noqa: E501
            logger.error(msg)
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=msg,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        for d_req_id, (transfer_id, _) in meta.req_blocks.items():
            if transfer_id not in self.reqs_need_send:
                # This req is not enqueued in P side yet, create it here.
""",
"""
        pending_reqs: dict[ReqId, SendBlockMeta] = {}
        remote_tp_ranks = self.kv_topo.get_target_remote_ranks(meta.remote_tp_size)
        for d_req_id, (transfer_id, _) in meta.req_blocks.items():
            if transfer_id not in self.reqs_need_send:
                # This req is not enqueued in P side yet, create it here.
""",
),

# Relax send-side heterogeneous TP requirement handling
(
"""
    def resolve_need_send(self, send_meta: SendBlockMeta, remote_tp_ranks: list[int]):
        # Prepare for heterogeneous TP (one P pairs to multiple D)
        send_meta.need_send = len(remote_tp_ranks)
        if send_meta.need_send != 1:
            logger.error("Mooncake: Heterogeneous TP is not supported yet.")
            raise NotImplementedError(
                "Mooncake: Heterogeneous TP is not supported yet."
            )

    async def _build_transfer_params(
        self,
        ready_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeXferMetadata,
    ) -> tuple[list[int], list[int], list[int], list[ReqId]]:
""",
"""
    def resolve_need_send(self, send_meta: SendBlockMeta, remote_tp_ranks: list[int]):
        # Prepare for heterogeneous TP (one P pairs to multiple D)
        send_meta.need_send = len(remote_tp_ranks)
        if send_meta.need_send > 1:
            logger.debug(
                "Heterogeneous TP: P tp_rank=%d pairs with %d D workers %s",
                self.tp_rank, send_meta.need_send, remote_tp_ranks,
            )

    async def _build_transfer_params(
        self,
        ready_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeXferMetadata,
    ) -> tuple[list[int], list[int], list[int], list[ReqId]]:
""",
),

# Build heterogeneous TP/PP transfer parameters and assertions
(
"""
        lengths = []
        err_reqs: list[ReqId] = []
        local_base_addr = self.kv_caches_base_addr
        remote_base_addr = agent_meta.kv_caches_base_addr
        block_len = self.block_len
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        for d_req_id, send_meta in ready_reqs:
            _, remote_block_ids = agent_meta.req_blocks[d_req_id]
            num_remote_blocks = len(remote_block_ids)
""",
"""
        lengths = []
        err_reqs: list[ReqId] = []
        local_base_addr = self.kv_caches_base_addr
        # For asymmetric PP (P_SIZE < D_SIZE): P's single rank owns more
        # layers than the target D rank, so we need to skip past the layers
        # that belong to earlier D PP ranks in the global layer order.
        src_offset = agent_meta.src_layer_offset
        if src_offset:
            local_base_addr = local_base_addr[src_offset:]
        remote_base_addr = agent_meta.kv_caches_base_addr
        block_len = self.block_len
        remote_block_len = agent_meta.block_len
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        # --- Heterogeneous TP support ---
        # Compute tp_ratio: how many D workers share one P worker's KV cache.
        # P worker has full KV heads; each D worker pulls its chunk.
        # tp_ratio = D_TP / P_TP (from P's perspective, this is the number of
        # D workers that will pull from this P worker).
        remote_tp_size = agent_meta.remote_tp_size
        if self.tp_size > remote_tp_size:
            assert self.tp_size % remote_tp_size == 0, (
                f"Local TP size {self.tp_size} is not divisible by remote TP "
                f"size {remote_tp_size}."
            )
            tp_ratio = - (self.tp_size // remote_tp_size)
        else:
            assert remote_tp_size % self.tp_size == 0, (
                f"Remote TP size {remote_tp_size} is not divisible by local TP "
                f"size {self.tp_size}."
            )
            tp_ratio = remote_tp_size // self.tp_size
        split_ratio = tp_ratio if tp_ratio > 1 else (-tp_ratio if tp_ratio < 0 else 1)
        split_kv = self.kv_topo.split_k_and_v
        split_kv_layout = self.split_kv_cache_layout if split_kv else None
        split_kv_is_nhd = split_kv_layout == "NHD"
        logger.debug(
            "xfer params: tp_ratio=%d split_ratio=%d split_kv=%s use_mla=%s "
            "split_kv_layout=%s block_len=%d remote_block_len=%d slot_size_bytes=%d "
            "remote_tp_rank=%d src_layer_offset=%d",
            tp_ratio, split_ratio, split_kv, self.use_mla, split_kv_layout,
            block_len, remote_block_len, self.slot_size_bytes,
            agent_meta.remote_tp_rank, src_offset,
        )

        # Compute the head chunk offset for this P worker.
        # With heterogeneous TP (D_TP > P_TP), the P worker's KV cache has
        # more heads than each D worker needs. Each D worker pulls a contiguous
        # chunk of heads. The offset is determined by which D workers pair
        # with this P worker.
        #
        # For MLA: KV is replicated across TP, no head splitting needed.
        # For MHA/MQA/GQA: KV heads are sharded across TP.
        if self.use_mla:
            # With MLA the only difference is in the number of blocks.
            # [num_blocks, block_size, latent_dim]
            remote_block_size = remote_block_len / self.slot_size_bytes
            assert self.block_len == remote_block_len
        else:
            # Combined K+V layout: [num_blocks, 2, block_size, kv_heads, head_dim]
            # Split K/V layout: each region is [num_blocks, block_size, kv_heads, head_dim]
            if tp_ratio < 0:
                # P > D: D.block_len = num_splits * P.block_len, but
                # block_size (tokens per block) is the same on both sides.
                num_splits = -tp_ratio
                remote_block_size = self.block_len / self.slot_size_bytes
                if not split_kv:
                    remote_block_size /= 2
                assert remote_block_len == self.block_len * num_splits, (
                    f"P>D block_len mismatch: local={self.block_len}, "
                    f"remote={remote_block_len}, num_splits={num_splits}"
                )
            else:
                remote_block_size = remote_block_len / (
                    self.slot_size_bytes / tp_ratio
                )
                if not split_kv:
                    remote_block_size /= 2
                assert remote_block_len == self.block_len / tp_ratio, (
                    "Remote D worker KV layer cache has incompatible shape/dtype "
                    "for the current TP ratio."
                )

        assert self.block_size == remote_block_size, (
            "Remote P worker with different block size is not supported"
        )

        rank_offset = 0
        # For heterogeneous TP (D > P or P > D), we need to split by heads
        # For D > P (tp_ratio > 1): calculate offset into the heads
        # For P > D (tp_ratio < 0): we also use head splitting, offset = chunk_idx * remote_block_len
        if split_ratio > 1:
            if not self.use_mla:
                rank_offset = (agent_meta.remote_tp_rank % split_ratio) * remote_block_len

        for d_req_id, send_meta in ready_reqs:
            _, remote_block_ids = agent_meta.req_blocks[d_req_id]
            num_remote_blocks = len(remote_block_ids)
""",
),

# Implement three transfer strategies depending on TP topology
(
"""
                for group_local_block_id, group_remote_block_id in zip(
                    group_local_block_ids, group_remote_block_ids
                ):
                    src_ptrs.append(
                        local_layer_addr + group_local_block_id[0] * block_len
                    )
                    dst_ptrs.append(
                        remote_layer_addr + group_remote_block_id[0] * block_len
                    )
                    lengths.append(block_len * len(group_local_block_id))

            logger.debug(
                "Sending kv_caches for request %s (%d blocks) to %s",
                d_req_id,
                num_remote_blocks,
                remote_session,
            )
""",
"""
                for group_local_block_id, group_remote_block_id in zip(
                    group_local_block_ids, group_remote_block_ids
                ):
                    num_blocks = len(group_local_block_id)
                    # --- Transfer strategies depending on TP topology/layout ---
                    # 1. P_TP > D_TP: D receives multiple P head chunks.
                    #    Combined K+V and NHD split K/V copy each
                    #    token's head chunk into the matching D offset.
                    #    HND split K/V can still use a contiguous
                    #    block-level head slice.
                    # 2. P_TP == D_TP or MLA: grouped contiguous transfer.
                    # 3. D_TP > P_TP with HND split K/V: pull one head chunk per
                    #    block (cannot merge chunks across block boundaries).
                    # 4. D_TP > P_TP for combined K+V and NHD split K/V: each D
                    #    rank pulls a head chunk from every token in the P block.
                    if tp_ratio < 0 and not self.use_mla:
                        # Case 1: P_TP > D_TP (non-MLA)
                        if split_kv:
                            chunk_idx = agent_meta.remote_tp_rank % split_ratio
                            if split_kv_is_nhd:
                                # NHD split K/V stores heads inside each token;
                                # the TP slice must be copied token by token.
                                pos_stride_P = self.slot_size_bytes
                                pos_stride_D = pos_stride_P * split_ratio
                                h_off = chunk_idx * pos_stride_P
                                for l_idx, r_idx in zip(
                                    group_local_block_id, group_remote_block_id
                                ):
                                    for p in range(self.block_size):
                                        src_ptrs.append(
                                            local_layer_addr
                                            + l_idx * block_len
                                            + p * pos_stride_P
                                        )
                                        dst_ptrs.append(
                                            remote_layer_addr
                                            + r_idx * remote_block_len
                                            + p * pos_stride_D
                                            + h_off
                                        )
                                        lengths.append(pos_stride_P)
                            else:
                                # HND split K/V has a contiguous block-level
                                # head slice.
                                for l_idx, r_idx in zip(
                                    group_local_block_id, group_remote_block_id
                                ):
                                    src_ptrs.append(
                                        local_layer_addr + l_idx * block_len
                                    )
                                    dst_ptrs.append(
                                        remote_layer_addr
                                        + r_idx * remote_block_len
                                        + chunk_idx * block_len
                                    )
                                    lengths.append(block_len)
                        else:
                            # Combined K+V: per-position K/V split within
                            # each block.
                            pos_stride_P = self.slot_size_bytes
                            pos_stride_D = pos_stride_P * (-tp_ratio)
                            h_off = (
                                agent_meta.remote_tp_rank % split_ratio
                            ) * pos_stride_P
                            for l_idx, r_idx in zip(
                                group_local_block_id, group_remote_block_id
                            ):
                                for p in range(self.block_size):
                                    src_ptrs.append(
                                        local_layer_addr + l_idx * block_len
                                        + p * pos_stride_P
                                    )
                                    dst_ptrs.append(
                                        remote_layer_addr + r_idx * remote_block_len
                                        + p * pos_stride_D + h_off
                                    )
                                    lengths.append(pos_stride_P)
                                    src_ptrs.append(
                                        local_layer_addr + l_idx * block_len
                                        + block_len // 2 + p * pos_stride_P
                                    )
                                    dst_ptrs.append(
                                        remote_layer_addr + r_idx * remote_block_len
                                        + remote_block_len // 2 + p * pos_stride_D
                                        + h_off
                                    )
                                    lengths.append(pos_stride_P)

                    elif tp_ratio <= 1 or self.use_mla:
                        # Case 2: Homogeneous TP or MLA.
                        # rank_offset is only used in this contiguous-copy path.
                        src_ptrs.append(
                            local_layer_addr + group_local_block_id[0] * block_len + rank_offset
                        )
                        dst_ptrs.append(
                            remote_layer_addr + group_remote_block_id[0] * remote_block_len
                        )
                        lengths.append(remote_block_len * num_blocks)

                    elif split_kv and not split_kv_is_nhd:
                        # Case 3: HND split K/V with D_TP > P_TP.
                        # Each destination rank pulls one contiguous head chunk
                        # per source block; these chunks are not contiguous
                        # across block boundaries in source memory.
                        for l_idx, r_idx in zip(
                            group_local_block_id, group_remote_block_id
                        ):
                            src_ptrs.append(
                                local_layer_addr + l_idx * block_len + rank_offset
                            )
                            dst_ptrs.append(
                                remote_layer_addr + r_idx * remote_block_len
                            )
                            lengths.append(remote_block_len)

                    else:
                        # Case 4: D_TP > P_TP
                        # Split each source block by KV heads on every position.
                        # This works for combined K+V and NHD split K/V.
                        pos_stride_P = self.slot_size_bytes
                        pos_stride_D = pos_stride_P // tp_ratio
                        h_off_bytes = (
                            (agent_meta.remote_tp_rank % tp_ratio) * pos_stride_D
                        )
                        for l_idx, r_idx in zip(
                            group_local_block_id, group_remote_block_id
                        ):
                            for p in range(self.block_size):
                                src_ptrs.append(
                                    local_layer_addr + l_idx * block_len
                                    + p * pos_stride_P + h_off_bytes
                                )
                                dst_ptrs.append(
                                    remote_layer_addr + r_idx * remote_block_len
                                    + p * pos_stride_D
                                )
                                lengths.append(pos_stride_D)

                                if not split_kv:
                                    src_ptrs.append(
                                        local_layer_addr + l_idx * block_len
                                        + block_len // 2 + p * pos_stride_P + h_off_bytes
                                    )
                                    dst_ptrs.append(
                                        remote_layer_addr + r_idx * remote_block_len
                                        + remote_block_len // 2 + p * pos_stride_D
                                    )
                                    lengths.append(pos_stride_D)

            logger.debug(
                "Sending kv_caches for request %s (%d blocks) to %s",
                d_req_id,
                num_remote_blocks,
                remote_session,
            )
""",
),

# Improve KV cache registration and shape logging
(
"""
        split_k_and_v = self.kv_topo.split_k_and_v
        tensor_size_bytes = None
        for layer_name, cache_or_caches in kv_caches.items():
            logger.debug(
                "registering layer %s with shape %s", layer_name, cache_or_caches.shape
            )
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]

            for cache in cache_list:
                base_addr = cache.data_ptr()
""",
"""
        split_k_and_v = self.kv_topo.split_k_and_v
        tensor_size_bytes = None
        for layer_name, cache_or_caches in kv_caches.items():
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
            logger.debug(
                "registering layer %s with shapes %s",
                layer_name,
                [tuple(cache.shape) for cache in cache_list],
            )

            for cache in cache_list:
                base_addr = cache.data_ptr()
""",
),

# Validate kernel_block_size and derive slot_size_bytes from KV tensor shape assumptions
(
"""
                assert tensor_size_bytes == curr_tensor_size_bytes, (
                    "All kv cache tensors must have the same size"
                )
                kernel_block_size = cache.shape[-2 if self.use_mla else -3]
                assert self.block_size == kernel_block_size
                kv_data_ptrs.append(base_addr)
                kv_data_lens.append(tensor_size_bytes)
""",
"""
                assert tensor_size_bytes == curr_tensor_size_bytes, (
                    "All kv cache tensors must have the same size"
                )
                if self.use_mla:
                    kernel_block_size = cache.shape[-2]
                else:
                    # Different attention backends may use different KV layouts.
                    # Infer block_size by matching known configured block size
                    # from plausible axis positions.
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
                assert self.block_size == kernel_block_size
                kv_data_ptrs.append(base_addr)
                kv_data_lens.append(tensor_size_bytes)
""",
),

# Extend receive metadata with slicing and chunk routing fields
(
"""
        assert self.num_blocks != 0
        assert tensor_size_bytes % self.num_blocks == 0
        self.block_len = tensor_size_bytes // self.num_blocks
        self.device_kv_caches = kv_caches
        logger.debug(
            "registered num_blocks=%d block_len=%d", self.num_blocks, self.block_len
        )

        # No need to launch server for D node.
""",
"""
        assert self.num_blocks != 0
        assert tensor_size_bytes % self.num_blocks == 0
        self.block_len = tensor_size_bytes // self.num_blocks
        assert self.block_len % self.block_size == 0, (
            f"Invalid KV block layout: block_len={self.block_len} is not "
            f"divisible by block_size={self.block_size}."
        )
        # slot_size_bytes = per-token bytes for one KV tensor (K or V).
        # For combined K+V layout, each block stores both K and V, so divide by 2.
        per_token_bytes = self.block_len // self.block_size
        if self.use_mla:
            self.slot_size_bytes = per_token_bytes
        else:
            if split_k_and_v:
                self.slot_size_bytes = per_token_bytes
            else:
                assert per_token_bytes % 2 == 0, (
                    "Combined K+V layout expects even per-token bytes. "
                    f"got per_token_bytes={per_token_bytes}."
                )
                self.slot_size_bytes = per_token_bytes // 2
        self.device_kv_caches = kv_caches
        logger.debug(
            "registered num_blocks=%d block_len=%d slot_size_bytes=%d",
            self.num_blocks,
            self.block_len,
            self.slot_size_bytes,
        )

        # No need to launch server for D node.
""",
),

# Add notification-only receive path for replicated MLA flow
(
"""
    async def receive_kv_from_single_worker(
        self,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        req_ids = set(pull_metas)
        metadata = MooncakeXferMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            remote_tp_size=self.tp_size,
            remote_tp_rank=self.tp_rank,
            req_blocks={
                req_id: (pull_meta.transfer_id, pull_meta.local_block_ids)
                for req_id, pull_meta in pull_metas.items()
            },
            kv_caches_base_addr=self.kv_caches_base_addr,
        )

        encoded_data = self._encoder.encode(metadata)
""",
"""
    async def receive_kv_from_single_worker(
        self,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
        addr_slice: tuple[int, int] | None = None,
        src_layer_offset: int = 0,
        chunk_idx: int | None = None,
    ):
        req_ids = set(pull_metas)
        # If addr_slice is set, only send the overlapping layer addresses
        # for the target P PP rank. This is essential for asymmetric PP
        # where D has more layers per rank than P, and the request must
        # be split across multiple P ranks.
        base_addrs = self.kv_caches_base_addr
        if addr_slice is not None:
            base_addrs = base_addrs[addr_slice[0]:addr_slice[1]]

        metadata = MooncakeXferMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            remote_tp_size=self.tp_size,
            remote_tp_rank=chunk_idx if chunk_idx is not None else self.tp_rank,
            req_blocks={
                req_id: (pull_meta.transfer_id, pull_meta.local_block_ids)
                for req_id, pull_meta in pull_metas.items()
            },
            kv_caches_base_addr=base_addrs,
            block_len=self.block_len,
            slot_size_bytes=self.slot_size_bytes,
            src_layer_offset=src_layer_offset,
        )

        encoded_data = self._encoder.encode(metadata)
""",
),

# Add heterogeneous TP/PP receive task scheduling
(
"""
        if response.err_reqs:
            logger.error(
                "pulling kv_caches for %s failed: %s",
                response.err_reqs,
                response.err_msg,
            )

    async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr: str):
        url = remote_bootstrap_addr + "/query"
        try:
""",
"""
        if response.err_reqs:
            logger.error(
                "pulling kv_caches for %s failed: %s",
                response.err_reqs,
                response.err_msg,
            )

    async def _send_notification_only(
        self,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
        chunk_idx: int,
    ):
        \"\"\"Send notification to remote worker without actual data transfer.

        Used for MLA case where KV cache is replicated and only one chunk needs
        to be transferred, but other ranks need to be notified.
        \"\"\"
        req_ids = set(pull_metas)
        # Carry transfer_id for each request so producer side can run
        # the same bookkeeping path (need_send/sent/free).
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
            block_len=0,
            slot_size_bytes=0,
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
                        logger.error(
                            "Notification-only transfer failed for %s: %s",
                            req_ids,
                            response.err_msg,
                        )
                        return
                    self.process_pulling_result(response, pull_metas)
                    if response.status == MooncakeXferResponseStatus.FINISH:
                        break
                logger.debug("Notification-only finished for %s on %s", req_ids, worker_addr)
        except Exception as e:
            logger.warning(
                "Failed to send notification-only for %s to %s: %s",
                req_ids,
                worker_addr,
                e,
            )

    async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr: str):
        url = remote_bootstrap_addr + "/query"
        try:
""",
),

# Add asymmetric PP task counting and overlap-based slicing
(
"""
    def receive_kv(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_tp_ranks = self.kv_topo.get_target_remote_ranks_from_engine_id(
            remote_engine_id
        )
        count = len(remote_tp_ranks)
        if count != 1:
            logger.error("Mooncake: Heterogeneous TP is not supported yet.")
            raise NotImplementedError(
                "Mooncake: Heterogeneous TP is not supported yet."
            )
        for pull_meta in pull_metas.values():
            pull_meta.pull_tasks_count = count
        for remote_tp_rank in remote_tp_ranks:
            worker_addr = self._remote_agents[remote_engine_id][remote_tp_rank][0]
            asyncio.create_task(
                self.receive_kv_from_single_worker(worker_addr, pull_metas)
            )
""",
"""
    def receive_kv(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        # Determine which remote TP ranks to pull from.
        remote_tp_ranks = self.kv_topo.get_target_remote_ranks_from_engine_id(
            remote_engine_id
        )

        # Infer P's PP size from the bootstrap topology:
        # each tp_rank entry maps pp_rank -> worker_addr.
        # All tp_ranks should have the same number of PP ranks.
        p_pp_size = 1
        sample_tp = next(iter(self._remote_agents.get(remote_engine_id, {})), None)
        if sample_tp is None:
            logger.error(
                "Failed to infer remote PP size for engine %s: no remote TP "
                "topology found. Skip KV receive.",
                remote_engine_id,
            )
            return
        else:
            p_pp_size = len(self._remote_agents[remote_engine_id][sample_tp])
        if p_pp_size <= 0:
            logger.error(
                "Invalid remote PP size inferred for engine %s: p_pp_size=%d "
                "(sample_tp=%s). Skip KV receive.",
                remote_engine_id, p_pp_size, sample_tp)
            return

        if p_pp_size == self.pp_size:
            # Symmetric PP (P_pp_size == D_pp_size).
            count = len(remote_tp_ranks)
            if count != 1:
                # TP_P > TP_D scenario: read from multiple remote ranks
                logger.debug(
                    "Mooncake: TP_P > TP_D detected. D rank %d will read from "
                    "%d P ranks (remote_tp_ranks=%s)",
                    self.tp_rank, count, remote_tp_ranks
                )

                # Set pull_tasks_count for each request
                for pull_meta in pull_metas.values():
                    pull_meta.pull_tasks_count = count

                # Launch pull task for each remote TP rank
                for i, remote_tp_rank in enumerate(remote_tp_ranks):
                    worker_addr = self._remote_agents[remote_engine_id][remote_tp_rank][self.pp_rank]
                    # For MLA, KV cache is replicated across TP, only need first rank
                    if self.use_mla and i > 0:
                        # No actual data transfer is needed, but keep request-level
                        # transfer_id bookkeeping on producer side.
                        asyncio.create_task(
                            self._send_notification_only(
                                worker_addr, pull_metas, i
                            )
                        )
                        continue

                    asyncio.create_task(
                        self.receive_kv_from_single_worker(
                            worker_addr, pull_metas, None,
                            chunk_idx=i
                        )
                    )
                return
            for pull_meta in pull_metas.values():
                pull_meta.pull_tasks_count = count
            for remote_tp_rank in remote_tp_ranks:
                worker_addr = self._remote_agents[remote_engine_id][remote_tp_rank][self.pp_rank]
                asyncio.create_task(
                    self.receive_kv_from_single_worker(worker_addr, pull_metas)
                )
        else:
            # Asymmetric PP (P_pp_size != D_pp_size):
            # D may span more layers than a single P PP rank owns, so each
            # D request must be split into multiple sub-requests, one per
            # overlapping P PP rank, each transferring only the layers that
            # P rank owns.
            #
            # addr_stride = 2 when split_k_and_v=True (K and V have separate
            # entries per layer), otherwise = 1 (combined K+V per layer).
            addr_stride = 2 if self.kv_topo.split_k_and_v else 1
            assert len(self.kv_caches_base_addr) % addr_stride == 0, (
                "Local kv_caches_base_addr size must align with addr_stride. "
                f"got len={len(self.kv_caches_base_addr)} addr_stride={addr_stride}"
            )
            num_actual_layers = len(self.kv_caches_base_addr) // addr_stride
            total_layers = num_actual_layers * self.pp_size
            # Support uneven layer partition on P side:
            # first `remainder` ranks own `base+1` layers, others own `base`.
            base_layers_per_rank, remainder = divmod(total_layers, p_pp_size)

            def get_p_layer_range(pp_rank: int) -> tuple[int, int]:
                extra = 1 if pp_rank < remainder else 0
                start = pp_rank * base_layers_per_rank + min(pp_rank, remainder)
                end = start + base_layers_per_rank + extra
                return start, end

            # D's global layer range for this PP rank.
            d_layer_start = self.pp_rank * num_actual_layers
            d_layer_end = d_layer_start + num_actual_layers

            # Count total tasks = TP targets x PP targets (with overlap).
            tasks = 0
            for remote_tp_rank in remote_tp_ranks:
                tp_entry = self._remote_agents[remote_engine_id].get(remote_tp_rank, {})
                for pp_rank in range(p_pp_size):
                    worker_addr = tp_entry.get(pp_rank)
                    if worker_addr is None:
                        continue
                    p_layer_start, p_layer_end = get_p_layer_range(pp_rank)
                    if d_layer_start >= p_layer_end or p_layer_start >= d_layer_end:
                        continue
                    tasks += 1

            if tasks == 0:
                logger.error(
                    "receive_kv asymmetric pp has zero overlap tasks: "
                    "remote_engine_id=%s req_ids=%s; mark requests finished.",
                    remote_engine_id,
                    list(pull_metas.keys()),
                )
                for pull_meta in pull_metas.values():
                    pull_meta.pull_tasks_count = 0
                    self.finished_recving_reqs.add(pull_meta.d_req_id)
                return

            for pull_meta in pull_metas.values():
                pull_meta.pull_tasks_count = tasks

            needs_chunk_idx = len(remote_tp_ranks) > 1
            logger.debug(
                "receive_kv asymmetric pp: remote_tp_ranks=%s p_pp_size=%d "
                "d_pp_size=%d tasks=%d needs_chunk_idx=%s",
                remote_tp_ranks, p_pp_size, self.pp_size, tasks, needs_chunk_idx)
            for i, remote_tp_rank in enumerate(remote_tp_ranks):
                tp_entry = self._remote_agents[remote_engine_id].get(remote_tp_rank, {})
                for pp_rank in range(p_pp_size):
                    worker_addr = tp_entry.get(pp_rank)
                    if worker_addr is None:
                        continue
                    p_layer_start, p_layer_end = get_p_layer_range(pp_rank)
                    # Skip P PP ranks with no layer overlap.
                    if d_layer_start >= p_layer_end or p_layer_start >= d_layer_end:
                        continue
                    overlap_start = max(d_layer_start, p_layer_start)
                    overlap_end = min(d_layer_end, p_layer_end)
                    addr_start = (overlap_start - d_layer_start) * addr_stride
                    addr_end = (overlap_end - d_layer_start) * addr_stride
                    # When P_SIZE < D_SIZE, P's single rank owns more layers
                    # than D rank's subset. P must skip the layers belonging
                    # to earlier D ranks in global layer order.
                    p_src_offset = (overlap_start - p_layer_start) * addr_stride
                    asyncio.create_task(
                        self.receive_kv_from_single_worker(
                            worker_addr, pull_metas, (addr_start, addr_end),
                            src_layer_offset=p_src_offset,
                            chunk_idx=i if needs_chunk_idx else None,
                        )
                    )
""",
),

]
