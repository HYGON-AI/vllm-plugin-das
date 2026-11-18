# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU adapter for bounded Mooncake Store transfer batches."""

from copy import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from vllm.distributed.parallel_state import get_pcp_group
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import (
    MooncakeStoreConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    MooncakeStoreConnectorMetadata,
    PoolKey,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


_SINGLE_PCP_REPLICA = "pcp_single_replica"
_MAX_MULTI_BUFFER_BATCH = 32
_MAX_MULTI_BUFFER_GET_BATCH = 256


def _configure_single_pcp_replica(vllm_config, role):
    """Use PCP rank 0 as the sole Store writer when explicitly requested."""
    transfer = vllm_config.kv_transfer_config
    extra = transfer.kv_connector_extra_config
    if extra.get(_SINGLE_PCP_REPLICA) is not True:
        return vllm_config, False
    if role != KVConnectorRole.WORKER:
        return vllm_config, True

    pcp_group = get_pcp_group()
    if pcp_group.world_size <= 1:
        raise ValueError("pcp_single_replica requires PCP size greater than 1")
    if pcp_group.rank_in_group != 0:
        # Keep the original config intact for PCP validation and sibling
        # connectors; only this Store worker should skip creating a writer.
        vllm_config = copy(vllm_config)
        transfer = copy(transfer)
        transfer.kv_role = "kv_consumer"
        vllm_config.kv_transfer_config = transfer
    return vllm_config, True


def _use_canonical_pcp_key_namespace(worker) -> None:
    """Make every PCP rank address the key written by PCP rank 0."""
    worker.pcp_size = 1
    worker.pcp_rank = 0
    worker.metadata = replace(worker.metadata, pcp_rank=0)
    for db in worker.token_dbs:
        db.metadata = replace(db.metadata, pcp_rank=0)
        db._key_prefix = PoolKey.build_prefix(db.metadata)
    worker._init_lookup_key_prefixes()


class _BoundedStore:
    """Delegate to the native Store while limiting each multi-buffer RPC."""

    def __init__(self, store):
        self.store = store

    def __getattr__(self, name):
        return getattr(self.store, name)

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, replicate_config):
        results = []
        for start in range(0, len(keys), _MAX_MULTI_BUFFER_BATCH):
            end = start + _MAX_MULTI_BUFFER_BATCH
            results.extend(self.store.batch_put_from_multi_buffers(
                keys[start:end], addrs[start:end], sizes[start:end],
                replicate_config))
        return results

    def batch_get_into_multi_buffers(self, keys, addrs, sizes):
        results = []
        for start in range(0, len(keys), _MAX_MULTI_BUFFER_GET_BATCH):
            end = start + _MAX_MULTI_BUFFER_GET_BATCH
            results.extend(self.store.batch_get_into_multi_buffers(
                keys[start:end], addrs[start:end], sizes[start:end]))
        return results


class HcuMooncakeStoreConnector(MooncakeStoreConnector):
    """Use native Mooncake Store with bounded transfer batches."""

    def __init__(self, *args, **kwargs):
        config = args[0] if args else kwargs["vllm_config"]
        role = args[1] if len(args) > 1 else kwargs["role"]
        parallel_config = config.parallel_config
        self._singlecopy_send_count = parallel_config.world_size
        self._singlecopy_recv_count = parallel_config.world_size
        self._singlecopy_async_send_ids: set[str] = set()
        config, single_pcp_replica = _configure_single_pcp_replica(config, role)
        if args:
            args = (config, *args[1:])
        else:
            kwargs["vllm_config"] = config
        super().__init__(*args, **kwargs)
        self._single_pcp_replica = single_pcp_replica
        if self.connector_worker is not None:
            if single_pcp_replica:
                _use_canonical_pcp_key_namespace(self.connector_worker)
            self.connector_worker.store = _BoundedStore(self.connector_worker.store)

    def get_finished_send_count(self) -> int | None:
        if self._single_pcp_replica:
            return self._singlecopy_send_count
        return super().get_finished_send_count()

    def get_finished_recv_count(self) -> int | None:
        if self._single_pcp_replica:
            return self._singlecopy_recv_count
        return super().get_finished_recv_count()

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        async_save, transfer_params = super().request_finished_all_groups(
            request, block_ids
        )
        if (
            self._single_pcp_replica
            and self.connector_scheduler is not None
            and async_save
        ):
            self._singlecopy_async_send_ids.add(request.request_id)
        return async_save, transfer_params

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> MooncakeStoreConnectorMetadata:
        metadata = super().build_connector_meta(scheduler_output)
        if self._single_pcp_replica and self._singlecopy_async_send_ids:
            setattr(
                metadata,
                "_hcu_singlecopy_noop_send_ids",
                self._singlecopy_async_send_ids,
            )
            self._singlecopy_async_send_ids = set()
        return metadata

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        finished_sending, finished_recving = super().get_finished(finished_req_ids)
        worker = self.connector_worker
        if (
            not self._single_pcp_replica
            or worker is None
            or worker.kv_role != "kv_consumer"
        ):
            return finished_sending, finished_recving

        # MultiConnector counts async saves on the scheduler side. Consumer
        # ranks must report this no-op Store completion so their P2P completion
        # can drain the same per-request async-save count as the writer rank.
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MooncakeStoreConnectorMetadata)
        no_op_sends = getattr(metadata, "_hcu_singlecopy_noop_send_ids", set())
        if no_op_sends:
            finished_sending = (finished_sending or set()) | no_op_sends
        return finished_sending or None, finished_recving


__all__ = ["HcuMooncakeStoreConnector"]
