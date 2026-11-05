# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Step5 DSA summaries and unfinished regions carried with Mooncake KV.

Indexer K/Z projections and summaries are replicated across TP ranks, unlike
the sharded attention K/V. They must therefore use independent descriptors.
The fixed snapshot pool keeps source buffers alive until every paired D rank
acknowledges its transfer. No model prefill is repeated on the consumer.
"""
from __future__ import annotations

import os
import logging
import threading
from collections.abc import Hashable
from typing import Any

import torch
from vllm_hcu.runtime_compat.step5 import is_step5_pd_config


_DTYPE_CODES = {torch.bfloat16: 1, torch.float16: 2, torch.float32: 3}
_logger = logging.getLogger("vllm.step5.pd")


def _coalesced_block_runs(source: list[int], target: list[int]) -> list[tuple[int, int, int]]:
    """Merge paired adjacent blocks, preserving each source-to-target mapping."""
    runs = []
    direction = count = 0
    for src, dst in zip(source, target):
        step = src - previous_src if count else 0
        if (count and step in (-1, 1) and dst - previous_dst == step
                and direction in (0, step)):
            count += 1
            direction = step
            if step < 0:
                start_src, start_dst = src, dst
        else:
            if count:
                runs.append((start_src, start_dst, count))
            start_src, start_dst, count, direction = src, dst, 1, 0
        previous_src, previous_dst = src, dst
    if count:
        runs.append((start_src, start_dst, count))
    return runs


class _ImportLease:
    def __init__(self):
        self.owner = object()
        self.rows: list[tuple[Any, int]] = []
        self.received = False
        self.finished = False


class _SnapshotPool:
    def __init__(self, state: Any, capacity: int, device: torch.device):
        self.tensor = torch.empty(
            (state.num_layers, capacity, 2, state.geometry.region_size,
             state.geometry.proxy_dim), dtype=state.dtype, device=device,
        )
        self._rows: dict[Hashable, int] = {}
        self._free = list(reversed(range(capacity)))
        self._lock = threading.RLock()

    def acquire(self, first_block: Hashable) -> int:
        with self._lock:
            if first_block not in self._rows:
                if not self._free:
                    raise RuntimeError(
                        "Step5 PD auxiliary snapshot pool is full; too many "
                        "prefills are waiting for decode-side KV acknowledgements"
                    )
                self._rows[first_block] = self._free.pop()
            return self._rows[first_block]

    def get(self, first_block: Hashable) -> int:
        with self._lock:
            if first_block not in self._rows:
                raise RuntimeError(
                    f"Step5 PD has no indexer tail snapshot for block {first_block}"
                )
            return self._rows[first_block]

    def release(self, first_block: Hashable) -> None:
        with self._lock:
            row = self._rows.pop(first_block, None)
            if row is not None:
                self._free.append(row)


class Step5AuxManager:
    def __init__(self, worker: Any, indexers: dict[str, Any], kv_caches: dict):
        self.worker = worker
        self.indexers = indexers
        self._indexer_groups = {
            id(indexer): worker._layer_group_indices[name]
            for name, indexer in indexers.items()
        }
        self.pools: dict[int, _SnapshotPool] = {}
        self._sources: dict[str, list[list[int]]] = {}
        self._lock = threading.RLock()
        self._compiled_consumer = not worker.is_kv_producer and not getattr(
            getattr(worker.vllm_config, "model_config", None), "enforce_eager", True)
        self._imports: dict[str, _ImportLease] = {}
        # Transport and captured consumers retain these addresses. Keep the
        # allocations alive, and reject rebinding until KV is registered again.
        self._allocations: dict[int, tuple] = {}
        registered: dict[int, Any] = {}
        capacity = 4 * worker.vllm_config.scheduler_config.max_num_seqs + 32

        for name, indexer in indexers.items():
            state = indexer.state
            cache = kv_caches[name]
            state.ensure_allocated(cache, cache.device)
            allocation = (state.summary, state.pending_key, state.pending_z)
            self._allocations[id(state)] = (
                state, allocation, tuple(t.data_ptr() for t in allocation),
            )
            if state.dtype not in _DTYPE_CODES:
                raise ValueError(f"Unsupported Step5 indexer dtype: {state.dtype}")
            num_blocks = int(cache.shape[0] if cache.ndim == 4 else cache.shape[1])
            if state.summary.shape[1] != num_blocks * state.regions_per_page:
                raise ValueError("Step5 DSA summary capacity differs from its KV pool")
            tensors = [state.summary]
            if worker.is_kv_producer:
                if id(state) not in self.pools:
                    self.pools[id(state)] = _SnapshotPool(state, capacity, cache.device)
                    state._pd_aux_capture = self.capture
                tensors.append(self.pools[id(state)].tensor)
            else:
                tensors += [state.pending_key, state.pending_z]
            for tensor in tensors:
                storage = tensor.untyped_storage()
                registered[storage.data_ptr()] = storage

        # Storage references also survive an in-place Tensor.set_(). Keeping
        # only the Tensor object would not keep its old registered memory alive.
        self._registered_storages = tuple(registered.values())
        ptrs = list(registered)
        sizes = [storage.nbytes() for storage in self._registered_storages]
        if worker.engine.batch_register_memory(ptrs, sizes) != 0:
            raise RuntimeError("Step5 DSA auxiliary memory registration failed")
        if hasattr(worker, "seen_base_addresses"):
            worker.seen_base_addresses.extend(ptrs)
        first = next(iter(kv_caches[name] for name in indexers))
        if first.is_cuda:
            torch.cuda.synchronize(first.device)

    def _validate_allocations(self) -> None:
        for state, registered, pointers in self._allocations.values():
            current = (state.summary, state.pending_key, state.pending_z)
            if any(now is not old or old.data_ptr() != pointer
                   for now, old, pointer in zip(current, registered, pointers)):
                raise RuntimeError(
                    "Step5 PD auxiliary buffers changed after registration; "
                    "re-register KV caches before restoring or sending state"
                )

    def prepare_pulls(self, pulls: list[Any]):
        readiness = None
        if self._compiled_consumer and any(any(p.local_block_ids) for p in pulls):
            self._validate_allocations()
            first = next(iter(self._allocations.values()))[1][0]
            if first.is_cuda:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Step5 PD restoration must run before graph replay")
                # Do not block the model worker: the receiver waits for this
                # event before RDMA can overwrite a recycled pending row.
                readiness = torch.cuda.Event()
                readiness.record(torch.cuda.current_stream(first.device))
        for pull in pulls:
            self.prepare_pull(pull)
        return readiness

    def finish_pulls(self, received: set[str], finished: set[str]) -> None:
        # Called on the model worker, not the receiver thread. Pending-table
        # mutations stay ordered with metadata construction and graph replay.
        for request_id, lease in list(self._imports.items()):
            lease.received |= request_id in received
            lease.finished |= request_id in finished
            if not lease.received:
                continue
            if lease.finished:
                for state, slot in lease.rows:
                    state.pending.release_reservation(slot, lease.owner)
            if lease.finished or all(not state.pending.is_reserved(slot, lease.owner)
                                     for state, slot in lease.rows):
                self._imports.pop(request_id)
                _logger.debug("Step5 PD import lease released: request=%s finished=%s pending=%d",
                              request_id, lease.finished, len(self._imports))

    def capture(self, indexer: Any, md: Any, sched: Any,
                index_k: torch.Tensor, index_z: torch.Tensor) -> None:
        """Keep the last prompt region, including a region that just closed."""
        with self._lock:
            if not self._sources:
                return
        if index_k.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Step5 DSA P/D auxiliary capture requires eager execution")
        state = indexer.state
        key = ("step5_pd_rows", id(md), id(sched))
        rows = state.step_cache.get(key)
        if rows is None:
            starts_host = getattr(md, "query_start_loc_cpu", None)
            lengths_host = getattr(md, "seq_lens_cpu", None)
            table_host = getattr(md, "block_table_np", None)
            starts = (starts_host if starts_host is not None else md.query_start_loc.detach().cpu()).tolist()
            lengths = (lengths_host if lengths_host is not None else md.seq_lens.detach().cpu()).tolist()
            table = (table_host if table_host is not None else md.block_table.detach().cpu()).tolist()
            if sched.decode or sched.spec:
                pending = sched.pending_in.detach().cpu().tolist()
            else:
                pending = [0] * (len(starts) - 1)
                for plan in sched.requests:
                    pending[plan.row] = plan.pending_row_in
            rows = (starts, lengths, table, pending)
            state.step_cache[key] = rows
        starts, lengths, table, pending = rows
        region = state.geometry.region_size
        pool = self.pools[id(state)]
        group = self._indexer_groups[id(indexer)]
        for row, (start, end) in enumerate(zip(starts, starts[1:])):
            qlen = end - start
            if qlen <= 0 or lengths[row] <= 0:
                continue
            first_block = int(table[row][0])
            with self._lock:
                tracked = any(
                    group < len(groups) and groups[group]
                    and groups[group][0] == first_block
                    for groups in self._sources.values()
                )
            if not tracked:
                continue
            last_len = (lengths[row] - 1) % region + 1
            prefix = max(last_len - qlen, 0)
            destination = pool.tensor[indexer.slot, pool.acquire((group, first_block))]
            destination.zero_()
            if prefix:
                source_row = pending[row]
                if source_row < 0:
                    raise RuntimeError("Step5 PD capture is missing the indexer prefix")
                destination[0, :prefix].copy_(state.pending_key[indexer.slot, source_row, :prefix])
                destination[1, :prefix].copy_(state.pending_z[indexer.slot, source_row, :prefix])
                source = slice(start, end)
            else:
                source = slice(end - last_len, end)
            destination[0, prefix:last_len].copy_(index_k[source, 0])
            destination[1, prefix:last_len].copy_(index_z[source, 0])

    def track_source(self, transfer_id: str, block_ids: list[list[int]]) -> None:
        groups = self.worker._logical_to_kernel_block_ids(block_ids)
        with self._lock:
            old = self._sources.get(transfer_id)
            if old is not None:
                changed = [
                    previous if previous and (
                        i >= len(groups) or not groups[i]
                        or previous[0] != groups[i][0]
                    ) else []
                    for i, previous in enumerate(old)
                ]
                self._release_groups(changed)
            self._sources[transfer_id] = groups

    def _release_groups(self, groups: list[list[int]]) -> None:
        released = set()
        for name, indexer in self.indexers.items():
            group = self.worker._layer_group_indices[name]
            if group >= len(groups) or not groups[group]:
                continue
            key = (id(indexer.state), group, groups[group][0])
            if key not in released:
                self.pools[id(indexer.state)].release((group, groups[group][0]))
                released.add(key)

    def release(self, transfer_id: str) -> None:
        if not self.worker.is_kv_producer:
            return
        with self._lock:
            groups = self._sources.pop(transfer_id, None)
            if groups is not None:
                self._release_groups(groups)

    def prepare_pull(self, pull_meta: Any) -> None:
        # Aborted-before-allocation requests still notify P to release its KV.
        # They have no destination pages or prompt length to restore.
        if not any(pull_meta.local_block_ids):
            pull_meta.dsa_aux = {}
            return
        self._validate_allocations()
        first = next(iter(self._allocations.values()))[1][0]
        if first.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Step5 PD auxiliary restoration must run before graph replay")
        groups = self.worker._logical_to_kernel_block_ids(pull_meta.local_block_ids)
        n = int(pull_meta.num_prompt_tokens)
        if n <= 0:
            raise ValueError("Step5 DSA P/D requires the prompt token count")
        descriptors = {}
        rows: dict[tuple[int, int], int] = {}
        lease = None
        if self._compiled_consumer:
            lease = self._imports.setdefault(pull_meta.d_req_id, _ImportLease())
        for name, indexer in self.indexers.items():
            state = indexer.state
            group = self.worker._layer_group_indices[name]
            if group >= len(groups) or not groups[group]:
                continue
            blocks = groups[group]
            last = (n - 1) // state.block_size
            if last >= len(blocks):
                raise ValueError("Step5 DSA destination has too few prompt KV blocks")
            slot = (blocks[last] * state.regions_per_page
                    + ((n - 1) % state.block_size) // state.geometry.region_size)
            if not 0 <= slot < state.summary.shape[1]:
                raise ValueError("Step5 DSA destination block ID exceeds its registered pool")
            key = (id(state), slot)
            if key not in rows:
                # A just-finished request can leave a builder-owned cleanup
                # entry for a physical block that has since been reallocated.
                # Do not let that deferred cleanup erase this imported tail.
                previous = getattr(state, "_schedule", None)
                if previous is not None:
                    previous.completed_slots[:] = [
                        old_slot for old_slot in previous.completed_slots
                        if old_slot != slot
                    ]
                if lease is None:
                    rows[key] = state.pending.acquire(slot)
                else:
                    rows[key] = state.pending.reserve(slot, lease.owner)
                    if not any(old_state is state and old_slot == slot
                               for old_state, old_slot in lease.rows):
                        lease.rows.append((state, slot))
            row = rows[key]
            summary = state.summary[indexer.slot]
            tail_k = state.pending_key[indexer.slot, row]
            tail_z = state.pending_z[indexer.slot, row]
            descriptors[name] = (
                group, summary.data_ptr(),
                state.regions_per_page * state.geometry.proxy_dim * summary.element_size(),
                summary.shape[0] // state.regions_per_page,
                tail_k.data_ptr(), tail_z.data_ptr(),
                tail_k.numel() * tail_k.element_size(), _DTYPE_CODES[state.dtype],
            )
        pull_meta.dsa_aux = descriptors

    def transfer_parameters(self, request_id: str, agent_meta: Any,
                            source_groups: list[list[int]],
                            target_groups: list[list[int]]) -> tuple[list[int], list[int], list[int]]:
        # Each indexer's K/Z and summary are TP-replicated. When a D rank pulls
        # several P shards, one leader sends these bytes while all send main KV.
        ratio = self.worker.transfer_topo.tp_ratio(agent_meta.remote_tp_size)
        if ratio > 0 and self.worker.tp_rank % ratio:
            return [], [], []
        self._validate_allocations()
        peer = agent_meta.dsa_aux.get(request_id, {})
        missing = set(self.indexers) - set(peer)
        if missing:
            raise ValueError("Step5 P/D peer did not provide compatible DSA auxiliary state")
        sources, targets, sizes = [], [], []
        groups = {}
        summary_blocks = summary_runs = 0
        for name, indexer in self.indexers.items():
            state = indexer.state
            group = self.worker._layer_group_indices[name]
            remote_group, remote_base, stride, capacity, tail_k, tail_z, tail_bytes, code = peer[name]
            summary = state.summary[indexer.slot]
            expected_stride = state.regions_per_page * state.geometry.proxy_dim * summary.element_size()
            expected_tail = state.geometry.region_size * state.geometry.proxy_dim * state.pending_key.element_size()
            if (group != remote_group or stride != expected_stride
                    or tail_bytes != expected_tail or code != _DTYPE_CODES[state.dtype]):
                raise ValueError("Step5 DSA auxiliary geometry/dtype differs between P and D")
            if min(remote_base, tail_k, tail_z) <= 0 or capacity <= 0:
                raise ValueError("Step5 DSA destination memory descriptor is invalid")
            if group >= len(source_groups) or group >= len(target_groups):
                raise ValueError("Step5 DSA auxiliary KV group mapping is incomplete")
            src, dst = source_groups[group], target_groups[group]
            if len(src) != len(dst) or not src:
                raise ValueError("Step5 DSA auxiliary KV block mapping is incomplete")
            if group not in groups:
                groups[group] = (
                    _coalesced_block_runs(src, dst),
                    min(src), max(src), min(dst), max(dst),
                )
            runs, source_min, source_max, target_min, target_max = groups[group]
            source_capacity = summary.shape[0] // state.regions_per_page
            if (min(source_min, target_min) < 0
                    or source_max >= source_capacity or target_max >= capacity):
                raise ValueError("Step5 DSA auxiliary block ID exceeds its registered pool")
            summary_blocks += len(src)
            summary_runs += len(runs)
            for source_block, target_block, count in runs:
                sources.append(summary.data_ptr() + source_block * stride)
                targets.append(remote_base + target_block * stride)
                sizes.append(count * stride)
            row = self.pools[id(state)].get((group, src[0]))
            snapshot = self.pools[id(state)].tensor[indexer.slot, row]
            sources += [snapshot[0].data_ptr(), snapshot[1].data_ptr()]
            targets += [tail_k, tail_z]
            sizes += [tail_bytes, tail_bytes]
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "Step5 PD auxiliary transfer: request=%s blocks=%d runs=%d descriptors=%d bytes=%d",
                request_id, summary_blocks, summary_runs, len(sizes), sum(sizes),
            )
        return sources, targets, sizes


def create_step5_aux_manager(worker: Any, kv_caches: dict[str, torch.Tensor]):
    if not is_step5_pd_config(getattr(worker, "vllm_config", None)):
        return None
    from vllm_hcu.model_executor.layers.step4_dsa import _INDEXERS
    indexers = {name: indexer for name, indexer in _INDEXERS.items() if name in kv_caches}
    if not indexers:
        return None
    if not (worker.is_kv_producer or worker.is_kv_consumer):
        raise NotImplementedError("Step5 DSA auxiliary transfer requires separate producer/consumer roles")
    config = worker.vllm_config
    if not config.model_config.enforce_eager:
        if worker.is_kv_producer:
            raise NotImplementedError("Step5 DSA P/D producer snapshots require --enforce-eager")
        # Only producers install the Python snapshot callback. Consumers import
        # into fixed buffers before building/replaying the model's schedule.
        # Their sparse KV readers must also be true views: the paged bridge
        # performs eager-only materialization and cannot be captured.
        for name in indexers:
            cache = kv_caches[name]
            affine = (
                cache.ndim == 4 and cache.stride(0) == cache.shape[2] * cache.stride(2)
                or cache.ndim == 5 and cache.stride(1) == cache.shape[2] * cache.stride(2)
            )
            if not affine or cache.stride(-1) != 1:
                raise NotImplementedError(
                    "Compiled Step5 PD consumers require affine sparse KV views; "
                    "use --enforce-eager for this cache layout"
                )
    if config.cache_config.enable_prefix_caching:
        raise NotImplementedError("Step5 DSA P/D auxiliary transfer requires --no-enable-prefix-caching")
    parallel = config.parallel_config
    dbo_requested = (int(os.environ.get("HCU_DBO_MIN_TOKENS", "0")) > 0
                     or getattr(parallel, "enable_dbo", False))
    if dbo_requested and not (
        worker.is_kv_producer
        and getattr(parallel, "enable_dbo", False)
        and getattr(parallel, "_hcu_step5_dbo", False) is True
    ):
        raise NotImplementedError(
            "Step5 DSA auxiliary transfer supports DBO only with the HCU "
            "eager producer runtime"
        )
    if (getattr(parallel, "pipeline_parallel_size", 1) != 1
            or getattr(parallel, "prefill_context_parallel_size", 1) != 1
            or getattr(parallel, "decode_context_parallel_size", 1) != 1):
        raise NotImplementedError("Step5 DSA auxiliary transfer currently supports TP and DP only")
    return Step5AuxManager(worker, indexers, kv_caches)
