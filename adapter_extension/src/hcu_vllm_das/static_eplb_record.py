# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Calibration recording that turns observed routing load into an offline map.

A calibration run keeps the identity placement, so routing stays numerically
equal to a run without static EPLB, and only switches on the load counters Core's
routing kernel already maintains. At a fixed step interval the per-rank counters
are summed across the EP group and handed to Core's own rebalancing policy, so
the produced placement is the same kind of artifact the vLLM EPLB lifecycle would
have proposed. The file is rewritten in full each time and validated before it
replaces the previous one, so whenever the operator stops the server the map on
disk is complete and loadable.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch

from .static_eplb import StaticEplbPlan

_OFFLINE_EPLB_FORMAT = 'vllm_offline_eplb_physical_to_logical_by_model'


def propose_rows(load: torch.Tensor, *, num_physical: int, num_groups: int,
                 num_nodes: int, num_ranks: int) -> tuple[tuple[int, ...], ...]:
    """Run Core's rebalancing policy over cumulative logical-expert load."""
    from vllm.distributed.eplb.policy import EPLB_POLICIES

    if load.ndim != 2:
        raise ValueError('calibration load must be [layers, num_logical_experts]')
    if num_physical % num_ranks != 0:
        raise ValueError(
            f'num_physical_experts {num_physical} must be divisible by '
            f'num_ranks {num_ranks}')
    proposal = EPLB_POLICIES['default'].rebalance_experts(
        load.detach().to(device='cpu'),
        num_replicas=num_physical,
        num_groups=max(1, int(num_groups)),
        num_nodes=num_nodes,
        num_ranks=num_ranks,
    )
    rows = proposal.detach().to(device='cpu').tolist()
    return tuple(tuple(int(value) for value in row) for row in rows)


def _document(rows, *, model_key: str, model_name: str, num_logical: int,
              num_redundant: int) -> dict:
    return {
        'version': 2,
        'format': _OFFLINE_EPLB_FORMAT,
        'model_maps': {
            model_key: {
                'record_kind': 'proposal',
                'model_name': model_name,
                'model_class': model_key,
                'num_moe_layers': len(rows),
                'num_logical_experts': num_logical,
                'num_physical_experts': num_logical + num_redundant,
                'num_redundant_experts': num_redundant,
                'physical_to_logical_map': [list(row) for row in rows],
            }
        },
    }


def write_offline_map(path, rows, *, model_key: str, model_name: str,
                      num_logical: int, num_redundant: int) -> str:
    """Validate a proposal and atomically replace the map file. Returns its sha256."""
    import hashlib

    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_document(rows, model_key=model_key, model_name=model_name,
                                   num_logical=num_logical,
                                   num_redundant=num_redundant), sort_keys=True)
    raw = payload.encode()
    # Build the plan the loader would build, so an unusable proposal is rejected
    # here instead of at the next deployment.
    StaticEplbPlan(
        model_key=model_key,
        model_name=model_name,
        source_path=str(output.resolve().parent / output.name),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        record_kind='proposal',
        _map_values=tuple(rows),
        num_logical_experts=num_logical,
        num_physical_experts=num_logical + num_redundant,
        num_redundant_experts=num_redundant,
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{output.name}.',
                                                  suffix='.tmp', dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest()


class RecordSession:
    """Accumulating load counters and the periodic proposal writer.

    Core's routing kernel adds into ``expert_load`` on every forward, so the
    counters are cumulative for the whole calibration run and are never zeroed.
    """

    def __init__(self, *, model, cfg, options, load, unpadded, num_logical,
                 num_groups, model_key, logger=None):
        self.model = model
        self.cfg = cfg
        self.options = options
        self.load = load
        self.unpadded = unpadded
        self.num_logical = num_logical
        self.num_groups = num_groups
        self.model_key = model_key
        self.steps = 0
        self.writes = 0
        self.last_sha256 = None
        self._logger = logger

    def prepare(self, num_rows: int) -> bool:
        """Publish this forward's real token count before the routers run.

        Core's routing kernel only accumulates rows below
        ``num_unpadded_tokens``; left at zero it records nothing at all. The
        vLLM lifecycle refreshes this from ``EplbState.prepare_forward``, which
        a plugin-owned placement never reaches, so it is filled here instead.
        Padding rows and memory-profile forwards are excluded. Return whether
        this forward can advance the calibration interval.
        """
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
        target = self.unpadded[0]
        # Core profiles MoE with non-padding dummy tokens before KV caches and
        # attention metadata exist. Those rows must never enter the counters.
        if context.attn_metadata is None or num_rows == 0:
            target.zero_()
            return False
        if getattr(context, 'ubatch_slices', None):
            raise ValueError(
                'static EPLB calibration does not support DBO/ubatching: '
                'per-ubatch token ranges would need per-ubatch counters')
        padding = getattr(context, 'is_padding', None)
        if padding is None:
            target.fill_(int(num_rows))
            return True
        # Stay on device: a host sync here would serialize every forward.
        real = padding[:num_rows].logical_not().sum()
        target.copy_(real.to(dtype=target.dtype), non_blocking=True)
        return True

    def step(self) -> None:
        self.steps += 1
        if self.steps % self.options.record_step_interval:
            return
        self.flush()

    def flush(self) -> str | None:
        """Sum load across EP ranks and rewrite the proposal from rank 0."""
        from vllm.distributed import get_ep_group, get_node_count

        group = get_ep_group()
        total = self.load.detach().clone()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM,
                                         group=group.device_group)
        if int(total.sum().item()) == 0:
            # No routed token has been observed yet; a proposal built from an
            # all-zero load would be an arbitrary permutation, not a placement.
            return None
        device_group = group.device_group
        if device_group.rank() != 0:
            return None
        rows = propose_rows(
            total,
            num_physical=self.num_logical + self.options.num_redundant_experts,
            num_groups=self.num_groups,
            num_nodes=get_node_count(),
            num_ranks=device_group.size(),
        )
        self.last_sha256 = write_offline_map(
            self.options.expert_map_record_path, rows,
            model_key=self.model_key,
            model_name=self.cfg.model_config.model,
            num_logical=self.num_logical,
            num_redundant=self.options.num_redundant_experts,
        )
        self.writes += 1
        if self._logger is not None:
            self._logger.info(
                'Static EPLB calibration wrote %s after %d steps '
                '(write=%d sha256=%s layers=%d physical=%d redundant=%d)',
                self.options.expert_map_record_path, self.steps, self.writes,
                self.last_sha256, len(rows),
                self.num_logical + self.options.num_redundant_experts,
                self.options.num_redundant_experts)
        return self.last_sha256


__all__ = [
    'RecordSession',
    'propose_rows',
    'write_offline_map',
]
