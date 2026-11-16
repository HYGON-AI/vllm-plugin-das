# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Persistent EPLB staging formats, accounted during model loading."""

from __future__ import annotations

from collections.abc import Sequence

import torch

_POOL_ATTRIBUTE = "_vllm_hcu_eplb_buffer_pool"


def buffer_signature(tensors: Sequence[torch.Tensor]) -> tuple:
    return tuple((tuple(t.shape), t.dtype, t.device) for t in tensors)


class ExpertBufferPool:
    """One persistent staging group per layer format, shared by sync and async."""

    def __init__(self, expert_weights, expert_buffer):
        self.buffers = {buffer_signature(expert_buffer): list(expert_buffer)}
        self.layers = {}
        for layer in expert_weights:
            signature = buffer_signature(layer)
            self.layers.setdefault(signature, layer)
            if signature not in self.buffers:
                self.buffers[signature] = [torch.empty_like(w) for w in layer]

    def for_layer(self, layer):
        try:
            return self.buffers[buffer_signature(layer)]
        except KeyError as exc:
            raise RuntimeError(
                "Expert format changed after EPLB communicator initialization; "
                "recreate the communicator before transferring weights"
            ) from exc

    def registration_buffers(self):
        return [tensor for group in self.buffers.values() for tensor in group]


def create_communicator_with_buffers(
    factory, group_coordinator, backend, expert_weights, expert_buffer
):
    # This runs in EplbState.add_model, before KV cache sizing or async startup.
    # NIXL must see every receive address during its initial registration.
    pool = ExpertBufferPool(expert_weights, expert_buffer)
    communicator = factory(
        group_coordinator=group_coordinator,
        backend=backend,
        expert_weights=expert_weights,
        expert_buffer=(
            pool.registration_buffers() if backend == "nixl" else expert_buffer
        ),
    )
    setattr(communicator, _POOL_ATTRIBUTE, pool)
    return communicator


def get_buffer_pool(communicator, expert_weights, expert_buffer):
    pool = getattr(communicator, _POOL_ATTRIBUTE, None)
    if pool is None:
        # Preserve direct calls to the transfer API for unregistered backends.
        # An already-created NIXL communicator cannot register late buffers.
        from vllm.distributed.eplb.eplb_communicator import NixlEplbCommunicator

        if isinstance(communicator, NixlEplbCommunicator):
            # transfer_layer only sees one layer. Inspect all registered source
            # formats so a direct async call fails before its first transfer.
            if any(
                buffer_signature(layer) != buffer_signature(expert_buffer)
                for layer in communicator._all_expert_weights
            ):
                raise RuntimeError(
                    "Mixed-format NIXL EPLB buffers must be allocated and registered "
                    "through the HCU communicator factory before transfer"
                )
        pool = ExpertBufferPool(expert_weights, expert_buffer)
        setattr(communicator, _POOL_ATTRIBUTE, pool)
    for layer in expert_weights:
        pool.for_layer(layer)
    return pool
