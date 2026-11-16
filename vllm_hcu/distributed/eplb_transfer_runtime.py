# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

from dataclasses import dataclass, fields
from functools import wraps

from vllm.distributed.eplb.rebalance_execute import (
    EplbCommunicator,
    ProcessGroup,
    Sequence,
    TransferMetadata,
    _map_new_expert_indices_with_rank_mapping,
    _map_old_expert_indices_with_rank_mapping,
    all_gather,
    gpu_sync_allowed,
    move_from_buffer,
    move_to_buffer,
    torch,
    transfer_layer as _transfer_layer,
)

from vllm_hcu.distributed.eplb_buffer_runtime import get_buffer_pool


@dataclass
class BufferedTransferMetadata(TransferMetadata):
    # Kept alive by AsyncEplbLayerResult until completion/drain acknowledges it.
    expert_buffer: list[torch.Tensor]
    layer_idx: int


@wraps(_transfer_layer)
def transfer_layer(
    old_layer_indices: torch.Tensor,
    new_layer_indices: torch.Tensor,
    expert_weights: Sequence[torch.Tensor],
    expert_weights_buffer: Sequence[torch.Tensor],
    ep_group: ProcessGroup,
    communicator: EplbCommunicator,
    is_profile: bool = False,
    cuda_stream: torch.cuda.Stream | None = None,
    rank_mapping: dict[int, int] | None = None,
    layer_idx: int = 0,
) -> TransferMetadata:
    pool = get_buffer_pool(communicator, [expert_weights], expert_weights_buffer)
    layer_buffer = pool.for_layer(expert_weights)
    metadata = _transfer_layer(
        old_layer_indices=old_layer_indices,
        new_layer_indices=new_layer_indices,
        expert_weights=expert_weights,
        expert_weights_buffer=layer_buffer,
        ep_group=ep_group,
        communicator=communicator,
        is_profile=is_profile,
        cuda_stream=cuda_stream,
        rank_mapping=rank_mapping,
        layer_idx=layer_idx,
    )
    return BufferedTransferMetadata(
        **{field.name: getattr(metadata, field.name) for field in fields(TransferMetadata)},
        expert_buffer=layer_buffer,
        layer_idx=layer_idx,
    )


def rearrange_expert_weights_inplace(
    old_global_expert_indices: torch.Tensor,
    new_global_expert_indices: torch.Tensor,
    expert_weights: Sequence[Sequence[torch.Tensor]],
    expert_buffer: Sequence[torch.Tensor],
    ep_group: ProcessGroup,
    communicator: EplbCommunicator,
    is_profile: bool = False,
    rank_mapping: dict[int, int] | None = None,
) -> None:
    """
    Rearranges the expert weights in place according to the new expert indices.

    The value of the indices arguments are logical indices of the experts,
    while keys are physical.

    Args:
        old_global_expert_indices: Shape (num_moe_layers, num_physical_experts).
        new_global_expert_indices: Shape (num_moe_layers, num_physical_experts).
        expert_weights: A sequence of shape (num_moe_layers)(weight_count)
            of tensors of shape (num_local_physical_experts, hidden_size_i).
            For example, a linear layer may have up and down projection,
            so weight_count = 2. Each weight's hidden size can be different.
        expert_buffer: Pre-allocated receive buffer tensors (one per
            weight tensor in a single layer).
        ep_group: The device process group for expert parallelism.
        communicator: EplbCommunicator instance for P2P communication.
        is_profile (bool): If `True`, do not perform any actual weight copy.
            This is used during profile run, where we only perform dummy
            communications to reserve enough memory for the buffers.
        rank_mapping: A dictionary mapping old rank to new rank.
    """
    if rank_mapping is not None:
        if len(rank_mapping) == ep_group.size():
            # scale down
            new_global_expert_indices = _map_new_expert_indices_with_rank_mapping(
                new_global_expert_indices,
                rank_mapping,
            )
        else:
            # scale up
            old_global_expert_indices = _map_old_expert_indices_with_rank_mapping(
                old_global_expert_indices,
                rank_mapping,
                ep_group.size(),
            )

    assert old_global_expert_indices.shape[1] == new_global_expert_indices.shape[1]

    num_moe_layers, num_physical_experts = old_global_expert_indices.shape
    assert len(expert_weights) == num_moe_layers
    assert len(expert_weights[0]) >= 1

    num_local_physical_experts = expert_weights[0][0].shape[0]
    assert new_global_expert_indices.shape == (num_moe_layers, num_physical_experts)

    ep_size = ep_group.size()
    ep_rank = ep_group.rank()
    assert num_physical_experts == ep_size * num_local_physical_experts

    pool = get_buffer_pool(communicator, expert_weights, expert_buffer)

    if is_profile:
        if communicator.needs_profile_buffer_reservation:
            # All staging groups already live until communicator teardown.
            # Exercise communication for every distinct format using those
            # same buffers; no larger format can first appear after profiling.
            for layer in pool.layers.values():
                for weight, buffer in zip(layer, pool.for_layer(layer)):
                    dummy_recv_buffer = [buffer for _ in range(ep_size)]
                    torch.distributed.barrier()
                    all_gather(dummy_recv_buffer, weight, group=ep_group)
        return

    # The per-layer transfer plan is built in Python, so both maps have to
    # come back to the host. Once per rearrangement.
    with gpu_sync_allowed():
        old_global_expert_indices_cpu = old_global_expert_indices.cpu().numpy()
        new_global_expert_indices_cpu = new_global_expert_indices.cpu().numpy()

    for layer_idx in range(num_moe_layers):
        layer_buffer = pool.for_layer(expert_weights[layer_idx])
        transfer_metadata = move_to_buffer(
            num_local_experts=num_local_physical_experts,
            old_indices=old_global_expert_indices_cpu[layer_idx],
            new_indices=new_global_expert_indices_cpu[layer_idx],
            expert_weights=expert_weights[layer_idx],
            expert_weights_buffers=layer_buffer,
            cuda_stream=None,
            ep_rank=ep_rank,
            communicator=communicator,
            layer_idx=layer_idx,
        )

        move_from_buffer(
            expert_weights=expert_weights[layer_idx],
            expert_weights_buffers=layer_buffer,
            transfer_metadata=transfer_metadata,
            new_indices=new_global_expert_indices_cpu[layer_idx],
            ep_rank=ep_rank,
        )
