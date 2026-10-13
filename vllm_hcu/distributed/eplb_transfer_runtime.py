# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

from vllm.distributed.eplb.rebalance_execute import (
    EplbCommunicator,
    ProcessGroup,
    Sequence,
    _map_new_expert_indices_with_rank_mapping,
    _map_old_expert_indices_with_rank_mapping,
    all_gather,
    gpu_sync_allowed,
    move_from_buffer,
    move_to_buffer,
    torch,
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

    first_layer_weights = list(expert_weights[0])

    if is_profile:
        if communicator.needs_profile_buffer_reservation:
            # Reserve NCCL communication buffers via a dummy all_gather.
            # Backends that pre-allocate their own transfer buffers
            # skip this to avoid the extra memory spike during profiling.
            profile_buffer: list[torch.Tensor] = [
                torch.empty_like(w) for w in first_layer_weights
            ]
            for weight, buffer in zip(expert_weights[0], profile_buffer):
                dummy_recv_buffer = [buffer for _ in range(ep_size)]
                torch.distributed.barrier()
                all_gather(
                    dummy_recv_buffer,
                    weight,
                    group=ep_group,
                )
        return

    weights_buffer = list(expert_buffer)

    # The per-layer transfer plan is built in Python, so both maps have to
    # come back to the host. Once per rearrangement.
    with gpu_sync_allowed():
        old_global_expert_indices_cpu = old_global_expert_indices.cpu().numpy()
        new_global_expert_indices_cpu = new_global_expert_indices.cpu().numpy()

    def _signature(tensors: Sequence[torch.Tensor]) -> list:
        return [(t.shape, t.dtype) for t in tensors]

    # Models may mix layer formats (e.g. Step5 keeps its last MoE layers in
    # bf16 while the rest are fp8). The persistent buffer only matches the
    # first layer; a mismatched recv buffer makes the P2P counts disagree and
    # the transfer hangs, so give other formats their own temporary buffers.
    buffers_by_signature = {repr(_signature(weights_buffer)): weights_buffer}

    for layer_idx in range(num_moe_layers):
        signature = repr(_signature(expert_weights[layer_idx]))
        layer_buffer = buffers_by_signature.get(signature)
        if layer_buffer is None:
            layer_buffer = [torch.empty_like(w) for w in expert_weights[layer_idx]]
            buffers_by_signature[signature] = layer_buffer
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
