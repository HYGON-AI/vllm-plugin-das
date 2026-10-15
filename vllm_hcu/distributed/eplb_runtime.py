# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

from vllm.distributed.eplb.eplb_state import (
    EplbLayerState,
    EplbStats,
    StatelessGroupCoordinator,
    _commit_eplb_maps,
    _commit_eplb_maps_for_layer,
    _compute_eplb_load_stats,
    _node_count_with_rank_mapping,
    current_platform,
    get_ep_group,
    get_node_count,
    gpu_sync_allowed,
    logger,
    torch,
)

from vllm_hcu.distributed.eplb_buffer_runtime import buffer_signature
from vllm_hcu.distributed.eplb_transfer_runtime import (
    BufferedTransferMetadata,
    move_from_buffer,
    rearrange_expert_weights_inplace,
)
from vllm_hcu.runtime_compat.step5 import is_step5_config

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.model_executor.models.interfaces import MixtureOfExperts

_EPLB_TOUCHED_LOAD = os.environ.get("VLLM_EPLB_TOUCHED_LOAD", "0") == "1"
_EPLB_DUMP_PATH = os.environ.get("VLLM_EPLB_DUMP_PATH", "")


def _move_to_workspace(model_state, ep_rank):
    result = model_state.pending_result
    assert result is not None
    weights = model_state.model.expert_weights[result.layer_idx]
    metadata = result.transfer_metadata
    if isinstance(metadata, BufferedTransferMetadata):
        if metadata.layer_idx != result.layer_idx:
            raise RuntimeError("Async EPLB result belongs to a different layer")
        layer_buffer = metadata.expert_buffer
    else:
        layer_buffer = model_state.expert_buffer
    if buffer_signature(weights) != buffer_signature(layer_buffer):
        raise RuntimeError("Async EPLB result has incompatible staging buffers")
    move_from_buffer(
        expert_weights=weights,
        expert_weights_buffers=layer_buffer,
        transfer_metadata=metadata,
        new_indices=result.new_physical_to_logical_map.numpy(),
        ep_rank=ep_rank,
    )
    _commit_eplb_maps_for_layer(
        model_state,
        new_physical_to_logical_map=result.new_physical_to_logical_map,
        layer=result.layer_idx,
    )
    if result.layer_idx == model_state.model.num_moe_layers - 1:
        model_state.rebalanced = False
    # Keep the upstream consume ordering: clear ownership before waking worker.
    model_state.pending_result = None
    result.consumed_event.record()


def _rank_load_imbalance(
    mapping: torch.Tensor,
    logical_loads: torch.Tensor,
    ep_size: int,
    *,
    layerwise: bool,
) -> float:
    mapping = mapping.to(device=logical_loads.device, dtype=torch.long)
    replica_counts = torch.zeros_like(logical_loads)
    replica_counts.scatter_add_(
        dim=1, index=mapping, src=torch.ones_like(mapping, dtype=logical_loads.dtype)
    )
    loads_per_replica = torch.gather(
        logical_loads / replica_counts.clamp_min(1), dim=1, index=mapping
    )
    layer_rank_loads = loads_per_replica.reshape(
        logical_loads.shape[0], ep_size, -1
    )
    if layerwise:
        # Step5 accounts for each layer waiting for its slowest EP rank.
        loads_per_rank = layer_rank_loads.sum(dim=2)
        mean_load = loads_per_rank.mean(dim=1).sum()
        max_load = loads_per_rank.max(dim=1).values.sum()
    else:
        # Preserve the upstream placement metric for other models.
        loads_per_rank = layer_rank_loads.sum(dim=(0, 2))
        mean_load = loads_per_rank.mean()
        max_load = loads_per_rank.max()
    if mean_load == 0:
        return 1.0
    return (max_load / mean_load).item()


def step(
    self,
    is_dummy: bool = False,
    is_profile: bool = False,
    log_stats: bool = False,
) -> None:
    """
    Step the EPLB state.

    Args:
        is_dummy (bool): If `True`, this is a dummy step and the load
            metrics recorded in this forward pass will not count.
            Defaults to `False`.
        is_profile (bool): If `True`, perform a dummy rearrangement
            with maximum communication cost. This is used in
            `profile_run` to reserve enough memory
            for the communication buffer.
        log_stats (bool): If `True`, log the expert load metrics.

    # Stats
        The metrics are all summed up across layers.
        - `avg_tokens`: The average load across ranks.
        - `max_tokens`: The maximum load across ranks.
        - `balancedness`: The ratio of average load to maximum load.
    """
    ep_group = get_ep_group().device_group
    if is_profile:
        self.rearrange(is_profile=True)
        return

    if is_dummy:
        # Do not record load metrics for dummy steps
        for eplb_model_state in self.model_states.values():
            eplb_model_state.expert_load_pass.zero_()

    if (
        log_stats
        and self.expert_rearrangement_step
        % self.parallel_config.eplb_config.log_balancedness_interval
        == 0
    ):
        # Sync the expert load pass for each model (main and drafter).
        # expert_load_pass: (num_moe_layers, num_physical_experts)
        expert_load_pass_list = self._sync_load_pass()
        ep_group = get_ep_group().device_group
        for expert_load_pass, eplb_model_state in zip(
            expert_load_pass_list, self.model_states.values()
        ):
            # num_tokens_per_rank: (num_moe_layers, num_ranks)
            num_tokens_per_rank = (
                expert_load_pass.reshape(expert_load_pass.shape[0], ep_group.size(), -1)
                .sum(dim=-1)
                .float()
            )

            # Compute balancedness ratio:
            # for each layer:
            #   (mean load across ranks) / (max load across ranks)
            avg_tokens_tensor, max_tokens_tensor = _compute_eplb_load_stats(
                num_tokens_per_rank
            )

            # This gpu/cpu sync only happens with expert stat logging enabled.
            with gpu_sync_allowed():
                tokens_tensors: list[float] = torch.stack(
                    [avg_tokens_tensor, max_tokens_tensor]
                ).tolist()
            avg_tokens, max_tokens = tokens_tensors
            balancedness = avg_tokens / max_tokens if max_tokens > 0 else 0.0

            if ep_group.rank() == 0:
                logger.info(
                    "EPLB step: %d for model %s: avg_tokens=%.2f, "
                    "max_tokens=%d, balancedness=%.4f, "
                    "steps until the next rearrangement: %d",
                    self.expert_rearrangement_step,
                    eplb_model_state.model_name,
                    avg_tokens,
                    max_tokens,
                    balancedness,
                    self.expert_rearrangement_step_interval
                    - self.expert_rearrangement_step,
                )

    # Update the expert load sliding window
    if not is_dummy:
        should_record = self._should_record_current_step(log_stats=log_stats)
        for eplb_model_state in self.model_states.values():
            if should_record:
                load_pass = eplb_model_state.expert_load_pass
                if _EPLB_TOUCHED_LOAD:
                    load_pass = load_pass > 0
                eplb_model_state.expert_load_window[self.expert_load_window_step].copy_(
                    load_pass
                )
                eplb_model_state.expert_load_pass.zero_()

        if should_record:
            self.expert_load_window_step += 1
            if self.expert_load_window_step >= self.expert_load_window_size:
                self.expert_load_window_step = 0

    # Step the expert rearrangement step
    # Note that even if this is a dummy step, we still increment the
    # rearrangement step and perform rearrangement to ensure all ranks are
    # performing collective communication.
    self.expert_rearrangement_step += 1

    if self.is_async:
        # Run _move_to_workspace if all ranks have finished transferring the
        # new weights to the intermediate buffer
        for eplb_model_state in self.model_states.values():
            # rebalanced must remain consistent amongst all ranks otherwise the
            # all_reduce in _all_ranks_result_ready will hang
            if eplb_model_state.rebalanced and self._all_ranks_result_ready(
                eplb_model_state
            ):
                _move_to_workspace(
                    model_state=eplb_model_state,
                    ep_rank=ep_group.rank(),
                )

    if self.expert_rearrangement_step >= self.expert_rearrangement_step_interval:
        if self.is_async and any(
            eplb_model_state.rebalanced
            for eplb_model_state in self.model_states.values()
        ):
            # Still performing asynchronous rearrangement; update
            # should_record (step > step_interval, so always True) and
            # bail out before the step counter is reset.
            self._update_layer_should_record(log_stats=log_stats)
            return
        self.expert_rearrangement_step = 0
        self.rearrange()

    self._update_layer_should_record(log_stats=log_stats)


def _propagate_shared_tensors(
    self,
    model: "MixtureOfExperts",  # type: ignore[name-defined]
    num_unpadded_tokens_tensors: list[torch.Tensor],
) -> None:
    """Propagate shared tensors to every :class:`EplbLayerState`.

    Allocates ``should_record_tensor`` on the first call and then
    assigns both it and ``num_unpadded_tokens_tensors`` to every
    MoE layer's :class:`EplbLayerState`.  All layers reference the
    **same** objects so a single update is visible everywhere.

    Must be called after :meth:`model.set_eplb_state` so that each
    layer's ``eplb_state`` is already populated.
    """
    # Some models (e.g. Step5) list the MoE block, which wraps the runner
    # holding ``eplb_state`` in ``.experts``.
    layers = [
        layer if hasattr(layer, "eplb_state") else getattr(layer, "experts", layer)
        for layer in model.moe_layers
    ]
    layer_states = [
        layer.eplb_state
        for layer in layers
        if hasattr(layer, "eplb_state") and isinstance(layer.eplb_state, EplbLayerState)
    ]

    if self.should_record_tensor is None and layer_states:
        self.should_record_tensor = torch.ones((), dtype=torch.bool, device=self.device)

    for ls in layer_states:
        if ls is not None:
            ls.should_record_tensor = self.should_record_tensor
            ls.num_unpadded_tokens_tensors = num_unpadded_tokens_tensors


def rearrange(
    self,
    is_profile: bool = False,
    rank_mapping: dict[int, int] | None = None,
) -> torch.Tensor | None:
    """
    Rearrange the experts according to the current load.

    Args:
        is_profile (bool): If `True`, perform a dummy rearrangement.
            This is used in `profile_run` to reserve enough memory,
            no memory movement will be performed. Default is False.
        rank_mapping (dict[int, int] | None): The rank mapping
            when scaling is done in EEP.
    """

    ep_group = get_ep_group().device_group
    ep_rank = ep_group.rank()

    start_event = None
    end_event = None
    is_main_rank = ep_rank == 0
    if is_main_rank:
        if not self.is_async or is_profile:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        logger.info(
            "Rearranging experts %s %s...",
            "(async mode)" if self.is_async else "sync mode",
            "(profile)" if is_profile else "",
        )

    # Map the physical expert load to global logical experts
    global_expert_load_windows = []
    for eplb_model_state in self.model_states.values():
        expert_load_window = eplb_model_state.expert_load_window
        physical_to_logical = eplb_model_state.physical_to_logical_map
        invalid_idx = eplb_model_state.model.num_logical_experts
        logical_expert_load_window = torch.zeros(
            self.expert_load_window_size,
            eplb_model_state.model.num_moe_layers,
            invalid_idx + 1,
            dtype=eplb_model_state.expert_load_window.dtype,
            device=eplb_model_state.expert_load_window.device,
        )
        logical_expert_load_window.scatter_add_(
            dim=-1,
            index=physical_to_logical.masked_fill(physical_to_logical < 0, invalid_idx)
            .unsqueeze(0)
            .expand_as(expert_load_window)
            .long(),
            src=expert_load_window,
        )

        global_expert_load_window = logical_expert_load_window[..., :-1].sum(dim=0)
        global_expert_load_windows.append(global_expert_load_window)
    # Perform all-reduce to get the expert load across all ranks for each model
    global_expert_load_windows = self._allreduce_list(global_expert_load_windows)

    # TODO(bowen): Treat differently for prefill and decode nodes
    eplb_model_state = next(iter(self.model_states.values()))
    model = eplb_model_state.model
    num_replicas = model.num_physical_experts
    num_groups = model.num_expert_groups

    if rank_mapping is not None and len(rank_mapping) == ep_group.size():
        # NOTE(yongji): scale down, we need to rebalance the experts on
        # remaining GPUs, transfer the experts while we haven't shutdown
        # the GPUs to be released.
        coordinator = get_ep_group()
        assert isinstance(coordinator, StatelessGroupCoordinator)
        tcp_store_group = coordinator.tcp_store_group
        num_nodes = _node_count_with_rank_mapping(tcp_store_group, rank_mapping)
        num_gpus = sum(new_rank != -1 for new_rank in rank_mapping.values())
        num_replicas = (
            num_replicas // ep_group.size() * num_gpus
        )  # handle num replicas change
    else:
        num_nodes = get_node_count()
        num_gpus = ep_group.size()

    if num_gpus % num_nodes != 0:
        logger.warning_once(
            "num_gpus % num_nodes != 0, "
            "not using hierarchical rearrangement algorithm.\n"
            f"{num_gpus=}, {num_nodes=}"
        )
        num_nodes = 1

    # Get new expert mappings
    for eplb_model_state, global_expert_load_window in zip(
        self.model_states.values(), global_expert_load_windows
    ):
        if not self.is_async or is_profile:
            # Get new expert mappings for the model. The policy runs on the
            # host, so the load window and current map have to come back.
            with gpu_sync_allowed():
                new_physical_to_logical_map = self.policy.rebalance_experts(
                    global_expert_load_window.cpu(),
                    num_replicas,
                    num_groups,
                    num_nodes,
                    num_gpus,
                    eplb_model_state.physical_to_logical_map.cpu(),
                )

            skip_rearrange = False
            if (
                current_platform.is_rocm()
                and not is_profile
                and rank_mapping is None
                and bool((eplb_model_state.physical_to_logical_map >= 0).all())
            ):
                logical_loads = global_expert_load_window.float()
                ep_size = ep_group.size()

                layerwise = is_step5_config(
                    getattr(eplb_model_state.model, "config", None)
                )
                current_imbalance = _rank_load_imbalance(
                    eplb_model_state.physical_to_logical_map,
                    logical_loads,
                    ep_size,
                    layerwise=layerwise,
                )
                proposed_imbalance = _rank_load_imbalance(
                    new_physical_to_logical_map,
                    logical_loads,
                    ep_size,
                    layerwise=layerwise,
                )
                relative_improvement = (
                    current_imbalance - proposed_imbalance
                ) / current_imbalance
                skip_rearrange = relative_improvement < 0.05
                if skip_rearrange and is_main_rank:
                    logger.info(
                        "[EPLB] Skip rearrange: imbalance %.4f -> "
                        "%.4f (no material gain)",
                        current_imbalance,
                        proposed_imbalance,
                    )
                elif is_main_rank:
                    logger.info(
                        "[EPLB] Rearrange: imbalance %.4f -> %.4f",
                        current_imbalance,
                        proposed_imbalance,
                    )

            if _EPLB_DUMP_PATH and is_main_rank and not is_profile:
                # Saves the placement before the transfer so it can be
                # inspected or applied offline (e.g. at weight load).
                torch.save(
                    {
                        "physical_to_logical_map": new_physical_to_logical_map.cpu(),
                        "load_window": global_expert_load_window.cpu(),
                    },
                    _EPLB_DUMP_PATH,
                )
                logger.info("[EPLB] Placement saved to %s", _EPLB_DUMP_PATH)

            if not skip_rearrange:
                # Update expert weights
                rearrange_expert_weights_inplace(
                    eplb_model_state.physical_to_logical_map,
                    new_physical_to_logical_map,
                    eplb_model_state.model.expert_weights,
                    eplb_model_state.expert_buffer,
                    ep_group,
                    eplb_model_state.communicator,
                    is_profile,
                    rank_mapping,
                )

                if not is_profile:
                    _commit_eplb_maps(
                        eplb_model_state,
                        new_physical_to_logical_map=new_physical_to_logical_map,
                    )

            if is_main_rank:
                assert start_event is not None
                assert end_event is not None
                end_event.record()
                end_event.synchronize()
                gpu_elapsed = start_event.elapsed_time(end_event) / 1000.0
                logger.info(
                    "Rearranged experts %s in %.2f s.",
                    " (profile) " if is_profile else " ",
                    gpu_elapsed,
                )
        else:
            eplb_model_state.eplb_stats = EplbStats(
                # We copy the tensor to snapshot the global_expert_load_window
                # on the main thread so that async worker can access it safely
                # while the main thread is running.
                global_expert_load_window=global_expert_load_window.clone(),
                num_replicas=num_replicas,
                num_groups=num_groups,
                num_nodes=num_nodes,
                num_gpus=num_gpus,
            )
            eplb_model_state.rebalanced = True
    # Signal async thread to start transferring layers
    if self.is_async and (not is_profile):
        self.rearrange_event.record()
    return None
