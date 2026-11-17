# SPDX-License-Identifier: Apache-2.0
"""Validate and rebind MiniMax EP's deterministic map after IPC restoration.

Core ExpertMapManager already builds its map on CPU when constructed under
meta; this module does not change its constructor. The registered map may
be replaced by IPC while the manager retains its original Python reference.
"""
from .minimax31_preload import validate_parallel_config


def validate_ep_map(layer, ep_rank):
    import torch
    if type(ep_rank) is not int or not 0 <= ep_rank < 8:
        raise ValueError("invalid EP rank")
    manager = layer.expert_map_manager
    static=getattr(layer,"_static_placement",None)
    experts,local=(136,17) if static is not None else (128,16)
    if (manager.ep_size != 8 or manager.ep_rank != ep_rank
            or manager.global_num_experts != experts or manager.local_num_experts != local
            or manager.placement_strategy != "linear"):
        raise ValueError("unexpected MiniMax EP expert-manager placement")
    if "_expert_map" not in layer._buffers:
        raise ValueError("missing registered expert map")
    restored = layer._buffers["_expert_map"]
    if (not isinstance(restored, torch.Tensor) or restored.is_meta
            or restored.dtype != torch.int32 or tuple(restored.shape) != (experts,)):
        raise ValueError("invalid restored EP expert-map buffer")
    if layer._buffers.get("expert_mask") is not None or manager.expert_mask is not None:
        raise ValueError("AITER expert masks are outside this contract")
    if manager.routing_tables is not None:
        raise ValueError("EP routing-table placement is outside linear contract")
    expected = torch.full((experts,), -1, dtype=torch.int32)
    expected[local * ep_rank:local * (ep_rank + 1)] = torch.arange(local, dtype=torch.int32)
    if not torch.equal(restored.detach().to(device="cpu"), expected):
        raise ValueError("restored EP expert map does not match rank ownership")
    if static is not None:
        from .minimax31_static_contract import layer_id
        table=getattr(layer,"_static_physical_to_logical",None)
        if (not isinstance(table,torch.Tensor) or table.is_meta or table.dtype!=torch.int32
                or table.cpu().tolist()!=list(static.row(layer_id(layer.layer_name)))):
            raise ValueError("static physical-to-logical map differs from configured ownership")
    prior = manager.expert_map
    if (not isinstance(prior, torch.Tensor) or prior.is_meta
            or prior.dtype != torch.int32 or tuple(prior.shape) != (experts,)
            or not torch.equal(prior.detach().to(device="cpu"), expected)):
        raise ValueError("expert manager map does not match actual EP rank")
    return manager, restored


def rebind_ep_layers(layers, ep_rank):
    # Validate every layer before changing any alias.
    items = [validate_ep_map(module, ep_rank) for module in layers]
    if not items:
        raise ValueError("MiniMax model has no routed expert layers")
    changed = 0
    for manager, restored in items:
        if manager.expert_map is not restored:
            manager._expert_map = restored
            changed += 1
    return changed


def restore_ep_state(model, ep_rank):
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    return rebind_ep_layers(
        (module for module in model.modules() if isinstance(module, RoutedExperts)),
        ep_rank,
    )


def after_minimax31_ep_load(context):
    from vllm.distributed.parallel_state import get_ep_group, get_pcp_group
    config = context.vllm_config.parallel_config
    validate_parallel_config(config)
    ep, pcp = get_ep_group(), get_pcp_group()
    if ep.world_size != 8 or pcp.world_size != 8 or ep.rank_in_group != pcp.rank_in_group:
        raise ValueError("live PCP and EP group/rank identity mismatch")
    return restore_ep_state(context.worker.model_runner.model, ep.rank_in_group)
