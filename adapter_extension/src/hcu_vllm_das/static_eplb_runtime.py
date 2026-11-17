# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Static expert placement for owned V41 MoE layers, without Core EPLB state.

Core's ``EplbState`` owns load recording, policy proposals and periodic weight
transfer. Static EPLB needs none of that: the placement is fixed before any
checkpoint weight is read, so this module instead

* asks the owned MoE construction for ``enable_eplb``/``num_redundant_experts``,
  which is what makes Core build per-layer ``EplbLayerState`` and size the
  physical expert tables (``--enable-eplb`` stays off, so no ``EplbState``
  exists and no rearrangement can run);
* fans each logical checkpoint expert out to every physical slot the plan maps
  it to, before quantization reads the loaded parameters;
* publishes the routing tensors Core's router reads, pinning one primary
  replica per logical expert and disabling load recording.

Scope is fail-closed: the owned PCP target model with expert parallel and a
proposal-complete version-2 map. Anything else raises instead of silently
routing to the initial placement.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MethodType

import torch

from .static_eplb import StaticEplbPlan, load_static_eplb_plan
from .static_eplb_dispatch import REPLICA_ORDER_BUILDERS

_CONFIG_KEY = 'hcu_static_eplb'
_PLAN_ATTR = '_hcu_static_eplb_plan'
_ROW_ATTR = '_hcu_static_eplb_row'
_SESSION_ATTR = '_hcu_static_eplb_record_session'
_INSTALLED_ATTR = '_hcu_static_eplb_routing_installed'
_LOADERS_BOUND_ATTR = '_hcu_static_eplb_loaders_bound'
_FIELDS = ('expert_map_path', 'expert_map_record_path', 'num_redundant_experts',
           'static_dispatch_policy', 'record_step_interval')


class StaticEplbError(ValueError):
    """Raised before an unsupported static EPLB placement can be published."""


@dataclass(frozen=True, slots=True)
class StaticEplbOptions:
    expert_map_path: str | None
    expert_map_record_path: str | None
    num_redundant_experts: int
    static_dispatch_policy: str
    record_step_interval: int

    @property
    def recording(self) -> bool:
        return self.expert_map_record_path is not None


def _optional_path(payload, name):
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise StaticEplbError(f'{_CONFIG_KEY}.{name} must be a nonempty string')
    return value


def read_options(cfg):
    """Parse the owned ``additional_config`` sidecar, or return None."""
    additional = getattr(cfg, 'additional_config', None)
    if not isinstance(additional, Mapping):
        return None
    payload = additional.get(_CONFIG_KEY)
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise StaticEplbError(f'{_CONFIG_KEY} must be a JSON object')
    unknown = sorted(set(payload) - set(_FIELDS))
    if unknown:
        raise StaticEplbError(f'unknown {_CONFIG_KEY} fields: {", ".join(unknown)}')
    path = _optional_path(payload, 'expert_map_path')
    record_path = _optional_path(payload, 'expert_map_record_path')
    if (path is None) == (record_path is None):
        raise StaticEplbError(
            f'{_CONFIG_KEY} requires exactly one of expert_map_path '
            '(deploy a placement) or expert_map_record_path (calibrate one)')
    redundant = payload.get('num_redundant_experts', 0)
    if type(redundant) is not int or redundant < 0:
        raise StaticEplbError(
            f'{_CONFIG_KEY}.num_redundant_experts must be an integer >= 0')
    policy = payload.get('static_dispatch_policy', 'nearest')
    if policy not in REPLICA_ORDER_BUILDERS:
        supported = ', '.join(sorted(REPLICA_ORDER_BUILDERS))
        raise StaticEplbError(
            f'{_CONFIG_KEY}.static_dispatch_policy must be one of {supported}')
    interval = payload.get('record_step_interval', 200)
    if type(interval) is not int or interval < 1:
        raise StaticEplbError(
            f'{_CONFIG_KEY}.record_step_interval must be an integer >= 1')
    return StaticEplbOptions(path, record_path, redundant, policy, interval)


def enabled(cfg) -> bool:
    return read_options(cfg) is not None


def validate_config(cfg) -> bool:
    """Validate the static EPLB scope before allocating expert parameters."""
    options = read_options(cfg)
    if options is None:
        return False
    from .v41_pcp import validate_config as validate_pcp
    errors = []
    p = cfg.parallel_config
    if not p.enable_expert_parallel:
        errors.append('--enable-expert-parallel required by static expert placement')
    if p.enable_eplb:
        errors.append('static EPLB replaces --enable-eplb; both cannot be active')
    if p.enable_elastic_ep:
        errors.append('elastic EP would resize the fixed physical expert layout')
    if p.all2all_backend != 'deepep_high_throughput':
        errors.append('static EPLB is validated only with deepep_high_throughput')
    if cfg.kernel_config.moe_backend != 'deep_gemm':
        errors.append('static EPLB requires the owned DeepGEMM expert path')
    if options.recording and p.enable_dbo:
        errors.append('calibration cannot attribute load across DBO ubatches')
    # EP width is the flattened TP/PCP width; reject an unusable redundancy here
    # rather than at the first proposal, which would kill one rank mid-run.
    ep_size = p.prefill_context_parallel_size * p.tensor_parallel_size
    physical = cfg.model_config.hf_config.n_routed_experts + options.num_redundant_experts
    if physical % ep_size:
        errors.append(
            f'num_redundant_experts={options.num_redundant_experts} gives '
            f'{physical} physical experts, which EP width {ep_size} cannot '
            'divide evenly')
    if errors:
        raise StaticEplbError('HCU static EPLB: ' + '; '.join(errors))
    # The owned PCP target is the only reviewed placement owner; its own
    # validation raises for every unsupported topology.
    if not validate_pcp(cfg):
        raise StaticEplbError(
            'HCU static EPLB requires the owned V41 PCP target model')
    return True


def _backbone_layer(cfg, prefix: str) -> bool:
    """True for target backbone MoE layers; DSpark draft layers keep native EP."""
    from vllm.model_executor.models.utils import extract_layer_index
    return extract_layer_index(prefix) < cfg.model_config.hf_config.num_hidden_layers


def _deployed_redundancy(options) -> int:
    """Redundant slots this process allocates.

    A calibration run keeps the identity placement so routing matches a run
    without static EPLB; its ``num_redundant_experts`` describes the placement
    to propose, not the one to allocate now.
    """
    return 0 if options.recording else options.num_redundant_experts


def moe_eplb_kwargs(cfg, prefix: str) -> dict:
    """Return the Core MoE construction keywords that size physical experts."""
    options = read_options(cfg)
    if options is None:
        return {}
    if not _backbone_layer(cfg, prefix):
        return {}
    validate_config(cfg)
    return {
        'enable_eplb': True,
        'num_redundant_experts': _deployed_redundancy(options),
    }


def publish_physical_metadata(moe, cfg, prefix: str) -> None:
    """Replace the TP-derived MixtureOfExperts counts with the EP layout."""
    options = read_options(cfg)
    if options is None or not _backbone_layer(cfg, prefix):
        return
    redundant = _deployed_redundancy(options)
    routed = moe.experts.routed_experts
    num_physical = moe.n_logical_experts + redundant
    num_local = routed.local_num_experts
    ep_size = routed.moe_config.moe_parallel_config.ep_size
    if (routed.moe_config.num_experts != num_physical
            or num_local * ep_size != num_physical):
        raise StaticEplbError(
            f'static EPLB layout mismatch for {moe.prefix}: '
            f'logical={moe.n_logical_experts} redundant={redundant} '
            f'routed_global={routed.moe_config.num_experts} local={num_local} '
            f'ep_size={ep_size}')
    if moe.experts.eplb_state is None:
        raise StaticEplbError(
            f'static EPLB router state was not created for {moe.prefix}')
    moe.n_redundant_experts = redundant
    moe.n_physical_experts = num_physical
    moe.n_local_physical_experts = num_local


def _static_loader(original):
    """Load one logical checkpoint expert into every physical slot it maps to."""

    def load(self, param, loaded_weight, weight_name, shard_id, expert_id,
             return_success=False):
        row = getattr(self, _ROW_ATTR)
        logical_experts = self.moe_config.num_logical_experts
        fused_shared = self.expert_map_manager.num_fused_shared_experts
        if (type(expert_id) is int
                and logical_experts <= expert_id < logical_experts + fused_shared):
            # Fused shared experts occupy the physical tail and are not replicated.
            physical_tail_id = len(row) + expert_id - logical_experts
            return original(self, param, loaded_weight, weight_name, shard_id,
                            physical_tail_id, return_success=return_success)
        if type(expert_id) is not int or expert_id not in row:
            raise StaticEplbError(f'static EPLB invalid logical expert ID {expert_id!r}')
        if isinstance(loaded_weight, torch.Tensor) and loaded_weight.ndim >= 3:
            raise StaticEplbError(
                'static EPLB fused expert tensors must be split by the current loader')
        if (getattr(getattr(self, 'quant_method', None), 'use_global_sf', False)
                and 'input_scale' in weight_name):
            return original(self, param, loaded_weight, weight_name, shard_id,
                            expert_id, return_success=return_success)
        loaded_any = False
        for physical_expert_id, mapped_logical_id in enumerate(row):
            if mapped_logical_id != expert_id:
                continue
            loaded = original(self, param, loaded_weight, weight_name, shard_id,
                              physical_expert_id, return_success=True)
            loaded_any = bool(loaded) or loaded_any
        return loaded_any if return_success else None

    return load


def resolve_model_key(model) -> str:
    """Return the version-2 map key: the native model class this owner extends.

    Offline maps are recorded against Core's model class name, so an owned
    subclass must not change the key. Reusing a map across a plugin rename is
    exactly the behaviour the recorded expert counts and layer shape verify.
    """
    for klass in type(model).__mro__:
        if klass.__module__.startswith('vllm.models.'):
            return klass.__name__
    raise StaticEplbError(
        f'static EPLB cannot resolve a native map key for {type(model).__name__}')


def _real_device(moe_model) -> torch.device:
    """Device of a tensor the checkpoint loader is guaranteed to have placed.

    ``next(model.parameters()).device`` can return a meta-device placeholder
    first — e.g. a vision tower left unmaterialized under
    ``--language-model-only`` — which is a normal-looking Python tensor with
    no real backing storage. A kernel that reads or writes through it
    null-dereferences on the device instead of raising a catchable Python
    exception. Routed expert weights are always materialized by this point.
    """
    return moe_model.moe_mlp_layers[0].experts.routed_experts.get_expert_weights()[0].device


def _expert_storage_key(weight: torch.Tensor) -> tuple[object, ...]:
    return (weight.untyped_storage(), weight.storage_offset(), weight.numel(),
            weight.dtype)


def bind_placement(model, cfg, *, bind_weight_loaders=True):
    """Bind layout rows, and optionally the checkpoint weight fan-out loaders.

    Dummy loading initializes physical tensors directly. Its post-load hook
    needs the same validated rows without inspecting already-packed loaders.
    """
    options = read_options(cfg)
    if options is None or options.recording:
        # Calibration keeps the identity placement, which the native loaders
        # already produce; nothing may be rebound.
        return None
    validate_config(cfg)
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.models.interfaces import get_mixture_of_experts_model

    # The multimodal wrapper is not itself a MoE model; bind the language model.
    moe_model = get_mixture_of_experts_model(model)
    if moe_model is None:
        raise StaticEplbError('static EPLB requires a current MixtureOfExperts model')
    bound = getattr(moe_model, _PLAN_ATTR, None)
    if bound is not None and (
        not bind_weight_loaders or getattr(moe_model, _LOADERS_BOUND_ATTR, False)
    ):
        return bound

    layers = tuple(moe_model.moe_mlp_layers)
    if not layers or len(layers) != moe_model.num_moe_layers:
        raise StaticEplbError('static EPLB local MoE layer count mismatch')
    if moe_model.num_redundant_experts != options.num_redundant_experts:
        raise StaticEplbError(
            'static EPLB redundant expert count was not published by MoE construction')
    plan = bound if bound is not None else load_static_eplb_plan(
        options.expert_map_path,
        model_key=resolve_model_key(moe_model),
        expected_model_name=cfg.model_config.model,
        expected_shape=(len(layers), moe_model.num_physical_experts),
        num_logical_experts=moe_model.num_logical_experts,
        num_redundant_experts=moe_model.num_redundant_experts,
    )

    targets = []
    for layer in layers:
        experts = layer.experts.routed_experts
        if not isinstance(experts, RoutedExperts):
            raise StaticEplbError('static EPLB requires current RoutedExperts owners')
        if (experts.moe_config.num_logical_experts != plan.num_logical_experts
                or experts.moe_config.num_experts != plan.num_physical_experts):
            raise StaticEplbError('static EPLB RoutedExperts count mismatch')
        parameters = []
        if not bind_weight_loaders:
            targets.append((experts, parameters))
            continue
        expert_storage = {_expert_storage_key(weight)
                          for weight in experts.get_expert_weights()}
        local_experts = experts.local_num_experts
        for name, parameter in experts.named_parameters():
            if _expert_storage_key(parameter) not in expert_storage:
                # ``get_expert_weights`` excludes activation scales by name,
                # assuming they are global. A quantization method that makes
                # them genuinely per-expert would then keep a native loader and
                # write logical ids into physical slots; refuse instead.
                if parameter.dim() and parameter.shape[0] == local_experts:
                    raise StaticEplbError(
                        f'static EPLB cannot place expert-indexed parameter '
                        f'{name!r}: it is absent from get_expert_weights(), so '
                        'its loader would not follow the placement')
                continue
            loader = getattr(parameter, 'weight_loader', None)
            if getattr(loader, '__self__', None) is not experts:
                raise StaticEplbError(
                    f'static EPLB parameter {name!r} has an unsupported loader owner')
            parameters.append((parameter, loader.__func__))
        targets.append((experts, parameters))

    _verify_plan_across_ep_ranks(plan)
    for index, (experts, parameters) in enumerate(targets):
        setattr(experts, _ROW_ATTR, plan.layer_map(index))
        for parameter, original in parameters:
            parameter.weight_loader = MethodType(_static_loader(original), experts)
    setattr(moe_model, _PLAN_ATTR, plan)
    if bind_weight_loaders:
        setattr(moe_model, _LOADERS_BOUND_ATTR, True)
    return plan


def _verify_plan_across_ep_ranks(plan: StaticEplbPlan) -> None:
    """Fail unless every EP rank loaded the exact same plan fingerprint."""
    if not torch.distributed.is_initialized():
        return
    from vllm.distributed import get_ep_group
    group = get_ep_group()
    fingerprint = plan.fingerprint()
    fingerprints = [None] * group.world_size
    torch.distributed.all_gather_object(fingerprints, fingerprint,
                                        group=group.cpu_group)
    if any(value != fingerprint for value in fingerprints):
        raise StaticEplbError('static EPLB plan fingerprint mismatch across EP ranks')


def _publish(moe_model, cfg, *, logical_to_physical, replica_count,
             num_physical, record: bool):
    """Hand Core the routing tensors and the recording switch."""
    device = _real_device(moe_model)
    num_layers = logical_to_physical.shape[0]
    expert_load = torch.zeros((num_layers, num_physical), dtype=torch.int32,
                              device=device)
    moe_model.set_eplb_state(expert_load, logical_to_physical, replica_count)
    should_record = torch.full((), record, dtype=torch.bool, device=device)
    num_ubatches = max(1, cfg.parallel_config.num_ubatches)
    unpadded = [torch.tensor(0, dtype=torch.int32, device=device)
                for _ in range(num_ubatches)]
    for layer in moe_model.moe_layers:
        state = layer.eplb_state
        if state is None:
            raise StaticEplbError('static EPLB layer is missing its router state')
        state.should_record_tensor = should_record
        state.num_unpadded_tokens_tensors = unpadded
        if state.logical_replica_count is None or state.logical_to_physical_map is None:
            raise StaticEplbError('static EPLB routing tensors were not published')
    return expert_load


def _install_recording(moe_model, cfg, options):
    """Keep the identity placement and switch on Core's load counters."""
    from vllm.logger import init_logger
    from .static_eplb_record import RecordSession

    device = _real_device(moe_model)
    num_layers = moe_model.num_moe_layers
    num_logical = moe_model.num_logical_experts
    if moe_model.num_physical_experts != num_logical:
        raise StaticEplbError(
            'static EPLB calibration must run on the identity placement; '
            f'got {moe_model.num_physical_experts} physical experts')
    identity = torch.arange(num_logical, dtype=torch.long, device=device)
    logical_to_physical = identity.reshape(1, num_logical, 1).expand(
        num_layers, -1, -1).contiguous()
    replica_count = torch.ones((num_layers, num_logical), dtype=torch.long,
                               device=device)
    load = _publish(moe_model, cfg,
                    logical_to_physical=logical_to_physical,
                    replica_count=replica_count, num_physical=num_logical,
                    record=True)
    session = RecordSession(
        model=moe_model, cfg=cfg, options=options, load=load,
        unpadded=moe_model.moe_layers[0].eplb_state.num_unpadded_tokens_tensors,
        num_logical=num_logical,
        num_groups=getattr(moe_model, 'num_expert_groups', 1) or 1,
        model_key=resolve_model_key(moe_model),
        logger=init_logger('vllm.hcu.static_eplb'))
    setattr(moe_model, _SESSION_ATTR, session)
    return session


def record_session(model):
    """Return the calibration session bound to a model, if it is recording."""
    from vllm.model_executor.models.interfaces import get_mixture_of_experts_model
    moe_model = get_mixture_of_experts_model(model)
    return getattr(moe_model, _SESSION_ATTR, None) if moe_model is not None else None


def install_routing(model, cfg):
    """Publish the fixed routing tensors after checkpoint loading completes."""
    options = read_options(cfg)
    if options is None:
        return None
    from vllm.distributed import get_ep_group, get_node_count
    from vllm.model_executor.models.interfaces import get_mixture_of_experts_model

    moe_model = get_mixture_of_experts_model(model)
    if moe_model is None:
        raise StaticEplbError('static EPLB requires a current MixtureOfExperts model')
    # The post-load hook can run more than once; re-publishing would hand the
    # routers fresh tensors and discard a calibration run's accumulated load.
    if getattr(moe_model, _INSTALLED_ATTR, False):
        return getattr(moe_model, _SESSION_ATTR, None) or getattr(moe_model, _PLAN_ATTR, None)
    if options.recording:
        session = _install_recording(moe_model, cfg, options)
        setattr(moe_model, _INSTALLED_ATTR, True)
        return session

    plan = getattr(moe_model, _PLAN_ATTR, None)
    if not isinstance(plan, StaticEplbPlan):
        raise StaticEplbError('static EPLB routing requires a bound placement plan')
    for index, layer in enumerate(moe_model.moe_mlp_layers):
        row = getattr(layer.experts.routed_experts, _ROW_ATTR, None)
        if row != plan.layer_map(index):
            raise StaticEplbError('static EPLB consumer row changed after binding')

    device = _real_device(moe_model)
    num_layers = len(plan.physical_to_logical_map)
    num_logical = plan.num_logical_experts
    num_physical = plan.num_physical_experts
    physical_to_logical = plan.physical_to_logical_map.to(device=device)

    # Invert the placement without per-slot host syncs: ordering physical slots
    # by (logical id, slot id) makes each logical expert's replicas contiguous,
    # so a slot's replica column is its offset from its group's start.
    counts = torch.zeros((num_layers, num_logical), dtype=torch.long, device=device)
    counts.scatter_add_(1, physical_to_logical,
                        torch.ones_like(physical_to_logical))
    max_replicas = max(1, int(counts.max().item()))
    order = torch.argsort(physical_to_logical, dim=1, stable=True)
    sorted_logical = torch.gather(physical_to_logical, 1, order)
    group_start = torch.cumsum(counts, dim=1) - counts
    column = (torch.arange(num_physical, device=device).expand(num_layers, -1)
              - torch.gather(group_start, 1, sorted_logical))
    logical_to_physical = torch.full((num_layers, num_logical, max_replicas), -1,
                                     dtype=torch.long, device=device)
    logical_to_physical[
        torch.arange(num_layers, device=device).unsqueeze(1).expand(-1, num_physical),
        sorted_logical,
        column,
    ] = order
    # Every logical expert must have exactly the replicas the plan gives it.
    if not torch.equal((logical_to_physical >= 0).sum(dim=-1), counts):
        raise StaticEplbError('static EPLB replica accounting disagrees with the plan')

    group = get_ep_group().device_group
    ordered = REPLICA_ORDER_BUILDERS[options.static_dispatch_policy](
        logical_to_physical,
        ep_rank=group.rank(),
        ep_size=group.size(),
        num_nodes=get_node_count(),
        num_physical_experts=num_physical,
    )
    # Every token takes the primary replica in column 0; Core's mapping kernel
    # reads exactly ``logical_replica_count`` columns per logical expert, and
    # recording stays off because the placement never changes.
    primary_count = torch.ones((num_layers, num_logical), dtype=torch.long,
                               device=device)
    _publish(moe_model, cfg, logical_to_physical=ordered,
             replica_count=primary_count, num_physical=num_physical, record=False)
    setattr(moe_model, _INSTALLED_ATTR, True)
    return plan


__all__ = [
    'StaticEplbError',
    'StaticEplbOptions',
    'bind_placement',
    'enabled',
    'install_routing',
    'moe_eplb_kwargs',
    'publish_physical_metadata',
    'read_options',
    'record_session',
    'resolve_model_key',
    'validate_config',
]
