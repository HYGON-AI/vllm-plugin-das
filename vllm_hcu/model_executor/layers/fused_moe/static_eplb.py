# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Immutable offline plans and adapters for current logical expert loaders."""

from __future__ import annotations

import hashlib
import functools
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MethodType

import torch


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Static EPLB {name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class StaticEplbPlan:
    model_key: str
    source_path: str
    source_sha256: str
    _map_values: tuple[tuple[int, ...], ...]
    num_logical_experts: int
    num_physical_experts: int
    num_redundant_experts: int

    def __post_init__(self):
        if not isinstance(self.model_key, str) or not self.model_key:
            raise ValueError("Static EPLB model_key must be nonempty")
        if not isinstance(self.source_path, str) or not Path(self.source_path).is_absolute():
            raise ValueError("Static EPLB source_path must be absolute")
        if (not isinstance(self.source_sha256, str) or len(self.source_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.source_sha256)):
            raise ValueError("Static EPLB source_sha256 must be a SHA-256 hex digest")
        _integer(self.num_logical_experts, "num_logical_experts", 1)
        _integer(self.num_physical_experts, "num_physical_experts", 1)
        _integer(self.num_redundant_experts, "num_redundant_experts")
        if self.num_logical_experts + self.num_redundant_experts != self.num_physical_experts:
            raise ValueError("Static EPLB physical/logical/redundant counts disagree")
        if type(self._map_values) is not tuple or not self._map_values:
            raise ValueError("Static EPLB rows must be a nonempty immutable tuple")
        required = set(range(self.num_logical_experts))
        for row in self._map_values:
            if type(row) is not tuple or len(row) != self.num_physical_experts:
                raise ValueError("Static EPLB row shape or immutability mismatch")
            if any(type(i) is not int or not 0 <= i < self.num_logical_experts for i in row):
                raise ValueError("Static EPLB invalid logical expert ID")
            if set(row) != required:
                raise ValueError("Static EPLB row misses logical experts")

    @property
    def physical_to_logical_map(self):
        return torch.tensor(self._map_values, dtype=torch.int64, device="cpu")

    def layer_map(self, layer_idx):
        return self._map_values[layer_idx]

    def fingerprint(self):
        return (self.model_key, self.source_sha256,
                (len(self._map_values), self.num_physical_experts),
                self.num_logical_experts, self.num_physical_experts,
                self.num_redundant_experts)


@lru_cache(maxsize=16)
def _parse_plan(raw, canonical_path, model_key, expected_shape,
                num_logical_experts, num_redundant_experts):
    payload = json.loads(raw)
    maps = payload.get("model_maps") if isinstance(payload, dict) else None
    if not isinstance(maps, dict) or model_key not in maps:
        raise ValueError(f"Static EPLB missing model key {model_key!r}")
    selected = maps[model_key]
    if not isinstance(selected, dict):
        raise ValueError("Static EPLB model map must be an object")
    rows = selected.get("physical_to_logical_map")
    if (not isinstance(rows, list) or len(rows) != expected_shape[0]
            or any(not isinstance(row, list) for row in rows)):
        raise ValueError("Static EPLB map shape mismatch")
    expected_metadata = dict(num_moe_layers=expected_shape[0],
        num_physical_experts=expected_shape[1], num_logical_experts=num_logical_experts,
        num_redundant_experts=num_redundant_experts,
        model_class=model_key.split("#pp_rank=", 1)[0])
    for name, expected in expected_metadata.items():
        if name in selected and (type(selected[name]) is not type(expected)
                                  or selected[name] != expected):
            raise ValueError(f"Static EPLB declared {name} mismatch")
    return StaticEplbPlan(model_key, canonical_path, hashlib.sha256(raw).hexdigest(),
        tuple(tuple(row) for row in rows), num_logical_experts, expected_shape[1],
        num_redundant_experts)


def load_static_eplb_plan(path, *, model_key, expected_shape,
                          num_logical_experts, num_redundant_experts):
    if not isinstance(model_key, str) or not model_key:
        raise ValueError("Static EPLB model_key must be nonempty")
    if not isinstance(expected_shape, tuple) or len(expected_shape) != 2:
        raise ValueError("Static EPLB expected_shape must contain two dimensions")
    for dimension in expected_shape:
        _integer(dimension, "shape", 1)
    _integer(num_logical_experts, "num_logical_experts", 1)
    _integer(num_redundant_experts, "num_redundant_experts")
    canonical = Path(path).expanduser().resolve(strict=True)
    # Read bytes on every call: a replacement may preserve size and mtime.
    # Cache only parsing/validation, keyed by the bytes actually fingerprinted.
    return _parse_plan(canonical.read_bytes(), str(canonical), model_key,
                       expected_shape, num_logical_experts, num_redundant_experts)


def load_static_logical_expert(routed_experts, original_weight_loader, *, param,
                               loaded_weight, weight_name, shard_id,
                               logical_expert_id, return_success):
    row = routed_experts._vllm_hcu_static_eplb_row
    if type(logical_expert_id) is not int or logical_expert_id not in row:
        raise ValueError(f"Static EPLB invalid logical expert ID {logical_expert_id!r}")
    if isinstance(loaded_weight, torch.Tensor) and loaded_weight.ndim >= 3:
        raise ValueError("Static EPLB fused tensors must be split by the current expert loader")
    if (getattr(getattr(routed_experts, "quant_method", None), "use_global_sf", False)
            and "input_scale" in weight_name):
        return original_weight_loader(routed_experts, param=param,
            loaded_weight=loaded_weight, weight_name=weight_name, shard_id=shard_id,
            expert_id=logical_expert_id, return_success=return_success)
    loaded_any = False
    for physical_id, logical_id in enumerate(row):
        if logical_id == logical_expert_id:
            loaded = original_weight_loader(routed_experts, param=param,
                loaded_weight=loaded_weight, weight_name=weight_name,
                shard_id=shard_id, expert_id=physical_id, return_success=True)
            loaded_any = bool(loaded) or loaded_any
    return loaded_any if return_success else None


def resolve_offline_eplb_model_key(model, parallel_config):
    key = type(model).__name__
    size = _integer(getattr(parallel_config, "pipeline_parallel_size", 1),
                    "pipeline_parallel_size", 1)
    if size > 1:
        rank = getattr(parallel_config, "pipeline_parallel_rank", None)
        if rank is None:
            from vllm.distributed import get_pp_group
            rank = get_pp_group().rank_in_group
        _integer(rank, "pipeline_parallel_rank")
        if rank >= size:
            raise ValueError("Static EPLB PP rank is outside the pipeline")
        key += f"#pp_rank={rank}"
    return key


def verify_static_plan_across_ep_ranks(plan):
    if not torch.distributed.is_initialized():
        return
    from vllm.distributed import get_ep_group
    group = get_ep_group()
    fingerprints = [None] * group.world_size
    torch.distributed.all_gather_object(fingerprints, plan.fingerprint(), group=group.cpu_group)
    if any(value != plan.fingerprint() for value in fingerprints):
        raise RuntimeError("Static EPLB plan fingerprint mismatch across EP ranks")


def _static_loader(original):
    @functools.wraps(original)
    def load(self, param, loaded_weight, weight_name, shard_id, expert_id,
             return_success=False):
        logical = self.moe_config.num_logical_experts
        shared = self.expert_map_manager.num_fused_shared_experts
        if type(expert_id) is int and logical <= expert_id < logical + shared:
            return original(self, param, loaded_weight, weight_name, shard_id,
                len(self._vllm_hcu_static_eplb_row) + expert_id - logical,
                return_success=return_success)
        return load_static_logical_expert(self, original, param=param,
            loaded_weight=loaded_weight, weight_name=weight_name, shard_id=shard_id,
            logical_expert_id=expert_id, return_success=return_success)
    return load


def _static_expert_mapping(self, ckpt_gate_proj_name=None, ckpt_down_proj_name=None,
                           ckpt_up_proj_name=None, include_fused=False):
    return self.build_expert_params_mapping(
        ckpt_gate_proj_name or self.ckpt_gate_proj_name,
        ckpt_down_proj_name or self.ckpt_down_proj_name,
        ckpt_up_proj_name or self.ckpt_up_proj_name,
        self.moe_config.num_logical_experts + self.expert_map_manager.num_fused_shared_experts,
        num_redundant_experts=0,
        routed_experts_prefix="", lora_base_layer_prefix=self.lora_base_layer_prefix,
        include_fused=include_fused)


def _logical_model_mapping(original, logical_experts):
    @functools.wraps(original)
    def mapping(*args, **kwargs):
        return [entry for entry in original(*args, **kwargs) if entry[2] < logical_experts]
    return mapping


def bind_static_eplb_plan(vllm_config, model):
    """Attach rows to existing owners before their captured loaders run.

    No quantization method, MoE layout, weight tensor or global class is
    replaced. All targets are checked before publishing any instance state.
    """
    parallel = getattr(vllm_config, "parallel_config", None)
    path = getattr(parallel, "_vllm_hcu_expert_map_path", None)
    if not path:
        return None
    from vllm.model_executor.models.interfaces import is_mixture_of_experts
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    if (not getattr(parallel, "enable_eplb", False)
            or not getattr(parallel, "enable_expert_parallel", False)):
        raise ValueError("Static EPLB requires EPLB and expert parallel")
    if getattr(parallel, "enable_ep_weight_filter", False):
        raise ValueError("Static EPLB cannot use the upstream EP weight filter")
    if not is_mixture_of_experts(model):
        raise ValueError("Static EPLB requires a current MixtureOfExperts model")
    layers = tuple(model.moe_layers)
    if not layers:
        raise ValueError("Static EPLB found no local MoE layers")
    if getattr(parallel, "pipeline_parallel_size", 1) == 1 and len(layers) != model.num_moe_layers:
        raise ValueError("Static EPLB local MoE layer count mismatch")
    plan = load_static_eplb_plan(path,
        model_key=resolve_offline_eplb_model_key(model, parallel),
        expected_shape=(len(layers), model.num_physical_experts),
        num_logical_experts=model.num_logical_experts,
        num_redundant_experts=model.num_redundant_experts)
    bound = getattr(model, "_vllm_hcu_static_eplb_plan", None)
    if bound is not None:
        if bound != plan:
            raise ValueError("Static EPLB plan changed; cannot rebind loaded weights")
        for index, layer in enumerate(layers):
            if getattr(layer.routed_experts, "_vllm_hcu_static_eplb_row", None) != plan.layer_map(index):
                raise ValueError("Static EPLB consumer row changed after binding")
        return bound
    targets = []
    for layer in layers:
        experts = getattr(layer, "routed_experts", None)
        if not isinstance(experts, RoutedExperts):
            raise ValueError("Static EPLB requires current RoutedExperts owners")
        if not experts.quant_method.supports_eplb:
            raise ValueError(f"EPLB unsupported by {type(experts.quant_method).__name__}")
        if (experts.moe_config.num_logical_experts != plan.num_logical_experts
                or experts.moe_config.num_experts != plan.num_physical_experts):
            raise ValueError("Static EPLB RoutedExperts count mismatch")
        parameters = []
        for param in experts.parameters():
            loader = getattr(param, "weight_loader", None)
            if getattr(loader, "__self__", None) is not experts:
                raise ValueError("Static EPLB parameter has an unsupported loader owner")
            parameters.append((param, loader.__func__))
        targets.append((experts, parameters))
    verify_static_plan_across_ep_ranks(plan)
    owners = [model]
    for name in ("model", "language_model"):
        inner = getattr(model, name, None)
        if inner is not None and inner is not model and getattr(inner, "moe_layers", None) is model.moe_layers:
            owners.append(inner)
    for index, (experts, parameters) in enumerate(targets):
        experts._vllm_hcu_static_eplb_row = plan.layer_map(index)
        for param, original in parameters:
            param.weight_loader = MethodType(_static_loader(original), experts)
        experts.get_expert_mapping = MethodType(_static_expert_mapping, experts)
    for owner in owners:
        owner._vllm_hcu_static_eplb_plan = plan
        mapping = getattr(owner, "get_expert_mapping", None)
        if callable(mapping):
            shared = targets[0][0].expert_map_manager.num_fused_shared_experts
            owner.get_expert_mapping = _logical_model_mapping(mapping, plan.num_logical_experts + shared)
    return plan
