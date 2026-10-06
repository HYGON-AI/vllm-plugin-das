# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Immutable placement plans for offline expert load balancing."""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MethodType
from typing import Any

import torch


_OFFLINE_EPLB_FORMAT = "vllm_offline_eplb_physical_to_logical_by_model"
_RECORD_KINDS = frozenset({"initial", "proposal"})


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"Static EPLB {name} must be an integer >= {minimum}"
        )
    return value


@dataclass(frozen=True, slots=True)
class StaticEplbPlan:
    """A validated, immutable physical-to-logical expert placement."""

    model_key: str
    source_path: str
    source_sha256: str
    record_kind: str
    _map_values: tuple[tuple[int, ...], ...]
    num_logical_experts: int
    num_physical_experts: int
    num_redundant_experts: int

    def __post_init__(self) -> None:
        if not isinstance(self.model_key, str) or not self.model_key:
            raise ValueError("Static EPLB model_key must be nonempty")
        if (
            not isinstance(self.source_path, str)
            or not Path(self.source_path).is_absolute()
        ):
            raise ValueError("Static EPLB source_path must be absolute")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_sha256
            )
        ):
            raise ValueError(
                "Static EPLB source_sha256 must be a SHA-256 hex digest"
            )
        if (
            not isinstance(self.record_kind, str)
            or self.record_kind not in _RECORD_KINDS
        ):
            raise ValueError("Static EPLB record_kind must be initial or proposal")
        _integer(self.num_logical_experts, "num_logical_experts", 1)
        _integer(self.num_physical_experts, "num_physical_experts", 1)
        _integer(self.num_redundant_experts, "num_redundant_experts")
        if (
            self.num_logical_experts + self.num_redundant_experts
            != self.num_physical_experts
        ):
            raise ValueError(
                "Static EPLB physical/logical/redundant counts disagree"
            )
        if type(self._map_values) is not tuple or not self._map_values:
            raise ValueError(
                "Static EPLB rows must be a nonempty immutable tuple"
            )

        required = set(range(self.num_logical_experts))
        for row in self._map_values:
            if type(row) is not tuple or len(row) != self.num_physical_experts:
                raise ValueError(
                    "Static EPLB row shape or immutability mismatch"
                )
            if any(
                type(expert_id) is not int
                or not 0 <= expert_id < self.num_logical_experts
                for expert_id in row
            ):
                raise ValueError("Static EPLB invalid logical expert ID")
            if set(row) != required:
                raise ValueError("Static EPLB row misses logical experts")

    @property
    def physical_to_logical_map(self) -> torch.Tensor:
        """Return an isolated CPU tensor copy of the immutable placement."""

        return torch.tensor(self._map_values, dtype=torch.int64, device="cpu")

    def layer_map(self, layer_idx: int) -> tuple[int, ...]:
        return self._map_values[layer_idx]

    def fingerprint(self) -> tuple[object, ...]:
        return (
            self.model_key,
            self.source_sha256,
            (len(self._map_values), self.num_physical_experts),
            self.num_logical_experts,
            self.num_physical_experts,
            self.num_redundant_experts,
        )


def _required_field(payload: dict[str, Any], name: str) -> Any:
    if name not in payload:
        raise ValueError(f"Static EPLB model map is missing {name}")
    return payload[name]


@lru_cache(maxsize=16)
def _parse_plan(
    raw: bytes,
    canonical_path: str,
    model_key: str,
    expected_shape: tuple[int, int],
    num_logical_experts: int,
    num_redundant_experts: int,
    require_proposal: bool,
) -> StaticEplbPlan:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Static EPLB file must contain valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Static EPLB top-level payload must be an object")
    if type(payload.get("version")) is not int or payload["version"] != 2:
        raise ValueError("Static EPLB file must declare version 2")
    if payload.get("format") != _OFFLINE_EPLB_FORMAT:
        raise ValueError(
            f"Static EPLB file must declare format {_OFFLINE_EPLB_FORMAT!r}"
        )
    model_maps = payload.get("model_maps")
    if not isinstance(model_maps, dict):
        raise ValueError("Static EPLB model_maps must be an object")
    if model_key not in model_maps:
        raise ValueError(f"Static EPLB missing model key {model_key!r}")
    selected = model_maps[model_key]
    if not isinstance(selected, dict):
        raise ValueError("Static EPLB model map must be an object")

    record_kind = _required_field(selected, "record_kind")
    if not isinstance(record_kind, str) or record_kind not in _RECORD_KINDS:
        raise ValueError(
            "Static EPLB record_kind must be initial or proposal"
        )
    if require_proposal and record_kind != "proposal":
        raise ValueError(
            f"Static EPLB model key {model_key!r} has no calibration proposal"
        )
    model_name = _required_field(selected, "model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Static EPLB model_name must be a nonempty string")

    expected_metadata = {
        "model_class": model_key,
        "num_moe_layers": expected_shape[0],
        "num_logical_experts": num_logical_experts,
        "num_physical_experts": expected_shape[1],
        "num_redundant_experts": num_redundant_experts,
    }
    for name, expected in expected_metadata.items():
        value = _required_field(selected, name)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"Static EPLB declared {name} mismatch")

    rows = _required_field(selected, "physical_to_logical_map")
    if (
        not isinstance(rows, list)
        or len(rows) != expected_shape[0]
        or any(not isinstance(row, list) for row in rows)
    ):
        raise ValueError("Static EPLB map shape mismatch")

    return StaticEplbPlan(
        model_key=model_key,
        source_path=canonical_path,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        record_kind=record_kind,
        _map_values=tuple(tuple(row) for row in rows),
        num_logical_experts=num_logical_experts,
        num_physical_experts=expected_shape[1],
        num_redundant_experts=num_redundant_experts,
    )


def load_static_eplb_plan(
    path: str | Path,
    *,
    model_key: str,
    expected_shape: tuple[int, int],
    num_logical_experts: int,
    num_redundant_experts: int,
    require_proposal: bool = True,
) -> StaticEplbPlan:
    """Read and validate a version-2 offline plan.

    File bytes are read on every call. Parsing is cached only by those exact
    bytes, so replacing a file while preserving its size and mtime cannot
    return a stale placement.
    """

    if not isinstance(model_key, str) or not model_key:
        raise ValueError("Static EPLB model_key must be nonempty")
    if type(expected_shape) is not tuple or len(expected_shape) != 2:
        raise ValueError(
            "Static EPLB expected_shape must contain two dimensions"
        )
    for dimension in expected_shape:
        _integer(dimension, "shape", 1)
    _integer(num_logical_experts, "num_logical_experts", 1)
    _integer(num_redundant_experts, "num_redundant_experts")
    if type(require_proposal) is not bool:
        raise ValueError("Static EPLB require_proposal must be bool")
    if num_logical_experts + num_redundant_experts != expected_shape[1]:
        raise ValueError(
            "Static EPLB physical/logical/redundant counts disagree"
        )

    canonical = Path(path).expanduser().resolve(strict=True)
    return _parse_plan(
        canonical.read_bytes(),
        str(canonical),
        model_key,
        expected_shape,
        num_logical_experts,
        num_redundant_experts,
        require_proposal,
    )


def resolve_offline_eplb_model_key(model: object, parallel_config: object) -> str:
    """Resolve the version-2 map key for a current runtime model owner."""

    pipeline_parallel_size = _integer(
        getattr(parallel_config, "pipeline_parallel_size", 1),
        "pipeline_parallel_size",
        1,
    )
    if pipeline_parallel_size != 1:
        raise ValueError("Static EPLB does not support pipeline parallelism")
    model_key = type(model).__name__
    if not model_key:
        raise ValueError("Static EPLB runtime model class name must be nonempty")
    return model_key


def verify_static_plan_across_ep_ranks(plan: StaticEplbPlan) -> None:
    """Fail unless every EP rank loaded the exact same plan fingerprint."""

    if not torch.distributed.is_initialized():
        return
    from vllm.distributed import get_ep_group

    group = get_ep_group()
    fingerprint = plan.fingerprint()
    fingerprints: list[tuple[object, ...] | None] = [None] * group.world_size
    torch.distributed.all_gather_object(
        fingerprints,
        fingerprint,
        group=group.cpu_group,
    )
    if any(value != fingerprint for value in fingerprints):
        raise RuntimeError(
            "Static EPLB plan fingerprint mismatch across EP ranks"
        )


def load_static_logical_expert(
    routed_experts: object,
    original_weight_loader,
    *,
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    logical_expert_id: int,
    return_success: bool,
):
    """Load one logical checkpoint expert into every mapped physical slot."""

    row = routed_experts._vllm_hcu_static_eplb_row
    if type(logical_expert_id) is not int or logical_expert_id not in row:
        raise ValueError(
            f"Static EPLB invalid logical expert ID {logical_expert_id!r}"
        )
    if isinstance(loaded_weight, torch.Tensor) and loaded_weight.ndim >= 3:
        raise ValueError(
            "Static EPLB fused expert tensors must be split by the current loader"
        )
    if (
        getattr(
            getattr(routed_experts, "quant_method", None),
            "use_global_sf",
            False,
        )
        and "input_scale" in weight_name
    ):
        return original_weight_loader(
            routed_experts,
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            expert_id=logical_expert_id,
            return_success=return_success,
        )

    loaded_any = False
    for physical_expert_id, mapped_logical_id in enumerate(row):
        if mapped_logical_id != logical_expert_id:
            continue
        loaded = original_weight_loader(
            routed_experts,
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            expert_id=physical_expert_id,
            return_success=True,
        )
        loaded_any = bool(loaded) or loaded_any
    return loaded_any if return_success else None


def _static_loader(original):
    @functools.wraps(original)
    def load(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        logical_experts = self.moe_config.num_logical_experts
        fused_shared_experts = (
            self.expert_map_manager.num_fused_shared_experts
        )
        if (
            type(expert_id) is int
            and logical_experts
            <= expert_id
            < logical_experts + fused_shared_experts
        ):
            physical_tail_id = (
                len(self._vllm_hcu_static_eplb_row)
                + expert_id
                - logical_experts
            )
            return original(
                self,
                param,
                loaded_weight,
                weight_name,
                shard_id,
                physical_tail_id,
                return_success=return_success,
            )
        return load_static_logical_expert(
            self,
            original,
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            logical_expert_id=expert_id,
            return_success=return_success,
        )

    return load


def _static_expert_mapping(
    self,
    ckpt_gate_proj_name=None,
    ckpt_down_proj_name=None,
    ckpt_up_proj_name=None,
    include_fused=False,
):
    return self.build_expert_params_mapping(
        ckpt_gate_proj_name or self.ckpt_gate_proj_name,
        ckpt_down_proj_name or self.ckpt_down_proj_name,
        ckpt_up_proj_name or self.ckpt_up_proj_name,
        self.moe_config.num_logical_experts
        + self.expert_map_manager.num_fused_shared_experts,
        num_redundant_experts=0,
        routed_experts_prefix="",
        lora_base_layer_prefix=self.lora_base_layer_prefix,
        include_fused=include_fused,
    )


def _logical_model_mapping(original, logical_experts: int):
    @functools.wraps(original)
    def mapping(*args, **kwargs):
        return [
            entry
            for entry in original(*args, **kwargs)
            if entry[2] < logical_experts
        ]

    return mapping


def _expert_storage_key(weight: torch.Tensor) -> tuple[object, ...]:
    return (
        weight.untyped_storage(),
        weight.storage_offset(),
        weight.numel(),
        weight.dtype,
    )


def bind_static_eplb_plan(
    vllm_config: object,
    model: torch.nn.Module,
) -> StaticEplbPlan | None:
    """Attach validated static rows before checkpoint loaders run.

    All owners and parameter loaders are validated before any instance state is
    published. Quantization methods, tensor layouts, and global classes remain
    owned by the current vLLM implementation.
    """

    from vllm.model_executor.layers.fused_moe.routed_experts import (
        RoutedExperts,
    )
    from vllm.model_executor.models.interfaces import is_mixture_of_experts

    from vllm_hcu.patch.config import bind_hcu_eplb_config

    bind_hcu_eplb_config(vllm_config)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    path = getattr(parallel_config, "_vllm_hcu_expert_map_path", None)
    if not path:
        return None
    if not getattr(parallel_config, "enable_eplb", False) or not getattr(
        parallel_config,
        "enable_expert_parallel",
        False,
    ):
        raise ValueError("Static EPLB requires EPLB and expert parallel")
    if getattr(parallel_config, "enable_ep_weight_filter", False):
        raise ValueError("Static EPLB cannot use the upstream EP weight filter")
    if not is_mixture_of_experts(model):
        raise ValueError("Static EPLB requires a current MixtureOfExperts model")

    layers = tuple(model.moe_layers)
    if not layers:
        raise ValueError("Static EPLB found no local MoE layers")
    if len(layers) != model.num_moe_layers:
        raise ValueError("Static EPLB local MoE layer count mismatch")
    plan = load_static_eplb_plan(
        path,
        model_key=resolve_offline_eplb_model_key(model, parallel_config),
        expected_shape=(len(layers), model.num_physical_experts),
        num_logical_experts=model.num_logical_experts,
        num_redundant_experts=model.num_redundant_experts,
    )

    bound = getattr(model, "_vllm_hcu_static_eplb_plan", None)
    if bound is not None:
        if bound != plan:
            raise ValueError(
                "Static EPLB plan changed; cannot rebind loaded weights"
            )
        for index, layer in enumerate(layers):
            row = getattr(
                getattr(layer, "routed_experts", None),
                "_vllm_hcu_static_eplb_row",
                None,
            )
            if row != plan.layer_map(index):
                raise ValueError(
                    "Static EPLB consumer row changed after binding"
                )
        return bound

    targets: list[tuple[RoutedExperts, list[tuple[torch.Tensor, object]]]] = []
    for layer in layers:
        experts = getattr(layer, "routed_experts", None)
        if not isinstance(experts, RoutedExperts):
            raise ValueError(
                "Static EPLB requires current RoutedExperts owners"
            )
        if not experts.quant_method.supports_eplb:
            raise ValueError(
                "EPLB unsupported by "
                f"{type(experts.quant_method).__name__}"
            )
        if (
            experts.moe_config.num_logical_experts
            != plan.num_logical_experts
            or experts.moe_config.num_experts
            != plan.num_physical_experts
        ):
            raise ValueError("Static EPLB RoutedExperts count mismatch")

        parameters: list[tuple[torch.Tensor, object]] = []
        expert_storage = {
            _expert_storage_key(weight)
            for weight in experts.get_expert_weights()
        }
        for name, parameter in experts.named_parameters():
            if _expert_storage_key(parameter) not in expert_storage:
                continue
            loader = getattr(parameter, "weight_loader", None)
            if getattr(loader, "__self__", None) is not experts:
                raise ValueError(
                    f"Static EPLB parameter {name!r} has an unsupported loader owner"
                )
            parameters.append((parameter, loader.__func__))
        targets.append((experts, parameters))

    verify_static_plan_across_ep_ranks(plan)

    owners: list[torch.nn.Module] = [model]
    expert_owners = {experts for experts, _ in targets}
    for name in ("model", "language_model"):
        inner = getattr(model, name, None)
        if (
            isinstance(inner, torch.nn.Module)
            and inner is not model
            and {
                owner
                for owner in inner.modules()
                if isinstance(owner, RoutedExperts)
            }
            == expert_owners
        ):
            owners.append(inner)

    for index, (experts, parameters) in enumerate(targets):
        experts._vllm_hcu_static_eplb_row = plan.layer_map(index)
        for parameter, original in parameters:
            parameter.weight_loader = MethodType(
                _static_loader(original),
                experts,
            )
        experts.get_expert_mapping = MethodType(
            _static_expert_mapping,
            experts,
        )
    for owner in owners:
        owner._vllm_hcu_static_eplb_plan = plan
        mapping = getattr(owner, "get_expert_mapping", None)
        if callable(mapping):
            fused_shared_experts = (
                targets[0][0].expert_map_manager.num_fused_shared_experts
            )
            owner.get_expert_mapping = _logical_model_mapping(
                mapping,
                plan.num_logical_experts + fused_shared_experts,
            )
    return plan


__all__ = [
    "StaticEplbPlan",
    "bind_static_eplb_plan",
    "load_static_eplb_plan",
    "load_static_logical_expert",
    "resolve_offline_eplb_model_key",
    "verify_static_plan_across_ep_ranks",
]
