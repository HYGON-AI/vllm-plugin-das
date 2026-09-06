# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Validated, immutable static EPLB loading plans."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class StaticEplbPlan:
    """A checkpoint-loading layout that cannot be mutated by its consumers."""

    model_key: str
    source_path: str
    source_sha256: str
    _map_values: tuple[tuple[int, ...], ...]
    num_logical_experts: int
    num_physical_experts: int
    num_redundant_experts: int

    @property
    def physical_to_logical_map(self) -> torch.Tensor:
        """Return a private CPU copy so cached plan state remains immutable."""

        return torch.tensor(self._map_values, dtype=torch.int64, device="cpu")

    def layer_map(self, layer_idx: int) -> tuple[int, ...]:
        return self._map_values[layer_idx]

    def fingerprint(self) -> tuple[str, str, tuple[int, int], int, int, int]:
        return (
            self.model_key,
            self.source_sha256,
            (len(self._map_values), self.num_physical_experts),
            self.num_logical_experts,
            self.num_physical_experts,
            self.num_redundant_experts,
        )


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=16)
def _read_json(
    canonical_path: str,
    identity: tuple[int, int, int, int],
) -> tuple[dict[str, Any], str]:
    del identity
    raw = Path(canonical_path).read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Offline EPLB map {canonical_path!r} must be an object.")
    return payload, hashlib.sha256(raw).hexdigest()


def _select_model_payload(
    path: str,
    payload: dict[str, Any],
    model_key: str,
) -> dict[str, Any]:
    model_maps = payload.get("model_maps")
    if model_maps is None:
        return payload
    if not isinstance(model_maps, dict):
        raise ValueError(f"Offline EPLB map {path!r} has invalid model_maps.")
    if model_key not in model_maps:
        raise ValueError(
            f"Offline EPLB map {path!r} does not contain key {model_key!r}; "
            f"available keys: {sorted(model_maps)}."
        )
    selected = model_maps[model_key]
    if not isinstance(selected, dict):
        raise ValueError(f"Offline EPLB map {path!r} key {model_key!r} is invalid.")
    return selected


def _validate_raw_map(path: str, raw_map: Any) -> list[list[int]]:
    if not isinstance(raw_map, list) or not raw_map:
        raise ValueError(f"Offline EPLB map {path!r} must be a non-empty 2D list.")
    result: list[list[int]] = []
    for row in raw_map:
        if not isinstance(row, list):
            raise ValueError(f"Offline EPLB map {path!r} must be a 2D list.")
        checked_row: list[int] = []
        for expert_id in row:
            if isinstance(expert_id, bool) or not isinstance(expert_id, int):
                raise ValueError(
                    f"Offline EPLB map {path!r} must contain integer expert ids."
                )
            checked_row.append(expert_id)
        result.append(checked_row)
    return result


def load_static_eplb_plan(
    path: str | Path,
    *,
    model_key: str,
    expected_shape: tuple[int, int],
    num_logical_experts: int,
    num_redundant_experts: int,
) -> StaticEplbPlan:
    """Load and validate one model's physical-to-logical expert layout."""

    canonical_path = str(Path(path).expanduser().resolve(strict=True))
    payload, digest = _read_json(canonical_path, _file_identity(Path(canonical_path)))
    selected = _select_model_payload(canonical_path, payload, model_key)
    raw_map = selected.get("physical_to_logical_map", selected.get("expert_map"))
    if raw_map is None:
        raise ValueError(
            f"Offline EPLB map {canonical_path!r} must contain "
            "physical_to_logical_map."
        )
    rows = _validate_raw_map(canonical_path, raw_map)

    loaded_shape = (len(rows), len(rows[0]))
    if any(len(row) != loaded_shape[1] for row in rows):
        raise ValueError(f"Offline EPLB map {canonical_path!r} has ragged rows.")
    if loaded_shape != expected_shape:
        if loaded_shape[0] > expected_shape[0] and loaded_shape[1] == expected_shape[1]:
            rows = rows[-expected_shape[0] :]
            loaded_shape = expected_shape
        else:
            raise ValueError(
                f"Offline EPLB map {canonical_path!r} has shape {loaded_shape}, "
                f"expected {expected_shape}."
            )

    expected_redundant = expected_shape[1] - num_logical_experts
    if num_redundant_experts != expected_redundant:
        raise ValueError(
            f"Offline EPLB map {canonical_path!r} redundant expert count "
            f"{num_redundant_experts} does not match physical-logical count "
            f"{expected_redundant}."
        )
    for layer_idx, row in enumerate(rows):
        negative = [value for value in row if value < 0]
        if negative:
            raise ValueError(
                f"Offline EPLB map {canonical_path!r} contains negative expert ids."
            )
        invalid = [value for value in row if value >= num_logical_experts]
        if invalid:
            raise ValueError(
                f"Offline EPLB map {canonical_path!r} contains logical expert id "
                f">= {num_logical_experts}."
            )
        missing = sorted(set(range(num_logical_experts)) - set(row))
        if missing:
            raise ValueError(
                f"Offline EPLB map {canonical_path!r} layer {layer_idx} misses "
                f"logical experts {missing}."
            )

    return StaticEplbPlan(
        model_key=model_key,
        source_path=canonical_path,
        source_sha256=digest,
        _map_values=tuple(tuple(row) for row in rows),
        num_logical_experts=num_logical_experts,
        num_physical_experts=expected_shape[1],
        num_redundant_experts=num_redundant_experts,
    )


def maybe_load_static_eplb_plan(
    vllm_config: object,
    *,
    model_key: str,
    num_moe_layers: int,
    num_logical_experts: int,
    num_physical_experts: int,
    num_redundant_experts: int,
) -> StaticEplbPlan | None:
    """Build a static plan from the worker-propagated parallel config."""

    parallel_config = getattr(vllm_config, "parallel_config", None)
    path = getattr(parallel_config, "_vllm_hcu_expert_map_path", None)
    if not path:
        return None
    if not getattr(parallel_config, "enable_expert_parallel", False):
        raise ValueError("Static EPLB direct load requires expert parallel.")
    if not getattr(parallel_config, "enable_eplb", False):
        raise ValueError("Static EPLB direct load requires EPLB.")
    if getattr(parallel_config, "enable_ep_weight_filter", False):
        raise ValueError(
            "Static EPLB direct load does not support upstream EP weight "
            "filtering because the filter cannot express per-layer offline maps."
        )
    if num_moe_layers <= 0:
        raise ValueError(
            f"Static EPLB direct load found no MoE layers for {model_key}."
        )
    return load_static_eplb_plan(
        path,
        model_key=model_key,
        expected_shape=(num_moe_layers, num_physical_experts),
        num_logical_experts=num_logical_experts,
        num_redundant_experts=num_redundant_experts,
    )


def bind_static_eplb_plan(
    vllm_config: object,
    model: object,
) -> StaticEplbPlan | None:
    """Validate and bind a static EPLB plan before checkpoint loading."""

    parallel_config = getattr(vllm_config, "parallel_config", None)
    if not getattr(parallel_config, "_vllm_hcu_expert_map_path", None):
        return None

    from vllm.model_executor.models.interfaces import is_mixture_of_experts

    model_key = model.__class__.__name__
    if not is_mixture_of_experts(model):
        raise ValueError(
            "Static EPLB direct load requires a MixtureOfExperts model with "
            f"MoE layers; got {model_key}."
        )

    counts: dict[str, int] = {}
    for name in (
        "num_moe_layers",
        "num_logical_experts",
        "num_physical_experts",
        "num_redundant_experts",
    ):
        value = getattr(model, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "Static EPLB direct load requires "
                f"MixtureOfExperts.{name} to be an integer for {model_key}."
            )
        counts[name] = value

    moe_layers = tuple(model.moe_layers)
    if len(moe_layers) != counts["num_moe_layers"]:
        raise ValueError(
            "Static EPLB direct load found a MixtureOfExperts.moe_layers "
            f"count of {len(moe_layers)} for {model_key}; expected "
            f"{counts['num_moe_layers']}."
        )

    plan = maybe_load_static_eplb_plan(
        vllm_config,
        model_key=model_key,
        num_moe_layers=counts["num_moe_layers"],
        num_logical_experts=counts["num_logical_experts"],
        num_physical_experts=counts["num_physical_experts"],
        num_redundant_experts=counts["num_redundant_experts"],
    )
    if plan is None:
        return None

    routed_experts = []
    for layer_idx, moe_layer in enumerate(moe_layers):
        runner_routed_experts = getattr(moe_layer, "routed_experts", None)
        if runner_routed_experts is None:
            raise ValueError(
                "Static EPLB direct load requires "
                "MixtureOfExperts.moe_layers to expose routed_experts; "
                f"layer {layer_idx} of {model_key} does not."
            )
        routed_experts.append(runner_routed_experts)

    if len(plan._map_values) != len(routed_experts):
        raise ValueError(
            "Static EPLB direct-load plan row count does not match "
            f"MixtureOfExperts.moe_layers for {model_key}."
        )

    # Validate the entire model before publishing any plan state.  The routed
    # expert object is the checkpoint-loader consumer, while the model-level
    # plan is consumed later by EPLB-state initialization.
    rows = tuple(
        plan.layer_map(layer_idx) for layer_idx in range(len(routed_experts))
    )
    for routed_experts_layer, row in zip(routed_experts, rows, strict=True):
        setattr(routed_experts_layer, "_vllm_hcu_static_eplb_row", row)
    setattr(model, "_vllm_hcu_static_eplb_plan", plan)
    return plan


def load_static_logical_expert(
    routed_experts: object,
    original_weight_loader: Callable[..., bool | None],
    *,
    param: object,
    loaded_weight: object,
    weight_name: str,
    shard_id: str,
    logical_expert_id: int,
    return_success: bool,
) -> bool | None:
    """Load one checkpoint expert into every mapped local physical slot."""

    row = getattr(routed_experts, "_vllm_hcu_static_eplb_row")
    if (
        isinstance(logical_expert_id, bool)
        or not isinstance(logical_expert_id, int)
        or logical_expert_id < 0
        or logical_expert_id not in row
    ):
        raise ValueError(
            "Static EPLB logical expert id "
            f"{logical_expert_id!r} is not present in the bound plan row."
        )

    loaded_any = False
    for physical_expert_id, mapped_logical_expert_id in enumerate(row):
        if mapped_logical_expert_id != logical_expert_id:
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


def static_eplb_layer_map_for_weight(
    plan: StaticEplbPlan,
    moe_layer_indices: tuple[int, ...],
    weight_name: str,
) -> tuple[int, ...]:
    """Resolve an absolute checkpoint layer name to its static map row."""

    match = re.search(r"(?:^|\.)layers\.(\d+)\.", weight_name)
    if match is None:
        raise ValueError(
            f"Cannot resolve a HY V4 MoE layer from checkpoint weight {weight_name!r}."
        )
    absolute_layer_idx = int(match.group(1))
    try:
        moe_layer_idx = moe_layer_indices.index(absolute_layer_idx)
    except ValueError as error:
        raise ValueError(
            f"HY V4 checkpoint layer {absolute_layer_idx} is not an MoE layer "
            f"in static EPLB order {moe_layer_indices}."
        ) from error
    return plan.layer_map(moe_layer_idx)


def build_expert_params_mapping_for_row(
    model: torch.nn.Module,
    *,
    ckpt_gate_proj_name: str,
    ckpt_down_proj_name: str,
    ckpt_up_proj_name: str,
    physical_to_logical: tuple[int, ...],
    routed_experts_prefix: str = "routed_experts",
) -> list[tuple[str, str, int, str]]:
    """Build vLLM split-expert mappings from an authoritative map row."""

    has_base_layer = any(".base_layer." in name for name, _ in model.named_parameters())
    base_layer_prefix = "base_layer." if has_base_layer else ""
    routed_prefix = f"{routed_experts_prefix}." if routed_experts_prefix else ""
    w13 = f"experts.{routed_prefix}{base_layer_prefix}w13_"
    w2 = f"experts.{routed_prefix}{base_layer_prefix}w2_"
    result: list[tuple[str, str, int, str]] = []
    for physical_expert_id, logical_expert_id in enumerate(physical_to_logical):
        for shard_id, weight_name in (
            ("w1", ckpt_gate_proj_name),
            ("w2", ckpt_down_proj_name),
            ("w3", ckpt_up_proj_name),
        ):
            param_name = w13 if shard_id in ("w1", "w3") else w2
            result.append(
                (
                    param_name,
                    f"experts.{logical_expert_id}.{weight_name}.{base_layer_prefix}",
                    physical_expert_id,
                    shard_id,
                )
            )
    return result


__all__ = [
    "StaticEplbPlan",
    "bind_static_eplb_plan",
    "build_expert_params_mapping_for_row",
    "load_static_logical_expert",
    "load_static_eplb_plan",
    "maybe_load_static_eplb_plan",
    "static_eplb_layer_map_for_weight",
]
