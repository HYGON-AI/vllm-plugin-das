# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Durable offline-map recording and guarded vLLM EPLB lifecycle hooks."""

from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

import torch

from vllm_hcu.model_executor.layers.fused_moe.static_eplb import (
    StaticEplbPlan,
    load_static_eplb_plan,
    resolve_offline_eplb_model_key,
    verify_static_plan_across_ep_ranks,
)

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_class,
    require_exact_signature,
)


TARGET_MODULE = "vllm.distributed.eplb.eplb_state"
PATCH_ID = "worker.framework_opt.eplb.offline_expert_map"
TARGETS = tuple(
    f"{TARGET_MODULE}.EplbState.{name}"
    for name in ("add_model", "step", "rearrange")
)
TARGETS += (
    f"{TARGET_MODULE}._commit_eplb_maps",
    f"{TARGET_MODULE}.rearrange_expert_weights_inplace",
)
_MARKER = "_vllm_hcu_offline_eplb"
_OFFLINE_EPLB_FORMAT = "vllm_offline_eplb_physical_to_logical_by_model"
_RECORD_ONLY = ContextVar("hcu_eplb_record_only", default=False)
_RECORD_PROPOSAL_ALLOWED = ContextVar(
    "hcu_eplb_record_proposal_allowed",
    default=True,
)


def _ep_rank() -> int:
    if not torch.distributed.is_initialized():
        return 0
    from vllm.distributed import get_ep_group

    return get_ep_group().rank_in_group


def _strict_count(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"Offline EPLB {name} must be an integer >= {minimum}"
        )
    return value


def _validate_model_entry(model_key: str, entry: object) -> None:
    if not isinstance(model_key, str) or not model_key:
        raise ValueError("Offline EPLB model key must be nonempty")
    if not isinstance(entry, dict):
        raise ValueError(f"Offline EPLB model entry {model_key!r} is malformed")
    required = {
        "record_kind",
        "model_name",
        "model_class",
        "num_moe_layers",
        "num_logical_experts",
        "num_physical_experts",
        "num_redundant_experts",
        "physical_to_logical_map",
    }
    missing = required.difference(entry)
    if missing:
        raise ValueError(
            f"Offline EPLB model entry {model_key!r} is missing "
            + ", ".join(sorted(missing))
        )
    record_kind = entry["record_kind"]
    if record_kind not in ("initial", "proposal"):
        raise ValueError("Offline EPLB record_kind must be initial or proposal")
    if not isinstance(entry["model_name"], str) or not entry["model_name"]:
        raise ValueError("Offline EPLB model_name must be nonempty")
    if type(entry["model_class"]) is not str or entry["model_class"] != model_key:
        raise ValueError("Offline EPLB model_class does not match its key")

    num_layers = _strict_count(entry["num_moe_layers"], "num_moe_layers", minimum=1)
    logical = _strict_count(
        entry["num_logical_experts"],
        "num_logical_experts",
        minimum=1,
    )
    physical = _strict_count(
        entry["num_physical_experts"],
        "num_physical_experts",
        minimum=1,
    )
    redundant = _strict_count(
        entry["num_redundant_experts"],
        "num_redundant_experts",
    )
    rows = entry["physical_to_logical_map"]
    if not isinstance(rows, list) or len(rows) != num_layers:
        raise ValueError("Offline EPLB physical map layer count mismatch")
    StaticEplbPlan(
        model_key=model_key,
        source_path="/offline-eplb-record-validation",
        source_sha256="0" * 64,
        record_kind=record_kind,
        _map_values=tuple(
            tuple(row) if isinstance(row, list) else row for row in rows
        ),
        num_logical_experts=logical,
        num_physical_experts=physical,
        num_redundant_experts=redundant,
    )


def _validate_record_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Existing offline EPLB record is malformed")
    if type(payload.get("version")) is not int or payload["version"] != 2:
        raise ValueError("Offline EPLB record must declare version 2")
    if payload.get("format") != _OFFLINE_EPLB_FORMAT:
        raise ValueError("Offline EPLB record format is malformed")
    model_maps = payload.get("model_maps")
    if not isinstance(model_maps, dict):
        raise ValueError("Existing offline EPLB model_maps is malformed")
    for model_key, entry in model_maps.items():
        _validate_model_entry(model_key, entry)
    return payload


def _empty_record_payload() -> dict[str, Any]:
    return {
        "version": 2,
        "format": _OFFLINE_EPLB_FORMAT,
        "model_maps": {},
    }


def record_offline_expert_map(
    path: str | Path,
    *,
    model_key: str,
    model_name: str,
    model_class: str,
    physical_to_logical_map: torch.Tensor,
    num_logical_experts: int,
    num_redundant_experts: int,
    record_kind: Literal["initial", "proposal"],
) -> None:
    """Atomically merge one model's initial map or real proposal."""

    if _ep_rank() != 0:
        return
    if not isinstance(physical_to_logical_map, torch.Tensor):
        raise TypeError("physical_to_logical_map must be a torch.Tensor")
    if physical_to_logical_map.ndim != 2:
        raise ValueError("physical_to_logical_map must be two-dimensional")
    rows = physical_to_logical_map.detach().to(device="cpu").tolist()
    entry = {
        "record_kind": record_kind,
        "model_name": model_name,
        "model_class": model_class,
        "num_moe_layers": len(rows),
        "num_logical_experts": num_logical_experts,
        "num_physical_experts": physical_to_logical_map.shape[1],
        "num_redundant_experts": num_redundant_experts,
        "physical_to_logical_map": rows,
    }
    _validate_model_entry(model_key, entry)

    output = Path(path).expanduser()
    if not str(output):
        raise ValueError("Offline EPLB record path must be nonempty")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_suffix(output.suffix + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if output.exists():
            try:
                payload = json.loads(output.read_bytes())
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "Existing offline EPLB record contains malformed JSON"
                ) from exc
            payload = _validate_record_payload(payload)
        else:
            payload = _empty_record_payload()

        existing = payload["model_maps"].get(model_key)
        if (
            isinstance(existing, dict)
            and existing.get("record_kind") == "proposal"
            and record_kind == "initial"
        ):
            return
        merged = {
            **payload,
            "model_maps": {**payload["model_maps"], model_key: entry},
        }
        _validate_record_payload(merged)

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
                json.dump(merged, destination, sort_keys=True)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, output)
            directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)


def validate_offline_eplb_record_complete(
    path: str | Path,
    *,
    required_models: Mapping[str, int],
    num_logical_experts: int,
    num_redundant_experts: int,
) -> str:
    """Validate a stable, proposal-complete multi-model calibration file."""

    logical = _strict_count(
        num_logical_experts,
        "num_logical_experts",
        minimum=1,
    )
    redundant = _strict_count(
        num_redundant_experts,
        "num_redundant_experts",
    )
    if not isinstance(required_models, Mapping) or not required_models:
        raise ValueError("Offline EPLB required_models must be nonempty")

    source = Path(path).expanduser().resolve(strict=True)
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Offline EPLB record contains malformed JSON") from exc
    payload = _validate_record_payload(payload)
    physical = logical + redundant
    for model_key, expected_layers in required_models.items():
        layers = _strict_count(
            expected_layers,
            f"{model_key} num_moe_layers",
            minimum=1,
        )
        entry = payload["model_maps"].get(model_key)
        if entry is None:
            raise ValueError(
                f"Offline EPLB record is missing required model {model_key!r}"
            )
        if entry["record_kind"] != "proposal":
            raise ValueError(
                f"Offline EPLB model {model_key!r} has no proposal"
            )
        expected = {
            "model_class": model_key,
            "num_moe_layers": layers,
            "num_logical_experts": logical,
            "num_physical_experts": physical,
            "num_redundant_experts": redundant,
        }
        for name, value in expected.items():
            if type(entry.get(name)) is not type(value) or entry[name] != value:
                raise ValueError(
                    f"Offline EPLB model {model_key!r} {name} count mismatch"
                )
    if source.read_bytes() != raw:
        raise ValueError("Offline EPLB record changed during validation")
    return hashlib.sha256(raw).hexdigest()


def _paths(parallel_config: object) -> tuple[str | None, str | None]:
    load_path = getattr(
        parallel_config,
        "_vllm_hcu_expert_map_path",
        None,
    )
    record_path = getattr(
        parallel_config,
        "_vllm_hcu_expert_map_record_path",
        None,
    )
    if load_path and record_path:
        raise ValueError(
            "Static load and offline map recording are mutually exclusive"
        )
    return load_path, record_path


def _initialize_offline_counters(state: object) -> None:
    state._vllm_hcu_offline_proposal_events = 0
    state._vllm_hcu_offline_transfer_events = 0
    state._vllm_hcu_offline_rearrangement_events = 0


def apply_to_module(module) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    eplb_state_cls = require_class(
        target,
        "EplbState",
        f"{TARGET_MODULE}.EplbState",
    )
    originals = tuple(
        getattr(eplb_state_cls, name)
        for name in ("add_model", "step", "rearrange")
    )
    originals += (
        target._commit_eplb_maps,
        target.rearrange_expert_weights_inplace,
    )
    installed = getattr(target, _MARKER, None)
    if installed is not None:
        if (
            not isinstance(installed, tuple)
            or len(installed) != len(originals)
            or any(
                current is not wrapper
                for current, wrapper in zip(originals, installed)
            )
        ):
            raise PatchCompatibilityError(
                "Static EPLB state marker is stale; restart the process"
            )
        return False

    (
        original_add,
        original_step,
        original_rearrange,
        original_commit,
        original_transfer,
    ) = originals
    require_exact_signature(
        original_add,
        TARGETS[0],
        positional=("self", "model", "model_config"),
    )
    require_exact_signature(
        original_step,
        TARGETS[1],
        positional=("self", "is_dummy", "is_profile", "log_stats"),
        defaults={"is_dummy": False, "is_profile": False, "log_stats": False},
    )
    require_exact_signature(
        original_rearrange,
        TARGETS[2],
        positional=("self", "is_profile", "rank_mapping"),
        defaults={"is_profile": False, "rank_mapping": None},
    )
    require_exact_signature(
        original_commit,
        TARGETS[3],
        positional=("model_state", "new_physical_to_logical_map"),
    )
    require_exact_signature(
        original_transfer,
        TARGETS[4],
        positional=(
            "old_global_expert_indices",
            "new_global_expert_indices",
            "expert_weights",
            "expert_buffer",
            "ep_group",
            "communicator",
            "is_profile",
            "rank_mapping",
        ),
        defaults={"is_profile": False, "rank_mapping": None},
    )

    @functools.wraps(original_add)
    def add_model(self, model, model_config):
        load_path, record_path = _paths(self.parallel_config)
        if not load_path and not record_path:
            return original_add(self, model, model_config)

        _initialize_offline_counters(self)
        plan = None
        if load_path:
            plan = getattr(model, "_vllm_hcu_static_eplb_plan", None)
            if not isinstance(plan, StaticEplbPlan):
                raise ValueError(
                    "Static EPLB requires a bound direct-load plan"
                )
            current = load_static_eplb_plan(
                load_path,
                model_key=resolve_offline_eplb_model_key(
                    model,
                    self.parallel_config,
                ),
                expected_shape=(
                    len(tuple(model.moe_layers)),
                    model.num_physical_experts,
                ),
                num_logical_experts=model.num_logical_experts,
                num_redundant_experts=model.num_redundant_experts,
            )
            if current != plan:
                raise ValueError(
                    "Static EPLB bound plan changed before runtime commit"
                )
            for index, layer in enumerate(model.moe_layers):
                if (
                    getattr(
                        layer.routed_experts,
                        "_vllm_hcu_static_eplb_row",
                        None,
                    )
                    != plan.layer_map(index)
                ):
                    raise ValueError("Static EPLB bound layer row mismatch")
            expected_sha = getattr(
                self,
                "_vllm_hcu_static_source_sha256",
                plan.source_sha256,
            )
            if expected_sha != plan.source_sha256:
                raise ValueError(
                    "Static EPLB models use mixed source SHA-256 fingerprints"
                )
            verify_static_plan_across_ep_ranks(plan)

        model.num_moe_layers = len(tuple(model.moe_layers))
        if plan is not None:
            eplb_config = self.parallel_config.eplb_config
            communicator = eplb_config.communicator
            if communicator not in (
                "torch_nccl",
                "torch_gloo",
                "nixl",
                "pynccl",
            ):
                raise PatchCompatibilityError(
                    "Static EPLB communicator is unresolved or unsupported"
                )
            try:
                eplb_config.communicator = "torch_gloo"
                original_add(self, model, model_config)
            finally:
                eplb_config.communicator = communicator
        else:
            original_add(self, model, model_config)

        model_state = self.model_states[model_config.compute_hash()]
        model_state._hcu_offline_record_path = record_path
        model_state._hcu_offline_owner = self
        model_state._hcu_offline_model_key = resolve_offline_eplb_model_key(
            model,
            self.parallel_config,
        )
        self.is_async = False
        if self.should_record_tensor is not None:
            self.should_record_tensor.fill_(False)

        if plan is not None:
            original_commit(model_state, plan.physical_to_logical_map)
            from vllm_hcu.model_executor.layers.fused_moe.eplb_dispatch import (
                build_locality_fair_replica_order,
                build_nearest_replica_order,
            )

            policy = getattr(
                self.parallel_config,
                "_vllm_hcu_eplb_static_dispatch_policy",
                "nearest",
            )
            order = {
                "nearest": build_nearest_replica_order,
                "locality_fair": build_locality_fair_replica_order,
            }[policy]
            group = target.get_ep_group().device_group
            ordered = order(
                model_state.logical_to_physical_map,
                ep_rank=group.rank(),
                ep_size=group.size(),
                num_nodes=target.get_node_count(),
                num_physical_experts=plan.num_physical_experts,
            )
            model_state.logical_to_physical_map.copy_(ordered)
            self._vllm_hcu_static_source_sha256 = plan.source_sha256
        else:
            record_offline_expert_map(
                record_path,
                model_key=model_state._hcu_offline_model_key,
                model_name=model_state.model_name,
                model_class=type(model).__name__,
                physical_to_logical_map=model_state.physical_to_logical_map,
                num_logical_experts=model.num_logical_experts,
                num_redundant_experts=model.num_redundant_experts,
                record_kind="initial",
            )
        return None

    @functools.wraps(original_step)
    def step(self, is_dummy=False, is_profile=False, log_stats=False):
        load_path, record_path = _paths(self.parallel_config)
        if load_path:
            return None
        if not record_path:
            return original_step(
                self,
                is_dummy=is_dummy,
                is_profile=is_profile,
                log_stats=log_stats,
            )
        token = _RECORD_PROPOSAL_ALLOWED.set(not is_dummy and not is_profile)
        try:
            return original_step(
                self,
                is_dummy=is_dummy,
                is_profile=is_profile,
                log_stats=log_stats,
            )
        finally:
            _RECORD_PROPOSAL_ALLOWED.reset(token)

    @functools.wraps(original_rearrange)
    def rearrange(self, is_profile=False, rank_mapping=None):
        load_path, record_path = _paths(self.parallel_config)
        if load_path or (
            not is_profile
            and getattr(
                self.parallel_config,
                "_vllm_hcu_eplb_disable_rearrange",
                False,
            )
        ):
            return None
        if not record_path or is_profile:
            return original_rearrange(
                self,
                is_profile=is_profile,
                rank_mapping=rank_mapping,
            )
        if rank_mapping is not None:
            raise ValueError(
                "Offline EPLB recording does not support elastic rank mappings"
            )
        token = _RECORD_ONLY.set(True)
        try:
            return original_rearrange(
                self,
                is_profile=False,
                rank_mapping=None,
            )
        finally:
            _RECORD_ONLY.reset(token)

    @functools.wraps(original_transfer)
    def rearrange_expert_weights_inplace(*args, **kwargs):
        if _RECORD_ONLY.get():
            return None
        return original_transfer(*args, **kwargs)

    @functools.wraps(original_commit)
    def commit_eplb_maps(model_state, new_physical_to_logical_map):
        if not _RECORD_ONLY.get():
            return original_commit(
                model_state,
                new_physical_to_logical_map,
            )
        if not _RECORD_PROPOSAL_ALLOWED.get():
            return None
        model = model_state.model
        record_offline_expert_map(
            model_state._hcu_offline_record_path,
            model_key=model_state._hcu_offline_model_key,
            model_name=model_state.model_name,
            model_class=type(model).__name__,
            physical_to_logical_map=new_physical_to_logical_map,
            num_logical_experts=model.num_logical_experts,
            num_redundant_experts=model.num_redundant_experts,
            record_kind="proposal",
        )
        model_state._hcu_offline_owner._vllm_hcu_offline_proposal_events += 1
        return None

    wrappers = (
        add_model,
        step,
        rearrange,
        commit_eplb_maps,
        rearrange_expert_weights_inplace,
    )
    for wrapper in wrappers:
        setattr(wrapper, _MARKER, True)
    for name, wrapper in zip(
        ("add_model", "step", "rearrange"),
        wrappers[:3],
    ):
        setattr(eplb_state_cls, name, wrapper)
    target._commit_eplb_maps = commit_eplb_maps
    target.rearrange_expert_weights_inplace = rearrange_expert_weights_inplace
    setattr(target, _MARKER, wrappers)
    return True


def apply(module=None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
    "record_offline_expert_map",
    "validate_offline_eplb_record_complete",
]
