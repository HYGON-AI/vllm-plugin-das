# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Static/offline EPLB expert-map support for the audited vLLM runtime."""

from __future__ import annotations

import fcntl
import functools
import json
import os
import tempfile
import threading
from contextvars import ContextVar
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from vllm_hcu.model_executor.layers.fused_moe.static_eplb import (
    StaticEplbPlan,
    load_static_eplb_plan,
)

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
)


_FILE_LOCK = threading.RLock()
TARGET_MODULE = "vllm.distributed.eplb.eplb_state"
PATCH_ID = "worker.framework_opt.eplb.offline_expert_map"
TARGETS = (
    f"{TARGET_MODULE}.EplbState.add_model",
    f"{TARGET_MODULE}.EplbState.step",
    f"{TARGET_MODULE}.EplbState.rearrange",
    f"{TARGET_MODULE}.rearrange_expert_weights_inplace",
    f"{TARGET_MODULE}._commit_eplb_maps",
    f"{TARGET_MODULE}._move_to_workspace",
)
_MARKER = "_vllm_hcu_offline_eplb_patch_applied"
_WRAPPER_MARKER = "_vllm_hcu_offline_eplb_wrapper"
_RECORD_PATH_ATTR = "_vllm_hcu_expert_map_record_path"
_LOAD_PATH_ATTR = "_vllm_hcu_expert_map_path"
_MODEL_RECORD_PATH_ATTR = "_vllm_hcu_expert_map_record_path"
_MODEL_KEY_ATTR = "_vllm_hcu_expert_map_key"
_RECORD_ONLY_REARRANGE = ContextVar(
    "vllm_hcu_record_only_eplb_rearrange",
    default=False,
)


def load_offline_expert_map(
    path: str | Path,
    *,
    model_key: str,
    expected_shape: tuple[int, int],
    num_logical_experts: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Load and validate one model's physical-to-logical expert map."""

    plan = load_static_eplb_plan(
        path,
        model_key=model_key,
        expected_shape=expected_shape,
        num_logical_experts=num_logical_experts,
        num_redundant_experts=expected_shape[1] - num_logical_experts,
    )
    return plan.physical_to_logical_map.to(dtype=dtype, device=device)


def _merge_record_payload(
    output_path: Path,
    model_key: str,
    model_payload: dict,
) -> dict:
    payload = {
        "version": 2,
        "format": "vllm_offline_eplb_physical_to_logical_by_model",
        "model_maps": {},
    }
    if output_path.exists():
        try:
            with output_path.open(encoding="utf-8") as source:
                existing = json.load(source)
        except (json.JSONDecodeError, OSError):
            existing = {}
        if isinstance(existing, dict) and isinstance(
            existing.get("model_maps"), dict
        ):
            payload = existing
        elif isinstance(existing, dict) and "physical_to_logical_map" in existing:
            legacy_key = existing.get("model_class", "legacy")
            payload["model_maps"][legacy_key] = existing
    payload["model_maps"][model_key] = model_payload
    return payload


def record_offline_expert_map(
    path: str | Path,
    *,
    model_key: str,
    model_name: str,
    model_class: str,
    physical_to_logical_map: torch.Tensor,
    num_logical_experts: int,
    num_redundant_experts: int,
) -> None:
    """Atomically merge one model's candidate map into an offline JSON file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_cpu = physical_to_logical_map.detach().to(device="cpu")
    model_payload = {
        "version": 1,
        "format": "vllm_offline_eplb_physical_to_logical",
        "model_name": model_name,
        "model_class": model_class,
        "num_moe_layers": int(map_cpu.shape[0]),
        "num_logical_experts": num_logical_experts,
        "num_physical_experts": int(map_cpu.shape[1]),
        "num_redundant_experts": num_redundant_experts,
        "physical_to_logical_map": map_cpu.tolist(),
    }
    with _FILE_LOCK:
        lock_path = output_path.with_suffix(output_path.suffix + ".lock")
        with lock_path.open("a", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            payload = _merge_record_payload(output_path, model_key, model_payload)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                dir=output_path.parent,
            )
            temporary_path = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
                    json.dump(payload, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
                temporary_path.replace(output_path)
            finally:
                temporary_path.unlink(missing_ok=True)


def _parallel_offline_paths(parallel_config: object) -> tuple[str | None, str | None]:
    record_path = getattr(parallel_config, _RECORD_PATH_ATTR, None)
    load_path = getattr(parallel_config, _LOAD_PATH_ATTR, None)
    if record_path and load_path:
        raise ValueError(
            "expert_map_record_path and expert_map_path are mutually exclusive"
        )
    return record_path, load_path


def _record_model_state(
    module: ModuleType,
    model_state: object,
    *,
    physical_to_logical_map: torch.Tensor | None = None,
) -> None:
    path = getattr(model_state, _MODEL_RECORD_PATH_ATTR, None)
    if not path or module.get_ep_group().device_group.rank() != 0:
        return
    model = model_state.model
    model_key = getattr(model_state, _MODEL_KEY_ATTR)
    record_offline_expert_map(
        path,
        model_key=model_key,
        model_name=model_state.model_name,
        model_class=model.__class__.__name__,
        physical_to_logical_map=(
            model_state.physical_to_logical_map
            if physical_to_logical_map is None
            else physical_to_logical_map
        ),
        num_logical_experts=model.num_logical_experts,
        num_redundant_experts=model.num_redundant_experts,
    )
    module.logger.info(
        "Recorded offline EPLB expert map to %s for model key %s.",
        path,
        model_key,
    )


def _wrapped_is_valid(owner: object, name: str) -> bool:
    function = getattr(owner, name, None)
    return callable(function) and bool(getattr(function, _WRAPPER_MARKER, False))


def _find_static_eplb_plan(model: object) -> StaticEplbPlan | None:
    plan = getattr(model, "_vllm_hcu_static_eplb_plan", None)
    if plan is None:
        inner_model = getattr(model, "model", None)
        plan = getattr(inner_model, "_vllm_hcu_static_eplb_plan", None)
    if plan is not None and not isinstance(plan, StaticEplbPlan):
        raise PatchCompatibilityError(
            "HY V4 static EPLB model published an invalid direct-load plan."
        )
    return plan


def _validate_direct_load_plan(
    plan: StaticEplbPlan,
    *,
    load_path: str,
    model_key: str,
    model: object,
    expected_shape: tuple[int, int],
) -> None:
    configured_path = str(Path(load_path).expanduser().resolve(strict=True))
    if plan.source_path != configured_path:
        raise PatchCompatibilityError(
            "HY V4 static EPLB direct-load plan path does not match the runtime "
            f"configuration: plan={plan.source_path!r}, configured={configured_path!r}."
        )
    if plan.model_key != model_key:
        raise PatchCompatibilityError(
            "HY V4 static EPLB direct-load model key mismatch: "
            f"plan={plan.model_key!r}, runtime={model_key!r}."
        )
    observed = (
        plan.num_logical_experts,
        plan.num_physical_experts,
        plan.num_redundant_experts,
    )
    expected = (
        int(getattr(model, "num_logical_experts")),
        int(getattr(model, "num_physical_experts")),
        int(getattr(model, "num_redundant_experts")),
    )
    if observed != expected or tuple(plan.physical_to_logical_map.shape) != expected_shape:
        raise PatchCompatibilityError(
            "HY V4 static EPLB direct-load plan metadata does not match EPLB state: "
            f"plan_counts={observed}, model_counts={expected}, "
            f"plan_shape={tuple(plan.physical_to_logical_map.shape)}, "
            f"state_shape={expected_shape}."
        )


def _verify_static_plan_across_ep_ranks(
    module: ModuleType,
    plan: StaticEplbPlan,
) -> None:
    ep_group = module.get_ep_group()
    world_size = int(getattr(ep_group, "world_size", 1))
    if world_size <= 1:
        return
    fingerprints: list[object] = [None] * world_size
    torch.distributed.all_gather_object(
        fingerprints,
        plan.fingerprint(),
        group=ep_group.cpu_group,
    )
    if any(fingerprint != fingerprints[0] for fingerprint in fingerprints[1:]):
        raise RuntimeError(
            "Static EPLB plan fingerprints differ across EP ranks: "
            f"local_rank={getattr(ep_group, 'rank_in_group', 'unknown')}, "
            f"fingerprints={fingerprints}."
        )


def apply_to_module(module: ModuleType) -> bool:
    """Patch vLLM EPLB state transitions with offline save/load behavior."""

    eplb_module = load_exact_module(TARGET_MODULE, module)
    eplb_state_cls = require_class(
        eplb_module,
        "EplbState",
        f"{TARGET_MODULE}.EplbState",
    )
    require_class(
        eplb_module,
        "EplbModelState",
        f"{TARGET_MODULE}.EplbModelState",
    )
    if getattr(eplb_module, _MARKER, False):
        wrapped = (
            (eplb_state_cls, "add_model"),
            (eplb_state_cls, "step"),
            (eplb_state_cls, "rearrange"),
            (eplb_module, "rearrange_expert_weights_inplace"),
            (eplb_module, "_commit_eplb_maps"),
            (eplb_module, "_move_to_workspace"),
        )
        if not all(_wrapped_is_valid(owner, name) for owner, name in wrapped):
            raise PatchCompatibilityError(
                "required HCU offline EPLB patch marker is stale; restart the process"
            )
        return False

    original_add_model = require_callable(eplb_state_cls, "add_model", TARGETS[0])
    original_step = require_callable(eplb_state_cls, "step", TARGETS[1])
    original_rearrange = require_callable(
        eplb_state_cls,
        "rearrange",
        TARGETS[2],
    )
    original_rearrange_weights = require_callable(
        eplb_module,
        "rearrange_expert_weights_inplace",
        TARGETS[3],
    )
    original_commit = require_callable(eplb_module, "_commit_eplb_maps", TARGETS[4])
    original_move = require_callable(eplb_module, "_move_to_workspace", TARGETS[5])

    @functools.wraps(original_add_model)
    def hcu_add_model(self, model, model_config) -> None:
        original_add_model(self, model, model_config)
        model_hash = model_config.compute_hash()
        model_state = self.model_states.get(model_hash)
        if model_state is None:
            raise PatchCompatibilityError(
                "vLLM EPLB add_model did not publish the expected model state"
            )
        record_path, load_path = _parallel_offline_paths(self.parallel_config)
        model_key = model.__class__.__name__
        setattr(model_state, _MODEL_RECORD_PATH_ATTR, record_path)
        setattr(model_state, _MODEL_KEY_ATTR, model_key)

        if record_path:
            # Recording is an offline planning operation. Running the async
            # worker would transfer expert weights and block later layer
            # commits even though the candidate map must not become live.
            self.is_async = False
            eplb_module.logger.info(
                "EPLB expert-map recording for model %s uses plan-only mode; "
                "online expert rearrangement is disabled.",
                model_key,
            )

        if load_path:
            direct_plan = _find_static_eplb_plan(model)
            if direct_plan is not None:
                _validate_direct_load_plan(
                    direct_plan,
                    load_path=load_path,
                    model_key=model_key,
                    model=model,
                    expected_shape=tuple(model_state.physical_to_logical_map.shape),
                )
                _verify_static_plan_across_ep_ranks(eplb_module, direct_plan)
                target_map = direct_plan.physical_to_logical_map.to(
                    dtype=model_state.physical_to_logical_map.dtype,
                    device="cpu",
                )
                original_commit(
                    model_state,
                    new_physical_to_logical_map=target_map,
                )
                eplb_module.get_ep_group().barrier()
                eplb_module.logger.info(
                    "Static EPLB direct-loaded model %s with map SHA-256 %s; "
                    "committed routing metadata with zero expert rearrangement.",
                    model_key,
                    direct_plan.source_sha256,
                )
            else:
                target_map = load_offline_expert_map(
                    load_path,
                    model_key=model_key,
                    expected_shape=tuple(model_state.physical_to_logical_map.shape),
                    num_logical_experts=model.num_logical_experts,
                    dtype=model_state.physical_to_logical_map.dtype,
                    device=torch.device("cpu"),
                )
                eplb_module.logger.info(
                    "Loading offline EPLB expert map from %s for model %s "
                    "with key %s through the compatibility rearrangement path.",
                    load_path,
                    model_config.model,
                    model_key,
                )
                original_rearrange_weights(
                    model_state.physical_to_logical_map,
                    target_map,
                    model_state.model.expert_weights,
                    model_state.expert_buffer,
                    eplb_module.get_ep_group().device_group,
                    model_state.communicator,
                    False,
                    None,
                )
                original_commit(
                    model_state,
                    new_physical_to_logical_map=target_map,
                )
            should_record = getattr(self, "should_record_tensor", None)
            if should_record is not None:
                should_record.fill_(False)
            # A loaded map is static: do not start the asynchronous EPLB
            # worker or collect data for a later dynamic rearrangement.
            self.is_async = False

        _record_model_state(eplb_module, model_state)

    setattr(hcu_add_model, _WRAPPER_MARKER, True)

    @functools.wraps(original_step)
    def hcu_step(
        self,
        is_dummy: bool = False,
        is_profile: bool = False,
        log_stats: bool = False,
    ) -> Any:
        _, load_path = _parallel_offline_paths(self.parallel_config)
        if load_path:
            return None
        return original_step(
            self,
            is_dummy=is_dummy,
            is_profile=is_profile,
            log_stats=log_stats,
        )

    setattr(hcu_step, _WRAPPER_MARKER, True)

    @functools.wraps(original_rearrange)
    def hcu_rearrange(
        self,
        is_profile: bool = False,
        rank_mapping: dict[int, int] | None = None,
    ) -> Any:
        record_path, _ = _parallel_offline_paths(self.parallel_config)
        if not record_path or is_profile:
            return original_rearrange(
                self,
                is_profile=is_profile,
                rank_mapping=rank_mapping,
            )

        token = _RECORD_ONLY_REARRANGE.set(True)
        try:
            result = original_rearrange(
                self,
                is_profile=False,
                rank_mapping=rank_mapping,
            )
        finally:
            _RECORD_ONLY_REARRANGE.reset(token)
        eplb_module.logger.info(
            "Recorded candidate EPLB expert maps without transferring weights "
            "or changing live routing metadata."
        )
        return result

    setattr(hcu_rearrange, _WRAPPER_MARKER, True)

    @functools.wraps(original_rearrange_weights)
    def hcu_rearrange_weights(*args, **kwargs) -> None:
        if _RECORD_ONLY_REARRANGE.get():
            return None
        return original_rearrange_weights(*args, **kwargs)

    setattr(hcu_rearrange_weights, _WRAPPER_MARKER, True)

    @functools.wraps(original_commit)
    def hcu_commit(model_state, new_physical_to_logical_map) -> None:
        if _RECORD_ONLY_REARRANGE.get():
            _record_model_state(
                eplb_module,
                model_state,
                physical_to_logical_map=new_physical_to_logical_map,
            )
            return
        original_commit(
            model_state,
            new_physical_to_logical_map=new_physical_to_logical_map,
        )
        _record_model_state(eplb_module, model_state)

    setattr(hcu_commit, _WRAPPER_MARKER, True)

    @functools.wraps(original_move)
    def hcu_move_to_workspace(model_state, ep_rank) -> None:
        pending_result = model_state.pending_result
        is_last_layer = bool(
            pending_result is not None
            and pending_result.layer_idx == model_state.model.num_moe_layers - 1
        )
        original_move(model_state, ep_rank)
        if is_last_layer:
            _record_model_state(eplb_module, model_state)

    setattr(hcu_move_to_workspace, _WRAPPER_MARKER, True)

    setattr(eplb_state_cls, "_vllm_hcu_original_offline_add_model", original_add_model)
    setattr(eplb_state_cls, "_vllm_hcu_original_offline_step", original_step)
    setattr(
        eplb_state_cls,
        "_vllm_hcu_original_offline_rearrange",
        original_rearrange,
    )
    setattr(eplb_state_cls, "add_model", hcu_add_model)
    setattr(eplb_state_cls, "step", hcu_step)
    setattr(eplb_state_cls, "rearrange", hcu_rearrange)
    setattr(
        eplb_module,
        "_vllm_hcu_original_offline_rearrange_weights",
        original_rearrange_weights,
    )
    setattr(eplb_module, "_vllm_hcu_original_offline_commit", original_commit)
    setattr(eplb_module, "_vllm_hcu_original_offline_move", original_move)
    setattr(eplb_module, "rearrange_expert_weights_inplace", hcu_rearrange_weights)
    setattr(eplb_module, "_commit_eplb_maps", hcu_commit)
    setattr(eplb_module, "_move_to_workspace", hcu_move_to_workspace)
    setattr(eplb_module, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "PatchCompatibilityError",
    "apply",
    "apply_to_module",
    "load_offline_expert_map",
    "record_offline_expert_map",
]
