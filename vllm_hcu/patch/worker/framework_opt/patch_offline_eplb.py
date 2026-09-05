# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Static/offline EPLB expert-map support for the audited vLLM runtime."""

from __future__ import annotations

import functools
import json
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

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
    f"{TARGET_MODULE}._commit_eplb_maps",
    f"{TARGET_MODULE}._move_to_workspace",
    f"{TARGET_MODULE}.EplbState.rearrange",
)
_MARKER = "_vllm_hcu_offline_eplb_patch_applied"
_WRAPPER_MARKER = "_vllm_hcu_offline_eplb_wrapper"
_RECORD_PATH_ATTR = "_vllm_hcu_expert_map_record_path"
_LOAD_PATH_ATTR = "_vllm_hcu_expert_map_path"
_MODEL_RECORD_PATH_ATTR = "_vllm_hcu_expert_map_record_path"
_MODEL_KEY_ATTR = "_vllm_hcu_expert_map_key"


def _select_model_payload(
    path: Path,
    payload: dict,
    model_key: str,
) -> dict:
    model_maps = payload.get("model_maps")
    if model_maps is None:
        return payload
    if not isinstance(model_maps, dict):
        raise ValueError(f"Offline EPLB map {str(path)!r} has invalid model_maps.")
    if model_key not in model_maps:
        raise ValueError(
            f"Offline EPLB map {str(path)!r} does not contain key "
            f"{model_key!r}; available keys: {sorted(model_maps)}."
        )
    selected = model_maps[model_key]
    if not isinstance(selected, dict):
        raise ValueError(
            f"Offline EPLB map {str(path)!r} key {model_key!r} is invalid."
        )
    return selected


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

    input_path = Path(path)
    with input_path.open(encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"Offline EPLB map {str(input_path)!r} must be an object.")
    selected = _select_model_payload(input_path, payload, model_key)
    raw_map = selected.get(
        "physical_to_logical_map",
        selected.get("expert_map"),
    )
    if raw_map is None:
        raise ValueError(
            f"Offline EPLB map {str(input_path)!r} must contain "
            "physical_to_logical_map."
        )

    loaded = torch.tensor(raw_map, device="cpu")
    if (
        loaded.dtype == torch.bool
        or loaded.is_floating_point()
        or loaded.is_complex()
    ):
        raise ValueError(
            f"Offline EPLB map {str(input_path)!r} must contain integer expert ids."
        )
    loaded = loaded.to(dtype=dtype)
    loaded_shape = tuple(loaded.shape)
    if loaded_shape != expected_shape:
        if (
            loaded.ndim == 2
            and loaded.shape[0] > expected_shape[0]
            and loaded.shape[1] == expected_shape[1]
        ):
            loaded = loaded[-expected_shape[0] :]
        else:
            raise ValueError(
                f"Offline EPLB map {str(input_path)!r} has shape "
                f"{loaded_shape}, expected {expected_shape}."
            )
    if loaded.numel() == 0:
        raise ValueError(f"Offline EPLB map {str(input_path)!r} is empty.")
    if loaded.min().item() < 0:
        raise ValueError(
            f"Offline EPLB map {str(input_path)!r} contains negative expert ids."
        )
    if loaded.max().item() >= num_logical_experts:
        raise ValueError(
            f"Offline EPLB map {str(input_path)!r} contains logical expert id "
            f">= {num_logical_experts}."
        )
    for layer_idx, layer_map in enumerate(loaded):
        counts = torch.bincount(
            layer_map.to(torch.long),
            minlength=num_logical_experts,
        )
        missing = torch.nonzero(counts[:num_logical_experts] == 0).flatten()
        if missing.numel() > 0:
            raise ValueError(
                f"Offline EPLB map {str(input_path)!r} layer {layer_idx} "
                f"misses logical experts {missing.tolist()}."
            )
    return loaded.to(device=device)


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
    """Atomically merge one model's committed map into an offline JSON file."""

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
        payload = _merge_record_payload(output_path, model_key, model_payload)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as destination:
            json.dump(payload, destination)
        tmp_path.replace(output_path)


def _parallel_offline_paths(parallel_config: object) -> tuple[str | None, str | None]:
    record_path = getattr(parallel_config, _RECORD_PATH_ATTR, None)
    load_path = getattr(parallel_config, _LOAD_PATH_ATTR, None)
    if record_path and load_path:
        raise ValueError(
            "expert_map_record_path and expert_map_path are mutually exclusive"
        )
    if (record_path or load_path) and getattr(
        parallel_config, "pipeline_parallel_size", 1
    ) > 1:
        raise ValueError("Offline EPLB requires pipeline_parallel_size=1")
    return record_path, load_path


def _record_model_state(module: ModuleType, model_state: object) -> None:
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
        physical_to_logical_map=model_state.physical_to_logical_map,
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
            (eplb_module, "_commit_eplb_maps"),
            (eplb_module, "_move_to_workspace"),
            (eplb_state_cls, "rearrange"),
        )
        if not all(_wrapped_is_valid(owner, name) for owner, name in wrapped):
            raise PatchCompatibilityError(
                "required HCU offline EPLB patch marker is stale; restart the process"
            )
        return False

    original_add_model = require_callable(eplb_state_cls, "add_model", TARGETS[0])
    original_step = require_callable(eplb_state_cls, "step", TARGETS[1])
    original_commit = require_callable(eplb_module, "_commit_eplb_maps", TARGETS[2])
    original_move = require_callable(eplb_module, "_move_to_workspace", TARGETS[3])
    original_rearrange = require_callable(eplb_state_cls, "rearrange", TARGETS[4])
    rearrange = require_callable(
        eplb_module,
        "rearrange_expert_weights_inplace",
        f"{TARGET_MODULE}.rearrange_expert_weights_inplace",
    )

    @functools.wraps(original_add_model)
    def hcu_add_model(self, model, model_config) -> None:
        record_path, load_path = _parallel_offline_paths(self.parallel_config)
        original_add_model(self, model, model_config)
        model_hash = model_config.compute_hash()
        model_state = self.model_states.get(model_hash)
        if model_state is None:
            raise PatchCompatibilityError(
                "vLLM EPLB add_model did not publish the expected model state"
            )
        model_key = model.__class__.__name__
        setattr(model_state, _MODEL_RECORD_PATH_ATTR, record_path)
        setattr(model_state, _MODEL_KEY_ATTR, model_key)

        if load_path:
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
                "with key %s.",
                load_path,
                model_config.model,
                model_key,
            )
            rearrange(
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
        if is_profile and load_path:
            return None
        return original_step(
            self,
            is_dummy=is_dummy,
            is_profile=is_profile,
            log_stats=log_stats,
        )

    setattr(hcu_step, _WRAPPER_MARKER, True)

    @functools.wraps(original_rearrange)
    def hcu_rearrange(self, is_profile=False, rank_mapping=None):
        _, load_path = _parallel_offline_paths(self.parallel_config)
        # Keep step statistics and explicit elastic-EP remapping intact.
        if load_path and rank_mapping is None:
            return None
        return original_rearrange(
            self, is_profile=is_profile, rank_mapping=rank_mapping
        )

    setattr(hcu_rearrange, _WRAPPER_MARKER, True)

    @functools.wraps(original_commit)
    def hcu_commit(model_state, new_physical_to_logical_map) -> None:
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
    setattr(eplb_state_cls, "add_model", hcu_add_model)
    setattr(eplb_state_cls, "step", hcu_step)
    eplb_state_cls.rearrange = hcu_rearrange
    setattr(eplb_module, "_vllm_hcu_original_offline_commit", original_commit)
    setattr(eplb_module, "_vllm_hcu_original_offline_move", original_move)
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
