# SPDX-License-Identifier: Apache-2.0
"""Static map commits and explicit offline recording on current EPLB state."""
from __future__ import annotations

import fcntl
import functools
import json
import os
import tempfile
from contextvars import ContextVar
from pathlib import Path

from vllm_hcu.model_executor.layers.fused_moe.static_eplb import (
    StaticEplbPlan, load_static_eplb_plan, resolve_offline_eplb_model_key,
    verify_static_plan_across_ep_ranks,
)
from ._common import load_exact_module, require_exact_signature, PatchCompatibilityError

TARGET_MODULE = "vllm.distributed.eplb.eplb_state"
PATCH_ID = "worker.framework_opt.eplb.offline_expert_map"
TARGETS = tuple(f"{TARGET_MODULE}.EplbState.{name}" for name in ("add_model", "step", "rearrange"))
TARGETS += (f"{TARGET_MODULE}._commit_eplb_maps", f"{TARGET_MODULE}.rearrange_expert_weights_inplace")
_MARKER = "_vllm_hcu_offline_eplb"
_RECORD_ONLY = ContextVar("hcu_eplb_record_only", default=False)


def record_offline_expert_map(path, *, model_key, model_name, model_class,
                            physical_to_logical_map, num_logical_experts,
                            num_redundant_experts):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = physical_to_logical_map.cpu().tolist()
    # The same immutable validator governs published plans and recorded maps.
    StaticEplbPlan(model_key, str(output.resolve()), "0" * 64,
        tuple(tuple(row) for row in rows), num_logical_experts,
        physical_to_logical_map.shape[1], num_redundant_experts)
    with output.with_suffix(output.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        payload = json.loads(output.read_text()) if output.exists() else {"version": 2, "model_maps": {}}
        if not isinstance(payload, dict) or not isinstance(payload.get("model_maps"), dict):
            raise ValueError("Existing offline EPLB record is malformed")
        payload["model_maps"][model_key] = dict(model_name=model_name,
            model_class=model_class, num_logical_experts=num_logical_experts,
            num_physical_experts=physical_to_logical_map.shape[1],
            num_moe_layers=len(rows), num_redundant_experts=num_redundant_experts,
            physical_to_logical_map=rows)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
        try:
            with os.fdopen(descriptor, "w") as destination:
                json.dump(payload, destination)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, output)
        finally:
            Path(temporary).unlink(missing_ok=True)


def _paths(parallel):
    load = getattr(parallel, "_vllm_hcu_expert_map_path", None)
    record = getattr(parallel, "_vllm_hcu_expert_map_record_path", None)
    if load and record:
        raise ValueError("Static load and map recording are mutually exclusive")
    return load, record


def apply_to_module(module):
    target = load_exact_module(TARGET_MODULE, module)
    cls = target.EplbState
    originals = tuple(getattr(cls, name) for name in ("add_model", "step", "rearrange"))
    originals += (target._commit_eplb_maps, target.rearrange_expert_weights_inplace)
    installed = getattr(target, _MARKER, None)
    if installed:
        if (not isinstance(installed, tuple) or len(installed) != len(originals)
                or any(current is not wrapper for current, wrapper in zip(originals, installed))):
            raise PatchCompatibilityError("Static EPLB state marker is stale")
        return False
    original_add, original_step, original_rearrange, original_commit, original_transfer = originals
    require_exact_signature(original_add, TARGETS[0], positional=("self", "model", "model_config"))
    require_exact_signature(original_step, TARGETS[1],
        positional=("self", "is_dummy", "is_profile", "log_stats"),
        defaults=dict(is_dummy=False, is_profile=False, log_stats=False))
    require_exact_signature(original_rearrange, TARGETS[2],
        positional=("self", "is_profile", "rank_mapping"),
        defaults=dict(is_profile=False, rank_mapping=None))
    require_exact_signature(original_commit, TARGETS[3],
        positional=("model_state", "new_physical_to_logical_map"))
    require_exact_signature(original_transfer, TARGETS[4], positional=(
        "old_global_expert_indices", "new_global_expert_indices", "expert_weights",
        "expert_buffer", "ep_group", "communicator", "is_profile", "rank_mapping"),
        defaults=dict(is_profile=False, rank_mapping=None))

    @functools.wraps(original_add)
    def add(self, model, model_config):
        load_path, record_path = _paths(self.parallel_config)
        plan = None
        if load_path:
            plan = getattr(model, "_vllm_hcu_static_eplb_plan", None)
            if not isinstance(plan, StaticEplbPlan):
                raise ValueError("Static EPLB requires a bound direct-load plan")
            current = load_static_eplb_plan(load_path,
                model_key=resolve_offline_eplb_model_key(model, self.parallel_config),
                expected_shape=(len(tuple(model.moe_layers)), model.num_physical_experts),
                num_logical_experts=model.num_logical_experts,
                num_redundant_experts=model.num_redundant_experts)
            if current != plan:
                raise ValueError("Static EPLB bound plan changed before runtime commit")
            for index, layer in enumerate(model.moe_layers):
                if getattr(layer.routed_experts, "_vllm_hcu_static_eplb_row", None) != plan.layer_map(index):
                    raise ValueError("Static EPLB bound layer row mismatch")
            verify_static_plan_across_ep_ranks(plan)
        if load_path or record_path:
            model.num_moe_layers = len(tuple(model.moe_layers))
        if plan is not None:
            # Static direct-load never transfers weights. Keep the official
            # state/buffer ABI, but avoid NIXL's eager registration of every
            # layer (which corrupts HCU free-memory accounting). The official
            # Gloo owner allocates transfer staging only if a transfer executes.
            # This local construction policy also applies to explicit transfer
            # preferences; restore them for every other model/call, even on error.
            eplb_config = self.parallel_config.eplb_config
            communicator = eplb_config.communicator
            if communicator not in ("torch_nccl", "torch_gloo", "nixl", "pynccl"):
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
        state = self.model_states[model_config.compute_hash()]
        state._hcu_offline_record_path = record_path
        state._hcu_offline_model_key = resolve_offline_eplb_model_key(model, self.parallel_config)
        if plan is not None:
            original_commit(state, plan.physical_to_logical_map)
            from vllm_hcu.model_executor.layers.fused_moe.eplb_dispatch import (
                build_locality_fair_replica_order, build_nearest_replica_order,
            )
            policy = getattr(self.parallel_config, "_vllm_hcu_eplb_static_dispatch_policy", "nearest")
            order = {"nearest": build_nearest_replica_order,
                     "locality_fair": build_locality_fair_replica_order}[policy]
            group = target.get_ep_group().device_group
            ordered = order(state.logical_to_physical_map,
                ep_rank=group.rank(), ep_size=group.size(), num_nodes=target.get_node_count(),
                num_physical_experts=plan.num_physical_experts)
            state.logical_to_physical_map.copy_(ordered)
            self.is_async = False
            if self.should_record_tensor is not None:
                self.should_record_tensor.fill_(False)
        if record_path:
            self.is_async = False
            if target.get_ep_group().rank_in_group == 0:
                record_offline_expert_map(record_path,
                    model_key=resolve_offline_eplb_model_key(model, self.parallel_config),
                    model_name=model_config.model, model_class=type(model).__name__,
                    physical_to_logical_map=state.physical_to_logical_map,
                    num_logical_experts=model.num_logical_experts,
                    num_redundant_experts=model.num_redundant_experts)

    @functools.wraps(original_step)
    def step(self, is_dummy=False, is_profile=False, log_stats=False):
        if _paths(self.parallel_config)[0]:
            return None
        return original_step(self, is_dummy=is_dummy, is_profile=is_profile, log_stats=log_stats)

    @functools.wraps(original_rearrange)
    def rearrange(self, is_profile=False, rank_mapping=None):
        load_path, record_path = _paths(self.parallel_config)
        if load_path or (not is_profile and getattr(self.parallel_config, "_vllm_hcu_eplb_disable_rearrange", False)):
            return None
        if not record_path or is_profile:
            return original_rearrange(self, is_profile=is_profile, rank_mapping=rank_mapping)
        if rank_mapping is not None:
            raise ValueError("Offline EPLB recording does not support elastic EP")
        token = _RECORD_ONLY.set(True)
        try:
            return original_rearrange(self, is_profile=False, rank_mapping=None)
        finally:
            _RECORD_ONLY.reset(token)

    @functools.wraps(original_transfer)
    def transfer(*args, **kwargs):
        if not _RECORD_ONLY.get():
            return original_transfer(*args, **kwargs)

    @functools.wraps(original_commit)
    def commit(model_state, new_physical_to_logical_map):
        if not _RECORD_ONLY.get():
            return original_commit(model_state, new_physical_to_logical_map)
        model = model_state.model
        if target.get_ep_group().rank_in_group == 0:
            record_offline_expert_map(model_state._hcu_offline_record_path,
                model_key=model_state._hcu_offline_model_key,
                model_name=model_state.model_name, model_class=type(model).__name__,
                physical_to_logical_map=new_physical_to_logical_map,
                num_logical_experts=model.num_logical_experts,
                num_redundant_experts=model.num_redundant_experts)

    wrappers = (add, step, rearrange, commit, transfer)
    for wrapper in wrappers:
        setattr(wrapper, _MARKER, True)
    for name, wrapper in zip(("add_model", "step", "rearrange"), wrappers[:3]):
        setattr(cls, name, wrapper)
    target._commit_eplb_maps = commit
    target.rearrange_expert_weights_inplace = transfer
    # Retain exact identities: functools.wraps copies boolean markers, so a
    # marker alone cannot prove that the guarded mutation hooks remain installed.
    setattr(target, _MARKER, wrappers)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
