# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Expose Kimi-K3's local routed experts through vLLM's EPLB protocol."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
)

TARGET_MODULE = "vllm.models.kimi_k3.amd.linear"
PATCH_ID = "worker.op_opt.kimi_k3.eplb_model_protocol"
TARGETS = (
    f"{TARGET_MODULE}.KimiLinearForCausalLM.__init__",
    f"{TARGET_MODULE}.KimiMoE.__init__",
    f"{TARGET_MODULE}.KimiLinearModel.load_weights",
)
_MARKER = "_vllm_hcu_kimi_k3_eplb_model_protocol_applied"
_WRAPPER = "_vllm_hcu_kimi_k3_eplb_init_wrapper"
_MOE_INIT_MARKER = "_vllm_hcu_kimi_k3_w4a8_native_width_applied"
_MOE_INIT_WRAPPER = "_vllm_hcu_kimi_k3_w4a8_native_width_wrapper"
_LOAD_WRAPPER = "_vllm_hcu_kimi_k3_redundant_expert_loader"


def _wrap_model_load_weights(model_cls) -> None:
    original_load = require_callable(model_cls, "load_weights", TARGETS[2])

    @functools.wraps(original_load)
    def load_weights(self, weights):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        # Delegate redundant experts to their owning loader. The legacy Kimi
        # inline loader neither requests redundant mappings nor visits multiple
        # physical slots for one checkpoint expert (it breaks on first match).
        # RoutedExperts already handles both, including static-plan fanout and
        # the rank-local physical ID conversion in each parameter's loader.
        owners = {
            name.removesuffix(".routed_experts"): (name, owner)
            for name, owner in self.named_modules()
            if isinstance(owner, RoutedExperts)
            and owner.moe_config.num_experts > owner.moe_config.num_logical_experts
        }
        if not owners:
            return original_load(self, weights)
        loaded = set()

        def remaining_weights():
            for entry in weights:
                name, weight = entry[:2]
                parent, separator, expert_name = name.partition(".experts.")
                destination = owners.get(parent + ".experts") if separator else None
                if destination is None:
                    yield entry
                    continue
                module_name, owner = destination
                # Kimi's inline loader calls these w1/w2/w3 even when the
                # factory leaves the owner's conventional checkpoint names.
                parts = expert_name.split(".", 2)
                if len(parts) == 3 and parts[0].isdigit():
                    projections = {
                        "w1": owner.ckpt_gate_proj_name,
                        "w2": owner.ckpt_down_proj_name,
                        "w3": owner.ckpt_up_proj_name,
                    }
                    projection = projections.get(parts[1], parts[1])
                    expert_name = f"{parts[0]}.{projection}.{parts[2]}"
                if (
                    expert_name.endswith(".weight_packed")
                    and not hasattr(owner, "w13_weight_packed")
                ):
                    expert_name = expert_name.removesuffix("_packed")
                for param_name in owner.load_weights(((expert_name, weight),)):
                    loaded.add(f"{module_name}.{param_name}")

        result = original_load(self, remaining_weights())
        return result | loaded

    setattr(load_weights, _LOAD_WRAPPER, True)
    model_cls.load_weights = load_weights


def _init_moe_metadata(self) -> None:
    from vllm.models.kimi_k3.amd.linear import KimiMoE

    layers = [
        layer.mlp.experts
        for layer in self.model.layers[
            self.model.start_layer : self.model.end_layer
        ]
        if isinstance(getattr(layer, "mlp", None), KimiMoE)
    ]
    logical = int(self.config.num_experts or 0)
    counts = {
        (
            int(runner.routed_experts.moe_config.num_experts),
            int(runner.routed_experts.local_num_experts),
        )
        for runner in layers
    }
    if len(counts) > 1:
        raise ValueError("Kimi-K3 local MoE layers disagree on expert counts")
    physical, local = next(iter(counts), (logical, 0))
    if physical < logical or (layers and not 0 < local <= physical):
        raise ValueError("Kimi-K3 has invalid physical/local expert counts")
    self.moe_layers = layers
    self.num_moe_layers = len(layers)
    self.num_expert_groups = int(self.config.num_expert_group or 1)
    self.num_logical_experts = logical
    self.num_routed_experts = logical
    self.num_physical_experts = physical
    self.num_local_physical_experts = local
    self.num_shared_experts = int(self.config.num_shared_experts or 0)
    self.num_redundant_experts = physical - logical
    # The list is populated only after post-load HIPC packing, when EPLB asks
    # for it. Keeping no early snapshot is required for stable ownership.
    self.expert_weights = []


def _set_eplb_state(
    self,
    expert_load_view,
    logical_to_physical_map,
    logical_replica_count,
) -> None:
    self.expert_weights.clear()
    for layer_idx, layer in enumerate(self.moe_layers):
        self.expert_weights.append(list(layer.get_expert_weights()))
        layer.set_eplb_state(
            moe_layer_idx=layer_idx,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
        )


def _update_physical_experts_metadata(
    self, num_physical_experts: int, num_local_physical_experts: int
) -> None:
    if (num_physical_experts, num_local_physical_experts) != (
        self.num_physical_experts,
        self.num_local_physical_experts,
    ):
        raise ValueError("Kimi-K3 cannot resize packed physical expert storage")
    for layer in self.moe_layers:
        layer.update_expert_map()


def _wrap_kimi_moe_init(kimi_moe_cls) -> None:
    """Keep aligned K3 W4A8 TP shards at their native width.

    KimiMoE's upstream initializer may pad an expert partition up to its
    legacy minimum. The HCU W4A8 MoE kernel only requires widths aligned to
    32, so preserve the native TP partition when it already meets that
    requirement. Lower the initializer's minimum only while constructing
    that K3 W4A8 layer.
    """
    if getattr(kimi_moe_cls, _MOE_INIT_MARKER, False):
        return
    original_init = require_callable(
        kimi_moe_cls,
        "__init__",
        f"{TARGET_MODULE}.KimiMoE.__init__",
    )

    @functools.wraps(original_init)
    def hcu_kimi_moe_init(self, config, *args, **kwargs):
        quant_config = kwargs.get("quant_config")
        if quant_config is None and args:
            quant_config = args[0]
        get_name = getattr(quant_config, "get_name", None)
        quant_name = get_name() if callable(get_name) else None
        moe_size = getattr(config, "moe_intermediate_size", None)
        if quant_name != "slimquant_w4a8" or moe_size is None:
            return original_init(self, config, *args, **kwargs)

        from vllm.distributed import get_tensor_model_parallel_world_size

        tp_size = int(get_tensor_model_parallel_world_size())
        moe_size = int(moe_size)
        if tp_size < 1 or moe_size % tp_size:
            return original_init(self, config, *args, **kwargs)
        per_partition = moe_size // tp_size
        if per_partition % 32:
            return original_init(self, config, *args, **kwargs)

        attr = "min_moe_intermediate_per_partition"
        missing = object()
        previous = getattr(config, attr, missing)
        try:
            setattr(config, attr, per_partition)
        except (AttributeError, TypeError) as exc:
            raise PatchCompatibilityError(
                "Cannot preserve the native SlimQuant W4A8 Kimi-MoE width: "
                f"{type(config).__name__}.{attr} is not writable"
            ) from exc
        try:
            return original_init(self, config, *args, **kwargs)
        finally:
            if previous is missing:
                delattr(config, attr)
            else:
                setattr(config, attr, previous)

    setattr(hcu_kimi_moe_init, _MOE_INIT_WRAPPER, True)
    setattr(kimi_moe_cls, "__init__", hcu_kimi_moe_init)
    setattr(kimi_moe_cls, _MOE_INIT_MARKER, True)


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    cls = require_class(
        target, "KimiLinearForCausalLM", f"{TARGET_MODULE}.KimiLinearForCausalLM"
    )
    kimi_moe_cls = require_class(target, "KimiMoE", f"{TARGET_MODULE}.KimiMoE")
    model_cls = require_class(
        target, "KimiLinearModel", f"{TARGET_MODULE}.KimiLinearModel"
    )
    if already_applied(
        target,
        _MARKER,
        (
            (cls, "__init__", TARGETS[0], _WRAPPER),
            (
                kimi_moe_cls,
                "__init__",
                f"{TARGET_MODULE}.KimiMoE.__init__",
                _MOE_INIT_WRAPPER,
            ),
            (model_cls, "load_weights", TARGETS[2], _LOAD_WRAPPER),
        ),
    ):
        return False
    original_init = require_callable(cls, "__init__", TARGETS[0])

    @functools.wraps(original_init)
    def hcu_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _init_moe_metadata(self)

    setattr(hcu_init, _WRAPPER, True)
    setattr(cls, "__init__", hcu_init)
    _wrap_kimi_moe_init(kimi_moe_cls)
    _wrap_model_load_weights(model_cls)
    cls.set_eplb_state = _set_eplb_state
    cls.update_physical_experts_metadata = _update_physical_experts_metadata
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
