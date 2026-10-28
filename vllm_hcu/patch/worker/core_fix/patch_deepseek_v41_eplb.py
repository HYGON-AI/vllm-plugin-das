# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Connect the DeepSeek V4.1 ROCm MoE to EPLB.

The V4.1 ROCm MoE reuses the V4 ROCm constructor, which builds the routed
experts without ``enable_eplb``/``num_redundant_experts``.  A global
``--enable-eplb`` therefore registers a model whose routers carry no EPLB
state: weights could be rearranged while every router keeps routing to the
initial physical layout.  The V4.1 subclass also publishes TP-derived expert
counts and an expert mapping without redundant slots.

When EPLB is enabled this adapter, for backbone MoE layers only:

* passes ``enable_eplb=True`` and the configured redundant expert count to
  the routed experts built by the V4 constructor;
* publishes logical/physical/local-physical counts from the constructed EP
  layout instead of the TP split;
* builds the checkpoint mapping over physical slots, so redundant replicas
  load the weights of their logical expert;
* checks, when EPLB registers the model, that the packed DeepGEMM weights and
  scales used by the kernel share storage with the tensors EPLB moves.

With EPLB disabled every call delegates to the official implementation.  The
supported scope is the DeepEP high-throughput path, including DSpark with
offline/static EPLB on the target only. The draft keeps its original experts
and routing; other speculative methods and online DSpark rearrangement fail closed.
"""

from __future__ import annotations

import functools
import importlib
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v41.amd.model"
BASE_MODULE = "vllm.models.deepseek_v4.amd.model"
PATCH_ID = "worker.core_fix.deepseek_v41_amd.eplb"
TARGETS = (
    f"{TARGET_MODULE}.DeepseekV4MoE.__init__",
    f"{TARGET_MODULE}.DeepseekV4Model.get_expert_mapping",
    f"{TARGET_MODULE}.DeepseekV41LLMForCausalLM.set_eplb_state",
)
_CLASS_MARKER = "_vllm_hcu_dsv41_eplb_applied"
_WRAPPER_MARKER = "_vllm_hcu_dsv41_eplb_wrapper"
_SUPPORTED_ALL2ALL_BACKENDS = ("deepep_high_throughput",)


def _validate_eplb_scope(vllm_config) -> None:
    parallel_config = vllm_config.parallel_config
    speculative_config = vllm_config.speculative_config
    if speculative_config is not None:
        if getattr(speculative_config, "method", None) != "dspark":
            raise NotImplementedError(
                "HCU DeepSeek V4.1 EPLB supports only DSpark speculative decoding."
            )
        from vllm_hcu.patch.config import get_hcu_config

        hcu = get_hcu_config(vllm_config)
        if not (
            hcu.eplb_disable_rearrange
            and (hcu.expert_map_path or hcu.expert_map_record_path)
        ):
            raise NotImplementedError(
                "HCU DeepSeek V4.1 DSpark EPLB requires offline calibration or "
                "a static expert map with online rearrangement disabled."
            )
    if parallel_config.enable_elastic_ep:
        raise NotImplementedError(
            "HCU DeepSeek V4.1 EPLB does not support elastic EP."
        )
    if parallel_config.all2all_backend not in _SUPPORTED_ALL2ALL_BACKENDS:
        raise NotImplementedError(
            "HCU DeepSeek V4.1 EPLB is validated only with "
            f"--all2all-backend {'/'.join(_SUPPORTED_ALL2ALL_BACKENDS)}, "
            f"got {parallel_config.all2all_backend!r}."
        )


def _publish_physical_metadata(moe, num_redundant_experts: int) -> None:
    routed = moe.experts.routed_experts
    ep_size = routed.moe_config.moe_parallel_config.ep_size
    num_physical = moe.n_logical_experts + num_redundant_experts
    num_local = routed.local_num_experts
    if routed.global_num_experts != num_physical or num_local * ep_size != (
        num_physical
    ):
        raise RuntimeError(
            f"DeepSeek V4.1 EPLB layout mismatch for {moe.prefix}: "
            f"logical={moe.n_logical_experts} redundant={num_redundant_experts} "
            f"routed_global={routed.global_num_experts} local={num_local} "
            f"ep_size={ep_size}"
        )
    if moe.experts.eplb_state is None:
        raise RuntimeError(
            f"DeepSeek V4.1 EPLB router state was not created for {moe.prefix}"
        )
    moe.n_redundant_experts = num_redundant_experts
    moe.n_physical_experts = num_physical
    moe.n_local_physical_experts = num_local


_KERNEL_TENSORS = ("_deepgemm_w13", "_deepgemm_w2", "w1_scale", "w2_scale")


def _validate_expert_weight_aliases(model) -> None:
    """Fail if EPLB would move tensors other than the ones the kernel reads."""

    for layer_idx, (layer, weights) in enumerate(
        zip(model.moe_layers, model.expert_weights, strict=True)
    ):
        routed = layer.routed_experts
        moe_kernel = getattr(routed.quant_method, "moe_kernel", None)
        experts = getattr(moe_kernel, "fused_experts", None)
        used = {name: getattr(experts, name, None) for name in _KERNEL_TENSORS}
        missing = [name for name, tensor in used.items() if tensor is None]
        if missing:
            raise NotImplementedError(
                "HCU DeepSeek V4.1 EPLB is validated only with the DeepEP HT "
                f"DeepGEMM experts; MoE layer {layer_idx} uses "
                f"{type(experts).__name__} without {', '.join(missing)}."
            )
        bad_shape = [
            tuple(w.shape) for w in weights if w.shape[0] != routed.local_num_experts
        ]
        if bad_shape:
            raise RuntimeError(
                f"EPLB expert weights of MoE layer {layer_idx} are not indexed "
                f"by {routed.local_num_experts} local experts: {bad_shape}"
            )
        moved = {w.untyped_storage().data_ptr() for w in weights}
        stale = [
            name
            for name, tensor in used.items()
            if tensor.untyped_storage().data_ptr() not in moved
        ]
        if stale:
            raise RuntimeError(
                f"EPLB would not move kernel tensors {stale} of MoE layer "
                f"{layer_idx}; routing would use stale expert weights."
            )


def _model_redundant_experts(model) -> int:
    for layer in model.layers:
        ffn = getattr(layer, "ffn", None)
        if ffn is not None and hasattr(ffn, "n_redundant_experts"):
            return ffn.n_redundant_experts
    return 0


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    moe_class = require_class(
        target, "DeepseekV4MoE", f"{TARGET_MODULE}.DeepseekV4MoE"
    )
    model_class = require_class(
        target, "DeepseekV4Model", f"{TARGET_MODULE}.DeepseekV4Model"
    )
    causal_class = require_class(
        target,
        "DeepseekV41LLMForCausalLM",
        f"{TARGET_MODULE}.DeepseekV41LLMForCausalLM",
    )
    original_init = require_callable(moe_class, "__init__", TARGETS[0])
    original_mapping = require_callable(
        model_class, "get_expert_mapping", TARGETS[1]
    )
    original_set_state = require_callable(causal_class, "set_eplb_state", TARGETS[2])
    if getattr(moe_class, _CLASS_MARKER, False):
        if not (
            getattr(vars(moe_class).get("__init__"), _WRAPPER_MARKER, False)
            and getattr(
                vars(model_class).get("get_expert_mapping"), _WRAPPER_MARKER, False
            )
            and getattr(
                vars(causal_class).get("set_eplb_state"), _WRAPPER_MARKER, False
            )
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale"
            )
        return False
    require_exact_signature(
        original_init,
        TARGETS[0],
        positional=("self", "vllm_config", "prefix", "use_sequence_parallel"),
        defaults={"prefix": "", "use_sequence_parallel": False},
    )
    require_exact_signature(original_mapping, TARGETS[1], positional=("self",))
    require_exact_signature(
        original_set_state,
        TARGETS[2],
        positional=(
            "self",
            "expert_load_view",
            "logical_to_physical_map",
            "logical_replica_count",
        ),
    )
    require_callable(
        target,
        "fused_moe_make_expert_params_mapping",
        f"{TARGET_MODULE}.fused_moe_make_expert_params_mapping",
    )
    require_callable(
        target, "extract_layer_index", f"{TARGET_MODULE}.extract_layer_index"
    )

    @functools.wraps(original_init)
    def hcu_init(self, vllm_config, prefix="", use_sequence_parallel=False):
        parallel_config = vllm_config.parallel_config
        config = vllm_config.model_config.hf_config
        if (
            not parallel_config.enable_eplb
            or target.extract_layer_index(prefix) >= config.num_hidden_layers
        ):
            return original_init(
                self,
                vllm_config,
                prefix=prefix,
                use_sequence_parallel=use_sequence_parallel,
            )

        _validate_eplb_scope(vllm_config)
        num_redundant = parallel_config.eplb_config.num_redundant_experts
        base = importlib.import_module(BASE_MODULE)
        factory = base.FusedMoEFactory
        base.FusedMoEFactory = functools.partial(
            factory, enable_eplb=True, num_redundant_experts=num_redundant
        )
        try:
            original_init(
                self,
                vllm_config,
                prefix=prefix,
                use_sequence_parallel=use_sequence_parallel,
            )
        finally:
            base.FusedMoEFactory = factory
        _publish_physical_metadata(self, num_redundant)

    @functools.wraps(original_mapping)
    def hcu_get_expert_mapping(self):
        num_redundant = _model_redundant_experts(self)
        if num_redundant == 0:
            return original_mapping(self)
        return target.fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=num_redundant,
        )

    @functools.wraps(original_set_state)
    def hcu_set_eplb_state(
        self, expert_load_view, logical_to_physical_map, logical_replica_count
    ):
        original_set_state(
            self, expert_load_view, logical_to_physical_map, logical_replica_count
        )
        _validate_expert_weight_aliases(self)

    setattr(hcu_init, _WRAPPER_MARKER, True)
    setattr(hcu_get_expert_mapping, _WRAPPER_MARKER, True)
    setattr(hcu_set_eplb_state, _WRAPPER_MARKER, True)
    setattr(moe_class, "_vllm_hcu_original_dsv41_moe_init", original_init)
    setattr(
        model_class, "_vllm_hcu_original_dsv41_get_expert_mapping", original_mapping
    )
    setattr(moe_class, "__init__", hcu_init)
    setattr(model_class, "get_expert_mapping", hcu_get_expert_mapping)
    setattr(causal_class, "set_eplb_state", hcu_set_eplb_state)
    setattr(moe_class, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
