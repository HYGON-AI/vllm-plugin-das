# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Make the official DeepSeek-V4 AMD model participate in EPLB.

Upstream ``DeepseekV4ForCausalLM`` does not implement ``MixtureOfExperts`` and
``DeepseekV4MoE`` never forwards ``enable_eplb``/``num_redundant_experts`` to
``FusedMoEFactory``.  ``--enable-eplb`` was therefore a silent no-op for V4:
no redundant experts were allocated and ``EplbState`` never registered the
model.  When EPLB is enabled this adapter (mirroring ``deepseek_v2``):

* builds the routed experts with the configured redundant experts;
* loads every physical replica from its logical checkpoint expert;
* exposes the ``MixtureOfExperts`` interface on the CausalLM and native MTP;
* maps redundant checkpoint experts for both target and draft models.

With EPLB disabled every wrapped path is the original one.
"""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v4.amd.model"
PATCH_ID = "worker.core_fix.deepseek_v4_amd.eplb"
TARGETS = (
    f"{TARGET_MODULE}.FusedMoEFactory",
    f"{TARGET_MODULE}.DeepseekV4MoE.__init__",
    f"{TARGET_MODULE}.DeepseekV4Model.get_expert_mapping",
    f"{TARGET_MODULE}.DeepseekV4ForCausalLM.__init__",
)
_MARKER = "_vllm_hcu_deepseek_v4_eplb_applied"
_WRAPPER = "_vllm_hcu_deepseek_v4_eplb_wrapper"
_CONFIG_ATTR = "_vllm_hcu_deepseek_v4_eplb_config"


def _eplb_config() -> tuple[bool, int]:
    from vllm.config import get_current_vllm_config

    parallel_config = get_current_vllm_config().parallel_config
    if not parallel_config.enable_eplb:
        return False, 0
    return True, int(parallel_config.eplb_config.num_redundant_experts)


def _capture_eplb_config(*owners) -> tuple[bool, int]:
    config = _eplb_config()
    for owner in owners:
        if owner is not None:
            setattr(owner, _CONFIG_ATTR, config)
    return config


def _instance_eplb_config(owner) -> tuple[bool, int]:
    config = getattr(owner, _CONFIG_ATTR, None)
    if config is not None:
        return config
    # Initialization-time callers still have the temporary vLLM config scope.
    return _eplb_config()


def _collect_moe(model) -> None:
    """Populate the ``MixtureOfExperts`` attributes (see deepseek_v2)."""

    # The registered HCU MTP compatibility class inherits the patched native
    # AMD MTP class, so its MoE marker may be inherited rather than present in
    # the concrete class dictionary.
    moe_cls = getattr(type(model), "_vllm_hcu_moe_cls", None)
    model.num_expert_groups = getattr(model.config, "n_group", None) or 1
    model.moe_layers = []
    model.moe_mlp_layers = []
    layers = model.model.layers
    if hasattr(layers, "values"):
        layers = layers.values()
    for layer in layers:
        decoder = getattr(layer, "mtp_block", layer)
        ffn = getattr(decoder, "ffn", None)
        if isinstance(ffn, moe_cls):
            model.moe_mlp_layers.append(ffn)
            model.moe_layers.append(ffn.experts)
    if not model.moe_mlp_layers:
        raise PatchCompatibilityError("DeepSeek-V4 EPLB found no DeepseekV4MoE layer")
    example = model.moe_mlp_layers[-1]
    model.num_moe_layers = len(model.moe_layers)
    model.num_logical_experts = example.n_logical_experts
    model.num_physical_experts = example.n_physical_experts
    model.num_local_physical_experts = example.n_local_physical_experts
    model.num_routed_experts = example.n_routed_experts
    model.num_shared_experts = example.n_shared_experts
    model.num_redundant_experts = example.n_redundant_experts
    # MixtureOfExperts is a runtime Protocol: isinstance() requires every
    # declared member; set_eplb_state() fills this in.
    model.expert_weights = []


def _update_physical_experts_metadata(
    self, num_physical_experts: int, num_local_physical_experts: int
) -> None:
    assert self.num_local_physical_experts == num_local_physical_experts
    self.num_physical_experts = num_physical_experts
    self.num_redundant_experts = num_physical_experts - self.num_logical_experts
    for moe in self.moe_mlp_layers:
        moe.n_physical_experts = num_physical_experts
        moe.n_redundant_experts = self.num_redundant_experts
        moe.experts.update_expert_map()


def apply_to_module(module: ModuleType) -> bool:
    amd_model = load_exact_module(TARGET_MODULE, module)
    moe_cls = require_class(amd_model, "DeepseekV4MoE", f"{TARGET_MODULE}.DeepseekV4MoE")
    model_cls = require_class(
        amd_model, "DeepseekV4Model", f"{TARGET_MODULE}.DeepseekV4Model"
    )
    lm_cls = require_class(
        amd_model, "DeepseekV4ForCausalLM", f"{TARGET_MODULE}.DeepseekV4ForCausalLM"
    )
    if getattr(amd_model, _MARKER, False):
        for owner, name, target in (
            (amd_model, "FusedMoEFactory", TARGETS[0]),
            (moe_cls, "__init__", TARGETS[1]),
            (model_cls, "get_expert_mapping", TARGETS[2]),
            (lm_cls, "__init__", TARGETS[3]),
        ):
            if not getattr(require_callable(owner, name, target), _WRAPPER, False):
                raise PatchCompatibilityError(
                    f"required HCU patch marker for {target} is stale; restart the process"
                )
        return False

    factory = require_callable(amd_model, "FusedMoEFactory", TARGETS[0])
    moe_init = require_callable(moe_cls, "__init__", TARGETS[1])
    get_expert_mapping = require_callable(model_cls, "get_expert_mapping", TARGETS[2])
    require_exact_signature(get_expert_mapping, TARGETS[2], positional=("self",))
    lm_init = require_callable(lm_cls, "__init__", TARGETS[3])
    make_mapping = require_callable(
        amd_model,
        "fused_moe_make_expert_params_mapping",
        f"{TARGET_MODULE}.fused_moe_make_expert_params_mapping",
    )
    from vllm.model_executor.models.interfaces import MixtureOfExperts

    @functools.wraps(factory)
    def hcu_factory(*args, **kwargs):
        enable_eplb, num_redundant = _eplb_config()
        if enable_eplb:
            if kwargs.get("fuse_shared_experts"):
                raise ValueError(
                    "DeepSeek-V4 EPLB does not support fused shared experts"
                )
            kwargs.setdefault("enable_eplb", True)
            kwargs.setdefault("num_redundant_experts", num_redundant)
        return factory(*args, **kwargs)

    @functools.wraps(moe_init)
    def hcu_moe_init(self, *args, **kwargs):
        moe_init(self, *args, **kwargs)
        enable_eplb, num_redundant = _eplb_config()
        if not enable_eplb:
            return
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = num_redundant
        self.n_physical_experts = self.n_logical_experts + num_redundant
        ep_size = self.experts.moe_config.ep_size
        if self.n_physical_experts % ep_size != 0:
            raise ValueError(
                f"DeepSeek-V4 EPLB needs (n_routed_experts + num_redundant_experts)"
                f"={self.n_physical_experts} divisible by EP size {ep_size}"
            )
        self.n_local_physical_experts = self.n_physical_experts // ep_size
        from vllm.config import get_current_vllm_config

        hf_config = get_current_vllm_config().model_config.hf_config
        self.n_shared_experts = getattr(hf_config, "n_shared_experts", 0) or 0

    @functools.wraps(get_expert_mapping)
    def hcu_get_expert_mapping(self):
        enable_eplb, num_redundant = _instance_eplb_config(self)
        if not enable_eplb or num_redundant == 0:
            return get_expert_mapping(self)
        # EPLB disables shared-expert fusion, so only routed experts remain.
        return make_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=num_redundant,
        )

    @functools.wraps(lm_init)
    def hcu_lm_init(self, *args, **kwargs):
        lm_init(self, *args, **kwargs)
        enable_eplb, _ = _capture_eplb_config(
            self,
            getattr(self, "model", None),
        )
        if enable_eplb:
            _collect_moe(self)

    for function in (
        hcu_factory,
        hcu_moe_init,
        hcu_get_expert_mapping,
        hcu_lm_init,
    ):
        setattr(function, _WRAPPER, True)
    amd_model.FusedMoEFactory = hcu_factory
    moe_cls.__init__ = hcu_moe_init
    model_cls.get_expert_mapping = hcu_get_expert_mapping
    lm_cls.__init__ = hcu_lm_init
    # Methods are class-level; the data attributes exist only when EPLB is on,
    # so ``is_mixture_of_experts`` stays False with EPLB disabled.
    lm_cls._vllm_hcu_moe_cls = moe_cls
    lm_cls.set_eplb_state = MixtureOfExperts.set_eplb_state
    lm_cls.update_physical_experts_metadata = _update_physical_experts_metadata
    setattr(amd_model, _MARKER, True)
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
