# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Register native DeepSeek-V4 MTP for EPLB after its module finishes loading."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
)
from .patch_deepseek_v4_eplb import (
    _capture_eplb_config,
    _collect_moe,
    _instance_eplb_config,
    _update_physical_experts_metadata,
)

TARGET_MODULE = "vllm.models.deepseek_v4.amd.mtp"
PATCH_ID = "worker.core_fix.deepseek_v4_amd.mtp_eplb"
TARGETS = (
    f"{TARGET_MODULE}.DeepSeekV4MTP.__init__",
    f"{TARGET_MODULE}.fused_moe_make_expert_params_mapping",
)
_MARKER = "_vllm_hcu_deepseek_v4_mtp_eplb_applied"
_WRAPPER = "_vllm_hcu_deepseek_v4_mtp_eplb_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    amd_mtp = load_exact_module(TARGET_MODULE, module)
    mtp_cls = require_class(amd_mtp, "DeepSeekV4MTP", TARGETS[0].rsplit(".", 1)[0])
    if getattr(amd_mtp, _MARKER, False):
        for owner, name, target in (
            (mtp_cls, "__init__", TARGETS[0]),
            (amd_mtp, "fused_moe_make_expert_params_mapping", TARGETS[1]),
        ):
            if not getattr(require_callable(owner, name, target), _WRAPPER, False):
                raise PatchCompatibilityError(
                    f"required HCU patch marker for {target} is stale; restart the process"
                )
        return False

    mtp_init = require_callable(mtp_cls, "__init__", TARGETS[0])
    mtp_make_mapping = require_callable(
        amd_mtp, "fused_moe_make_expert_params_mapping", TARGETS[1]
    )
    from vllm.model_executor.models.interfaces import MixtureOfExperts
    from vllm.models.deepseek_v4.amd.model import DeepseekV4MoE

    @functools.wraps(mtp_init)
    def hcu_mtp_init(self, *args, **kwargs):
        mtp_init(self, *args, **kwargs)
        enable_eplb, _ = _capture_eplb_config(
            self,
            getattr(self, "model", None),
        )
        if enable_eplb:
            _collect_moe(self)

    @functools.wraps(mtp_make_mapping)
    def hcu_mtp_make_mapping(*args, **kwargs):
        model = args[0] if args else kwargs.get("model")
        enable_eplb, num_redundant = _instance_eplb_config(model)
        if enable_eplb:
            if getattr(model, "_vllm_hcu_static_eplb_plan", None) is None:
                kwargs.setdefault("num_redundant_experts", num_redundant)
            else:
                # Static checkpoint mappings contain logical expert IDs only.
                # The bound static loaders fan each logical weight out to every
                # physical replica selected by the placement plan.
                kwargs.pop("num_redundant_experts", None)
        return mtp_make_mapping(*args, **kwargs)

    for function in (hcu_mtp_init, hcu_mtp_make_mapping):
        setattr(function, _WRAPPER, True)
    mtp_cls.__init__ = hcu_mtp_init
    amd_mtp.fused_moe_make_expert_params_mapping = hcu_mtp_make_mapping
    mtp_cls._vllm_hcu_moe_cls = DeepseekV4MoE
    mtp_cls.set_eplb_state = MixtureOfExperts.set_eplb_state
    mtp_cls.update_physical_experts_metadata = _update_physical_experts_metadata
    setattr(amd_mtp, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
