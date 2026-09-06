# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Bind Qwen GDN's captured RMSNormGated class to the HCU OOT class."""

from __future__ import annotations

from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_class,
    require_exact_signature,
)


TARGET_MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
PATCH_ID = "worker.op_opt.mamba.gdn.rms_norm_gated"
TARGETS = (f"{TARGET_MODULE}.RMSNormGated",)
_MARKER = "_vllm_hcu_gdn_rms_norm_gated_applied"


def apply_to_module(module: ModuleType) -> bool:
    qwen = load_exact_module(TARGET_MODULE, module)

    from vllm.model_executor.layers.layernorm import RMSNormGated
    captured = require_class(qwen, "RMSNormGated", TARGETS[0])
    if getattr(qwen, _MARKER, False):
        from vllm_hcu.ops.rms_norm_gated import HcuRMSNormGated

        if captured is not HcuRMSNormGated:
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGETS[0]} is stale; "
                "restart the process"
            )
        return False
    if captured is not RMSNormGated:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} is not the canonical "
            "vLLM RMSNormGated class"
        )
    require_exact_signature(
        RMSNormGated.__init__,
        f"{TARGETS[0]}.__init__",
        positional=(
            "self",
            "hidden_size",
            "eps",
            "group_size",
            "norm_before_gate",
            "device",
            "dtype",
            "activation",
        ),
        defaults={
            "eps": 1e-5,
            "group_size": None,
            "norm_before_gate": False,
            "device": None,
            "dtype": None,
            "activation": "swish",
        },
    )
    require_exact_signature(
        RMSNormGated.forward_cuda,
        f"{TARGETS[0]}.forward_cuda",
        positional=("self", "x", "z"),
        defaults={"z": None},
    )

    from vllm_hcu.ops.rms_norm_gated import HcuRMSNormGated

    qwen._vllm_hcu_original_rms_norm_gated = captured
    qwen.RMSNormGated = HcuRMSNormGated
    setattr(qwen, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
