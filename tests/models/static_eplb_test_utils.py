# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import inspect
from importlib import import_module
from typing import Callable

from pytest import MonkeyPatch

from vllm_hcu.patch.worker.op_opt.moe import patch_layer


def _track_direct_patch(
    monkeypatch: MonkeyPatch,
    owner: object,
    name: str,
) -> None:
    """Record an attribute so pytest restores a direct patch-layer mutation."""

    try:
        value = inspect.getattr_static(owner, name)
    except AttributeError:
        monkeypatch.setattr(owner, name, None, raising=False)
    else:
        monkeypatch.setattr(owner, name, value)


def apply_real_moe_layer_patch(
    monkeypatch: MonkeyPatch,
) -> Callable[..., list[tuple[str, str, int, str]]]:
    """Apply the production mapping patch and arrange complete test cleanup."""

    fused_moe = import_module(patch_layer.TARGET_MODULE)
    layer = import_module(patch_layer.LAYER_MODULE)
    if getattr(fused_moe, patch_layer._MARKER, False):
        return fused_moe.fused_moe_make_expert_params_mapping

    routed_experts = fused_moe.RoutedExperts
    tracked: tuple[tuple[object, str], ...] = (
        (fused_moe, "FusedMoE"),
        (fused_moe, patch_layer._MARKER),
        (fused_moe, "_vllm_hcu_original_fused_moe_factory"),
        (layer, "FusedMoE"),
        (layer, "_vllm_hcu_original_fused_moe_factory"),
        (routed_experts, "get_expert_weights"),
        (routed_experts, "load_weights"),
        (routed_experts, "weight_loader"),
        (routed_experts, "get_expert_mapping"),
        (routed_experts, "make_expert_params_mapping"),
        (routed_experts, "_vllm_hcu_original_get_expert_weights"),
        (routed_experts, "_vllm_hcu_original_load_weights"),
        (routed_experts, "_vllm_hcu_original_weight_loader"),
        (routed_experts, "_vllm_hcu_original_get_expert_mapping"),
        (routed_experts, "_vllm_hcu_original_make_expert_params_mapping"),
    )
    for owner, name in tracked:
        _track_direct_patch(monkeypatch, owner, name)

    assert patch_layer.apply_to_module(fused_moe) is True
    return fused_moe.fused_moe_make_expert_params_mapping


__all__ = ["apply_real_moe_layer_patch"]
