# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Tolerate compressed-tensors ``.weight_scale`` naming in DeepSeek V4 loader.

The AMD variant of ``DeepseekV4Model.load_weights`` maps checkpoint scales
through ``.scale$`` -> ``.weight_scale_inv`` and then feeds every renamed
name straight into ``params_dict[name]``.  Compressed-tensors W8A8 FP8 always
registers the linear scale as ``weight_scale`` (never ``weight_scale_inv``),
so any FP8-Channel checkpoint fails at layer 0 with
``KeyError: 'layers.0.attn.fused_wqa_wkv.weight_scale_inv'``.

This runtime patch swaps ``DeepseekV4Model.load_weights`` for a variant that
falls back to the ``.weight_scale`` / ``_weight_scale`` name when the mapped
``_inv`` name is not registered on the module, mirroring the vllm-hcu
compatibility shim.  When the original name resolves normally the shim is a
no-op, so the patch is safe for FP8-block checkpoints as well.
"""

from __future__ import annotations

import functools
import typing
from collections.abc import Callable, Iterable
from types import ModuleType

import torch

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
)

TARGET_MODULE = "vllm.models.deepseek_v4.amd.model"
PATCH_ID = "worker.core_fix.deepseek_v4_amd.load_weights_scale_remap"
TARGET_SYMBOL = f"{TARGET_MODULE}.DeepseekV4Model.load_weights"
_CLASS_MARKER = "_vllm_hcu_deepseek_v4_load_weights_applied"
_WRAPPER_MARKER = "_vllm_hcu_deepseek_v4_load_weights_wrapper"
_STACKED_PARAMS_MAPPING = (
    # (param_name, shard_name, shard_id)
    ("gate_up_proj", "w1", 0),
    ("gate_up_proj", "w3", 1),
    ("attn.fused_wqa_wkv", "attn.wq_a", 0),
    ("attn.fused_wqa_wkv", "attn.wkv", 1),
    ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
    ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
)


def _maybe_remap_scale(param_name: str, params_dict: dict) -> str:
    """Return an alternative registered scale name if the mapped name is absent.

    Compressed-tensors linear schemes register ``weight_scale``; the model's
    weight mapper rewrites checkpoint ``.scale`` to ``.weight_scale_inv``, so
    only the ``_inv`` variant reaches the loader.  When the mapped name is
    missing from the parameter registry, look for the drop-``_inv`` sibling.
    """

    if param_name in params_dict:
        return param_name
    if param_name.endswith(".weight_scale_inv"):
        candidate = param_name[: -len(".weight_scale_inv")] + ".weight_scale"
        if candidate in params_dict:
            return candidate
    if param_name.endswith("_weight_scale_inv"):
        candidate = param_name[: -len("_weight_scale_inv")] + "_weight_scale"
        if candidate in params_dict:
            return candidate
    return param_name


def _load_stacked_parameter(
    self,
    name: str,
    loaded_weight: torch.Tensor,
    params_dict: dict,
    is_pp_missing_parameter: Callable[[str, object], bool],
) -> tuple[bool, str | None]:
    """Load a stacked parameter and report whether the name was handled."""
    if ".experts." in name:
        return False, None

    for param_name, weight_name, shard_id in _STACKED_PARAMS_MAPPING:
        if weight_name not in name:
            continue
        name = name.replace(weight_name, param_name)
        if is_pp_missing_parameter(name, self):
            return True, None

        name = _maybe_remap_scale(name, params_dict)
        param = params_dict[name]
        param.weight_loader(param, loaded_weight, shard_id)
        return True, name

    return False, None


def _load_expert_parameter(
    self,
    name: str,
    loaded_weight: torch.Tensor,
    params_dict: dict,
    expert_mapping: list[tuple],
    is_pp_missing_parameter: Callable[[str, object], bool],
) -> str:
    """Load one expert parameter using the first successful replica mapping."""
    if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
        loaded_weight = loaded_weight.view(torch.uint8)

    name_mapped = name
    for param_name, weight_name, expert_id, expert_shard_id in expert_mapping:
        if weight_name not in name:
            continue
        name_mapped = name.replace(weight_name, param_name)
        if is_pp_missing_parameter(name_mapped, self):
            continue

        name_mapped = _maybe_remap_scale(name_mapped, params_dict)
        param = params_dict[name_mapped]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        success = weight_loader(
            param,
            loaded_weight,
            name_mapped,
            shard_id=expert_shard_id,
            expert_id=expert_id,
            return_success=True,
        )
        if success:
            break

    return name_mapped


def _load_attention_sink(
    self,
    name: str,
    loaded_weight: torch.Tensor,
    params_dict: dict,
    head_rank_start: int,
    head_rank_end: int,
    is_pp_missing_parameter: Callable[[str, object], bool],
) -> str | None:
    """Load the tensor-parallel slice of an attention-sink parameter."""
    if is_pp_missing_parameter(name, self):
        return None

    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
    params_dict[name][: narrow_weight.shape[0]].copy_(narrow_weight)
    return name


def _load_default_parameter(
    self,
    name: str,
    loaded_weight: torch.Tensor,
    params_dict: dict,
    default_weight_loader: Callable,
    is_pp_missing_parameter: Callable[[str, object], bool],
) -> str | None:
    """Load a non-stacked, non-expert parameter."""
    if is_pp_missing_parameter(name, self):
        return None

    name = _maybe_remap_scale(name, params_dict)
    param = params_dict[name]
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded_weight)
    return name


def _hcu_load_weights(
    self,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """DeepseekV4Model.load_weights with the compressed-tensors scale shim.

    Behavior tracks the audited AMD upstream implementation exactly: same
    stacked_params_mapping, same expert-mapping call, same attn_sink narrowing,
    and same default fallback in the ``else`` branch.  The only additions are
    ``_maybe_remap_scale`` lookups immediately before each ``params_dict[name]``.
    """

    # Import here so this patch adapter stays import-free at registration.
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.models.utils import is_pp_missing_parameter

    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()

    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    n_head = self.config.num_attention_heads
    n_local_head = n_head // tp_size
    head_rank_start = n_local_head * tp_rank
    head_rank_end = n_local_head * (tp_rank + 1)

    expert_mapping = self.get_expert_mapping()

    for name, loaded_weight in weights:
        handled, loaded_name = _load_stacked_parameter(
            self,
            name,
            loaded_weight,
            params_dict,
            is_pp_missing_parameter,
        )
        if handled:
            if loaded_name is not None:
                loaded_params.add(loaded_name)
            continue

        if ".experts." in name:
            loaded_name = _load_expert_parameter(
                self,
                name,
                loaded_weight,
                params_dict,
                expert_mapping,
                is_pp_missing_parameter,
            )
        elif "attn_sink" in name:
            loaded_name = _load_attention_sink(
                self,
                name,
                loaded_weight,
                params_dict,
                head_rank_start,
                head_rank_end,
                is_pp_missing_parameter,
            )
        else:
            loaded_name = _load_default_parameter(
                self,
                name,
                loaded_weight,
                params_dict,
                default_weight_loader,
                is_pp_missing_parameter,
            )

        if loaded_name is not None:
            loaded_params.add(loaded_name)

    return loaded_params


def apply_to_module(module: ModuleType) -> bool:
    amd_model = load_exact_module(TARGET_MODULE, module)
    model_class = require_class(
        amd_model,
        "DeepseekV4Model",
        f"{TARGET_MODULE}.DeepseekV4Model",
    )
    original = require_callable(
        model_class,
        "load_weights",
        TARGET_SYMBOL,
    )
    if getattr(model_class, _CLASS_MARKER, False):
        current = vars(model_class).get("load_weights")
        if not getattr(current, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {TARGET_SYMBOL} is stale"
            )
        return False

    @functools.wraps(original)
    def hcu_load_weights(self, weights):
        return _hcu_load_weights(self, weights)

    setattr(hcu_load_weights, _WRAPPER_MARKER, True)
    setattr(model_class, "_vllm_hcu_original_load_weights", original)
    setattr(model_class, "load_weights", hcu_load_weights)
    setattr(model_class, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
