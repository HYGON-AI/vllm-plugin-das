# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Shared adapter for AITER-owned MoE solution selection and execution."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch

_CACHE_LIMIT = 8
_SELECTION_CACHE_ATTR = "_hcu_aiter_moe_selection_cache"
_WEIGHT_CACHE_ATTR = "_hcu_aiter_moe_weight_cache"
_SCALE_CACHE_ATTR = "_hcu_aiter_moe_scale_cache"


class HcuAiterMoeDispatchError(RuntimeError):
    """The AITER MoE API violated the adapter contract."""


@dataclass(frozen=True)
class AiterMoeProblem:
    """Shape and model metadata used by AITER to select an MoE solution."""

    M: int
    E: int
    N1: int
    N2: int
    K: int
    top_k: int
    block_size: int
    dtype: torch.dtype
    device: torch.device
    quant_type: object
    activation: str = "silu"
    use_shuffle: bool = True

    def describe(self) -> str:
        return (
            f"M={self.M}, E={self.E}, N1={self.N1}, N2={self.N2}, "
            f"K={self.K}, top_k={self.top_k}, block_size={self.block_size}, "
            f"dtype={self.dtype}, device={self.device}, "
            f"quant_type={self.quant_type!r}, activation={self.activation!r}, "
            f"use_shuffle={self.use_shuffle}"
        )


def _owner_cache(owner: object, name: str) -> OrderedDict[Any, Any]:
    cache = getattr(owner, name, None)
    if isinstance(cache, OrderedDict):
        return cache
    cache = OrderedDict()
    try:
        setattr(owner, name, cache)
    except (AttributeError, TypeError):
        # Some short-lived test or compatibility owners do not allow dynamic
        # attributes. Correctness does not depend on caching in that case.
        pass
    return cache


def _cache_put(cache: OrderedDict[Any, Any], key: Any, value: Any) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_LIMIT:
        cache.popitem(last=False)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    enum_value = getattr(value, "value", value)
    try:
        hash(enum_value)
    except TypeError:
        return repr(enum_value)
    return enum_value


def _solution_token(config: object) -> str:
    solution = getattr(config, "solution_type", None)
    solution = getattr(solution, "value", solution)
    return str(solution).rsplit(".", 1)[-1].upper()


def _config_generation(config: object) -> tuple[Any, ...]:
    return (
        _freeze(getattr(config, "quant_type", None)),
        _solution_token(config),
        bool(getattr(config, "need_shuffle", False)),
        bool(getattr(config, "need_shuffle_scale", False)),
        _freeze(getattr(config, "config", None)),
    )


def _tensor_generation(tensor: torch.Tensor | None) -> tuple[Any, ...] | None:
    if tensor is None:
        return None
    return (
        id(tensor),
        tensor.data_ptr(),
        tensor._version,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _validate_derived_tensor(
    original: torch.Tensor | None,
    derived: object,
    *,
    label: str,
    config: object,
) -> torch.Tensor | None:
    if original is None:
        if derived is not None:
            raise HcuAiterMoeDispatchError(
                f"AITER returned {label} although the input was None; "
                f"solution={_solution_token(config)}"
            )
        return None
    if not isinstance(derived, torch.Tensor):
        raise HcuAiterMoeDispatchError(
            f"AITER returned a non-tensor {label}; "
            f"solution={_solution_token(config)}"
        )
    if derived.shape != original.shape:
        raise HcuAiterMoeDispatchError(
            f"AITER returned incompatible {label} shape {tuple(derived.shape)} "
            f"for {tuple(original.shape)}; solution={_solution_token(config)}"
        )
    return derived


def select_aiter_moe_config(
    problem: AiterMoeProblem,
    cache_owner: object,
) -> object | None:
    """Ask AITER to route the problem, preserving explicit no-solution status."""

    cache = _owner_cache(cache_owner, _SELECTION_CACHE_ATTR)
    if problem in cache:
        cache.move_to_end(problem)
        return cache[problem]

    moe_module = import_module("aiter.moe")
    selector = getattr(moe_module, "get_aiter_moe_config", None)
    if not callable(selector):
        raise HcuAiterMoeDispatchError(
            "aiter.moe exposes no callable get_aiter_moe_config; "
            + problem.describe()
        )

    result = selector(
        M=problem.M,
        E=problem.E,
        N1=problem.N1,
        N2=problem.N2,
        K=problem.K,
        top_k=problem.top_k,
        block_size=problem.block_size,
        dtype=problem.dtype,
        quant_type=problem.quant_type,
        activation=problem.activation,
        use_shuffle=int(problem.use_shuffle),
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise HcuAiterMoeDispatchError(
            "get_aiter_moe_config must return (status, config); "
            + problem.describe()
        )
    status, config = result
    if not status:
        _cache_put(cache, problem, None)
        return None
    if config is None:
        raise HcuAiterMoeDispatchError(
            "get_aiter_moe_config returned status=True without a config; "
            + problem.describe()
        )

    _cache_put(cache, problem, config)
    return config


def prepare_aiter_moe_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    config: object,
    cache_owner: object,
    block_shape: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive and cache the weight layout requested by an AITER config."""

    if not bool(getattr(config, "need_shuffle", False)):
        return w1, w2

    cache = _owner_cache(cache_owner, _WEIGHT_CACHE_ATTR)
    cache_key = (
        _tensor_generation(w1),
        _tensor_generation(w2),
        _config_generation(config),
        _freeze(block_shape),
    )
    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]

    moe_module = import_module("aiter.moe")
    shuffle = getattr(moe_module, "aiter_moe_shfl_weight", None)
    if not callable(shuffle):
        raise HcuAiterMoeDispatchError(
            "aiter.moe exposes no callable aiter_moe_shfl_weight; "
            f"solution={_solution_token(config)}"
        )
    with torch.no_grad():
        if block_shape is None:
            derived_w1, derived_w2 = shuffle(w1, w2, config)
        else:
            derived_w1, derived_w2 = shuffle(
                w1,
                w2,
                config,
                block_shape=block_shape,
            )
    validated_w1 = _validate_derived_tensor(
        w1,
        derived_w1,
        label="w1",
        config=config,
    )
    validated_w2 = _validate_derived_tensor(
        w2,
        derived_w2,
        label="w2",
        config=config,
    )
    assert validated_w1 is not None and validated_w2 is not None
    result = (validated_w1, validated_w2)
    _cache_put(cache, cache_key, result)
    return result


def prepare_aiter_moe_scales(
    scale1: torch.Tensor | None,
    scale2: torch.Tensor | None,
    config: object,
    cache_owner: object,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Derive and cache the scale layout requested by an AITER config."""

    if not bool(getattr(config, "need_shuffle_scale", False)):
        return scale1, scale2

    cache = _owner_cache(cache_owner, _SCALE_CACHE_ATTR)
    cache_key = (
        _tensor_generation(scale1),
        _tensor_generation(scale2),
        _config_generation(config),
    )
    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]

    moe_module = import_module("aiter.moe")
    shuffle = getattr(moe_module, "aiter_moe_shfl_scale", None)
    if not callable(shuffle):
        raise HcuAiterMoeDispatchError(
            "aiter.moe exposes no callable aiter_moe_shfl_scale; "
            f"solution={_solution_token(config)}"
        )
    with torch.no_grad():
        derived_scale1, derived_scale2 = shuffle(scale1, scale2, config)
    result = (
        _validate_derived_tensor(
            scale1,
            derived_scale1,
            label="w1_scale",
            config=config,
        ),
        _validate_derived_tensor(
            scale2,
            derived_scale2,
            label="w2_scale",
            config=config,
        ),
    )
    _cache_put(cache, cache_key, result)
    return result


def aiter_expert_map_for_solution(
    expert_map: torch.Tensor | None,
    config: object,
    global_num_experts: int,
) -> torch.Tensor | None:
    """Convert vLLM's expert map only for AITER ASM's mask ABI."""

    if expert_map is None or _solution_token(config) != "ASM":
        return expert_map
    if expert_map.ndim != 1 or expert_map.dtype not in (torch.int32, torch.int64):
        raise HcuAiterMoeDispatchError(
            "AITER ASM requires a rank-1 integer expert_map"
        )
    if global_num_experts > 0 and expert_map.numel() != global_num_experts:
        raise HcuAiterMoeDispatchError(
            "AITER ASM expert_map does not match global_num_experts: "
            f"{expert_map.numel()} != {global_num_experts}"
        )
    return (expert_map >= 0).to(torch.int32)


def execute_aiter_moe(
    config: object,
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    routed_scaling_factor: float | None = 1.0,
    use_weight_shuffle: bool = False,
    output_dtype: torch.dtype | None = None,
    gemm1_alpha: float | None = None,
    gemm1_limit: float | None = None,
) -> torch.Tensor:
    """Execute the selected config through AITER's public MoE entry point."""

    moe_module = import_module("aiter.moe")
    operation = getattr(moe_module, "aiter_moe", None)
    if not callable(operation):
        raise HcuAiterMoeDispatchError(
            "aiter.moe exposes no callable aiter_moe; "
            f"solution={_solution_token(config)}"
        )
    result = operation(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        moe_config=config,
        inplace=inplace,
        activation=activation,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        routed_scaling_factor=routed_scaling_factor,
        use_weight_shuffle=use_weight_shuffle,
        output_dtype=output_dtype,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
    )
    if not isinstance(result, torch.Tensor):
        raise HcuAiterMoeDispatchError(
            "aiter_moe returned a non-tensor result; "
            f"solution={_solution_token(config)}"
        )
    return result
