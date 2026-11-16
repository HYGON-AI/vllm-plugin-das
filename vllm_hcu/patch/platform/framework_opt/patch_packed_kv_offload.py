# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Preserve packed KV descriptor fields through CPU offload sizing."""

from __future__ import annotations

import dataclasses
import functools
import inspect
from types import ModuleType
from typing import Any

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
)


TARGET_MODULE = "vllm.v1.simple_kv_offload.manager"
PATCH_ID = "platform.framework_opt.packed_kv_offload"
TARGETS = (f"{TARGET_MODULE}.SimpleCPUOffloadScheduler._derive_cpu_config",)
_MARKER = "_vllm_hcu_packed_kv_offload_applied"
_WRAPPER = "_vllm_hcu_packed_kv_offload_wrapper"


def _layer_names(tensor: Any) -> list[str]:
    names = getattr(tensor, "shared_by", None)
    if names is None:
        names = getattr(tensor, "layers", None)
    if not names:
        raise ValueError("KV cache tensor must name at least one layer")
    return list(names)


def _is_packed(tensor: Any) -> bool:
    """HCU packed descriptors always carry ``packed_kv_backing_id``.

    Native ``KVCacheTensor`` also has a positive ``block_stride`` (bytes
    between manager blocks). Treating that as packed makes PCP=1 / layer-split
    off raise when several native descriptors lack a backing id. Only the
    explicit HCU identity enters the packed export path.
    """
    return _packed_backing_id(tensor) is not None


def _packed_backing_id(tensor: Any) -> Any | None:
    from vllm_hcu.v1.dsa_cache_layer_split import packed_kv_backing_id

    return packed_kv_backing_id(tensor)


def _hashable_backing_id(value: Any) -> Any:
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(
            "Packed KV backing_id must be hashable; refusing ambiguous "
            "CPU offload sizing"
        ) from exc
    return value


def _packed_pool_keys(tensors: list[Any]) -> dict[int, Any]:
    """Return one explicit allocation key per packed descriptor pool.

    HCU packed descriptors must carry an identity because independent pools
    can have equal ``size`` and ``block_stride``.  A single legacy descriptor
    remains unambiguous; multiple legacy descriptors fail closed rather than
    silently aliasing unrelated allocations.
    """
    packed = [tensor for tensor in tensors if _is_packed(tensor)]
    if not packed:
        return {}
    explicit = [_packed_backing_id(tensor) for tensor in packed]
    missing = [index for index, value in enumerate(explicit) if value is None]
    if missing and (len(packed) > 1 or any(value is not None for value in explicit)):
        raise ValueError(
            "Packed KV descriptors must carry explicit backing_id metadata "
            "when more than one packed descriptor is present"
        )
    result: dict[int, Any] = {}
    for index, value in enumerate(explicit):
        result[index] = (
            ("packed", _hashable_backing_id(value))
            if value is not None
            else ("legacy-single-packed",)
        )
    return result


def _scaled_tensor(tensor: Any, num_gpu_blocks: int, num_cpu_blocks):
    size = int(tensor.size) // num_gpu_blocks * num_cpu_blocks
    names = _layer_names(tensor)
    fields = getattr(type(tensor), "__dataclass_fields__", {})
    packed_fields = {"block_stride", "offset"}
    backing_id = _packed_backing_id(tensor)
    if _is_packed(tensor) and (
        not packed_fields.issubset(fields)
        or (backing_id is not None and "backing_id" not in fields)
    ):
        # A dynamically augmented stock KVCacheTensor cannot retain HCU's
        # addressing fields through dataclasses.replace().  Convert it to the
        # HCU descriptor rather than silently producing a flat CPU view.
        from vllm_hcu.v1.dsa_cache_layer_split import make_packed_kv_tensor

        return make_packed_kv_tensor(
            type(tensor),
            size=size,
            layer_names=names,
            page_size=int(getattr(tensor, "layer_stride", 0) or 0),
            offset=int(getattr(tensor, "offset", 0) or 0),
            block_stride=int(getattr(tensor, "block_stride", 0) or 0),
            backing_id=_packed_backing_id(tensor),
        )
    try:
        replace_kwargs: dict[str, Any] = {"size": size}
        if "shared_by" in fields:
            replace_kwargs["shared_by"] = list(names)
        elif "layers" in fields:
            replace_kwargs["layers"] = list(names)
        return dataclasses.replace(tensor, **replace_kwargs)
    except (TypeError, ValueError):
        kwargs = {
            name: getattr(tensor, name)
            for name in fields
            if name != "size" and hasattr(tensor, name)
        }
        if "shared_by" in fields:
            kwargs["shared_by"] = list(names)
        elif "layers" in fields:
            kwargs["layers"] = list(names)
        kwargs["size"] = size
        return type(tensor)(**kwargs)


def apply_to_module(module: ModuleType) -> bool:
    manager = load_exact_module(TARGET_MODULE, module)
    scheduler = getattr(manager, "SimpleCPUOffloadScheduler", None)
    if scheduler is None:
        raise PatchCompatibilityError(f"missing {TARGETS[0].rsplit('.', 1)[0]}")
    original = require_callable(
        scheduler,
        "_derive_cpu_config",
        TARGETS[0],
    )
    if already_applied(
        scheduler,
        _MARKER,
        ((scheduler, "_derive_cpu_config", _WRAPPER),),
    ):
        return False
    signature = inspect.signature(original)
    if tuple(signature.parameters) != ("gpu_config", "cpu_capacity_bytes"):
        raise PatchCompatibilityError(
            f"required HCU patch target has incompatible signature {signature}"
        )

    @functools.wraps(original)
    def hcu_derive_cpu_config(gpu_config, cpu_capacity_bytes):
        tensors = list(gpu_config.kv_cache_tensors)
        if not tensors or gpu_config.num_blocks <= 0:
            return original(gpu_config, cpu_capacity_bytes)

        # Packed descriptors are multiple views into one backing allocation;
        # count each explicitly identified allocation once while retaining
        # ordinary descriptors as independent allocations.  Never infer the
        # identity from numeric size alone: equal-sized pools are independent
        # unless the planner marked them as aliases.
        packed_keys = _packed_pool_keys(tensors)
        if not packed_keys:
            return original(gpu_config, cpu_capacity_bytes)
        packed_sizes: dict[Any, int] = {}
        packed_index = 0
        gpu_total_bytes = 0
        for tensor in tensors:
            if not _is_packed(tensor):
                gpu_total_bytes += int(tensor.size)
                continue
            key = packed_keys[packed_index]
            packed_index += 1
            size = int(tensor.size)
            previous = packed_sizes.get(key)
            if previous is not None and previous != size:
                raise ValueError(
                    "Packed KV descriptors sharing backing_id disagree on size"
                )
            packed_sizes[key] = size
        gpu_total_bytes += sum(packed_sizes.values())
        if gpu_total_bytes <= 0:
            return original(gpu_config, cpu_capacity_bytes)
        num_cpu_blocks = max(
            1,
            int(gpu_config.num_blocks)
            * int(cpu_capacity_bytes)
            // gpu_total_bytes,
        )
        cpu_tensors = [
            _scaled_tensor(tensor, gpu_config.num_blocks, num_cpu_blocks)
            for tensor in tensors
        ]
        try:
            return dataclasses.replace(
                gpu_config,
                num_blocks=num_cpu_blocks,
                kv_cache_tensors=cpu_tensors,
            )
        except (TypeError, ValueError):
            config_type = type(gpu_config)
            fields = getattr(config_type, "__dataclass_fields__", {})
            if fields:
                config_kwargs = {
                    name: getattr(gpu_config, name)
                    for name in fields
                    if hasattr(gpu_config, name)
                }
                config_kwargs.update(
                    num_blocks=num_cpu_blocks,
                    kv_cache_tensors=cpu_tensors,
                )
            else:
                try:
                    parameters = inspect.signature(config_type).parameters
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        "Cannot reconstruct CPU KVCacheConfig while preserving "
                        "packed descriptor metadata"
                    ) from exc
                config_kwargs = {
                    name: getattr(gpu_config, name)
                    for name in parameters
                    if name != "self" and hasattr(gpu_config, name)
                }
                config_kwargs.update(
                    num_blocks=num_cpu_blocks,
                    kv_cache_tensors=cpu_tensors,
                )
            return config_type(**config_kwargs)

    setattr(hcu_derive_cpu_config, _WRAPPER, True)
    setattr(scheduler, "_vllm_hcu_original_derive_cpu_config", original)
    setattr(scheduler, "_derive_cpu_config", staticmethod(hcu_derive_cpu_config))
    setattr(scheduler, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
