# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Avoid cold ``hipHostMalloc`` in ``async_tensor_h2d`` on HCU.

Upstream stages every unpinned H2D with a fresh
``torch.empty(..., pin_memory=True)``. On HCU the first CachingHostAllocator
slab costs ~30ms and leaves a GPU bubble before ``expand_idx_mapping``.

Mitigations:

1. Round-robin registered (``cudaHostRegister``) staging pools so the hot
   path does not call ``hipHostMalloc``. Pools grow lazily up to
   ``_POOL_DEPTH`` so outstanding ``non_blocking`` H2Ds do not clobber a
   buffer still owned by DMA.
2. One deferred ``pin_memory`` slab warmup as fallback if register is
   unavailable (still keeps upstream empty semantics on that path).
"""

from __future__ import annotations

import functools
import importlib
from types import ModuleType

import numpy as np
import torch

from ._common import (
    load_exact_module,
    require_callable,
    require_exact_signature,
)
from .patch_hcu_uva_buffer import _allocate_registered_buffer

TARGET_MODULE = "vllm.utils.torch_utils"
PATCH_ID = "worker.core_fix.async_tensor_h2d"
TARGETS = (f"{TARGET_MODULE}.async_tensor_h2d",)
_MARKER = "_vllm_hcu_async_tensor_h2d_applied"
_WRAPPER = "_vllm_hcu_async_tensor_h2d_wrapper"

# Hot decode / prepare importers that bind the symbol by name.
_REBIND_MODULES = (
    "vllm.utils.torch_utils",
    "vllm.v1.worker.gpu.model_runner",
    "vllm.v1.worker.gpu.buffer_utils",
    "vllm.v1.worker.gpu.pcp_manager",
    "vllm.v1.worker.gpu.pp_utils",
    "vllm.v1.worker.gpu.structured_outputs",
    "vllm.v1.worker.gpu.spec_decode.adaptive_verification",
    "vllm.v1.worker.gpu.sample.penalties",
    "vllm.v1.worker.gpu.sample.thinking_budget",
    "vllm.v1.worker.utils",
    "vllm.v1.attention.backends.utils",
    "vllm.v1.attention.backends.mla.indexer",
)

class _StagingOwner:
    """Must support weakref for ``cudaHostRegister`` finalizers."""


_STAGING_OWNER = _StagingOwner()
_STAGING_POOLS: dict[tuple[torch.dtype, int], list[torch.Tensor]] = {}
_STAGING_CURSORS: dict[tuple[torch.dtype, int], int] = {}
_STAGING_MAPPINGS: list[object] = []
_POOL_DEPTH = 64
_REGISTER_OK = True
_PIN_FALLBACK_WARMED = False


def _rebind(name: str, value) -> None:
    for mod_name in _REBIND_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if hasattr(mod, name):
            setattr(mod, name, value)


def _capacity(numel: int) -> int:
    return max(256, 1 << max(numel - 1, 0).bit_length())


def _get_staging(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor | None:
    global _REGISTER_OK
    if not _REGISTER_OK:
        return None
    numel = 1
    for dim in shape:
        numel *= int(dim)
    cap = _capacity(numel)
    key = (dtype, cap)
    pool = _STAGING_POOLS.setdefault(key, [])
    cursor = _STAGING_CURSORS.get(key, 0)
    if len(pool) < _POOL_DEPTH:
        try:
            raw, mapping = _allocate_registered_buffer(cap, dtype, _STAGING_OWNER)
        except Exception:
            _REGISTER_OK = False
            return None
        _STAGING_MAPPINGS.append(mapping)
        pool.append(raw)
        # Prefer the freshly allocated slot so early calls never alias.
        _STAGING_CURSORS[key] = len(pool)
        return raw[:numel].view(shape)
    buf = pool[cursor % len(pool)]
    _STAGING_CURSORS[key] = cursor + 1
    return buf[:numel].view(shape)


def _warm_pin_fallback_slabs() -> None:
    """Touch CachingHostAllocator only when registered staging is unavailable."""

    global _PIN_FALLBACK_WARMED
    if _PIN_FALLBACK_WARMED:
        return
    _PIN_FALLBACK_WARMED = True
    try:
        for dtype in (torch.int32, torch.int64):
            buf = torch.empty(2 * 1024 * 1024, dtype=dtype, pin_memory=True)
            del buf
    except Exception:
        return


def _stage_unpinned(
    t: torch.Tensor, dtype: torch.dtype, pin_memory: bool
) -> torch.Tensor:
    staging = _get_staging(tuple(t.shape), dtype)
    if staging is not None:
        if t.dtype == dtype:
            staging.copy_(t)
        else:
            staging.copy_(t.to(dtype=dtype))
        return staging
    if pin_memory:
        _warm_pin_fallback_slabs()
    out = torch.empty(t.shape, dtype=dtype, device="cpu", pin_memory=pin_memory)
    if t.dtype == dtype:
        out.copy_(t)
    else:
        out.copy_(t.to(dtype=dtype))
    return out


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    original = require_callable(target, "async_tensor_h2d", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("data", "device", "dtype", "out"),
        defaults={"device": None, "dtype": None, "out": None},
    )
    if getattr(original, _MARKER, False):
        return False

    pin_memory = bool(getattr(target, "PIN_MEMORY", False))

    @functools.wraps(original)
    def hcu_async_tensor_h2d(data, device=None, dtype=None, out=None):
        if dtype is None and out is not None:
            dtype = out.dtype
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if isinstance(data, torch.Tensor):
            t = data
            if pin_memory and not t.is_pinned():
                t = _stage_unpinned(t, dtype or t.dtype, pin_memory)
            elif dtype is not None and t.dtype != dtype:
                t = t.to(dtype=dtype)
        else:
            tmp = torch.tensor(data, dtype=dtype, device="cpu")
            if pin_memory:
                t = _stage_unpinned(tmp, tmp.dtype, pin_memory)
            else:
                t = tmp
        assert t.is_cpu

        if out is not None:
            assert out.dtype == dtype
            return out.copy_(t, non_blocking=True)
        assert device is not None, "must provide destination tensor or device"
        return t.to(device=device, dtype=dtype, non_blocking=True)

    setattr(hcu_async_tensor_h2d, _MARKER, True)
    setattr(hcu_async_tensor_h2d, _WRAPPER, True)
    setattr(target, "async_tensor_h2d", hcu_async_tensor_h2d)
    setattr(target, _MARKER, True)
    _rebind("async_tensor_h2d", hcu_async_tensor_h2d)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
