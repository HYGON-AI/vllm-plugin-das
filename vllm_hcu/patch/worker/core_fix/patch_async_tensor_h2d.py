# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Avoid cold ``hipHostMalloc`` in ``async_tensor_h2d`` on HCU.

Upstream stages every unpinned H2D with a fresh
``torch.empty(..., pin_memory=True)``. On HCU the first CachingHostAllocator
slab costs ~30ms and leaves a GPU bubble before ``expand_idx_mapping``.

Mitigations:

1. Round-robin registered (``cudaHostRegister``) staging pools so the hot
   path does not call ``hipHostMalloc``. Pools grow lazily up to
   ``_POOL_DEPTH``.
2. Each slot records a CUDA event after its non_blocking H2D; the slot is
   not overwritten until that event completes (CPU-side synchronize). CUDA
   stream waits alone cannot protect host memory from CPU stores.
3. One deferred ``pin_memory`` slab warmup as fallback if register is
   unavailable (still keeps upstream empty semantics on that path).
"""

from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass
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


@dataclass
class _StagingSlot:
    buf: torch.Tensor
    event: torch.cuda.Event | None = None
    # Device index that last recorded ``event`` (for correct device context).
    event_device: int | None = None


_STAGING_OWNER = _StagingOwner()
_STAGING_POOLS: dict[tuple[torch.dtype, int], list[_StagingSlot]] = {}
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


def _device_index(device: torch.device | str | int | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if getattr(device, "type", None) != "cuda":
        return torch.cuda.current_device()
    idx = device.index
    return torch.cuda.current_device() if idx is None else int(idx)


def _wait_slot_ready(slot: _StagingSlot) -> None:
    """Block CPU until any in-flight H2D from this host slot has finished."""

    ev = slot.event
    if ev is None:
        return
    # Host stores are not ordered by CUDA streams; must sync the event.
    ev.synchronize()


def _record_slot_h2d(slot: _StagingSlot, device: torch.device | str | int | None) -> None:
    """Record completion of the just-enqueued H2D that reads ``slot.buf``."""

    idx = _device_index(device)
    with torch.cuda.device(idx):
        if slot.event is None or slot.event_device != idx:
            slot.event = torch.cuda.Event()
            slot.event_device = idx
        slot.event.record()


def _acquire_staging(
    shape: tuple[int, ...], dtype: torch.dtype
) -> tuple[torch.Tensor, _StagingSlot] | None:
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
        slot = _StagingSlot(buf=raw)
        pool.append(slot)
        # Prefer the freshly allocated slot so early calls never alias.
        _STAGING_CURSORS[key] = len(pool)
        return raw[:numel].view(shape), slot
    slot = pool[cursor % len(pool)]
    _STAGING_CURSORS[key] = cursor + 1
    _wait_slot_ready(slot)
    return slot.buf[:numel].view(shape), slot


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
) -> tuple[torch.Tensor, _StagingSlot | None]:
    acquired = _acquire_staging(tuple(t.shape), dtype)
    if acquired is not None:
        staging, slot = acquired
        if t.dtype == dtype:
            staging.copy_(t)
        else:
            staging.copy_(t.to(dtype=dtype))
        return staging, slot
    if pin_memory:
        _warm_pin_fallback_slabs()
    out = torch.empty(t.shape, dtype=dtype, device="cpu", pin_memory=pin_memory)
    if t.dtype == dtype:
        out.copy_(t)
    else:
        out.copy_(t.to(dtype=dtype))
    return out, None


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
        slot: _StagingSlot | None = None
        if dtype is None and out is not None:
            dtype = out.dtype
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if isinstance(data, torch.Tensor):
            t = data
            if pin_memory and not t.is_pinned():
                t, slot = _stage_unpinned(t, dtype or t.dtype, pin_memory)
            elif dtype is not None and t.dtype != dtype:
                t = t.to(dtype=dtype)
        else:
            tmp = torch.tensor(data, dtype=dtype, device="cpu")
            if pin_memory:
                t, slot = _stage_unpinned(tmp, tmp.dtype, pin_memory)
            else:
                t = tmp
        assert t.is_cpu

        if out is not None:
            assert out.dtype == dtype
            result = out.copy_(t, non_blocking=True)
            if slot is not None:
                _record_slot_h2d(slot, out.device)
            return result
        assert device is not None, "must provide destination tensor or device"
        result = t.to(device=device, dtype=dtype, non_blocking=True)
        if slot is not None:
            _record_slot_h2d(slot, device)
        return result

    setattr(hcu_async_tensor_h2d, _MARKER, True)
    setattr(hcu_async_tensor_h2d, _WRAPPER, True)
    setattr(target, "async_tensor_h2d", hcu_async_tensor_h2d)
    setattr(target, _MARKER, True)
    _rebind("async_tensor_h2d", hcu_async_tensor_h2d)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
