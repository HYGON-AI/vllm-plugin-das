# SPDX-License-Identifier: Apache-2.0
"""Registered private HCU host storage for source-owned Engram tables.

Registration lifetime follows numpy-backed Torch storage (including Parameters
and native UVA views). Never substitute a pageable host address on failure.
"""
from __future__ import annotations

import ctypes
import ctypes.util
from functools import lru_cache
import logging
import mmap
import weakref
import numpy as np
import torch

logger = logging.getLogger(__name__)


class _HipRuntime:
    def __init__(self):
        path = ctypes.util.find_library('amdhip64') or 'libamdhip64.so'
        self.lib = ctypes.CDLL(path)
        self._register = self.lib.hipHostRegister
        self._register.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint)
        self._register.restype = ctypes.c_int
        self._unregister = self.lib.hipHostUnregister
        self._unregister.argtypes = (ctypes.c_void_p,)
        self._unregister.restype = ctypes.c_int
        self._pointer = self.lib.hipHostGetDevicePointer
        self._pointer.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint)
        self._pointer.restype = ctypes.c_int

    def register(self, pointer, nbytes):
        # hipHostRegisterMapped exposes the mapping through the native UVA op.
        result = self._register(pointer, nbytes, 2)
        if result:
            raise RuntimeError(f'Engram hipHostRegister failed: code={result}, bytes={nbytes}')

    def unregister(self, pointer):
        result = self._unregister(pointer)
        if result:
            raise RuntimeError(f'Engram hipHostUnregister failed: code={result}')

    def device_pointer(self, pointer):
        device = ctypes.c_void_p()
        result = self._pointer(ctypes.byref(device), pointer, 0)
        if result or not device.value:
            raise RuntimeError(f'Engram hipHostGetDevicePointer failed: code={result}')
        return device.value


@lru_cache(maxsize=1)
def runtime():
    """Load HIP only when allocating a non-meta production table."""
    return _HipRuntime()


def _release(mapping, pointer, hip):
    # Mapping is deliberately captured: unregister before its last reference is
    # released. numpy itself may be finalizing, so do not force mmap.close().
    try:
        hip.unregister(pointer)
    except Exception:
        logger.exception('HCU Engram host unregister failed')


def allocate_registered(nbytes):
    if not isinstance(nbytes, int) or nbytes <= 0:
        raise ValueError('HCU Engram storage requires a positive byte count')
    hip = runtime()
    mapping = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    owner = np.frombuffer(mapping, dtype=np.uint8)
    tensor = torch.from_numpy(owner)
    pointer = tensor.data_ptr()
    if pointer % mmap.PAGESIZE:
        raise RuntimeError('HCU Engram host mapping is not page aligned')
    hip.register(pointer, nbytes)
    try:
        if not tensor.is_pinned():
            raise RuntimeError('HCU did not recognize registered Engram memory as pinned')
        if not hip.device_pointer(pointer):
            raise RuntimeError('HCU Engram host memory has no device pointer')
    except BaseException:
        hip.unregister(pointer)
        raise
    finalizer = weakref.finalize(owner, _release, mapping, pointer, hip)
    finalizer.atexit = False
    return tensor


def allocate_table(table):
    """Return native FP8/E8M0 views of one registered private allocation."""
    scale_dim = table.dim // table.block_size
    weight_bytes = table.part_num_embeddings * table.dim
    table._packed = allocate_registered(weight_bytes + table.part_num_embeddings * scale_dim)
    weight = table._packed[:weight_bytes].view(torch.float8_e4m3fn).view(-1, table.dim)
    scales = table._packed[weight_bytes:].view(-1, scale_dim)
    table._hcu_storage = (weight.data_ptr(), scales.data_ptr(), tuple(weight.shape), tuple(scales.shape))
    return weight, scales


def check_table_storage(table):
    if table.weight.is_meta:
        raise RuntimeError('HCU Engram cannot look up meta storage')
    actual = (table.weight.data_ptr(), table.weight_scale_inv.data_ptr(),
              tuple(table.weight.shape), tuple(table.weight_scale_inv.shape))
    if actual != table._hcu_storage:
        raise RuntimeError('HCU Engram registered weight/scale storage was replaced')
