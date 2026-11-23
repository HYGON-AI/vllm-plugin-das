# SPDX-License-Identifier: Apache-2.0
"""Opt-in Mooncake Fabric pool for HCU KV backing allocation.

Importing this module does not import Mooncake or initialize a device. The
worker retains the pool and pluggable allocator until its KV caches are gone.
Transport selection remains the responsibility of the existing connector.
"""

import atexit
import ctypes
import gc
import logging
import os
import re
from contextlib import contextmanager

logger = logging.getLogger(__name__)
_SWITCH = "VLLM_HCU_MOONCAKE_FABRIC_MEM_POOL"


def validate_fabric_pool_config(config):
    """Fail early for unsupported combinations, only on the opt-in path."""
    transfer = getattr(config, "kv_transfer_config", None)
    if (
        transfer is None
        or transfer.kv_connector != "MooncakeConnector"
        or transfer.kv_role not in ("kv_producer", "kv_consumer", "kv_both")
    ):
        raise ValueError(f"{_SWITCH}=1 requires the MooncakeConnector.")
    model = config.model_config
    if getattr(model, "enable_sleep_mode", False) or getattr(
        model, "enable_cumem_allocator", False
    ):
        raise ValueError(
            f"{_SWITCH}=1 cannot be combined with sleep mode or the CuMem "
            "allocator: Fabric KV buffers are not managed by CuMem."
        )
    for name in (
        "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF"
    ):
        if re.search(r"expandable_segments\s*:\s*true", os.getenv(name, ""), re.I):
            raise ValueError(
                f"{_SWITCH}=1 requires expandable_segments:False in {name}."
            )


class MooncakeFabricPool:
    """Own a device-specific MemPool without changing the global allocator."""

    def __init__(self, device):
        import torch

        try:
            from mooncake.allocator import MemoryBackend, NVLinkAllocator
        except ImportError as exc:
            raise RuntimeError(
                f"{_SWITCH}=1 requires a Mooncake wheel with NVLinkAllocator "
                "and DTK Fabric support."
            ) from exc

        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("Mooncake Fabric pool requires an explicit cuda:N device.")
        self._torch = torch
        self._allocator = None
        self._pool = None

        with torch.cuda.device(self.device):
            # NVLinkAllocator.detect_mem_backend() in the deployed wheels
            # caches a probe of device 0. Probe THIS worker's visible device
            # instead, and reject cudaMalloc fallback/unknown/unsupported.
            library = ctypes.CDLL(NVLinkAllocator._get_so_path())
            try:
                probe = library.mc_allocator_probe
            except AttributeError as exc:
                raise RuntimeError("Mooncake allocator has no Fabric probe symbol.") from exc
            probe.argtypes = [ctypes.c_int]
            probe.restype = ctypes.c_int
            backend = probe(self.device.index)
            if backend != int(MemoryBackend.USE_CUMEMCREATE):
                raise RuntimeError(
                    f"Mooncake Fabric allocation is unavailable on {self.device} "
                    f"(probe={backend}); refusing ordinary-memory fallback. "
                    "Check the Hylink wheel and Fabric DTK, or disable " + _SWITCH
                )
            self._allocator = NVLinkAllocator.get_allocator(self.device)
            self._pool = torch.cuda.MemPool(
                self._allocator.allocator(), use_on_oom=False
            )

        # Keep the wrapper alive through MemPool teardown, including an early
        # worker exit that skips shutdown (PyTorch pluggable-allocator lifetime).
        atexit.register(self.close)
        logger.info(
            "Mooncake Fabric KV memory pool enabled on %s, pool_id=%s; "
            "transport configuration is unchanged", self.device, self._pool.id
        )

    @contextmanager
    def use(self):
        if self._pool is None:
            raise RuntimeError("Mooncake Fabric pool is already closed.")
        # Device binding is thread-local. use_mem_pool restores the original
        # allocation route on both normal and exceptional context exits.
        with self._torch.cuda.device(self.device):
            with self._torch.cuda.use_mem_pool(self._pool, device=self.device):
                yield

    def close(self):
        if self._pool is None:
            return
        with self._torch.cuda.device(self.device):
            self._torch.cuda.synchronize(self.device)
            # Drop the pool before its allocator wrapper/native library.
            self._pool = None
            gc.collect()
        self._allocator = None
        atexit.unregister(self.close)
