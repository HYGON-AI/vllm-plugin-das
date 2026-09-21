# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Offload DeepSeek V4.1 Engram tables to pinned host memory on HCU.

The AMD model uses the common TP-only Engram implementation.  Keep its
sharding, lookup kernel, staging, and collectives unchanged; replace only the
large FP8 table storage when ``EngramConfig.cpu_offload`` is enabled.
"""

from __future__ import annotations

import functools
import weakref
from types import ModuleType

import torch

from vllm.logger import init_logger
from vllm.model_executor.utils import set_weight_attrs

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

logger = init_logger(__name__)

TARGET_MODULE = "vllm.models.deepseek_v41.common.engram"
PATCH_ID = "worker.core_fix.deepseek_v41.engram_offload"
TARGETS = (f"{TARGET_MODULE}.Engram._create_embedding",)
_CLASS_MARKER = "_vllm_hcu_dsv41_engram_offload_storage"
_CREATE_MARKER = "_vllm_hcu_dsv41_engram_offload_create_embedding"


def _engram_config():
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    return None if config is None else getattr(config, "engram_config", None)


def _cpu_offload_enabled() -> bool:
    config = _engram_config()
    return bool(config is not None and getattr(config, "cpu_offload", False))


def _require_supported_config() -> None:
    config = _engram_config()
    if config is None:
        raise RuntimeError(
            "DeepSeek V4.1 Engram CPU offload requires EngramConfig, but no "
            "active engram_config was found"
        )
    if getattr(config, "dp_shared_memory", False):
        raise ValueError(
            "HCU DeepSeek V4.1 Engram CPU offload does not yet support "
            "dp_shared_memory=true; use dp_shared_memory=false"
        )
    if getattr(config, "embedding_across_dp", False):
        raise ValueError(
            "HCU DeepSeek V4.1 Engram CPU offload keeps TP-only sharding; "
            "use embedding_across_dp=false"
        )


def _is_pin_memory_available() -> bool:
    try:
        from vllm.utils.platform_utils import is_pin_memory_available

        return bool(is_pin_memory_available())
    except (ImportError, AttributeError):
        try:
            probe = torch.empty(1, device="cpu", pin_memory=True)
            del probe
            return True
        except (RuntimeError, TypeError):
            return False


def _is_uva_available() -> bool:
    return hasattr(torch.ops._C, "get_cuda_view_from_cpu_tensor")


def _accelerator_view_from_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.ops._C.get_cuda_view_from_cpu_tensor(tensor)


def _host_register(pointer: int, num_bytes: int) -> None:
    """Register an existing host allocation as pinned device-visible memory."""
    result = torch.cuda.cudart().cudaHostRegister(pointer, num_bytes, 0)
    value = getattr(result, "value", result)
    if value != 0:
        raise RuntimeError(f"cudaHostRegister failed: {result}")


def _host_unregister(pointer: int) -> None:
    try:
        result = torch.cuda.cudart().cudaHostUnregister(pointer)
        value = getattr(result, "value", result)
        if value != 0:
            logger.warning("HCU Engram cudaHostUnregister failed: %s", result)
    except Exception as exc:  # pragma: no cover - best effort teardown
        logger.warning("HCU Engram cudaHostUnregister raised: %r", exc)


def _register_host_tensor(tensor: torch.Tensor, owner: object) -> torch.Tensor:
    """Pin an anonymous host tensor for zero-copy lookup.

    Large ``hipHostMalloc`` requests are unreliable on HCU (observed
    ``hipErrorOutOfMemory`` for 12-47 GiB blocks), so the offload path
    allocates anonymous host memory and registers it with
    ``cudaHostRegister`` instead.  The finalizer is anchored to *owner* (the
    embedding module) because ``nn.Parameter`` shares storage with, but does
    not keep alive, the tensor returned here.
    """
    pointer = tensor.data_ptr()
    num_bytes = tensor.numel() * tensor.element_size()
    _host_register(pointer, num_bytes)
    try:
        if not tensor.is_pinned():
            raise RuntimeError(
                "CUDA did not recognize the registered Engram storage"
            )
    except BaseException:
        # Roll back so a failed check cannot leave a stale registration.
        _host_unregister(pointer)
        raise
    finalizer = weakref.finalize(owner, _host_unregister, pointer)
    finalizer.atexit = False  # type: ignore[misc]
    return tensor


def _make_offload_embedding_class(base: type) -> type:
    if getattr(base, _CLASS_MARKER, False):
        return base

    class HcuDeepseekV41ParallelEngramEmbedding(base):
        """The common TP-only Engram embedding with pinned-host storage."""

        _vllm_hcu_dsv41_engram_offload_storage = True

        def __init__(self, *args, **kwargs):
            self._hcu_uva_views = None
            self._hcu_uva_view_src = None
            super().__init__(*args, **kwargs)
            # Match the official offload implementation for dummy-weight mode.
            set_weight_attrs(self.weight, {"dummy_weight_value": 1.0})
            set_weight_attrs(self.weight_scale_inv, {"dummy_weight_value": 127})
            offloaded_bytes = (
                self.weight.numel() * self.weight.element_size()
                + self.weight_scale_inv.numel()
                * self.weight_scale_inv.element_size()
            )
            logger.info(
                "HCU DeepSeek V4.1 Engram CPU offload active: %.2f GiB per "
                "TP rank (weight=%s, weight_scale_inv=%s, TP-only sharding, "
                "dp_shared_memory=false)",
                offloaded_bytes / 1024**3,
                tuple(self.weight.shape),
                tuple(self.weight_scale_inv.shape),
            )

        def _allocate_weights(self):
            if not _is_pin_memory_available():
                raise RuntimeError(
                    "HCU DeepSeek V4.1 Engram CPU offload requires CPU pinned "
                    "memory, but pinned allocation is unavailable"
                )
            requested = (
                self.part_num_embeddings
                * (self.dim + self.dim // self.block_size)
                / 1024**3
            )
            try:
                weight = _register_host_tensor(
                    torch.empty(
                        self.part_num_embeddings,
                        self.dim,
                        dtype=torch.float8_e4m3fn,
                        device="cpu",
                    ),
                    self,
                )
                scales = _register_host_tensor(
                    torch.empty(
                        self.part_num_embeddings,
                        self.dim // self.block_size,
                        dtype=torch.uint8,
                        device="cpu",
                    ),
                    self,
                )
            except (RuntimeError, TypeError) as exc:
                raise RuntimeError(
                    "HCU DeepSeek V4.1 Engram CPU offload failed to allocate "
                    f"{requested:.2f} GiB of registered host memory for this "
                    "TP rank"
                ) from exc
            return weight, scales

        def _storage(self):
            weight = self.weight.data
            scales = self.weight_scale_inv.data
            if weight.device.type != "cpu" or scales.device.type != "cpu":
                raise RuntimeError(
                    "HCU DeepSeek V4.1 Engram offloaded parameters were moved "
                    "off CPU after loading"
                )
            if not weight.is_pinned() or not scales.is_pinned():
                raise RuntimeError(
                    "HCU DeepSeek V4.1 Engram offloaded parameters must remain "
                    "in pinned CPU memory"
                )
            src = (weight.data_ptr(), scales.data_ptr())
            if self._hcu_uva_view_src != src:
                if not _is_uva_available():
                    raise RuntimeError(
                        "HCU DeepSeek V4.1 Engram CPU offload requires "
                        "torch.ops._C.get_cuda_view_from_cpu_tensor"
                    )
                self._hcu_uva_views = (
                    _accelerator_view_from_cpu_tensor(weight),
                    _accelerator_view_from_cpu_tensor(scales),
                )
                self._hcu_uva_view_src = src
            assert self._hcu_uva_views is not None
            return self._hcu_uva_views

    HcuDeepseekV41ParallelEngramEmbedding.__name__ = (
        "HcuDeepseekV41ParallelEngramEmbedding"
    )
    HcuDeepseekV41ParallelEngramEmbedding.__qualname__ = (
        "HcuDeepseekV41ParallelEngramEmbedding"
    )
    setattr(HcuDeepseekV41ParallelEngramEmbedding, _CLASS_MARKER, True)
    return HcuDeepseekV41ParallelEngramEmbedding


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    engram_class = require_class(target, "Engram", f"{TARGET_MODULE}.Engram")
    embedding_base = require_class(
        target,
        "ParallelEngramEmbedding",
        f"{TARGET_MODULE}.ParallelEngramEmbedding",
    )
    original = require_callable(engram_class, "_create_embedding", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=("self", "layout", "layer_hash_index"),
    )
    if getattr(original, _CREATE_MARKER, False):
        return False

    offload_embedding = _make_offload_embedding_class(embedding_base)

    @functools.wraps(original)
    def hcu_create_embedding(self, layout, layer_hash_index):
        if not _cpu_offload_enabled():
            return original(self, layout, layer_hash_index)
        _require_supported_config()
        if not _is_pin_memory_available():
            raise RuntimeError(
                "HCU DeepSeek V4.1 Engram CPU offload requires CPU pinned "
                "memory, but pinned allocation is unavailable"
            )
        if not _is_uva_available():
            raise RuntimeError(
                "HCU DeepSeek V4.1 Engram CPU offload requires the HCU UVA "
                "operator torch.ops._C.get_cuda_view_from_cpu_tensor, but it "
                "is unavailable"
            )
        return offload_embedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            tuple(
                size
                for order in layout.primes[layer_hash_index]
                for size in order
            ),
        )

    setattr(hcu_create_embedding, _CREATE_MARKER, True)
    setattr(engram_class, "_vllm_hcu_original_create_embedding", original)
    setattr(engram_class, "_create_embedding", hcu_create_embedding)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
