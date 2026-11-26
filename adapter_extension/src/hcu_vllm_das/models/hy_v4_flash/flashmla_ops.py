# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Explicit, same-source FlashMLA operations; no Core or Adapter rebinding."""

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

from .runtime import is_hcu_platform


@dataclass(frozen=True)
class FlashMLAOps:
    sparse_fwd: Callable
    with_kvcache: Callable
    get_metadata: Callable
    source: str
    supports_out: bool
    get_sparse_kv_cache_formats: Callable | None = None


@dataclass(frozen=True)
class FlashMLAKVCacheFormat:
    """One worker's coherent writer/reader codec; IndexPool is independent."""

    name: str
    reason: str
    reader_formats: tuple[str, ...] = ()

    def __post_init__(self):
        if self.name not in ("e4m3", "e5m2"):
            raise ValueError("Unsupported sparse656 cache format")

    @property
    def writer_dtype(self):
        return "fp8_" + self.name

    @property
    def reader_kwargs(self):
        # Omit the E4 keyword so old Python/native APIs remain callable.
        return {"kv_cache_format": "e5m2"} if self.name == "e5m2" else {}


def _is_missing(value):
    if value is None:
        return True
    from hcu_adapter._base import Missing

    return isinstance(value, Missing)


def select_fp8_ds_mla_format(arch, *, eligible, ops, writer):
    """Prefer E5 only for the qualified ordinary gfx936 worker domain.

    Called at worker construction, never during Dynamo/capture. The same-source
    frontend getter reads the binary's capability, not a Python version or hash.
    """
    if arch != "gfx936":
        return FlashMLAKVCacheFormat("e4m3", "existing-architecture-policy")
    if not eligible:
        return FlashMLAKVCacheFormat("e4m3", "legacy-e4-unsupported-domain")
    getter = getattr(ops, "get_sparse_kv_cache_formats", None)
    if _is_missing(getter):
        return FlashMLAKVCacheFormat("e4m3", "legacy-e4-reader-capability-unavailable")
    if not callable(getter):
        raise RuntimeError("Sparse656 capability getter is not callable")
    formats = getter()
    if (type(formats) is not tuple or not formats
            or any(type(name) is not str or name not in ("e4m3", "e5m2")
                   for name in formats)
            or len(set(formats)) != len(formats) or "e4m3" not in formats):
        raise RuntimeError("Malformed native sparse656 cache-format capability")
    if "e5m2" not in formats:
        return FlashMLAKVCacheFormat("e4m3", "legacy-e4-reader-only", formats)
    for operation in (
        getattr(ops, "with_kvcache", None), getattr(ops, "get_metadata", None),
    ):
        if _is_missing(operation) or not callable(operation):
            return FlashMLAKVCacheFormat("e4m3", "legacy-e4-reader-unavailable", formats)
    if _is_missing(writer) or not callable(writer):
        return FlashMLAKVCacheFormat("e4m3", "legacy-e4-writer-unavailable", formats)
    return FlashMLAKVCacheFormat("e5m2", "gfx936-native-e5m2", formats)


def get_flashmla_ops() -> FlashMLAOps:
    """Resolve once at construction, pairing kernels with their own metadata.

    The HCU vendor API allocates prefill output and has no ``out`` parameter.
    The caller must copy that result into Core's workspace when one is supplied.
    Kernel import is not evidence of device/shape qualification.
    """
    hcu = is_hcu_platform()
    source = (
        "flash_mla.flash_mla_interface"
        if hcu
        else "vllm.v1.attention.ops.flashmla"
    )
    try:
        module = import_module(source)
        sparse = module.flash_mla_sparse_fwd
        decode = module.flash_mla_with_kvcache
        metadata = module.get_mla_metadata
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            f"HYV4 Flash requires sparse/decode/metadata FlashMLA API from {source}"
        ) from error
    if not all(callable(op) for op in (sparse, decode, metadata)):
        raise RuntimeError(f"HYV4 FlashMLA has a non-callable operation in {source}")
    formats = getattr(module, "get_sparse_kv_cache_formats", None) if hcu else None
    return FlashMLAOps(sparse, decode, metadata, source, supports_out=not hcu,
                       get_sparse_kv_cache_formats=formats)
