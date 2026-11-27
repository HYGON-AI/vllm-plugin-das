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
    return FlashMLAOps(sparse, decode, metadata, source, supports_out=not hcu)
