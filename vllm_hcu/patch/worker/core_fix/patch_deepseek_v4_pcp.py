# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DeepSeek-V4/V4.1 PCP capability gate for the HCU sparse-MLA path.

The generic vLLM PCP validator (``check_attention_cp_compatibility``) walks
every ``AttentionLayerBase`` in ``static_forward_context`` and asserts
``backend.supports_pcp()``.  ``AttentionBackend.supports_pcp()`` resolves the
capability through ``get_impl_cls()``, but the DeepSeek-V4/V4.1 sparse-MLA
backend deliberately has no implementation class:
``DeepseekV4SparseMLABackend.get_impl_cls()`` raises ``NotImplementedError``
because the platform attention layer (``DeepseekV4Attention``) owns the forward
path.  The validator therefore rejects every V4/V4.1 PCP configuration before
the attention wiring is ever reached.

The auxiliary cache owners V4/V4.1 register are checked by the same walk:

* the SWA cache (``DeepseekV4SWACache`` -> ``DeepseekSparseSWABackend``), and
* the compressor state cache (``CompressorStateCache`` -> ``CompressorBackend``).

Neither defines ``get_impl_cls()``, so both report ``supports_pcp() == False``
and trip the validator as well.  The sparse indexer cache is unaffected because
``DeepseekV41IndexerBackend`` inherits the V3.2 ``supports_pcp()`` contract.

This adapter only publishes the capability; it does not implement any of the
partitioning, cache-gather or output-restore logic.  It is deliberately inert by
default because the V4/V4.1 attention, compressed-KV, sparse-indexer and SWA
wiring is still under validation: enabling PCP before that wiring is proven must
keep failing closed instead of silently producing wrong tokens.  Set
``VLLM_HCU_DSV4_PCP_EXPERIMENTAL=1`` to arm the capability for bring-up.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
)

TARGET_MODULE = "vllm.models.deepseek_v41.sparse_mla"
PATCH_ID = "worker.core_fix.deepseek_v4.pcp_capability"
_ENV_FLAG = "VLLM_HCU_DSV4_PCP_EXPERIMENTAL"
_ORIGINAL_ATTR = "_vllm_hcu_original_supports_pcp"
_MARKER = "_vllm_hcu_dsv4_pcp_capability_applied"

_SWA_MODULE = "vllm.v1.attention.backends.mla.sparse_swa"
_V41_COMPRESSOR_MODULE = "vllm.models.deepseek_v41.compressor"
_V4_COMPRESSOR_MODULE = "vllm.models.deepseek_v4.compressor"

# Backends the validator resolves through ``get_impl_cls()``.  The sparse-MLA
# entries live in this adapter's own target module; the rest are cache owners
# that share the same missing-impl-class contract.  ``(module, class name)``
# pairs are resolved lazily so a tree that lacks an optional variant (for
# example the SM100-only mega-attention backend) still applies cleanly.
_SPARSE_MLA_BACKENDS = (
    "DeepseekV4SparseMLABackend",
    "DeepseekV4FlashMLABackend",
    "FlashMLAMegaAttnBackend",
)
_AUXILIARY_BACKENDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (_SWA_MODULE, ("DeepseekSparseSWABackend",)),
    (_V41_COMPRESSOR_MODULE, ("CompressorBackend",)),
    (_V4_COMPRESSOR_MODULE, ("CompressorBackend",)),
)
_OPTIONAL_BACKENDS = frozenset(
    {
        "FlashMLAMegaAttnBackend",
        # V4.0's compressor module is absent on trees that only ship V4.1.
        f"{_V4_COMPRESSOR_MODULE}.CompressorBackend",
    }
)


def _experimental_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "").lower() in ("true", "1")


def _publish_capability(backend: type, target: str) -> None:
    """Advertise PCP on a backend that intentionally has no impl class."""

    if "supports_pcp" in backend.__dict__:
        raise PatchCompatibilityError(
            f"required HCU patch target {target} already defines "
            "supports_pcp; the HCU adapter must not mask a real capability"
        )
    if not hasattr(backend, _ORIGINAL_ATTR):
        setattr(backend, _ORIGINAL_ATTR, True)
    backend.supports_pcp = classmethod(lambda cls: True)


def _resolve_backend(module: ModuleType, name: str, target: str) -> None:
    backend = getattr(module, name, None)
    if backend is None:
        if target in _OPTIONAL_BACKENDS:
            return
        raise PatchCompatibilityError(
            f"required HCU patch target {module.__name__}.{name} is missing"
        )
    if not isinstance(backend, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {module.__name__}.{name} is not a class"
        )
    _publish_capability(backend, f"{module.__name__}.{name}")


def _patch_auxiliary_modules() -> None:
    for module_name, class_names in _AUXILIARY_BACKENDS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            if f"{module_name}.CompressorBackend" in _OPTIONAL_BACKENDS:
                continue
            raise PatchCompatibilityError(
                f"required HCU PCP patch target {module_name!r} could not be "
                "imported while arming DeepSeek-V4/V4.1 PCP"
            )
        for name in class_names:
            _resolve_backend(module, name, f"{module_name}.{name}")


def apply_to_module(module: ModuleType) -> bool:
    sparse_mla = load_exact_module(TARGET_MODULE, module)
    if getattr(sparse_mla, _MARKER, False):
        return False
    if not _experimental_enabled():
        # Fail closed: keep the upstream rejection until the attention wiring
        # and its Gate evidence land.
        return False

    for name in _SPARSE_MLA_BACKENDS:
        _resolve_backend(sparse_mla, name, f"{TARGET_MODULE}.{name}")
    _patch_auxiliary_modules()
    setattr(sparse_mla, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
