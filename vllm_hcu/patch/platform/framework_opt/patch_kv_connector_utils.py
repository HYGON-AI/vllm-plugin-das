# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Restore TransferTopology KV-layout helpers required by HCU Mooncake.

vLLM 0.28 dropped ``get_transfer_cache_regions`` /
``virtually_split_kv_in_blocks`` / ``split_k_and_v`` from
``TransferTopology`` while the HCU-owned Mooncake connector (ported from
0.25) still depends on them for ``register_kv_caches``.  Re-attach the
0.25 layout detection and region-split APIs onto the 0.28 class.
"""

from __future__ import annotations

import functools
from types import ModuleType
from typing import TYPE_CHECKING

import torch

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_class,
    require_callable,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheSpec

TARGET_MODULE = "vllm.distributed.kv_transfer.kv_connector.utils"
PATCH_ID = "platform.framework_opt.transfer_topology_kv_layout"
TARGETS = (
    f"{TARGET_MODULE}.TransferTopology",
    f"{TARGET_MODULE}.TransferTopology.get_transfer_cache_regions",
)
_MARKER = "_vllm_hcu_transfer_topology_kv_layout_applied"
_POST_INIT_WRAPPER = "_vllm_hcu_transfer_topology_kv_layout_post_init"
_LAYOUT_METHOD_MARKER = "_vllm_hcu_transfer_topology_kv_layout_method"


def _layout_apis_present(topo: type) -> bool:
    return all(
        hasattr(topo, name)
        for name in (
            "get_transfer_cache_regions",
            "virtually_split_kv_in_blocks",
            "split_k_and_v",
            "is_kv_layout_blocks_first",
            "cross_layers_blocks",
        )
    )


def _init_kv_layout(self, logger) -> None:
    """Populate layout flags matching vLLM 0.25 TransferTopology.__post_init__.

    vLLM 0.28's DSA indexer backend may appear first in ``attn_backends`` but
    no longer implements ``get_kv_cache_shape`` (0.25 did).  Probe the first
    backend that still exposes the shape API so MLA/FlashAttn layout detection
    keeps working under Hy4 layer-split / DSA.
    """
    attn_backend = next(
        (
            backend
            for backend in self.attn_backends
            if callable(getattr(backend, "get_kv_cache_shape", None))
        ),
        None,
    )
    kv_cache_shape: tuple[int, ...] | None = None
    if not self.is_mamba:
        if attn_backend is None:
            # MLA / indexer-only: treat as non-blocks-first (Mooncake MLA path).
            self._is_kv_layout_blocks_first = False
            self._cross_layers_blocks = False
            logger.debug(
                "No attn backend exposes get_kv_cache_shape; "
                "defaulting to non-blocks-first layout (is_mla=%s)",
                self.is_mla,
            )
            return
        _MOCK_BLOCK_SIZE = 16
        kv_cache_shape = attn_backend.get_kv_cache_shape(
            num_blocks=1,
            block_size=_MOCK_BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
        )
        logger.debug("Test kv_cache_shape: %s", kv_cache_shape)

    # Non-MLA backends caches have 5 dims [num_blocks, 2, H,N,D] or
    # [2, num_blocks, ...]; mock num_blocks=1 for the dimension check.
    # Hybrid SSM models assume a single blocks_first layout.
    self._is_kv_layout_blocks_first = self.is_mamba or (
        kv_cache_shape is not None
        and len(kv_cache_shape) == 5
        and kv_cache_shape[0] == 1
    )

    self._cross_layers_blocks = False
    if self.tensor_shape is not None and kv_cache_shape is not None:
        self._cross_layers_blocks = (
            len(self.tensor_shape) == len(kv_cache_shape) + 1
        )

    if self._cross_layers_blocks:
        logger.debug("Using cross-layer KV cache")
        assert kv_cache_shape is not None and attn_backend is not None
        _MOCK_NUM_LAYERS = 80
        shaped = (_MOCK_NUM_LAYERS,) + kv_cache_shape
        try:
            kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                include_num_layers_dimension=self._cross_layers_blocks
            )
        except (AttributeError, NotImplementedError):
            assert self.tensor_shape is not None
            kv_cache_stride_order = tuple(range(len(self.tensor_shape)))
        # Keep parity with 0.25: recompute ordered shape for debug/side effects.
        _ = tuple(shaped[i] for i in kv_cache_stride_order)


def _is_kv_layout_blocks_first(self) -> bool:
    return self._is_kv_layout_blocks_first


def _cross_layers_blocks(self) -> bool:
    return self._cross_layers_blocks


def _virtually_split_kv_in_blocks(self) -> bool:
    # Whether to logically split each block into K and V halves.
    # Applies when K/V are interleaved within each block (blocks-first),
    # but NOT when cross-layer blocks are used.
    return self._is_kv_layout_blocks_first and not self._cross_layers_blocks


def _split_k_and_v(self) -> bool:
    # Whether to register regions for K and V separately (when present).
    return not (
        self._cross_layers_blocks
        or self.is_mla
        or self.is_kv_layout_blocks_first
    )


def _get_transfer_cache_regions(
    self, cache: torch.Tensor, layer_spec: "KVCacheSpec"
) -> list[torch.Tensor] | torch.Tensor:
    """Return the cache tensor(s) to register as transfer memory regions."""
    from vllm.v1.kv_cache_interface import MambaSpec

    if isinstance(layer_spec, MambaSpec):
        conv, _ssm = cache
        return [conv]

    # Matching ``_update_hybrid_attention_mamba_layout``.
    if self.is_mamba and cache.shape[0] == 2:
        cache = cache.transpose(0, 1)

    return cache if self.split_k_and_v else [cache]


def apply_to_module(module: ModuleType) -> bool:
    utils = load_exact_module(TARGET_MODULE, module)
    topo = require_class(utils, "TransferTopology", TARGETS[0])

    if already_applied(
        utils,
        _MARKER,
        ((topo, "__post_init__", _POST_INIT_WRAPPER),),
    ):
        return False

    # Upstream already has the 0.25 layout APIs — mark and skip.
    if _layout_apis_present(topo) and not getattr(
        getattr(topo, "get_transfer_cache_regions", None),
        _LAYOUT_METHOD_MARKER,
        False,
    ):
        # already_applied requires the post_init wrapper marker on re-entry.
        if not getattr(topo.__post_init__, _POST_INIT_WRAPPER, False):
            setattr(topo.__post_init__, _POST_INIT_WRAPPER, True)
        setattr(utils, _MARKER, True)
        return False

    if getattr(utils, _MARKER, False):
        raise PatchCompatibilityError(
            f"required HCU patch marker for {PATCH_ID} is stale; "
            "restart the process"
        )

    original_post_init = require_callable(
        topo, "__post_init__", f"{TARGETS[0]}.__post_init__"
    )
    logger = utils.logger

    @functools.wraps(original_post_init)
    def hcu_post_init(self):
        original_post_init(self)
        _init_kv_layout(self, logger)

    setattr(hcu_post_init, _POST_INIT_WRAPPER, True)
    setattr(topo, "_vllm_hcu_original_post_init", original_post_init)
    topo.__post_init__ = hcu_post_init  # type: ignore[method-assign]

    topo.is_kv_layout_blocks_first = property(  # type: ignore[assignment]
        _is_kv_layout_blocks_first
    )
    topo.cross_layers_blocks = property(_cross_layers_blocks)  # type: ignore[assignment]
    topo.virtually_split_kv_in_blocks = property(  # type: ignore[assignment]
        _virtually_split_kv_in_blocks
    )
    topo.split_k_and_v = property(_split_k_and_v)  # type: ignore[assignment]
    setattr(_get_transfer_cache_regions, _LAYOUT_METHOD_MARKER, True)
    topo.get_transfer_cache_regions = (  # type: ignore[method-assign]
        _get_transfer_cache_regions
    )

    setattr(utils, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
