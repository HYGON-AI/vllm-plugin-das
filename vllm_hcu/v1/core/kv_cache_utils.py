# SPDX-License-Identifier: Apache-2.0
"""vLLM v0.25.1 target hybrid-KV page sizing plus HCU unification policy."""

from __future__ import annotations

from dataclasses import replace
from math import lcm
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def _rebuild_spec_with_target_page(
    layer_spec: Any,
    target_page_size: int,
    ratio: int,
) -> Any:
    """Grow block size and use explicit padding only when the spec supports it."""

    replace_kwargs: dict[str, object] = {
        "block_size": layer_spec.block_size * ratio
    }
    if hasattr(layer_spec, "page_size_padded"):
        replace_kwargs["page_size_padded"] = None

    new_spec = replace(layer_spec, **replace_kwargs)
    if new_spec.page_size_bytes == target_page_size:
        return new_spec

    if hasattr(new_spec, "page_size_padded"):
        padded_spec = replace(new_spec, page_size_padded=target_page_size)
        if padded_spec.page_size_bytes == target_page_size:
            return padded_spec

    raise AssertionError(
        "Failed to rebuild KV cache spec to target page size "
        f"{target_page_size} from {layer_spec.page_size_bytes}."
    )


def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, Any],
) -> dict[str, Any]:
    """Unify heterogeneous page sizes using max-page or safe LCM alignment."""

    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        return kv_cache_spec

    if any(not isinstance(page_size, int) or page_size <= 0 for page_size in page_sizes):
        raise ValueError(f"KV cache page sizes must be positive integers: {page_sizes}")

    target_page_size = max(page_sizes)
    if any(target_page_size % page_size != 0 for page_size in page_sizes):
        target_page_size = lcm(*sorted(page_sizes))
        logger.info(
            "Using generic KV page-size LCM alignment: page_sizes=%s, target=%d",
            sorted(page_sizes),
            target_page_size,
        )

    new_kv_cache_spec: dict[str, Any] = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == target_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
            continue
        if target_page_size % layer_spec.page_size_bytes:
            raise NotImplementedError(
                "Cannot unify KV page sizes by adjusting block_size: "
                f"target={target_page_size}, layer={layer_spec.page_size_bytes}."
            )
        new_kv_cache_spec[layer_name] = _rebuild_spec_with_target_page(
            layer_spec,
            target_page_size,
            target_page_size // layer_spec.page_size_bytes,
        )
    return new_kv_cache_spec


__all__ = ["unify_kv_cache_spec_page_size"]
