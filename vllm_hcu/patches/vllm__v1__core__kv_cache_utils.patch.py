# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.v1.core.kv_cache_utils

General fix for hybrid KV page-size unification:
- Keep fast path for divisible max-page alignment.
- If not divisible, fall back to LCM page-size alignment.
- Rebuild specs with safe page padding fallback when necessary.
"""

PATCHES = []

REGEX_PATCHES = [
    (
        r"def unify_kv_cache_spec_page_size\([\s\S]*?\n\s*return new_kv_cache_spec\n",
        '''def _rebuild_spec_with_target_page(
    layer_spec: KVCacheSpec,
    target_page_size: int,
    ratio: int,
) -> KVCacheSpec:
    new_block_size = layer_spec.block_size * ratio

    # Clear stale page padding first; otherwise page_size_bytes may stay pinned
    # to an old padded value and fail to track block_size growth.
    replace_kwargs: dict[str, object] = {"block_size": new_block_size}
    if hasattr(layer_spec, "page_size_padded"):
        replace_kwargs["page_size_padded"] = None

    new_spec = replace(layer_spec, **replace_kwargs)
    if new_spec.page_size_bytes == target_page_size:
        return new_spec

    # Generic strict fallback: allow explicit page padding to the exact
    # target page size only when the padded spec is internally valid.
    if hasattr(new_spec, "page_size_padded"):
        padded_spec = replace(new_spec, page_size_padded=target_page_size)
        if padded_spec.page_size_bytes == target_page_size:
            return padded_spec

    raise AssertionError(
        "Failed to rebuild KV cache spec to target page size "
        f"{target_page_size} from {layer_spec.page_size_bytes}."
    )


def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """
    Unify the page size of the given KVCacheSpec. If the page size of all layers
    are the same, return the original KVCacheSpec. If not same, unify the page
    size by increasing the block size of layers with smaller page size. Raise
    NotImplementedError if failed to unify the page size.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        The updated KVCacheSpec with the same page_size_bytes.
    """
    from math import lcm

    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        # All layers have the same page size, no need to unify.
        return kv_cache_spec

    # Fast path: keep original max-page divisibility behavior.
    target_page_size = max(page_sizes)
    use_lcm_fallback = any(target_page_size % p != 0 for p in page_sizes)

    if use_lcm_fallback:
        target_page_size = lcm(*sorted(page_sizes))
        logger.info(
            "Using generic KV page-size LCM alignment: page_sizes=%s, target=%d",
            sorted(page_sizes),
            target_page_size,
        )

    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == target_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
            continue

        layer_page_size = layer_spec.page_size_bytes
        if target_page_size % layer_page_size != 0:
            raise NotImplementedError(
                "The page size of the layer is not divisible by the "
                "maximum page size. Cannot unify by adjusting block_size."
            )

        ratio = target_page_size // layer_page_size
        new_spec = _rebuild_spec_with_target_page(
            layer_spec,
            target_page_size=target_page_size,
            ratio=ratio,
        )
        new_kv_cache_spec[layer_name] = new_spec

    return new_kv_cache_spec
''',
    ),
]
