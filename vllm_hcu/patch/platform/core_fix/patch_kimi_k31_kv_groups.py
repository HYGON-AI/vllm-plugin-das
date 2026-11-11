# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Deterministic KV-cache grouping for the Kimi-K3.1 hybrid layout.

K3.1 carries three record geometries in one model: sparse MLA records
(656 B/token), the per-layer DSA indexer cache (132 B/token, an
``MLAAttentionSpec`` that upstream refuses to pad) and KDA state pools
(``MambaSpec``).  Upstream ``get_kv_cache_groups`` only reaches the
packed-group path under conditions that not every K3.1 configuration
satisfies and otherwise fails in ``unify_kv_cache_spec_page_size`` on the
indexer layers.

This patch wraps ``get_kv_cache_groups`` and, only for ``kimi_k31`` when the
upstream grouping failed with the known page-unification error, rebuilds the
groups structurally: uniform-type buckets are split into one PAGE-UNIFORM
group per page size (never a mixed-page group, so the layout stays safe for
any resolved kv-cache layout), then hidden-state groups are re-added with the
upstream page-alignment arithmetic.
"""

from __future__ import annotations

import functools
import logging
from collections import defaultdict
from dataclasses import replace as dc_replace
from math import gcd as _gcd
from types import ModuleType
from typing import Any

from ._common import load_exact_module

logger = logging.getLogger(__name__)

TARGET_MODULE = "vllm.v1.core.kv_cache_utils"
PATCH_ID = "platform.core_fix.kimi_k31_kv_cache_groups"
TARGETS = (f"{TARGET_MODULE}.get_kv_cache_groups",)
_MARKER = "_vllm_hcu_kimi_k31_kv_groups_applied"


def _largest_divisor_at_most(value: int, cap: int) -> int:
    for candidate in range(min(value, cap), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False

    original = target.get_kv_cache_groups

    @functools.wraps(original)
    def get_kv_cache_groups(vllm_config: Any, kv_cache_spec: Any) -> list:
        try:
            return original(vllm_config, kv_cache_spec)
        except NotImplementedError as exc:
            if "page size is not divisible" not in str(exc):
                raise
            hf_config = getattr(
                getattr(vllm_config, "model_config", None), "hf_config", None
            )
            if getattr(hf_config, "model_type", None) != "kimi_k31":
                raise

        # --- structural fallback (kimi_k31 only) ---
        from vllm.v1.kv_cache_interface import (
            HiddenStateCacheSpec,
            KVCacheGroupSpec,
            UniformTypeKVCacheSpecs,
        )

        hidden = {
            name: spec
            for name, spec in kv_cache_spec.items()
            if isinstance(spec, HiddenStateCacheSpec)
        }
        filtered = {
            name: spec
            for name, spec in kv_cache_spec.items()
            if not isinstance(spec, HiddenStateCacheSpec)
        }

        summary = sorted(
            (
                type(spec).__name__,
                spec.page_size_bytes,
                spec.block_size,
            )
            for spec in filtered.values()
        )
        counts: dict = defaultdict(int)
        for entry in summary:
            counts[entry] += 1
        logger.warning(
            "Kimi-K3.1: upstream KV-cache grouping failed; rebuilding groups "
            "structurally from specs %s",
            {f"{t}/ps={p}/bs={b}": c for (t, p, b), c in counts.items()},
        )

        groups: list = []
        # Greedy uniform-type buckets (same accumulation upstream uses).
        buckets: list[dict] = []
        for name, spec in filtered.items():
            for bucket in buckets:
                candidate = {**bucket, name: spec}
                if UniformTypeKVCacheSpecs.is_uniform_type(candidate):
                    bucket[name] = spec
                    break
            else:
                buckets.append({name: spec})

        for bucket in buckets:
            uniform = UniformTypeKVCacheSpecs.from_specs(bucket)
            by_page: dict = defaultdict(list)
            for name, spec in bucket.items():
                by_page[spec.page_size_bytes].append(name)
            if uniform is None:
                groups.extend(
                    KVCacheGroupSpec([name], spec)
                    for name, spec in bucket.items()
                )
                continue
            if len(by_page) == 1:
                groups.append(KVCacheGroupSpec(list(bucket), uniform))
                continue
            # One group PER PAGE SIZE, layers merged within it.  Mixed-page
            # groups are only legal for block-outermost layouts; keeping
            # groups page-uniform stays safe under any resolved layout.
            for _page, names in sorted(by_page.items()):
                sub = {name: bucket[name] for name in names}
                groups.append(
                    KVCacheGroupSpec(
                        list(sub),
                        UniformTypeKVCacheSpecs.from_specs(sub),
                    )
                )

        if hidden and groups:
            from vllm.utils.torch_utils import get_dtype_size
            from vllm.v1.core.kv_cache_utils import get_uniform_page_size

            common_page = get_uniform_page_size(
                [group.kv_cache_spec for group in groups]
            )
            group_block_size = _gcd(
                *(group.kv_cache_spec.block_size for group in groups)
            )
            for name, spec in hidden.items():
                per_token = (
                    spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
                )
                max_block = max(common_page // per_token, 1)
                new_bs = _largest_divisor_at_most(group_block_size, max_block)
                aligned = dc_replace(
                    spec, block_size=new_bs, page_size_padded=common_page
                )
                logger.info(
                    "Kimi-K3.1: hidden-state cache %s aligned to block %d "
                    "(page padded to %d bytes)",
                    name,
                    new_bs,
                    common_page,
                )
                groups.append(KVCacheGroupSpec([name], aligned))
        elif hidden:
            groups.extend(
                KVCacheGroupSpec([name], spec)
                for name, spec in hidden.items()
            )

        if hasattr(target, "_annotate_eagle_groups"):
            target._annotate_eagle_groups(vllm_config, kv_cache_spec, groups)
        return groups

    target.get_kv_cache_groups = get_kv_cache_groups
    # Modules that did ``from ... import get_kv_cache_groups`` BEFORE this
    # patch applied keep a stale direct reference (e.g.
    # vllm.v1.worker.gpu.cudagraph_utils).  Rebind every already-imported
    # module that still holds the original function object.
    import sys

    rebound = 0
    for owner in list(sys.modules.values()):
        try:
            if getattr(owner, "get_kv_cache_groups", None) is original:
                owner.get_kv_cache_groups = get_kv_cache_groups
                rebound += 1
        except Exception:  # pragma: no cover - inert module quirks
            continue
    if rebound:
        logger.info(
            "Kimi-K3.1: rebound %d stale get_kv_cache_groups reference(s)",
            rebound,
        )
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
