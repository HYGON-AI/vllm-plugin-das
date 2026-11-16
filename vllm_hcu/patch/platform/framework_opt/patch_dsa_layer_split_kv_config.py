# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Packed KV shrink to SGLang owned+scratch layer slots.

When DSA cache layer-split is active, shrink each page-size family to
``max_owned + 1 (+ MTP)`` physical slots (SGLang ``effective_num_layers`` /
``remote_kv_buffer``). Emits vLLM 0.25.1 V2 ``KVCacheTensor`` fields
(``layers`` / ``layer_stride`` / ``block_stride`` / ``offset``) when present,
falling back to legacy ``shared_by`` + ``offset`` + ``block_stride``.
"""

from __future__ import annotations

import functools
import inspect
import os
from collections import defaultdict
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
)

TARGET_MODULE = "vllm.v1.core.kv_cache_utils"
PATCH_ID = "platform.framework_opt.dsa_layer_split_kv_config"
TARGETS = (f"{TARGET_MODULE}.get_kv_cache_config_from_groups",)
_MARKER = "_vllm_hcu_dsa_layer_split_kv_config_applied"
_WRAPPER_PACKED = "_vllm_hcu_dsa_layer_split_kv_config_packed_wrapper"
_WRAPPER_GROUPS = "_vllm_hcu_dsa_layer_split_kv_config_groups_wrapper"
_WRAPPER_CONFIGS = "_vllm_hcu_dsa_layer_split_kv_config_configs_wrapper"
_WRAPPER_MAX_USAGE = "_vllm_hcu_dsa_layer_split_kv_config_max_usage_wrapper"
_WRAPPER_POOL_BYTES = "_vllm_hcu_dsa_layer_split_kv_config_pool_bytes_wrapper"

# vLLM 0.28 `_pool_bytes_per_block(groups)` drops vllm_config; equalization in
# `get_kv_cache_configs` still needs accounted bytes. Bind config while that
# call runs (025 passed vllm_config as an explicit argument).
_ACTIVE_VLLM_CONFIG = None


def _optional_config_groups_fn(owner: ModuleType, name: str):
    """Return ``owner.name`` if it is a known sizing-helper signature.

    025 / older: ``(vllm_config, kv_cache_groups)``.
    028 pool bytes: ``(kv_cache_groups,)`` only — still wrapped via
    ``_ACTIVE_VLLM_CONFIG`` set by ``get_kv_cache_configs``.
    """
    function = getattr(owner, name, None)
    if not callable(function):
        return None
    try:
        params = tuple(inspect.signature(function).parameters)
    except (TypeError, ValueError):
        return None
    if params not in (
        ("vllm_config", "kv_cache_groups"),
        ("kv_cache_groups",),
    ):
        return None
    return function


def _layer_split_active(vllm_config) -> bool:
    from vllm_hcu.v1.dsa_cache_layer_split import (
        is_dsa_cache_layer_split_requested,
    )

    if not is_dsa_cache_layer_split_requested(vllm_config):
        return False
    parallel = getattr(vllm_config, "parallel_config", None)
    pcp = int(getattr(parallel, "prefill_context_parallel_size", 1) or 1)
    return pcp > 1


def _layer_split_runtime_reserve_bytes() -> int:
    """Extra bytes kept out of KV for layer-split runtime peak.

    Bounce scratch is already amortized into accounted bytes/block. The
    remaining gap is profile-invisible activation that only shows up with
    layer-split (owner broadcast overlapping MoE / attention). Override:
      VLLM_HCU_DSA_LAYER_SPLIT_RESERVE_BYTES=<int>
      VLLM_HCU_DSA_LAYER_SPLIT_RESERVE_GIB=<float>
    Default: 2 GiB (this workload OOMd needing ~1 GiB more free).
    """
    raw_bytes = os.environ.get("VLLM_HCU_DSA_LAYER_SPLIT_RESERVE_BYTES")
    if raw_bytes is not None and raw_bytes.strip() != "":
        return max(0, int(raw_bytes))
    raw_gib = os.environ.get("VLLM_HCU_DSA_LAYER_SPLIT_RESERVE_GIB")
    if raw_gib is not None and raw_gib.strip() != "":
        return max(0, int(float(raw_gib) * (1 << 30)))
    return 2 << 30


def _available_after_layer_split_reserve(available_memory: int) -> int:
    reserve = _layer_split_runtime_reserve_bytes()
    if reserve <= 0:
        return int(available_memory)
    return max(0, int(available_memory) - reserve)


def _bucket_layers_by_page_size(kv_cache_groups) -> dict[int, list[list[str]]]:
    """page_size -> list of single-layer slots (stable within each family)."""
    buckets: dict[int, list[list[str]]] = defaultdict(list)
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        per_layer = getattr(spec, "kv_cache_specs", None)
        if per_layer:
            # Preserve insertion order of the UniformType map.
            for layer_name, layer_spec in per_layer.items():
                buckets[int(layer_spec.page_size_bytes)].append([str(layer_name)])
            continue
        page_size = int(spec.page_size_bytes)
        for layer_name in getattr(group, "layer_names", []) or []:
            buckets[page_size].append([str(layer_name)])
    return dict(buckets)


def _bounce_bytes_per_block(kv_cache_groups, page_sizes) -> int:
    """Per-block bytes of the sync + prefetch broadcast scratch pools.

    ``_scratch_for`` keys one buffer per (pool, device, dtype) and grows it to
    the largest cache seen, so same-dtype page families share one buffer sized
    to their max page. Families whose spec dtype is unknown are charged
    standalone (conservative sum).
    """
    page_dtypes: dict[int, set] = {}
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        per_layer = getattr(spec, "kv_cache_specs", None)
        layer_specs = per_layer.values() if per_layer else (spec,)
        for layer_spec in layer_specs:
            page = int(getattr(layer_spec, "page_size_bytes", 0) or 0)
            if page:
                page_dtypes.setdefault(page, set()).add(
                    getattr(layer_spec, "dtype", None)
                )
    max_by_pool: dict[object, int] = {}
    for page in page_sizes:
        for dtype in page_dtypes.get(int(page)) or {None}:
            key = ("standalone", page) if dtype is None else dtype
            max_by_pool[key] = max(max_by_pool.get(key, 0), int(page))
    return 2 * sum(max_by_pool.values())


def apply_to_module(module: ModuleType) -> bool:
    kv_cache_utils = load_exact_module(TARGET_MODULE, module)
    has_packed = callable(
        getattr(kv_cache_utils, "_get_kv_cache_config_packed", None)
    )
    wrapped = [
        (
            kv_cache_utils,
            "get_kv_cache_config_from_groups",
            _WRAPPER_GROUPS,
        )
    ]
    if has_packed:
        wrapped.insert(
            0,
            (
                kv_cache_utils,
                "_get_kv_cache_config_packed",
                _WRAPPER_PACKED,
            ),
        )
    if callable(getattr(kv_cache_utils, "get_kv_cache_configs", None)):
        wrapped.append(
            (kv_cache_utils, "get_kv_cache_configs", _WRAPPER_CONFIGS)
        )
    for name, wrapper_marker in (
        ("_max_memory_usage_bytes_from_groups", _WRAPPER_MAX_USAGE),
        ("_pool_bytes_per_block", _WRAPPER_POOL_BYTES),
    ):
        if _optional_config_groups_fn(kv_cache_utils, name) is not None:
            wrapped.append((kv_cache_utils, name, wrapper_marker))
    if already_applied(kv_cache_utils, _MARKER, wrapped):
        return False

    original_groups = require_callable(
        kv_cache_utils, "get_kv_cache_config_from_groups", TARGETS[0]
    )
    signature = inspect.signature(original_groups)
    if tuple(signature.parameters) != (
        "vllm_config",
        "kv_cache_groups",
        "available_memory",
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target has incompatible signature {signature}"
        )

    original_packed = None
    if has_packed:
        original_packed = require_callable(
            kv_cache_utils,
            "_get_kv_cache_config_packed",
            f"{TARGET_MODULE}._get_kv_cache_config_packed",
        )

    original_max_usage = _optional_config_groups_fn(
        kv_cache_utils, "_max_memory_usage_bytes_from_groups"
    )
    original_pool_bytes = _optional_config_groups_fn(
        kv_cache_utils, "_pool_bytes_per_block"
    )

    may_override = require_callable(
        kv_cache_utils,
        "may_override_num_blocks",
        f"{TARGET_MODULE}.may_override_num_blocks",
    )
    bucket_fn = getattr(kv_cache_utils, "_bucket_layers_by_page_size", None)
    if not callable(bucket_fn):
        bucket_fn = _bucket_layers_by_page_size

    kv_cache_config_cls = getattr(kv_cache_utils, "KVCacheConfig", None)
    if kv_cache_config_cls is None:
        from vllm.v1.kv_cache_interface import KVCacheConfig as kv_cache_config_cls

    kv_cache_tensor_cls = getattr(kv_cache_utils, "KVCacheTensor", None)
    if kv_cache_tensor_cls is None:
        from vllm.v1.kv_cache_interface import KVCacheTensor as kv_cache_tensor_cls

    def _shrunk_layout(vllm_config, kv_cache_groups):
        from vllm_hcu.v1.dsa_cache_layer_split import (
            _total_layers_from_config,
            parse_layer_index,
            phys_slots_for_layer_family,
        )

        total_layers = _total_layers_from_config(vllm_config)
        if total_layers is None or total_layers <= 0:
            return None

        parallel = getattr(vllm_config, "parallel_config", None)
        pcp = int(getattr(parallel, "prefill_context_parallel_size", 1) or 1)

        buckets = bucket_fn(kv_cache_groups)
        page_sizes = sorted(buckets)
        phys_slots: dict[int, int] = {}
        for page_size in page_sizes:
            slots = buckets[page_size]
            indices: list[int] = []
            for slot in slots:
                if not slot:
                    continue
                try:
                    indices.append(parse_layer_index(slot[0]))
                except Exception:
                    indices.append(-1)
            phys_slots[page_size] = phys_slots_for_layer_family(
                indices, int(total_layers), pcp
            )

        total_num_bytes_per_block = sum(
            page_size * phys_slots[page_size] for page_size in page_sizes
        )
        if total_num_bytes_per_block <= 0:
            return None

        full_bytes = sum(
            page_size * len(buckets[page_size]) for page_size in page_sizes
        )
        # No shrink (tiny models): the original planner keeps its layout, but
        # the runtime still bounces through the broadcast scratch, so the
        # reservation below applies either way.
        shrunk = total_num_bytes_per_block < full_bytes
        if not shrunk:
            total_num_bytes_per_block = full_bytes
        else:
            # Interleaving page-size families does not guarantee that the
            # byte pitch is a whole number of BF16 MLA rows. Pad with the
            # same helper the worker remap uses.
            from vllm_hcu.v1.dsa_cache_layer_split import (
                align_packed_block_stride,
            )

            total_num_bytes_per_block = align_packed_block_stride(
                total_num_bytes_per_block, kv_cache_groups
            )
        # ``_scratch_for`` keeps separate sync and prefetch pools; see
        # ``_bounce_bytes_per_block`` for the per-dtype sharing model.
        bounce_bytes_per_block = _bounce_bytes_per_block(
            kv_cache_groups, page_sizes
        )
        return (
            buckets,
            page_sizes,
            phys_slots,
            total_num_bytes_per_block,
            full_bytes,
            bounce_bytes_per_block,
            shrunk,
        )

    def _accounted_bytes(layout) -> int:
        return layout[3] + layout[5]

    def _budget_blocks(vllm_config, layout, available_memory) -> int:
        """Block count that fits packed KV plus bounce; rejects larger overrides.

        Bounce scratch is allocated lazily on the first broadcast, so an
        override that only fits the packed tensor passes startup and OOMs on
        the first prefill instead.
        """
        budget = available_memory // _accounted_bytes(layout)
        num_blocks = may_override(vllm_config, budget)
        if num_blocks > budget:
            raise ValueError(
                f"num_gpu_blocks_override={num_blocks} exceeds the DSA cache "
                f"layer-split budget of {budget} blocks (packed KV "
                f"{layout[3]} + broadcast bounce {layout[5]} bytes/block); "
                "lower the override or the layer-split bounce buffers OOM "
                "on the first prefill."
            )
        return num_blocks

    def _bounce_reserved_memory(vllm_config, kv_cache_groups, available_memory):
        """``available_memory`` minus bounce reservation for the original planner."""
        layout = _shrunk_layout(vllm_config, kv_cache_groups)
        if layout is None:
            return available_memory
        _budget_blocks(vllm_config, layout, available_memory)
        return available_memory * layout[3] // _accounted_bytes(layout)

    def _shrunk_packed_plan(vllm_config, kv_cache_groups, available_memory):
        from vllm_hcu.v1.dsa_cache_layer_split import make_packed_kv_tensor
        from vllm.logger import init_logger

        logger = init_logger("vllm_hcu.dsa_layer_split_kv")

        layout = _shrunk_layout(vllm_config, kv_cache_groups)
        if layout is None or not layout[6]:
            return None
        (
            buckets,
            page_sizes,
            phys_slots,
            total_num_bytes_per_block,
            full_bytes,
            bounce_bytes_per_block,
            _,
        ) = layout

        # Reserve the memory used by the two persistent broadcast bounce pools;
        # it is not part of the packed KV tensor returned below.
        # Runtime peak headroom is deducted once in get_kv_cache_configs — not
        # here — so equalization re-plans (min_blocks * pool_bytes) stay stable.
        accounted_bytes_per_block = _accounted_bytes(layout)
        num_blocks = _budget_blocks(vllm_config, layout, available_memory)
        total_size = total_num_bytes_per_block * num_blocks

        kv_cache_tensors = []
        byte_offset_base = 0
        for page_size in page_sizes:
            slots = buckets[page_size]
            n_phys = phys_slots[page_size]
            for slot_idx, slot in enumerate(slots):
                # Provisional clamp; workers rewrite via remap_packed_*.
                phys_idx = min(slot_idx, n_phys - 1)
                kv_cache_tensors.append(
                    make_packed_kv_tensor(
                        kv_cache_tensor_cls,
                        size=total_size,
                        layer_names=list(slot),
                        page_size=page_size,
                        offset=byte_offset_base + phys_idx * page_size,
                        block_stride=total_num_bytes_per_block,
                        backing_id=(
                            "dsa-layer-split",
                            tuple(page_sizes),
                            int(total_num_bytes_per_block),
                            int(num_blocks),
                        ),
                    )
                )
            byte_offset_base += n_phys * page_size

        logger.warning(
            "DSA layer-split packed KV plan: full_bytes/block=%d "
            "owned_scratch_bytes/block=%d bounce_bytes/block=%d "
            "accounted_bytes/block=%d ratio=%.3f num_blocks=%d "
            "phys_slots=%s",
            full_bytes,
            total_num_bytes_per_block,
            bounce_bytes_per_block,
            accounted_bytes_per_block,
            (total_num_bytes_per_block / full_bytes) if full_bytes else 0.0,
            num_blocks,
            dict(phys_slots),
        )
        return num_blocks, kv_cache_tensors

    def _config_from_plan(plan, kv_cache_groups, vllm_config):
        num_blocks, kv_cache_tensors = plan
        kwargs = dict(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )
        fields = getattr(kv_cache_config_cls, "__dataclass_fields__", {})
        if "prefix_cache_retention_interval" in fields:
            cache_config = getattr(vllm_config, "cache_config", None)
            kwargs["prefix_cache_retention_interval"] = getattr(
                cache_config, "prefix_cache_retention_interval", None
            )
        return kv_cache_config_cls(**kwargs)

    if original_packed is not None:

        @functools.wraps(original_packed)
        def hcu_get_kv_cache_config_packed(
            vllm_config, kv_cache_groups, available_memory
        ):
            if not _layer_split_active(vllm_config):
                return original_packed(
                    vllm_config, kv_cache_groups, available_memory
                )
            plan = _shrunk_packed_plan(
                vllm_config, kv_cache_groups, available_memory
            )
            if plan is None:
                # The groups wrapper already reserved bounce memory before
                # delegating here; reserving again would double-charge it.
                return original_packed(
                    vllm_config, kv_cache_groups, available_memory
                )
            return plan

        setattr(hcu_get_kv_cache_config_packed, _WRAPPER_PACKED, True)
        setattr(
            kv_cache_utils,
            "_vllm_hcu_original_get_kv_cache_config_packed",
            original_packed,
        )
        setattr(
            kv_cache_utils,
            "_get_kv_cache_config_packed",
            hcu_get_kv_cache_config_packed,
        )
        if hasattr(kv_cache_utils, "_get_kv_cache_config_deepseek_v4"):
            setattr(
                kv_cache_utils,
                "_get_kv_cache_config_deepseek_v4",
                hcu_get_kv_cache_config_packed,
            )

    @functools.wraps(original_groups)
    def hcu_get_kv_cache_config_from_groups(
        vllm_config, kv_cache_groups, available_memory
    ):
        if not _layer_split_active(vllm_config) or not kv_cache_groups:
            return original_groups(vllm_config, kv_cache_groups, available_memory)

        plan = _shrunk_packed_plan(
            vllm_config, kv_cache_groups, available_memory
        )
        if plan is None:
            return original_groups(
                vllm_config,
                kv_cache_groups,
                _bounce_reserved_memory(
                    vllm_config, kv_cache_groups, available_memory
                ),
            )
        return _config_from_plan(plan, kv_cache_groups, vllm_config)

    def _shrink_bytes(vllm_config, kv_cache_groups, full_value):
        if not _layer_split_active(vllm_config) or not kv_cache_groups:
            return full_value
        layout = _shrunk_layout(vllm_config, kv_cache_groups)
        if layout is None:
            return full_value
        full_bytes = layout[4]
        return -(-full_value * _accounted_bytes(layout) // full_bytes)

    if original_max_usage is not None:

        @functools.wraps(original_max_usage)
        def hcu_max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups):
            # 0.28 max-usage multiplies ``_pool_bytes_per_block``. That helper is
            # already the shrunk wrapper, so shrinking the product again
            # under-counts max-model-len. A stub that returns full bytes and
            # never calls the pool still gets one shrink.
            try:
                uses_pool = "_pool_bytes_per_block" in inspect.getsource(
                    original_max_usage
                )
            except (OSError, TypeError):
                uses_pool = False
            full_value = original_max_usage(vllm_config, kv_cache_groups)
            if uses_pool:
                return full_value
            return _shrink_bytes(vllm_config, kv_cache_groups, full_value)

        setattr(hcu_max_memory_usage_bytes_from_groups, _WRAPPER_MAX_USAGE, True)
        setattr(
            kv_cache_utils,
            "_vllm_hcu_original_max_memory_usage_bytes_from_groups",
            original_max_usage,
        )
        setattr(
            kv_cache_utils,
            "_max_memory_usage_bytes_from_groups",
            hcu_max_memory_usage_bytes_from_groups,
        )

    if original_pool_bytes is not None:
        try:
            pool_params = tuple(inspect.signature(original_pool_bytes).parameters)
        except (TypeError, ValueError):
            pool_params = ()

        if pool_params == ("vllm_config", "kv_cache_groups"):
            # 025 / older upstream: explicit vllm_config argument.
            @functools.wraps(original_pool_bytes)
            def hcu_pool_bytes_per_block(vllm_config, kv_cache_groups):
                return _shrink_bytes(
                    vllm_config,
                    kv_cache_groups,
                    original_pool_bytes(vllm_config, kv_cache_groups),
                )

        elif pool_params == ("kv_cache_groups",):
            # 028: equalization / override pass only groups; use the config
            # bound by get_kv_cache_configs (see hcu_get_kv_cache_configs).
            @functools.wraps(original_pool_bytes)
            def hcu_pool_bytes_per_block(kv_cache_groups):
                vllm_config = _ACTIVE_VLLM_CONFIG
                if vllm_config is None:
                    return original_pool_bytes(kv_cache_groups)
                return _shrink_bytes(
                    vllm_config,
                    kv_cache_groups,
                    original_pool_bytes(kv_cache_groups),
                )

        else:
            hcu_pool_bytes_per_block = None

        setattr(
            kv_cache_utils,
            "_vllm_hcu_original_pool_bytes_per_block",
            original_pool_bytes,
        )
        if hcu_pool_bytes_per_block is not None:
            setattr(hcu_pool_bytes_per_block, _WRAPPER_POOL_BYTES, True)
            setattr(
                kv_cache_utils, "_pool_bytes_per_block", hcu_pool_bytes_per_block
            )

    original_configs = getattr(kv_cache_utils, "get_kv_cache_configs", None)
    if callable(original_configs):

        @functools.wraps(original_configs)
        def hcu_get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory):
            global _ACTIVE_VLLM_CONFIG
            previous = _ACTIVE_VLLM_CONFIG
            _ACTIVE_VLLM_CONFIG = vllm_config
            try:
                # Deduct once per worker *before* planning/equalization so the
                # min_blocks * pool_bytes re-plan does not subtract again.
                if _layer_split_active(vllm_config):
                    reserve = _layer_split_runtime_reserve_bytes()
                    if reserve > 0:
                        from vllm.logger import init_logger

                        init_logger("vllm_hcu.dsa_layer_split_kv").warning(
                            "DSA layer-split runtime reserve: %d bytes "
                            "(%.2f GiB) deducted from each worker KV budget",
                            reserve,
                            reserve / (1 << 30),
                        )
                        available_memory = [
                            max(0, int(m) - reserve) for m in available_memory
                        ]
                return original_configs(
                    vllm_config, kv_cache_specs, available_memory
                )
            finally:
                _ACTIVE_VLLM_CONFIG = previous

        setattr(hcu_get_kv_cache_configs, _WRAPPER_CONFIGS, True)
        setattr(
            kv_cache_utils,
            "_vllm_hcu_original_get_kv_cache_configs",
            original_configs,
        )
        setattr(kv_cache_utils, "get_kv_cache_configs", hcu_get_kv_cache_configs)

    setattr(hcu_get_kv_cache_config_from_groups, _WRAPPER_GROUPS, True)
    setattr(
        kv_cache_utils,
        "_vllm_hcu_original_get_kv_cache_config_from_groups",
        original_groups,
    )
    setattr(
        kv_cache_utils,
        "get_kv_cache_config_from_groups",
        hcu_get_kv_cache_config_from_groups,
    )
    setattr(kv_cache_utils, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
