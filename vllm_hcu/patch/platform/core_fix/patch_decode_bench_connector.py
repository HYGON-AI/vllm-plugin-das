# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Fill-aware DecodeBenchConnector quantization fix (FP8 byte-view caches).

Upstream ``DecodeBenchConnectorWorker`` fills synthetic KV blocks with
``torch.full(...)``/``normal_(...)`` using the *storage* dtype of the cache
tensor.  vLLM allocates plain per-tensor FP8 KV caches as ``uint8``/``int8``
byte storage, so a ``fill_mean=0.015`` constant truncates to byte 0x00: the
synthetic "non-trivial history" silently becomes all-zero context.

The deployed attention read path reinterprets those bytes as
``torch.float8_e4m3fn`` (hard-coded; see ``lightop_concat_runtime.py`` and
``flashmla.py`` ``view(torch.float8_e4m3fn)``), so this patch encodes fill
values with e4m3fn semantics *before* writing bytes:

    0.015 -> float8_e4m3fn 0.015625 -> byte 0x08

Classification is per GROUP and per LAYER, on the real configuration surface:

* the runtime ``cache_config.cache_dtype`` (the field the HCU platform reads
  and rewrites, e.g. "fp8_ds_mla" at ``platforms/hcu.py:646``);
* each layer's ``cache_dtype_str`` from the group's ``KVCacheSpec``, expanded
  with the same semantics as ``vllm.v1.kv_cache_interface.iter_layer_specs``
  (a bare spec, or a ``UniformTypeKVCacheSpecs`` wrapper carrying a
  ``kv_cache_specs`` mapping).

A group is filled through the FP8 byte path only when EVERY layer in it is an
explicit scalar per-tensor e4m3 flavour (``fp8`` / ``fp8_e4m3``); layers with
no flavour string fall back to the global dtype.  ``fp8_ds_mla`` groups
(DeepSeek/Kimi 656-byte block-scaled layout) are filled explicitly: e4m3 value
bytes per 128-value tile, unit fp32 tile scales, and bf16 RoPE bytes, so the
synthetic context decodes back to ``fill_mean`` instead of zero.  Any other
packed flavour (``nvfp4*``, per-token-head scales, ...) -- alone or MIXED with
scalar layers -- refuses the fill with a clear PatchCompatibilityError instead
of writing wrong bytes.  Float caches, native-FP8 tensors, state tensors and
the circular-buffer zero-fill groups keep upstream semantics exactly.

Registered through the standard platform/core import coordinator via
``vllm_hcu.patch.platform.core_fix``; compatibility is validated (module,
attributes, positional signatures of all four wrapped methods) before arming,
and arming is idempotent.  All wrappers preserve the upstream call signatures;
the per-group mode travels through an instance marker set by the
``_fill_blocks`` wrapper, never through a changed signature.
"""

from __future__ import annotations

import functools
import logging
from types import ModuleType
from typing import Any, Mapping

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_positional_signature,
)

PATCH_ID = "platform.core.decode_bench_fp8_fill"
TARGET_MODULE = (
    "vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector"
)
_TARGETS_BASE = f"{TARGET_MODULE}.DecodeBenchConnectorWorker"
TARGETS = (
    f"{_TARGETS_BASE}.__init__",
    f"{_TARGETS_BASE}._fill_blocks",
    f"{_TARGETS_BASE}._fill_block_tensor",
    f"{_TARGETS_BASE}._fill_state_tensor",
)

_MODULE_NAME = "DecodeBenchConnectorWorker"
_WRAPPER_FLAG = "_hcu_fp8_fill_wrapper"
_PATCHED_FLAG = "_hcu_fp8_fill_patched"
_CURRENT_MODE_ATTR = "_hcu_current_fill_mode"
_CURRENT_GROUP_ATTR = "_hcu_current_fill_group"
_GROUP_MODES_ATTR = "_hcu_group_fill_modes"
_MAMBA_SPECS_ATTR = "_hcu_mamba_group_specs"

# Scalar per-tensor FP8 flavours whose byte storage is exactly e4m3fn
# (upstream ``get_kv_quant_mode`` treats the bare "fp8"/"fp8_e4m3" family as
# plain per-tensor FP8; everything fp8-flavoured beyond that is packed).
_SCALAR_FP8_FLAVOURS = frozenset({"fp8", "fp8_e4m3"})
# Known quantised/packed flavour markers this patch refuses to byte-encode.
_PACKED_FLAVOUR_PREFIXES = (
    "fp8_",       # fp8_per_token_head, fp8_ds_mla variants, ...
    "nvfp4",      # nvfp4, nvfp4_ds_mla, ...
    "turboquant_",
    "int8_per_token_head",
    "int4_per_token_head",
)

# DeepSeek/Kimi ``fp8_ds_mla`` 656-byte block-scaled entry layout:
#   [tile0: 128 x e4m3 values + 1 x fp32 scale] x 4 tiles
#   [64 x bf16 RoPE values (128 bytes)]
# The cache tensor is uint8 with last dim 656; the read path views the same
# buffer as float8_e4m3fn (values), float32 (tile scales) and bfloat16 (RoPE).
_DS_MLA_FLAVOURS = frozenset({"fp8_ds_mla"})
_DS_MLA_FLAVOUR_PREFIX = "fp8_ds_mla"
_DS_MLA_WIDTH = 656
_DS_MLA_TILES = 4
_DS_MLA_TILE_VALUES = 128
_DS_MLA_TILE_SCALE_BYTES = 4
_DS_MLA_ROPE_VALUES = 64

_GROUP_MODE_PLAIN = "plain_fp8"
_GROUP_MODE_PACKED = "packed"
_GROUP_MODE_MIXED = "mixed"
_GROUP_MODE_UNKNOWN = "unknown"
_GROUP_MODE_FLOAT = "float"
_GROUP_MODE_DS_MLA = "packed_ds_mla"
_GROUP_MODE_MAMBA = "mamba_state"

_LOG = logging.getLogger("vllm_hcu.patch.decode_bench_fp8_fill")


def _iter_layer_specs(kv_cache_spec: Any):
    """Expand a group spec the way ``vllm.v1.kv_cache_interface`` does.

    ``UniformTypeKVCacheSpecs`` groups carry ``kv_cache_specs`` (a mapping of
    per-layer specs); any other spec describes its group on its own.
    """
    inner = getattr(kv_cache_spec, "kv_cache_specs", None)
    if isinstance(inner, Mapping):
        return tuple(inner.values())
    if inner is not None and not isinstance(inner, str):
        try:
            return tuple(inner)
        except TypeError:
            pass
    return (kv_cache_spec,)


def _flavour_mode(flavour: str) -> str | None:
    """Classify one layer flavour string.

    Returns ``None`` when the spec carries no flavour information (callers
    fall back to the global ``cache_dtype``), otherwise one of the group
    modes.
    """
    if not flavour:
        return None
    if flavour in _DS_MLA_FLAVOURS or flavour.startswith(_DS_MLA_FLAVOUR_PREFIX):
        return _GROUP_MODE_DS_MLA
    if flavour in _SCALAR_FP8_FLAVOURS:
        return _GROUP_MODE_PLAIN
    if flavour.startswith(_PACKED_FLAVOUR_PREFIXES):
        return _GROUP_MODE_PACKED
    return _GROUP_MODE_UNKNOWN


def _global_fill_mode(vllm_config) -> str:
    """Classify ``cache_config.cache_dtype`` -- the real runtime field."""
    cache_config = getattr(vllm_config, "cache_config", None)
    dtype_str = str(getattr(cache_config, "cache_dtype", "auto") or "auto")
    if dtype_str in _DS_MLA_FLAVOURS or dtype_str.startswith(_DS_MLA_FLAVOUR_PREFIX):
        return _GROUP_MODE_DS_MLA
    if dtype_str in _SCALAR_FP8_FLAVOURS:
        return _GROUP_MODE_PLAIN
    if dtype_str.startswith(_PACKED_FLAVOUR_PREFIXES):
        return _GROUP_MODE_UNKNOWN
    return _GROUP_MODE_FLOAT


def _group_fill_mode(group, global_mode: str) -> str:
    """Resolve one group's fill mode from its per-layer specs."""
    layer_flavours = [
        str(getattr(spec, "cache_dtype_str", "") or "")
        for spec in _iter_layer_specs(getattr(group, "kv_cache_spec", None))
    ]
    modes = [_flavour_mode(flavour) for flavour in layer_flavours]
    resolved = [mode if mode is not None else global_mode for mode in modes]
    if not resolved:
        return _GROUP_MODE_UNKNOWN
    if any(mode == _GROUP_MODE_PACKED for mode in resolved):
        return _GROUP_MODE_PACKED
    if any(mode == _GROUP_MODE_UNKNOWN for mode in resolved):
        return _GROUP_MODE_UNKNOWN
    if all(mode == _GROUP_MODE_PLAIN for mode in resolved):
        return _GROUP_MODE_PLAIN
    if all(mode == _GROUP_MODE_DS_MLA for mode in resolved):
        return _GROUP_MODE_DS_MLA
    if any(
        mode in (_GROUP_MODE_PLAIN, _GROUP_MODE_DS_MLA) for mode in resolved
    ):
        # scalar-FP8 or fp8_ds_mla layers mixed with float/non-quantised
        # layers: encoding would split one synthetic fill into incompatible
        # semantics, and the legacy path would silently truncate the byte
        # layers.
        return _GROUP_MODE_MIXED
    # Only float modes remain; a scalar-byte fill never applies.
    return _GROUP_MODE_FLOAT


def _encode_scalar(fill_mean: float) -> int:
    """Quantize one constant to e4m3fn and return its byte (unsigned form)."""
    import torch

    value = torch.tensor([fill_mean], dtype=torch.float32).to(torch.float8_e4m3fn)
    return int(value.view(torch.uint8).item())


def _encode_random(fill_mean: float, fill_std: float, shape, device):
    """Sample float normals, quantize to e4m3fn, and return byte storage."""
    import torch

    samples = torch.normal(
        mean=fill_mean,
        std=fill_std,
        size=shape,
        dtype=torch.float32,
        device=device,
    )
    return samples.to(torch.float8_e4m3fn).view(torch.uint8)


def _encode_like(fill_values_uint8, storage_dtype):
    """Reinterpret e4m3fn bytes as the cache storage dtype bit-exactly."""
    import torch

    if storage_dtype == torch.uint8:
        return fill_values_uint8
    if storage_dtype == torch.int8:
        return fill_values_uint8.view(torch.int8)
    raise PatchCompatibilityError(
        "DecodeBenchConnector FP8 fill supports uint8/int8 byte storage; got "
        f"{storage_dtype}."
    )


def _ds_mla_pattern(fill_mean: float, fill_std: float, device):
    """Build one fp8_ds_mla 656-byte entry as uint8 bytes.

    The layout is DeepSeek's block-scaled MLA cache: four tiles of
    ``128 x e4m3`` values each followed by one fp32 tile scale, then 64 bf16
    RoPE values.  Tile scales stay at 1.0 so the encoded value round-trips
    unchanged, which keeps the synthetic context representative (the whole
    point of this patch) instead of all-zero bytes.
    """
    import torch

    if fill_std > 0:
        values = torch.normal(
            mean=fill_mean,
            std=fill_std,
            size=(_DS_MLA_TILES, _DS_MLA_TILE_VALUES),
            dtype=torch.float32,
            device=device,
        ).clamp(-448.0, 448.0)
        rope = torch.normal(
            mean=fill_mean,
            std=fill_std,
            size=(_DS_MLA_ROPE_VALUES,),
            dtype=torch.float32,
            device=device,
        )
    else:
        values = torch.full(
            (_DS_MLA_TILES, _DS_MLA_TILE_VALUES),
            float(fill_mean),
            dtype=torch.float32,
            device=device,
        )
        rope = torch.full(
            (_DS_MLA_ROPE_VALUES,),
            float(fill_mean),
            dtype=torch.float32,
            device=device,
        )
    value_bytes = values.to(torch.float8_e4m3fn).view(torch.uint8)
    scale_bytes = torch.ones(1, dtype=torch.float32, device=device).view(torch.uint8)
    rope_bytes = rope.to(torch.bfloat16).view(torch.uint8)
    tile_bytes = _DS_MLA_TILE_VALUES + _DS_MLA_TILE_SCALE_BYTES
    entry = torch.empty(_DS_MLA_WIDTH, dtype=torch.uint8, device=device)
    for tile in range(_DS_MLA_TILES):
        start = tile * tile_bytes
        entry[start:start + _DS_MLA_TILE_VALUES] = value_bytes[tile]
        entry[start + _DS_MLA_TILE_VALUES:start + tile_bytes] = scale_bytes
    rope_start = _DS_MLA_TILES * tile_bytes
    entry[rope_start:rope_start + _DS_MLA_ROPE_VALUES * 2] = rope_bytes
    return entry


def _mamba_spec_of(group):
    """Return ``(shapes, dtypes)`` for a Mamba/KDA state group, else None."""
    spec = getattr(group, "kv_cache_spec", None)
    shapes = getattr(spec, "shapes", None)
    dtypes = getattr(spec, "dtypes", None)
    if shapes is None or dtypes is None:
        return None
    try:
        shapes = tuple(tuple(int(dim) for dim in shape) for shape in shapes)
        dtypes = tuple(dtypes)
    except (TypeError, ValueError):
        return None
    if len(shapes) != len(dtypes):
        return None
    return shapes, dtypes


def _mamba_page_pattern(shapes, dtypes, width, fill_mean, fill_std, device):
    """Build one Mamba/KDA state page from its spec shapes/dtypes.

    vLLM packs the layer states (e.g. KDA conv + recurrent) into one int8 page;
    each region is filled with ``fill_mean`` in its own dtype so the synthetic
    state decodes back to a representative value instead of all-zero bytes.
    """
    import torch

    pieces = []
    total = 0
    for shape, dtype in zip(shapes, dtypes):
        count = 1
        for dim in shape:
            count *= dim
        if dtype.is_floating_point:
            if fill_std > 0:
                values = torch.normal(
                    mean=fill_mean, std=fill_std, size=(count,),
                    dtype=torch.float32, device=device,
                )
            else:
                values = torch.full(
                    (count,), float(fill_mean), dtype=torch.float32, device=device
                )
            piece = values.to(dtype).view(torch.uint8)
        else:
            if fill_std > 0:
                piece = _encode_random(fill_mean, fill_std, (count,), device)
            else:
                piece = torch.full(
                    (count,), _encode_scalar(fill_mean),
                    dtype=torch.uint8, device=device,
                )
            piece = _encode_like(piece, dtype).view(torch.uint8)
        pieces.append(piece)
        total += count * torch.empty((), dtype=dtype).element_size()
    if total > width:
        raise PatchCompatibilityError(
            "DecodeBenchConnector mamba state spec needs "
            f"{total} bytes but the cache page is {width} bytes; refusing to "
            "truncate a state page."
        )
    entry = (
        torch.cat(pieces)
        if pieces
        else torch.empty(0, dtype=torch.uint8, device=device)
    )
    if total < width:
        entry = torch.cat(
            [entry, torch.zeros(width - total, dtype=torch.uint8, device=device)]
        )
    return entry


def _raise_for_mode(mode: str) -> None:
    if mode == _GROUP_MODE_PACKED:
        raise PatchCompatibilityError(
            "DecodeBenchConnector cannot fill packed FP8 KV group layouts "
            "(e.g. per-token-head scales / nvfp4) with one byte value: the "
            "storage mixes quantized values, scales, and/or BF16 RoPE, or "
            "packs heterogeneous layers. Configure a scalar FP8 cache or fill "
            "explicitly."
        )
    if mode == _GROUP_MODE_MIXED:
        raise PatchCompatibilityError(
            "DecodeBenchConnector refuses a mixed KV cache group: scalar-FP8 "
            "layers combined with float/non-quantised layers cannot share one "
            "synthetic fill without either truncating the byte layers or "
            "changing the float layers; configure a uniform cache dtype."
        )
    raise PatchCompatibilityError(
        "DecodeBenchConnector byte cache with unrecognised KV cache layout: "
        "cannot decide FP8 encoding; refusing ambiguous fill (add the layout "
        "to platform.core.decode_bench_fp8_fill explicitly if needed)."
    )


def _apply(module: ModuleType) -> bool:
    import torch

    module = load_exact_module(TARGET_MODULE, module)
    worker_cls = require_callable(
        module, _MODULE_NAME, f"{TARGET_MODULE}.{_MODULE_NAME}"
    )

    for method in (
        "__init__",
        "_fill_blocks",
        "_fill_block_tensor",
        "_fill_state_tensor",
    ):
        require_callable(worker_cls, method, f"{_TARGETS_BASE}.{method}")
    require_positional_signature(
        worker_cls.__init__,
        f"{_TARGETS_BASE}.__init__",
        ["self", "vllm_config", "kv_cache_config"],
    )
    require_positional_signature(
        worker_cls._fill_blocks,
        f"{_TARGETS_BASE}._fill_blocks",
        ["self", "group_idx", "block_ids", "num_tokens"],
    )
    require_positional_signature(
        worker_cls._fill_block_tensor,
        f"{_TARGETS_BASE}._fill_block_tensor",
        ["self", "kv_cache", "block_ids", "fill_mean", "fill_std"],
    )
    require_positional_signature(
        worker_cls._fill_state_tensor,
        f"{_TARGETS_BASE}._fill_state_tensor",
        ["self", "kv_cache", "fill_mean", "fill_std"],
    )

    if getattr(worker_cls, _PATCHED_FLAG, False):
        return False  # already armed in a previous callback run
    wrapped = [
        name
        for name in ("__init__", "_fill_blocks", "_fill_block_tensor", "_fill_state_tensor")
        if getattr(getattr(worker_cls, name, None), _WRAPPER_FLAG, False)
    ]
    if wrapped:
        raise PatchCompatibilityError(
            "decode_bench_fp8_fill found pre-wrapped methods "
            f"({', '.join(wrapped)}) without the arming marker; refusing "
            "inconsistent state."
        )

    original_init = worker_cls.__init__
    original_fill_blocks = worker_cls._fill_blocks
    original_fill_block = worker_cls._fill_block_tensor
    original_fill_state = worker_cls._fill_state_tensor

    @functools.wraps(original_init)
    def init(self, vllm_config, kv_cache_config):
        self._hcu_vllm_config = vllm_config
        global_mode = _global_fill_mode(vllm_config)
        groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
        mamba_specs = {}
        modes = {}
        for group_idx, group in enumerate(groups):
            spec = _mamba_spec_of(group)
            if spec is not None:
                mamba_specs[group_idx] = spec
                modes[group_idx] = _GROUP_MODE_MAMBA
            else:
                modes[group_idx] = _group_fill_mode(group, global_mode)
        setattr(self, _GROUP_MODES_ATTR, modes)
        setattr(self, _MAMBA_SPECS_ATTR, mamba_specs)
        result = original_init(self, vllm_config, kv_cache_config)
        # TEMP DIAGNOSTIC: dump each group's mode/spec/flavours and layer names
        # so an unsupported cache layout can be identified from the serve log.
        try:
            group_to_layers = getattr(self, "group_to_layers", None) or {}
            for group_idx, group in enumerate(groups):
                spec = getattr(group, "kv_cache_spec", None)
                flavours = [
                    str(getattr(layer_spec, "cache_dtype_str", "") or "")
                    for layer_spec in _iter_layer_specs(spec)
                ]
                _LOG.info(
                    "[decode-bench-kvdbg] group=%d mode=%s spec=%s "
                    "page_size_bytes=%s block_size=%s flavours=%s "
                    "mamba_shapes=%s layers=%s",
                    group_idx,
                    modes.get(group_idx),
                    type(spec).__name__,
                    getattr(spec, "page_size_bytes", None),
                    getattr(spec, "block_size", None),
                    flavours,
                    mamba_specs.get(group_idx, (None,))[0],
                    list(group_to_layers.get(group_idx, ())),
                )
        except Exception:  # pragma: no cover - diagnostics must never fail arm
            _LOG.exception("[decode-bench-kvdbg] group dump failed")
        return result

    @functools.wraps(original_fill_blocks)
    def _fill_blocks(self, group_idx, block_ids, num_tokens):
        modes = getattr(self, _GROUP_MODES_ATTR, {})
        mode = modes.get(group_idx, _GROUP_MODE_UNKNOWN)
        if mode in (_GROUP_MODE_PACKED, _GROUP_MODE_MIXED):
            # Refuse BEFORE upstream touches any layer in this group so the
            # refusal guarantees zero writes for byte, float and state caches
            # alike, regardless of layer iteration order.
            _raise_for_mode(mode)
        previous = getattr(self, _CURRENT_MODE_ATTR, None)
        previous_group = getattr(self, _CURRENT_GROUP_ATTR, None)
        setattr(self, _CURRENT_MODE_ATTR, mode)
        setattr(self, _CURRENT_GROUP_ATTR, group_idx)
        try:
            return original_fill_blocks(self, group_idx, block_ids, num_tokens)
        finally:
            setattr(self, _CURRENT_MODE_ATTR, previous)
            setattr(self, _CURRENT_GROUP_ATTR, previous_group)

    @functools.wraps(original_fill_block)
    def _fill_block_tensor(self, kv_cache, block_ids, fill_mean, fill_std):
        mode = getattr(self, _CURRENT_MODE_ATTR, _GROUP_MODE_UNKNOWN)
        if kv_cache.is_floating_point() or kv_cache.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ):
            # Native float/FP8 storage: the tensor dtype quantizes already.
            return original_fill_block(self, kv_cache, block_ids, fill_mean, fill_std)
        if kv_cache.dtype not in (torch.uint8, torch.int8):
            return original_fill_block(self, kv_cache, block_ids, fill_mean, fill_std)
        if mode == _GROUP_MODE_MAMBA:
            group_idx = getattr(self, _CURRENT_GROUP_ATTR, -1)
            spec = getattr(self, _MAMBA_SPECS_ATTR, {}).get(group_idx)
            if spec is None:
                _raise_for_mode(_GROUP_MODE_UNKNOWN)
            shapes, dtypes = spec
            entry = _mamba_page_pattern(
                shapes, dtypes, int(kv_cache.shape[-1]),
                fill_mean, fill_std, kv_cache.device,
            )
            block_ids_tensor = torch.tensor(
                block_ids, dtype=torch.long, device=kv_cache.device
            )
            valid_block_ids = block_ids_tensor[
                block_ids_tensor < kv_cache.shape[0]
            ]
            if len(valid_block_ids) == 0:
                return None
            kv_cache[valid_block_ids] = _encode_like(entry, kv_cache.dtype)
            return None
        if mode == _GROUP_MODE_DS_MLA:
            if kv_cache.shape[-1] != _DS_MLA_WIDTH:
                _LOG.info(
                    "[decode-bench-kvdbg] ds_mla group reached a tensor with "
                    "unexpected entry width: shape=%s dtype=%s -> falling "
                    "back to upstream fill (layout support pending)",
                    tuple(kv_cache.shape),
                    kv_cache.dtype,
                )
                return original_fill_block(
                    self, kv_cache, block_ids, fill_mean, fill_std
                )
            entry = _ds_mla_pattern(fill_mean, fill_std, kv_cache.device)
            block_ids_tensor = torch.tensor(
                block_ids, dtype=torch.long, device=kv_cache.device
            )
            valid_block_ids = block_ids_tensor[
                block_ids_tensor < kv_cache.shape[0]
            ]
            if len(valid_block_ids) == 0:
                return None
            kv_cache[valid_block_ids] = _encode_like(entry, kv_cache.dtype)
            return None
        if mode != _GROUP_MODE_PLAIN:
            if mode in (_GROUP_MODE_PACKED, _GROUP_MODE_MIXED):
                _raise_for_mode(mode)
            # Unknown byte layouts keep upstream semantics (a warning marks the
            # gap) instead of aborting a serve on a bench-only fill path.
            _LOG.warning(
                "[decode-bench-fp8-fill] unrecognised byte cache layout "
                "(mode=%s shape=%s dtype=%s); falling back to upstream fill",
                mode,
                tuple(kv_cache.shape),
                kv_cache.dtype,
            )
            return original_fill_block(self, kv_cache, block_ids, fill_mean, fill_std)

        block_ids_tensor = torch.tensor(
            block_ids, dtype=torch.long, device=kv_cache.device
        )
        valid_mask = block_ids_tensor < kv_cache.shape[0]
        valid_block_ids = block_ids_tensor[valid_mask]
        if len(valid_block_ids) == 0:
            return None
        target_shape = (len(valid_block_ids),) + tuple(kv_cache.shape[1:])
        if fill_std > 0:
            fill_bytes = _encode_random(
                fill_mean, fill_std, target_shape, kv_cache.device
            )
        else:
            fill_bytes = torch.full(
                target_shape,
                _encode_scalar(fill_mean),
                dtype=torch.uint8,
                device=kv_cache.device,
            )
        kv_cache[valid_block_ids] = _encode_like(fill_bytes, kv_cache.dtype)
        return None

    @functools.wraps(original_fill_state)
    def _fill_state_tensor(self, kv_cache, fill_mean, fill_std):
        # Same call signature as upstream; state tensors (Mamba/KDA-style
        # buffers kept as float/complex storages) keep upstream semantics --
        # only per-group block-indexed scalar-FP8 byte caches are re-encoded.
        return original_fill_state(self, kv_cache, fill_mean, fill_std)

    for wrapper, original in (
        (init, original_init),
        (_fill_blocks, original_fill_blocks),
        (_fill_block_tensor, original_fill_block),
        (_fill_state_tensor, original_fill_state),
    ):
        wrapper.__name__ = original.__name__
        wrapper.__qualname__ = original.__qualname__
        wrapper.__doc__ = original.__doc__
        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
        setattr(wrapper, _WRAPPER_FLAG, True)

    worker_cls.__init__ = init
    worker_cls._fill_blocks = _fill_blocks
    worker_cls._fill_block_tensor = _fill_block_tensor
    worker_cls._fill_state_tensor = _fill_state_tensor
    setattr(worker_cls, _PATCHED_FLAG, True)

    _LOG.info(
        "[decode-bench-fp8-fill] armed: e4m3fn byte encoding for groups whose "
        "layers are all scalar per-tensor FP8 (0.015 -> byte 0x08 / value "
        "0.015625); fp8_ds_mla groups filled with the 656-byte block-scaled "
        "layout (tile e4m3 values + unit fp32 scales + bf16 RoPE); other "
        "mixed/packed groups refuse clearly; float, state and zero-fill paths "
        "keep upstream semantics."
    )
    return True


def apply_to_module(module: ModuleType) -> bool:
    try:
        return _apply(module)
    except Exception:
        _log = logging.getLogger("vllm_hcu.patch.decode_bench_fp8_fill")
        _log.exception("[decode-bench-fp8-fill] failed to arm (fail closed)")
        raise


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply_to_module",
]
