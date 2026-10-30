# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Trim GLM5Next kpool sparse-indexer decode elementwise kernels.

Two textual edits to ``sparse_attn_indexer_kpool``, enabled by default
(disable via ``VLLM_HCU_GLM5NEXT_KPOOL_ELEMENT_FUSION=0``):

1. Drop ``positions.to(torch.int32)`` before
   ``kpool_decode_update_and_maybe_write_cache_batched``. The kernel only does
   compare / ``%`` / subtract on the loaded position, so it reads int64
   directly; ``_scatter_decode_tokens_by_request`` is dtype-agnostic too.
   Removes one copy kernel per layer.
2. Pure-decode short-context batches (``max_seq_len <= topk_tokens``): replace
   ``topk_indices_buffer[:N] = -1`` followed by ``_fill_causal_indices``
   (arange / to / fill / gt / index_put) with one Triton kernel that writes the
   full-width rows directly: decode rows get ``col if col <= pos else -1``,
   padding rows get ``-1``. The result is bit-identical to upstream.

Prefill, mixed and long-context batches keep the upstream fill unchanged:
the long-context topk/expand path only writes ``[:topk + kpool - 1]`` columns
and relies on the fill for the 128-aligned padding columns and padding rows.
"""

from __future__ import annotations

import inspect
import linecache
from functools import cache
from types import ModuleType

import torch
from vllm.logger import init_logger

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)
from .patch_glm5next_kpool_fuse_expand import _DECORATOR_LINE, _INDEXER_SIGNATURE

TARGET_MODULE = "vllm.model_executor.layers.sparse_attn_indexer_kpool"
PATCH_ID = "worker.core_fix.glm5next.kpool_element_fusion"
ENV_VAR = "VLLM_HCU_GLM5NEXT_KPOOL_ELEMENT_FUSION"
_PATCH_MARKER = "_vllm_hcu_glm5next_kpool_element_fusion_applied"
_WRAPPER_MARKER = "_vllm_hcu_glm5next_kpool_element_fusion_wrapper"
_SOURCE_FILENAME = f"<{PATCH_ID}>"

logger = init_logger(__name__)


@cache
def _short_decode_fill_kernel():
    # Built lazily so importing the adapter never initialises Triton.
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _kernel(
        out_ptr,  # [n_rows, width] int32, row stride out_s0
        positions_ptr,  # [>= n_decode] any int dtype, 0-based position
        n_decode,
        width,
        out_s0,
        BLOCK_COLS: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        # Padding rows (row >= n_decode) load -1, so every column becomes -1.
        pos = tl.load(positions_ptr + row, mask=row < n_decode, other=-1)
        val = tl.where(cols <= pos, cols, -1).to(tl.int32)
        tl.store(out_ptr + row * out_s0 + cols, val, mask=cols < width)

    return _kernel


def fused_short_decode_fill(
    topk_indices_buffer: torch.Tensor,
    positions: torch.Tensor,
    num_decode_tokens: int,
    num_rows: int,
) -> None:
    """Equivalent to ``buf[:num_rows] = -1`` followed by
    ``_fill_causal_indices(buf[:num_decode_tokens], positions[:num_decode_tokens])``."""
    if num_rows == 0:
        return
    assert topk_indices_buffer.dtype == torch.int32
    assert topk_indices_buffer.stride(1) == 1
    assert 0 < num_decode_tokens <= num_rows <= topk_indices_buffer.shape[0]
    assert positions.ndim == 1 and positions.shape[0] >= num_decode_tokens
    from vllm.triton_utils import triton

    width = topk_indices_buffer.shape[1]
    block_cols = 256
    _short_decode_fill_kernel()[(num_rows, triton.cdiv(width, block_cols))](
        topk_indices_buffer,
        positions,
        num_decode_tokens,
        width,
        topk_indices_buffer.stride(0),
        BLOCK_COLS=block_cols,
    )


def _enabled() -> bool:
    import vllm_hcu.platforms.envs as henvs

    return henvs.VLLM_HCU_GLM5NEXT_KPOOL_ELEMENT_FUSION and henvs.VLLM_HCU_USE_CUSTOM_OPS


# (label, old, new). Every old block must match exactly once.
_EDITS: tuple[tuple[str, str, str], ...] = (
    (
        "fill",
        """\
    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
""",
        """\
    # HCU: pure short decode writes full-width rows in one kernel below, so
    # the whole-buffer -1 fill is only needed on the other paths.
    _hcu_short_decode = (
        has_decode
        and not has_prefill
        and positions is not None
        and positions.numel() > 0
        and current_platform.is_cuda_alike()
        and attn_metadata_narrowed.max_seq_len <= topk_tokens
    )
    if not _hcu_short_decode:
        topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
""",
    ),
    (
        "nonuniform_dec_pos",
        """\
                dec_pos = _scatter_decode_tokens_by_request(
                    positions[:num_decode_tokens].to(torch.int32),
""",
        """\
                dec_pos = _scatter_decode_tokens_by_request(
                    positions[:num_decode_tokens],
""",
    ),
    (
        "uniform_dec_pos",
        """\
                dec_pos = positions[:num_decode_tokens].to(torch.int32).view(shape2)
""",
        """\
                dec_pos = positions[:num_decode_tokens].view(shape2)
""",
    ),
    (
        "short_decode",
        """\
        if current_platform.is_cuda_alike() and _fill_short_decode_causal_indices(
""",
        """\
        if _hcu_short_decode:
            _vllm_hcu_fused_short_decode_fill(
                topk_indices_buffer,
                positions,
                num_decode_tokens,
                hidden_states.shape[0],
            )
            return topk_indices_buffer
        if current_platform.is_cuda_alike() and _fill_short_decode_causal_indices(
""",
    ),
)


def _rewrite_source(source: str) -> str:
    if source.startswith(_DECORATOR_LINE):
        source = source[len(_DECORATOR_LINE) :]
    for label, old, new in _EDITS:
        count = source.count(old)
        if count != 1:
            raise PatchCompatibilityError(
                f"{TARGET_MODULE}.sparse_attn_indexer_kpool {label} block "
                f"matched {count} times; expected exactly 1"
            )
        source = source.replace(old, new, 1)
    return source


def _patch_indexer(indexer_mod: ModuleType) -> None:
    target = f"{TARGET_MODULE}.sparse_attn_indexer_kpool"
    original = require_callable(indexer_mod, "sparse_attn_indexer_kpool", target)
    require_exact_signature(original, target, **_INDEXER_SIGNATURE)
    require_callable(
        indexer_mod,
        "_fill_short_decode_causal_indices",
        f"{TARGET_MODULE}._fill_short_decode_causal_indices",
    )
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError) as exc:
        raise PatchCompatibilityError(
            f"cannot read source of {target}; another patch may have replaced "
            "it without registering its source"
        ) from exc
    source = _rewrite_source(source)

    # Register the rewritten source so tracebacks and any later textual patch
    # (inspect.getsource) can still see it.
    linecache.cache[_SOURCE_FILENAME] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        _SOURCE_FILENAME,
    )
    # Compile against the module's own globals so private helpers resolve.
    indexer_mod.__dict__["_vllm_hcu_fused_short_decode_fill"] = (
        fused_short_decode_fill
    )
    namespace: dict[str, object] = {}
    # dont_inherit: keep this file's `from __future__ import annotations` from
    # turning the rewritten function's annotations into strings.
    code = compile(source, _SOURCE_FILENAME, "exec", dont_inherit=True)
    exec(code, indexer_mod.__dict__, namespace)
    patched = namespace["sparse_attn_indexer_kpool"]
    # Set before re-wrapping: functools.wraps copies __dict__ onto the wrapper.
    setattr(patched, _WRAPPER_MARKER, True)
    patched = indexer_mod.eager_break_during_capture(patched)

    setattr(
        indexer_mod, "_vllm_hcu_element_fusion_original_indexer_kpool", original
    )
    setattr(indexer_mod, "sparse_attn_indexer_kpool", patched)
    setattr(indexer_mod, _PATCH_MARKER, True)


def apply_to_module(module: ModuleType) -> bool:
    indexer_mod = load_exact_module(TARGET_MODULE, module)
    if getattr(indexer_mod, _PATCH_MARKER, False):
        if not getattr(indexer_mod.sparse_attn_indexer_kpool, _WRAPPER_MARKER, False):
            raise PatchCompatibilityError(
                "kpool element-fusion patch marker is stale"
            )
        # The coordinator requires repeated calls to return False.
        return False
    if not _enabled():
        logger.info("%s disabled (set %s=1 to enable)", PATCH_ID, ENV_VAR)
        return False
    _patch_indexer(indexer_mod)
    logger.info("%s applied", PATCH_ID)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "TARGET_MODULE",
    "PATCH_ID",
    "ENV_VAR",
    "apply",
    "apply_to_module",
    "fused_short_decode_fill",
]
