# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Env-gated fuse of the GLM5Next kpool indexer's pool-expand step.

Removes 4 host-side elementwise kernels (pool_topk int64 cast, positions
int32 cast + add, final topk_indices_buffer copy) from the sparse-indexer
decode path by pushing the `+1` into the Triton kernel and letting it write
straight into the caller's buffer. Gated by VLLM_HCU_GLM5NEXT_KPOOL_FUSE_EXPAND
(default on) so it can be disabled if a future vLLM version changes the
upstream function body underneath the textual patch below.
"""

from __future__ import annotations

import inspect
import linecache
from types import ModuleType

import torch
from vllm.logger import init_logger

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.layers.sparse_attn_indexer_kpool"
PATCH_ID = "worker.core_fix.glm5next.kpool_fuse_expand"
ENV_VAR = "VLLM_HCU_GLM5NEXT_KPOOL_FUSE_EXPAND"
_PATCH_MARKER = "_vllm_hcu_glm5next_kpool_fuse_expand_applied"
_WRAPPER_MARKER = "_vllm_hcu_glm5next_kpool_fuse_expand_wrapper"
_DECORATOR_LINE = "@eager_break_during_capture\n"

logger = init_logger(__name__)

_INDEXER_SIGNATURE = dict(
    positional=(
        "hidden_states",
        "k_cache_prefix",
        "kv_cache",
        "q_quant",
        "q_scale",
        "k",
        "weights",
        "quant_block_size",
        "scale_fmt",
        "topk_tokens",
        "head_dim",
        "max_model_len",
        "total_seq_lens",
        "topk_indices_buffer",
        "skip_k_cache_insert",
        "use_fp4_cache",
        "gate_score",
        "compress_ape",
        "index_kpool",
        "positions",
        "tail_kv_cache",
        "tail_prefix",
    ),
    defaults={
        "use_fp4_cache": False,
        "gate_score": None,
        "compress_ape": None,
        "index_kpool": 1,
        "positions": None,
        "tail_kv_cache": None,
        "tail_prefix": None,
    },
)

# Prefill branch: drop the int64 pool_ids cast and the separate
# positions.to(int32)+1 precompute (expand_pools_and_append_tail now derives
# seq_len = positions + 1 inside the Triton kernel).
_PREFILL_OLD = """\
            if index_kpool > 1:
                pool_ids = pool_topk.to(torch.int64)
                if positions is not None:
                    # Fused expand-pools + append-tail into one Triton kernel
                    # (replaces ~25 elementwise ops). seq_len is token-granular
                    # (pos+1); the kernel derives pool_len internally.
                    q_seq = (
                        positions[chunk.token_start : chunk.token_end].to(torch.int32)
                        + 1
                    )
                    expanded = kpool_ops.expand_pools_and_append_tail(
                        pool_ids, q_seq, index_kpool
                    )
"""
_PREFILL_NEW = """\
            if index_kpool > 1:
                pool_ids = pool_topk
                if positions is not None:
                    # Fused expand-pools + append-tail into one Triton kernel
                    # (replaces ~25 elementwise ops). The kernel derives
                    # seq_len = positions + 1 and pool_len internally.
                    q_pos = positions[chunk.token_start : chunk.token_end]
                    expanded = kpool_ops.expand_pools_and_append_tail(
                        pool_ids, q_pos, index_kpool
                    )
"""

# Decode branch: same int64/seq_len precompute removal, plus write directly
# into topk_indices_buffer (skip the final whole-buffer copy) whenever the
# non-padded layout lets the kernel's output land at the right offset already.
_DECODE_OLD = """\
        if index_kpool > 1:
            pool_ids = pool_topk.to(torch.int64)
            n = pool_topk.shape[0]
            # Decode seq_lens are pool-granular; recover token lengths from
            # positions using the padded [B, next_n] row layout when needed.
            if positions is not None:
                dec_seq = _decode_topk_seq_lens(
                    positions,
                    decode_lens,
                    num_decode_tokens,
                    batch_size,
                    next_n,
                    decode_metadata.requires_padding,
                )
            else:
                dec_seq = decode_metadata.seq_lens[:n]
                if dec_seq.ndim == 2:
                    dec_seq = dec_seq[:, -1]
                dec_seq = dec_seq.to(torch.int32)
            out = kpool_ops.expand_pools_and_append_tail(pool_ids, dec_seq, index_kpool)
        else:
            out = topk_dst

        if decode_metadata.requires_padding:
            # Drop padded query rows introduced by the next_n padding above.
            out = unpack_seq_triton(
                out.reshape(batch_size, -1, out.shape[-1]), decode_lens
            )
        topk_indices_buffer[: out.shape[0], : out.shape[-1]] = out
"""
_DECODE_NEW = """\
        _kpool_out_in_place = False
        if index_kpool > 1:
            pool_ids = pool_topk
            n = pool_topk.shape[0]
            # Decode seq_lens are pool-granular; recover token lengths from
            # positions using the padded [B, next_n] row layout when needed.
            # expand_pools_and_append_tail adds 1 internally and casts the
            # loaded position to int32 itself, so pass raw positions through
            # (no host-side cast).
            if positions is not None:
                if not decode_metadata.requires_padding:
                    dec_pos = positions[: batch_size * next_n]
                else:
                    _scatter_idx = _build_decode_scatter_indices(
                        decode_lens, batch_size, num_decode_tokens
                    )
                    dec_pos = _scatter_decode_tokens_by_request(
                        positions[:num_decode_tokens],
                        -1,
                        batch_size,
                        next_n,
                        _scatter_idx,
                    ).reshape(batch_size * next_n)
            else:
                dec_pos = decode_metadata.seq_lens[:n]
                if dec_pos.ndim == 2:
                    dec_pos = dec_pos[:, -1]
                dec_pos = dec_pos.to(torch.int32) - 1
            out_cols = pool_ids.shape[1] * index_kpool + index_kpool - 1
            # Write straight into topk_indices_buffer only when pool_ids does
            # not alias it: the kernel's column tiles run unordered, so an
            # aliased source could be overwritten before other tiles read it.
            dst = (
                topk_indices_buffer[:n, :out_cols]
                if not decode_metadata.requires_padding
                and pool_ids.untyped_storage().data_ptr()
                != topk_indices_buffer.untyped_storage().data_ptr()
                else None
            )
            out = kpool_ops.expand_pools_and_append_tail(
                pool_ids, dec_pos, index_kpool, out=dst
            )
            _kpool_out_in_place = dst is not None
        else:
            out = topk_dst

        if decode_metadata.requires_padding:
            # Drop padded query rows introduced by the next_n padding above.
            out = unpack_seq_triton(
                out.reshape(batch_size, -1, out.shape[-1]), decode_lens
            )
        if not _kpool_out_in_place:
            # The non-padded index_kpool > 1 path already wrote into
            # topk_indices_buffer via ``out=dst``; skip the self-copy.
            topk_indices_buffer[: out.shape[0], : out.shape[-1]] = out
"""


def _enabled() -> bool:
    import vllm_hcu.platforms.envs as henvs

    return henvs.VLLM_HCU_GLM5NEXT_KPOOL_FUSE_EXPAND and henvs.VLLM_HCU_USE_CUSTOM_OPS


def _already_fused_expand_signature(fn) -> bool:
    """True if `fn(pool_ids, positions, pool_size, out=None)` already."""
    try:
        params = tuple(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    names = tuple(p.name for p in params)
    if names != ("pool_ids", "positions", "pool_size", "out"):
        return False
    return params[-1].default is None


def _patch_kpool_ops(kpool_ops: ModuleType) -> None:
    """Swap expand_pools_and_append_tail for a positions+out-taking version."""
    import triton
    import triton.language as tl

    original = require_callable(
        kpool_ops,
        "expand_pools_and_append_tail",
        f"{kpool_ops.__name__}.expand_pools_and_append_tail",
    )
    if getattr(kpool_ops, _PATCH_MARKER, False):
        return
    if _already_fused_expand_signature(original):
        # The installed vLLM source already carries this fix directly (e.g.
        # hand-patched outside vllm_hcu). Nothing to do; mark so later
        # callers can tell this module was audited.
        logger.info(
            "%s: %s.expand_pools_and_append_tail already has the fused "
            "signature; skipping",
            PATCH_ID,
            kpool_ops.__name__,
        )
        setattr(kpool_ops, _PATCH_MARKER, True)
        return
    require_exact_signature(
        original,
        f"{kpool_ops.__name__}.expand_pools_and_append_tail",
        positional=("pool_ids", "seq_lens", "pool_size"),
    )

    @triton.jit
    def _fused_expand_pools_and_append_tail_kernel(
        pool_ids_ptr,
        positions_ptr,  # [rows], any int dtype (0-based token position)
        out_ptr,
        topk,
        out_cols,
        POOL_SIZE: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        pid_s0,
        out_s0,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        cols = tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        mask = cols < out_cols

        seq_len = tl.load(positions_ptr + row).to(tl.int32) + 1
        pool_len = seq_len // POOL_SIZE
        tail_start = pool_len * POOL_SIZE
        tail_count = seq_len - tail_start

        is_history = cols < topk
        g = cols // POOL_SIZE
        o = cols % POOL_SIZE
        pid = tl.load(
            pool_ids_ptr + row * pid_s0 + g, mask=mask & is_history, other=-1
        )
        hist_val = (pid * POOL_SIZE + o).to(tl.int32)
        hist_out = tl.where(pid >= 0, hist_val, -1)

        tail_off = cols - topk
        is_tail = (tail_off >= 0) & (tail_off < tail_count)
        tail_val = (tail_start + tail_off).to(tl.int32)
        tail_out = tl.where(is_tail, tail_val, -1)

        result = tl.where(is_history, hist_out, tail_out)
        tl.store(out_ptr + row * out_s0 + cols, result, mask=mask)

    def fused_expand_pools_and_append_tail(
        pool_ids: torch.Tensor,
        positions: torch.Tensor,
        pool_size: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rows, n_groups = pool_ids.shape
        topk = n_groups * pool_size
        out_cols = topk + pool_size - 1
        if out is None:
            out = torch.empty(
                (rows, out_cols), dtype=torch.int32, device=pool_ids.device
            )
        else:
            assert out.shape == (rows, out_cols) and out.dtype == torch.int32
        BLOCK_COLS = 128
        n_tiles = triton.cdiv(out_cols, BLOCK_COLS)
        _fused_expand_pools_and_append_tail_kernel[(rows, n_tiles)](
            pool_ids,
            positions,
            out,
            topk,
            out_cols,
            POOL_SIZE=pool_size,
            BLOCK_COLS=BLOCK_COLS,
            pid_s0=pool_ids.stride(0),
            out_s0=out.stride(0),
        )
        return out

    setattr(fused_expand_pools_and_append_tail, _WRAPPER_MARKER, True)
    setattr(kpool_ops, "_vllm_hcu_original_expand_pools_and_append_tail", original)
    setattr(
        kpool_ops, "expand_pools_and_append_tail", fused_expand_pools_and_append_tail
    )
    setattr(kpool_ops, _PATCH_MARKER, True)


def _patch_indexer_decode(indexer_mod: ModuleType) -> None:
    """Textually patch the 3 hot lines inside `sparse_attn_indexer_kpool`.

    `forward_cuda` looks the function up via LOAD_GLOBAL on every call, so
    reassigning the module attribute takes effect immediately. The
    replacement is compiled with `indexer_mod.__dict__` as its globals (NOT
    `original.__globals__` — `original` is `@eager_break_during_capture`-
    wrapped via functools.wraps, which copies __name__/__doc__/__dict__/
    __wrapped__ but never touches __globals__; wrapper.__globals__ is always
    breakable_cudagraph.py's own module dict, not this one's) so every
    private helper the function body references (`_resolve_layer_name`,
    `kpool_ops`, `_build_decode_scatter_indices`, ...) resolves normally
    instead of needing to be duplicated here.
    """
    original = require_callable(
        indexer_mod,
        "sparse_attn_indexer_kpool",
        f"{TARGET_MODULE}.sparse_attn_indexer_kpool",
    )
    if getattr(indexer_mod, _PATCH_MARKER, False):
        if not getattr(
            getattr(indexer_mod, "sparse_attn_indexer_kpool"), _WRAPPER_MARKER, False
        ):
            raise PatchCompatibilityError(
                "kpool fuse-expand patch marker is stale"
            )
        return
    require_exact_signature(
        original,
        f"{TARGET_MODULE}.sparse_attn_indexer_kpool",
        **_INDEXER_SIGNATURE,
    )

    source = inspect.getsource(original)
    # Drop the decorator line: the rewritten function is re-wrapped exactly
    # once below, so keeping it here would apply the decorator twice.
    if source.startswith(_DECORATOR_LINE):
        source = source[len(_DECORATOR_LINE) :]
    changed = False
    for label, old, new in (
        ("prefill", _PREFILL_OLD, _PREFILL_NEW),
        ("decode", _DECODE_OLD, _DECODE_NEW),
    ):
        old_count = source.count(old)
        if old_count == 1:
            source = source.replace(old, new, 1)
            changed = True
            continue
        if old_count > 1:
            raise PatchCompatibilityError(
                f"{TARGET_MODULE}.sparse_attn_indexer_kpool {label} block "
                f"matched {old_count} times; expected 0 or 1"
            )
        # old_count == 0: either already fused (hand-patched outside
        # vllm_hcu, or a previous run of this patch) or upstream changed
        # underneath us. Distinguish those two cases explicitly rather than
        # silently accepting any unrecognized source.
        if source.count(new) != 1:
            raise PatchCompatibilityError(
                f"{TARGET_MODULE}.sparse_attn_indexer_kpool {label} block no "
                "longer matches either the pre-fix or post-fix source text; "
                "vLLM must have changed this function"
            )
        logger.info(
            "%s: %s block already matches the fused source; leaving as-is",
            PATCH_ID,
            label,
        )

    if not changed:
        # Both blocks were already fused upstream (e.g. a direct, unconditional
        # edit to the installed vLLM source made outside this patch). There is
        # nothing left to rebind; mark it audited so re-imports are cheap.
        setattr(indexer_mod, _PATCH_MARKER, True)
        return

    # Register the rewritten source so later textual patches on the same
    # function (e.g. kpool_element_fusion) can still inspect.getsource it.
    filename = f"<{PATCH_ID}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace: dict[str, object] = {}
    code = compile(source, filename, "exec")
    exec(code, indexer_mod.__dict__, namespace)
    patched = namespace["sparse_attn_indexer_kpool"]
    # functools.wraps(fn) inside eager_break_during_capture copies fn.__dict__
    # onto the new wrapper, so setting the marker before re-wrapping makes it
    # survive onto whatever gets installed below (plain fn, if breakable
    # cudagraph capture is disabled, or the wrapper, if enabled).
    setattr(patched, _WRAPPER_MARKER, True)
    # The decorator line was stripped from the source above, so this is the
    # only application; it matches upstream's single @eager_break_during_capture.
    patched = indexer_mod.eager_break_during_capture(patched)

    setattr(indexer_mod, "_vllm_hcu_original_sparse_attn_indexer_kpool", original)
    setattr(indexer_mod, "sparse_attn_indexer_kpool", patched)
    setattr(indexer_mod, _PATCH_MARKER, True)


def apply_to_module(module: ModuleType) -> bool:
    indexer_mod = load_exact_module(TARGET_MODULE, module)
    if getattr(indexer_mod, _PATCH_MARKER, False):
        # Already applied (or audited as already-fused) by a prior call.
        # The coordinator's idempotency check requires every call after the
        # first to return False.
        return False
    if not _enabled():
        logger.info("%s disabled (set %s=1 to enable)", PATCH_ID, ENV_VAR)
        return False
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        from vllm.models.glm5next.amd.ops import kpool_compress as kpool_ops
    else:
        from vllm.models.glm5next.nvidia.ops import kpool_compress as kpool_ops
    _patch_kpool_ops(kpool_ops)
    _patch_indexer_decode(indexer_mod)
    logger.info("%s applied", PATCH_ID)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["TARGET_MODULE", "PATCH_ID", "ENV_VAR", "apply", "apply_to_module"]
