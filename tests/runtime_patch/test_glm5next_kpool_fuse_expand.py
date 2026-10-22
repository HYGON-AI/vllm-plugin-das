# SPDX-License-Identifier: Apache-2.0
"""Tests for the VLLM_HCU_GLM5NEXT_KPOOL_FUSE_EXPAND runtime patch."""

import functools
import inspect
import textwrap
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import (
    patch_glm5next_kpool_element_fusion as fusion_patch,
)
from vllm_hcu.patch.worker.core_fix import (
    patch_glm5next_kpool_fuse_expand as fuse_patch,
)
from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU for Triton kernels"
)


def _kpool_ops_module():
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        from vllm.models.glm5next.amd.ops import kpool_compress as kpool_ops
    else:
        from vllm.models.glm5next.nvidia.ops import kpool_compress as kpool_ops
    return kpool_ops


@pytest.fixture
def indexer_module(monkeypatch):
    # Import the model package first: importing the indexer module directly
    # re-enters it through glm5next/nvidia/attention.py (circular import).
    import vllm.models.glm5next  # noqa: F401
    import vllm.model_executor.layers.sparse_attn_indexer_kpool as module

    # Undo any rewrite this test performs so other tests see upstream code.
    monkeypatch.setattr(
        module, "sparse_attn_indexer_kpool", module.sparse_attn_indexer_kpool
    )
    monkeypatch.delattr(module, fuse_patch._PATCH_MARKER, raising=False)
    return module


@pytest.fixture
def kpool_ops(monkeypatch, indexer_module):
    ops = _kpool_ops_module()
    monkeypatch.setattr(
        ops, "expand_pools_and_append_tail", ops.expand_pools_and_append_tail
    )
    monkeypatch.delattr(ops, fuse_patch._PATCH_MARKER, raising=False)
    return ops


@pytest.fixture
def original_expand(kpool_ops):
    return kpool_ops.expand_pools_and_append_tail


@pytest.fixture
def fused_expand(kpool_ops):
    fuse_patch._patch_kpool_ops(kpool_ops)
    return kpool_ops.expand_pools_and_append_tail


def test_switch_disabled_by_default(monkeypatch):
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.delenv(fuse_patch.ENV_VAR, raising=False)
    assert henvs.VLLM_HCU_GLM5NEXT_KPOOL_FUSE_EXPAND is False


@pytest.mark.parametrize(
    ("switch", "custom_ops", "expected"),
    [("1", "1", True), ("true", "1", True), ("0", "1", False), ("1", "0", False)],
)
def test_enabled_requires_switch_and_custom_ops(
    monkeypatch, switch, custom_ops, expected
):
    monkeypatch.setenv(fuse_patch.ENV_VAR, switch)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", custom_ops)
    assert fuse_patch._enabled() is expected


def test_disabled_leaves_modules_untouched(monkeypatch, indexer_module, kpool_ops):
    monkeypatch.setenv(fuse_patch.ENV_VAR, "0")
    original_indexer = indexer_module.sparse_attn_indexer_kpool
    original_expand = kpool_ops.expand_pools_and_append_tail

    assert fuse_patch.apply_to_module(indexer_module) is False
    assert indexer_module.sparse_attn_indexer_kpool is original_indexer
    assert kpool_ops.expand_pools_and_append_tail is original_expand
    assert not hasattr(indexer_module, fuse_patch._PATCH_MARKER)


def test_custom_ops_off_leaves_module_untouched(monkeypatch, indexer_module):
    monkeypatch.setenv(fuse_patch.ENV_VAR, "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    original = indexer_module.sparse_attn_indexer_kpool

    assert fuse_patch.apply_to_module(indexer_module) is False
    assert indexer_module.sparse_attn_indexer_kpool is original


def test_upstream_source_matches_pre_fix_blocks(indexer_module):
    source = inspect.getsource(indexer_module.sparse_attn_indexer_kpool)
    assert source.count(fuse_patch._PREFILL_OLD) == 1
    assert source.count(fuse_patch._DECODE_OLD) == 1


def test_enabled_rewrites_indexer_and_kpool_ops_once(
    monkeypatch, indexer_module, kpool_ops
):
    monkeypatch.setenv(fuse_patch.ENV_VAR, "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    original = indexer_module.sparse_attn_indexer_kpool
    original_expand = kpool_ops.expand_pools_and_append_tail

    assert fuse_patch.apply_to_module(indexer_module) is True

    patched = indexer_module.sparse_attn_indexer_kpool
    assert patched is not original
    assert getattr(patched, fuse_patch._WRAPPER_MARKER, False)
    # compile() inherits the patch module's `from __future__ import
    # annotations`, so annotations become strings; names/kinds/defaults
    # must still match exactly.
    assert [
        (p.name, p.kind, p.default)
        for p in inspect.signature(patched).parameters.values()
    ] == [
        (p.name, p.kind, p.default)
        for p in inspect.signature(original).parameters.values()
    ]
    assert (
        indexer_module._vllm_hcu_original_sparse_attn_indexer_kpool is original
    )

    source = inspect.getsource(patched)
    assert fuse_patch._PREFILL_OLD not in source
    assert fuse_patch._DECODE_OLD not in source
    assert source.count(fuse_patch._PREFILL_NEW) == 1
    assert source.count(fuse_patch._DECODE_NEW) == 1
    assert "pool_topk.to(torch.int64)" not in source

    fused = kpool_ops.expand_pools_and_append_tail
    assert fused is not original_expand
    assert getattr(fused, fuse_patch._WRAPPER_MARKER, False)
    assert fuse_patch._already_fused_expand_signature(fused)
    assert kpool_ops._vllm_hcu_original_expand_pools_and_append_tail is (
        original_expand
    )

    # Idempotent: repeated calls return False and keep the same functions.
    assert fuse_patch.apply_to_module(indexer_module) is False
    assert indexer_module.sparse_attn_indexer_kpool is patched
    assert kpool_ops.expand_pools_and_append_tail is fused


def test_already_fused_indexer_source_is_audited_not_rebound(
    monkeypatch, indexer_module, kpool_ops
):
    monkeypatch.setenv(fuse_patch.ENV_VAR, "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    assert fuse_patch.apply_to_module(indexer_module) is True
    patched = indexer_module.sparse_attn_indexer_kpool

    # Simulate a vLLM install that already carries the fused source.
    delattr(indexer_module, fuse_patch._PATCH_MARKER)
    fuse_patch._patch_indexer_decode(indexer_module)

    assert indexer_module.sparse_attn_indexer_kpool is patched
    assert getattr(indexer_module, fuse_patch._PATCH_MARKER) is True


def _force_wrapping_decorator(monkeypatch, indexer_module):
    """Make eager_break_during_capture always wrap (as when breakable
    cudagraph capture is enabled) and record each application."""
    calls = []

    def eager_break_during_capture(fn):
        calls.append(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        indexer_module, "eager_break_during_capture", eager_break_during_capture
    )
    return calls


def _wrap_depth(fn):
    depth = 0
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
        depth += 1
    return depth


def test_rewritten_indexer_is_decorated_once(
    monkeypatch, indexer_module, kpool_ops
):
    monkeypatch.setenv(fuse_patch.ENV_VAR, "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    calls = _force_wrapping_decorator(monkeypatch, indexer_module)

    assert fuse_patch.apply_to_module(indexer_module) is True

    patched = indexer_module.sparse_attn_indexer_kpool
    assert len(calls) == 1
    assert _wrap_depth(patched) == 1
    assert getattr(patched, fuse_patch._WRAPPER_MARKER, False)
    assert not inspect.getsource(patched).startswith(fuse_patch._DECORATOR_LINE)


def test_stacked_with_element_fusion_is_decorated_once(
    monkeypatch, indexer_module, kpool_ops
):
    monkeypatch.setenv(fuse_patch.ENV_VAR, "1")
    monkeypatch.setenv(fusion_patch.ENV_VAR, "1")
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.delattr(indexer_module, fusion_patch._PATCH_MARKER, raising=False)
    calls = _force_wrapping_decorator(monkeypatch, indexer_module)

    assert fuse_patch.apply_to_module(indexer_module) is True
    assert len(calls) == 1
    assert fusion_patch.apply_to_module(indexer_module) is True
    assert len(calls) == 2

    patched = indexer_module.sparse_attn_indexer_kpool
    assert _wrap_depth(patched) == 1
    assert getattr(patched, fusion_patch._WRAPPER_MARKER, False)
    source = inspect.getsource(patched)
    assert fuse_patch._DECODE_NEW in source
    assert "_vllm_hcu_fused_short_decode_fill(" in source


def test_stale_indexer_marker_is_rejected(monkeypatch, indexer_module):
    monkeypatch.setattr(indexer_module, fuse_patch._PATCH_MARKER, True, raising=False)
    with pytest.raises(PatchCompatibilityError, match="stale"):
        fuse_patch._patch_indexer_decode(indexer_module)


def _unrecognised_indexer(
    hidden_states,
    k_cache_prefix,
    kv_cache,
    q_quant,
    q_scale,
    k,
    weights,
    quant_block_size,
    scale_fmt,
    topk_tokens,
    head_dim,
    max_model_len,
    total_seq_lens,
    topk_indices_buffer,
    skip_k_cache_insert,
    use_fp4_cache=False,
    gate_score=None,
    compress_ape=None,
    index_kpool=1,
    positions=None,
    tail_kv_cache=None,
    tail_prefix=None,
):
    return topk_indices_buffer


def test_rejects_unrecognised_indexer_source():
    module = ModuleType(fuse_patch.TARGET_MODULE)
    module.sparse_attn_indexer_kpool = _unrecognised_indexer
    with pytest.raises(PatchCompatibilityError, match="no\\s+longer matches"):
        fuse_patch._patch_indexer_decode(module)
    assert not hasattr(module, fuse_patch._PATCH_MARKER)


def test_rejects_indexer_signature_drift():
    def sparse_attn_indexer_kpool(hidden_states, topk_indices_buffer):
        return topk_indices_buffer

    module = ModuleType(fuse_patch.TARGET_MODULE)
    module.sparse_attn_indexer_kpool = sparse_attn_indexer_kpool
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        fuse_patch._patch_indexer_decode(module)


def test_already_fused_expand_signature_detection():
    def fused(pool_ids, positions, pool_size, out=None):
        pass

    def upstream(pool_ids, seq_lens, pool_size):
        pass

    def required_out(pool_ids, positions, pool_size, out):
        pass

    assert fuse_patch._already_fused_expand_signature(fused)
    assert not fuse_patch._already_fused_expand_signature(upstream)
    assert not fuse_patch._already_fused_expand_signature(required_out)
    assert not fuse_patch._already_fused_expand_signature(len)


def test_kpool_ops_with_fused_signature_is_left_as_is():
    def expand_pools_and_append_tail(pool_ids, positions, pool_size, out=None):
        pass

    ops = ModuleType("fake_kpool_ops")
    ops.expand_pools_and_append_tail = expand_pools_and_append_tail
    fuse_patch._patch_kpool_ops(ops)

    assert ops.expand_pools_and_append_tail is expand_pools_and_append_tail
    assert getattr(ops, fuse_patch._PATCH_MARKER) is True


def test_kpool_ops_signature_drift_is_rejected():
    def expand_pools_and_append_tail(pool_ids, seq_lens, pool_size, page_table):
        pass

    ops = ModuleType("fake_kpool_ops")
    ops.expand_pools_and_append_tail = expand_pools_and_append_tail
    with pytest.raises(PatchCompatibilityError, match="incompatible signature"):
        fuse_patch._patch_kpool_ops(ops)
    assert not hasattr(ops, fuse_patch._PATCH_MARKER)


def _random_pool_ids(rows, n_groups, positions, pool, dtype):
    # Valid ids are < pool_len of each row; sprinkle -1 for masked slots.
    pool_len = ((positions.to(torch.int64) + 1) // pool).clamp(min=1)
    ids = (torch.rand(rows, n_groups, device="cuda") * pool_len[:, None]).to(dtype)
    ids[torch.rand(rows, n_groups, device="cuda") < 0.2] = -1
    return ids


@requires_gpu
@pytest.mark.parametrize(
    ("rows", "n_groups", "pool", "pool_dtype", "pos_dtype"),
    [
        (1, 1, 4, torch.int32, torch.int64),
        (7, 16, 4, torch.int32, torch.int64),
        (16, 512, 4, torch.int32, torch.int32),
        (33, 64, 8, torch.int64, torch.int64),
        (5, 3, 2, torch.int32, torch.int64),
    ],
)
def test_fused_expand_matches_upstream(
    original_expand, fused_expand, rows, n_groups, pool, pool_dtype, pos_dtype
):
    torch.manual_seed(0)
    positions = torch.randint(0, 8192, (rows,), device="cuda", dtype=pos_dtype)
    positions[0] = 0
    if rows > 1:
        # Last token of a pool: tail is empty.
        positions[1] = pool * 7 - 1
    pool_ids = _random_pool_ids(rows, n_groups, positions, pool, pool_dtype)

    # Upstream call shape: int64 pool ids, token-granular int32 seq_len.
    expected = original_expand(
        pool_ids.to(torch.int64), positions.to(torch.int32) + 1, pool
    )
    actual = fused_expand(pool_ids, positions, pool)

    assert actual.dtype == torch.int32
    assert actual.shape == (rows, n_groups * pool + pool - 1)
    assert torch.equal(actual, expected)


@requires_gpu
def test_fused_expand_writes_into_strided_out(original_expand, fused_expand):
    torch.manual_seed(2)
    rows, n_groups, pool = 9, 32, 4
    out_cols = n_groups * pool + pool - 1
    positions = torch.randint(0, 4096, (rows,), device="cuda", dtype=torch.int32)
    pool_ids = _random_pool_ids(rows, n_groups, positions, pool, torch.int32)

    # Mirrors topk_indices_buffer[:n, :out_cols]: a wider, taller buffer
    # whose untouched rows/columns must keep their contents.
    buffer = torch.full((rows + 3, out_cols + 50), 12345, dtype=torch.int32,
                        device="cuda")
    sentinel = buffer.clone()
    dst = buffer[:rows, :out_cols]

    result = fused_expand(pool_ids, positions, pool, out=dst)

    assert result.data_ptr() == dst.data_ptr()
    assert torch.equal(
        buffer[:rows, :out_cols],
        original_expand(pool_ids.to(torch.int64), positions + 1, pool),
    )
    assert torch.equal(buffer[rows:], sentinel[rows:])
    assert torch.equal(buffer[:rows, out_cols:], sentinel[:rows, out_cols:])


@requires_gpu
def test_fused_expand_rejects_mismatched_out(fused_expand):
    # n_groups=4, pool=4 -> out_cols = 16 + 3 = 19.
    pool_ids = torch.zeros(2, 4, dtype=torch.int32, device="cuda")
    positions = torch.zeros(2, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError):
        fused_expand(
            pool_ids, positions, 4,
            out=torch.empty(2, 18, dtype=torch.int32, device="cuda"),
        )
    with pytest.raises(AssertionError):
        fused_expand(
            pool_ids, positions, 4,
            out=torch.empty(2, 19, dtype=torch.int64, device="cuda"),
        )


@requires_gpu
@pytest.mark.parametrize("requires_padding", [False, True])
def test_decode_positions_match_upstream_seq_lens(indexer_module, requires_padding):
    """The rewritten decode branch passes raw positions where upstream passed
    ``_decode_topk_seq_lens(...)`` (positions + 1, pad rows -> 0)."""
    batch_size, next_n = 5, 3
    if requires_padding:
        decode_lens = torch.tensor([3, 1, 2, 3, 1], device="cuda")
    else:
        decode_lens = torch.full((batch_size,), next_n, device="cuda")
    num_decode_tokens = int(decode_lens.sum())
    # Trailing prefill tokens must never leak into decode rows.
    positions = torch.randint(
        0, 4096, (num_decode_tokens + 6,), device="cuda", dtype=torch.int64
    )

    expected_seq = indexer_module._decode_topk_seq_lens(
        positions, decode_lens, num_decode_tokens, batch_size, next_n,
        requires_padding,
    )
    if not requires_padding:
        dec_pos = positions[: batch_size * next_n].to(torch.int32)
    else:
        scatter_idx = indexer_module._build_decode_scatter_indices(
            decode_lens, batch_size, num_decode_tokens
        )
        dec_pos = indexer_module._scatter_decode_tokens_by_request(
            positions[:num_decode_tokens].to(torch.int32),
            -1,
            batch_size,
            next_n,
            scatter_idx,
        ).reshape(batch_size * next_n)

    assert torch.equal(dec_pos + 1, expected_seq)


# (topk_tokens, index_kpool): the indexer selects topk_tokens // index_kpool
# pools, so non-divisible configs expand to fewer than topk_tokens columns.
_DECODE_TOPK_CONFIGS = [(2048, 4), (2049, 4), (2050, 4), (2051, 4), (4097, 8)]


def _run_decode_block(expand, pool_topk, positions, topk_tokens, index_kpool,
                      buffer):
    """Execute the rewritten non-padded decode block against ``buffer``."""
    namespace = dict(
        torch=torch,
        kpool_ops=SimpleNamespace(expand_pools_and_append_tail=expand),
        decode_metadata=SimpleNamespace(requires_padding=False),
        index_kpool=index_kpool,
        topk_tokens=topk_tokens,
        pool_topk=pool_topk,
        topk_dst=pool_topk,
        positions=positions,
        batch_size=pool_topk.shape[0],
        next_n=1,
        topk_indices_buffer=buffer,
    )
    exec(textwrap.dedent(fuse_patch._DECODE_NEW), namespace)
    return namespace["out"]


def _contract_expand(pool_ids, positions, pool_size, out=None):
    # Same shape contract as the fused helper, without the Triton launch.
    rows, n_groups = pool_ids.shape
    out_cols = n_groups * pool_size + pool_size - 1
    if out is None:
        out = torch.empty((rows, out_cols), dtype=torch.int32)
    else:
        assert out.shape == (rows, out_cols) and out.dtype == torch.int32
    out.fill_(7)
    return out


@pytest.mark.parametrize(("topk_tokens", "index_kpool"), _DECODE_TOPK_CONFIGS)
def test_decode_out_width_follows_selected_pools(topk_tokens, index_kpool):
    rows = 3
    n_groups = topk_tokens // index_kpool
    out_cols = n_groups * index_kpool + index_kpool - 1
    pool_topk = torch.zeros(rows, n_groups, dtype=torch.int32)
    positions = torch.full((rows,), topk_tokens * 2, dtype=torch.int64)
    buffer = torch.full(
        (rows + 2, topk_tokens + index_kpool - 1 + 32), -1, dtype=torch.int32
    )

    out = _run_decode_block(
        _contract_expand, pool_topk, positions, topk_tokens, index_kpool, buffer
    )

    # Written in place; columns past the expanded width keep the -1 fill.
    assert out.data_ptr() == buffer.data_ptr()
    assert out.shape == (rows, out_cols)
    assert torch.all(buffer[:rows, :out_cols] == 7)
    assert torch.all(buffer[:rows, out_cols:] == -1)
    assert torch.all(buffer[rows:] == -1)


@requires_gpu
@pytest.mark.parametrize(("topk_tokens", "index_kpool"), _DECODE_TOPK_CONFIGS)
def test_decode_block_matches_upstream_expand(
    original_expand, fused_expand, topk_tokens, index_kpool
):
    torch.manual_seed(3)
    rows = 6
    n_groups = topk_tokens // index_kpool
    # Long context: seq_len > topk_tokens so decode takes the pool-expand path.
    positions = torch.randint(
        topk_tokens, 16384, (rows,), device="cuda", dtype=torch.int64
    )
    pool_topk = _random_pool_ids(rows, n_groups, positions, index_kpool,
                                 torch.int32)
    width = topk_tokens + index_kpool - 1 + 32
    buffer = torch.full((rows + 2, width), -1, dtype=torch.int32, device="cuda")

    # Upstream: -1 fill, expand into a fresh tensor, copy its width back.
    upstream_out = original_expand(
        pool_topk.to(torch.int64), positions.to(torch.int32) + 1, index_kpool
    )
    expected = buffer.clone()
    expected[:rows, : upstream_out.shape[-1]] = upstream_out

    out = _run_decode_block(
        fused_expand, pool_topk, positions, topk_tokens, index_kpool, buffer
    )

    assert out.data_ptr() == buffer.data_ptr()
    assert torch.equal(buffer, expected)
