# SPDX-License-Identifier: Apache-2.0
"""Tests for the VLLM_HCU_GLM5NEXT_KPOOL_ELEMENT_FUSION runtime patch."""

import inspect

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import (
    patch_glm5next_kpool_element_fusion as fusion_patch,
)
from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU for Triton kernels"
)


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
    monkeypatch.delattr(module, fusion_patch._PATCH_MARKER, raising=False)
    return module


def test_switch_disabled_by_default(monkeypatch):
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.delenv(fusion_patch.ENV_VAR, raising=False)
    assert henvs.VLLM_HCU_GLM5NEXT_KPOOL_ELEMENT_FUSION is False


def test_disabled_leaves_module_untouched(monkeypatch, indexer_module):
    monkeypatch.setenv(fusion_patch.ENV_VAR, "0")
    original = indexer_module.sparse_attn_indexer_kpool
    assert fusion_patch.apply_to_module(indexer_module) is False
    assert indexer_module.sparse_attn_indexer_kpool is original


def test_enabled_rewrites_decode_path_once(monkeypatch, indexer_module):
    monkeypatch.setenv(fusion_patch.ENV_VAR, "1")
    original = indexer_module.sparse_attn_indexer_kpool

    assert fusion_patch.apply_to_module(indexer_module) is True
    patched = indexer_module.sparse_attn_indexer_kpool
    assert patched is not original
    assert inspect.signature(patched) == inspect.signature(original)

    source = inspect.getsource(patched)
    assert "positions[:num_decode_tokens].to(torch.int32)" not in source
    assert "positions[:num_decode_tokens].view(shape2)" in source
    assert "_vllm_hcu_fused_short_decode_fill(" in source
    # The upstream -1 fill is kept for prefill / mixed / long-context batches.
    assert "if not _hcu_short_decode:" in source

    # Idempotent: repeated calls return False and keep the same function.
    assert fusion_patch.apply_to_module(indexer_module) is False
    assert indexer_module.sparse_attn_indexer_kpool is patched


def test_rejects_unrecognised_source():
    with pytest.raises(PatchCompatibilityError):
        fusion_patch._rewrite_source("def sparse_attn_indexer_kpool():\n    pass\n")


@requires_gpu
@pytest.mark.parametrize(
    ("width", "n_decode", "n_rows", "pos_dtype"),
    [
        (2176, 3, 3, torch.int64),
        (2176, 7, 16, torch.int64),
        (2176, 1, 1, torch.int32),
        (128, 5, 8, torch.int64),
        (2176, 64, 96, torch.int64),
    ],
)
def test_fused_fill_matches_upstream(
    indexer_module, width, n_decode, n_rows, pos_dtype
):
    torch.manual_seed(0)
    positions = torch.randint(0, 2048, (n_rows,), device="cuda", dtype=pos_dtype)
    positions[0] = 0
    if n_decode > 1:
        positions[1] = 2047
    # Extra rows past n_rows must stay untouched.
    expected = torch.randint(
        -5, 9999, (n_rows + 4, width), device="cuda", dtype=torch.int32
    )
    actual = expected.clone()

    # Upstream: whole-buffer -1 fill, then _fill_causal_indices on decode rows.
    expected[:n_rows] = -1
    indexer_module._fill_causal_indices(expected[:n_decode], positions[:n_decode])
    fusion_patch.fused_short_decode_fill(actual, positions, n_decode, n_rows)

    assert torch.equal(actual, expected)


@requires_gpu
def test_kpool_decode_update_accepts_int64_positions():
    import vllm.models.glm5next  # noqa: F401
    from vllm.models.glm5next.amd.ops import kpool_compress as kpool_ops

    pool, head_dim, batch, next_n, page = 4, 128, 6, 3, 64

    def run(pos_dtype):
        torch.manual_seed(1)
        kv = torch.zeros(32, page, head_dim + 4, dtype=torch.uint8, device="cuda")
        tail = torch.randn(
            batch * 2, 2, pool, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        key = torch.randn(batch, next_n, head_dim, dtype=torch.bfloat16, device="cuda")
        gate = torch.randn_like(key)
        ape = torch.randn(pool, head_dim, dtype=torch.float32, device="cuda")
        positions = torch.tensor(
            [[s + i for i in range(next_n)] for s in (0, 1, 2, 3, 5, 9)],
            device="cuda",
        )
        # One invalid (padding) token to cover the pos < 0 branch.
        positions[5, 2] = -1
        tail_slot = torch.arange(batch, device="cuda")[:, None] * pool + (
            positions % pool
        )
        tail_slot[5, 2] = -1
        slot = torch.where(
            positions % pool == pool - 1,
            torch.arange(batch * next_n, device="cuda").view(batch, next_n) + page,
            torch.full_like(positions, -1),
        )
        kpool_ops.kpool_decode_update_and_maybe_write_cache_batched(
            kv,
            tail,
            tail_slot,
            key,
            gate,
            ape,
            slot,
            positions.to(pos_dtype),
            pool,
            head_dim,
            round_scale=True,
        )
        return kv, tail

    kv32, tail32 = run(torch.int32)
    kv64, tail64 = run(torch.int64)
    assert torch.count_nonzero(kv32) > 0
    assert torch.equal(kv32, kv64)
    assert torch.equal(tail32, tail64)
