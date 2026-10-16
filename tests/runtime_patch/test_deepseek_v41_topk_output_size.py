# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contract for compressed/SWA indices, including narrowed mixed batches."""

from __future__ import annotations

from types import ModuleType

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_hcu.patch.worker.op_opt import patch_deepseek_v41_topk_output_size as patch
from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError


def _oracle(indices, starts, seq_lens, gather_lens, window, ratio, topk, M, N):
    flat = indices.reshape(indices.shape[0], -1).tolist()
    width = ((topk + window + 127) // 128) * 128
    result = torch.full((len(flat), width), -1, dtype=torch.int32)
    lengths = torch.empty(len(flat), dtype=torch.int32)
    base = starts[0]
    for req, (begin, end) in enumerate(zip(starts[:-1], starts[1:])):
        query_len = end - begin
        for token in range(query_len):
            row = begin - base + token
            position = seq_lens[req] - query_len + token
            compressed = max(0, min((position + 1) // ratio, topk, len(flat[row])))
            swa = max(0, min(position + 1, window))
            for col in range(compressed):
                value = flat[row][col]
                if 0 <= value < N:
                    result[row, col] = M * req + value
            for col in range(swa):
                result[row, compressed + col] = (
                    M * req + N + col + position - swa + 1
                    - (seq_lens[req] - gather_lens[req])
                )
            lengths[row] = compressed + swa
    return result, lengths


@pytest.mark.parametrize(
    ("query_lens", "base", "past"),
    [
        ([1], 0, [0]),
        ([3, 1], 17, [0, 129]),
        ([0, 3, 0, 2], 9, [10, 5, 0, 128]),
        ([2, 1, 4], 0, [130, 1, 0]),
        ([4097], 31, [0]),
    ],
)
@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize(
    ("window", "topk"),
    [(0, 3), (8, 6), (8, 0), (1, 129), (129, 3), (256, 512)],
)
def test_combiner_preserves_request_offsets_and_padded_indices(
    query_lens, base, past, ratio, window, topk
):
    starts = [base]
    for length in query_lens:
        starts.append(starts[-1] + length)
    seq_lens = [n + old for n, old in zip(query_lens, past)]
    gather_lens = [min(seq, n + 3) for seq, n in zip(seq_lens, query_lens)]
    n = sum(query_lens)
    indices = (torch.arange(n * 8).reshape(n, 2, 4) % 20 - 2).to(torch.int32)
    args = (
        indices, torch.tensor(starts, dtype=torch.int32),
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.tensor(gather_lens, dtype=torch.int32),
        window, ratio, topk, 64, 16,
    )
    expected = _oracle(
        indices, starts, seq_lens, gather_lens, window, ratio, topk, 64, 16
    )
    actual = patch._combine_topk_swa_indices(*args)
    for result, reference in zip(actual, expected):
        assert result.dtype == torch.int32
        assert torch.equal(result, reference)


def test_combiner_avoids_dynamic_index_sizes_and_scalar_readback():
    calls = []

    class RecordOps(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            calls.append(str(func))
            return func(*args, **(kwargs or {}))

    args = (
        torch.tensor([[0, 1, -1], [2, 99, 1], [3, 0, 1]], dtype=torch.int32),
        torch.tensor([17, 19, 20], dtype=torch.int32),
        torch.tensor([2, 129], dtype=torch.int32),
        torch.tensor([2, 4], dtype=torch.int32),
        8, 4, 3, 64, 16,
    )
    with RecordOps():
        actual = patch._combine_topk_swa_indices(*args)
    expected = _oracle(args[0], [17, 19, 20], [2, 129], [2, 4], 8, 4, 3, 64, 16)
    for result, reference in zip(actual, expected):
        assert torch.equal(result, reference)
    assert "aten.scatter_.src" in calls
    assert not any("nonzero" in op or "_local_scalar_dense" in op for op in calls)


def _target(original):
    module = ModuleType(patch.TARGET_MODULE)
    module.combine_topk_swa_indices = original
    module._SPARSE_PREFILL_TOPK_ALIGNMENT = 128
    return module


def test_adapter_can_disable_optimization_and_rejects_stale_binding(monkeypatch):
    sentinel = object()
    calls = []

    def original(
        topk_indices, query_start_loc, seq_lens, gather_lens,
        window_size, compress_ratio, topk, M, N,
    ):
        calls.append(topk_indices)
        return sentinel

    module = _target(original)
    assert patch.apply_to_module(module)
    wrapper = module.combine_topk_swa_indices
    assert not patch.apply_to_module(module)
    assert module.combine_topk_swa_indices is wrapper
    monkeypatch.setenv(patch._FLAG, "0")
    assert wrapper(*range(9)) is sentinel
    assert calls == [0]
    module.combine_topk_swa_indices = original
    with pytest.raises(PatchCompatibilityError, match="stale"):
        patch.apply_to_module(module)


def test_adapter_routes_to_optimized_combiner_by_default(monkeypatch):
    def original(
        topk_indices, query_start_loc, seq_lens, gather_lens,
        window_size, compress_ratio, topk, M, N,
    ):
        raise AssertionError("original combiner unexpectedly called")

    module = _target(original)
    patch.apply_to_module(module)
    monkeypatch.delenv(patch._FLAG, raising=False)
    args = (
        torch.tensor([[1, -1, 99, 0]], dtype=torch.int32),
        torch.tensor([11, 12], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        2, 4, 4, 64, 16,
    )
    actual = module.combine_topk_swa_indices(*args)
    expected = _oracle(args[0], [11, 12], [4], [4], 2, 4, 4, 64, 16)
    for result, reference in zip(actual, expected):
        assert torch.equal(result, reference)


def test_adapter_rejects_incompatible_target():
    module = _target(lambda x: x)
    with pytest.raises(PatchCompatibilityError, match="signature"):
        patch.apply_to_module(module)


def test_combiner_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv(patch._BACKEND_FLAG, "unknown")
    with pytest.raises(ValueError, match="must be auto, torch or triton"):
        patch._combine_topk_swa_indices(*([None] * 9))


def test_swa_only_accepts_model_zero_compress_ratio():
    args = (
        torch.zeros(2, 8, dtype=torch.int32),
        torch.tensor([17, 19], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        128, 0, 0, 1024, 0,
    )
    actual = patch._combine_topk_swa_indices(*args)
    expected = _oracle(args[0], [17, 19], [2], [2], 128, 1, 0, 1024, 0)
    for result, reference in zip(actual, expected):
        assert torch.equal(result, reference)
