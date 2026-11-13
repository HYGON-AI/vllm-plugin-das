# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts for the opt-in fused per-token FP8 quant + DP IPC all-gather.

These tests do not need an HCU: they exercise the import contract without an active Triton
driver, the collective selection predicates, and the communicator helper's early-exit paths.
The GPU bitwise/graph tests stay in the integration harness (see the PR description).
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from vllm_hcu.distributed.device_communicators import ipc_fused_dispatch as fused
from vllm_hcu.patch.worker.framework_opt import patch_cuda_communicator as adapter


class _FakeTensor:
    """Minimal tensor stand-in for the selection predicates (no device allocation)."""

    def __init__(self, dtype, shape, contiguous=True):
        self.dtype = dtype
        self.shape = tuple(shape)
        self.contiguous = contiguous

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self.contiguous


def _subject(world_size=2, region_elems=512 * 1024):
    subject = object.__new__(fused.IpcQuantAllGather)
    subject.disabled = False
    subject.world_size = world_size
    subject.region_elems = region_elems
    return subject


def _valid_inputs(rows=32, hidden=4096, topk=8):
    x = _FakeTensor(torch.bfloat16, (rows, hidden))
    w = _FakeTensor(torch.float32, (rows, topk))
    ids = _FakeTensor(torch.int32, (rows, topk))
    return x, w, ids, [rows] * 2


def test_import_without_triton_driver(monkeypatch):
    """The module must import with the CPU Triton placeholders (no driver)."""
    import vllm.triton_utils as utils
    from vllm.triton_utils.importing import TritonLanguagePlaceholder, TritonPlaceholder

    monkeypatch.setattr(utils, "HAS_TRITON", False)
    monkeypatch.setattr(utils, "tl", TritonLanguagePlaceholder())
    monkeypatch.setattr(utils, "triton", TritonPlaceholder())
    spec = importlib.util.spec_from_file_location("fused_without_triton", fused.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.IpcQuantAllGather)
    assert callable(module._ipc_quant_gather_prod_kernel)


def test_defer_flag_only_set_for_world_two(monkeypatch):
    """The deferred-quant flag is enabled only by a world-2 fused communicator."""
    monkeypatch.setattr(fused, "UP02_DEFER_ALLOWED", False)

    class _Base:
        def __init__(self, *args, **kwargs):
            self.disabled = True
            self.world_size = world_size

    for world_size in (2, 8):
        monkeypatch.setattr(fused.IpcAllGather, "__init__", _Base.__init__)
        monkeypatch.setattr(fused, "UP02_DEFER_ALLOWED", False)
        fused.IpcQuantAllGather(None, None)
        assert fused.UP02_DEFER_ALLOWED is (world_size == 2)


def test_selection_predicate_world_two_valid():
    x, w, ids, sizes = _valid_inputs()
    assert _subject().can_quant_gather(x, w, ids, sizes) is True


def test_selection_predicate_rejects_other_world_sizes():
    x, w, ids, sizes = _valid_inputs()
    subject = _subject(world_size=8)
    assert subject.can_quant_gather(x, w, ids, [32] * 8) is False


def test_selection_predicate_rejects_capacity_and_dtypes():
    x, w, ids, sizes = _valid_inputs()
    tiny = _subject(region_elems=1024)  # capacity contract cannot hold 32 rows
    assert tiny.can_quant_gather(x, w, ids, sizes) is False

    bad_x = _FakeTensor(torch.float16, (32, 4096))
    assert _subject().can_quant_gather(bad_x, w, ids, sizes) is False

    bad_ids = _FakeTensor(torch.int32, (32, 7))
    assert _subject().can_quant_gather(x, w, bad_ids, sizes) is False

    assert _subject().can_quant_gather(x, w, ids, [32]) is False  # sizes must match world_size


def test_helper_early_exits_without_marker():
    """Non-dispatch shapes and unmarked tensors stay on the certified path."""
    t0 = torch.zeros(4, 8)
    t1 = torch.zeros(4, 8, dtype=torch.float32)
    t2 = torch.zeros(4, 8, dtype=torch.int32)
    t3 = torch.zeros(4, 1, dtype=torch.float32)

    outs, rew = adapter._up02_prepare_dispatch(None, [t0], [4])
    assert outs is None and rew == [t0]

    outs, rew = adapter._up02_prepare_dispatch(None, [t0, t1, t2, t3], [4])
    assert outs is None and rew == [t0, t1, t2, t3]
