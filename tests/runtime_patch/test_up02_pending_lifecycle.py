# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts for the UP-02 deferred-quant marker lifecycle.

The deferred path hands the all-gather a bf16 activation plus an *unwritten* scale
placeholder.  These tests pin the consumption protocol: a marked activation is either
quantized by the fused prologue or rewritten with a real local quantization, the marker is
retired once honoured, and any gather that cannot honour the deferral fails loudly instead of
gathering an uninitialized scale.  No HCU is needed.
"""

import pytest
import torch

from vllm_hcu.ops import token_fp8_quant as tq
from vllm_hcu.patch.worker.framework_opt import patch_cuda_communicator as adapter

MARKER = "_hcu_up02_quant_pending"


class _FakeAg:
    """All-gather stand-in recording which path was taken."""

    def __init__(self, fused_ok=True):
        self.fused_ok = fused_ok
        self.quant_gathered = False
        self.plain_gathered = None

    def can_quant_gather(self, x, w, ids, sizes):
        return self.fused_ok

    def gather_quant_prod(self, x, w, ids, scale_ph, sizes):
        self.quant_gathered = True
        return ["fused_q", w, ids, "fused_s"]

    def can_gather(self, tensors, sizes):
        return True

    def gather(self, tensors, sizes):
        self.plain_gathered = list(tensors)
        return list(tensors)


def _dispatch(rows=4, topk=8, hidden=16, dtype=torch.bfloat16, mark=True):
    t0 = torch.zeros(rows, hidden, dtype=dtype)
    t1 = torch.zeros(rows, topk, dtype=torch.float32)
    t2 = torch.zeros(rows, topk, dtype=torch.int32)
    t3 = torch.empty(rows, 1, dtype=torch.float32)  # unwritten placeholder
    if mark:
        setattr(t0, MARKER, True)
    return [t0, t1, t2, t3], [rows]


@pytest.fixture
def local_quant(monkeypatch):
    """Record local-quant fallbacks without needing the Triton kernel."""
    calls = []

    def fake(x):
        calls.append(x)
        rows = x.shape[0]
        return torch.zeros(rows, x.shape[1], dtype=torch.uint8), torch.ones(rows, 1)

    monkeypatch.setattr(tq, "token_fp8_quant", fake)
    return calls


def test_fused_path_consumes_marker():
    tensors, sizes = _dispatch()
    ag = _FakeAg(fused_ok=True)
    outs, rewritten = adapter._up02_prepare_dispatch(ag, tensors, sizes)
    assert ag.quant_gathered and outs[0] == "fused_q"
    assert getattr(tensors[0], MARKER) is False, "marker must be retired once honoured"
    assert rewritten is tensors


def test_capacity_decline_rewrites_and_retires(local_quant):
    tensors, sizes = _dispatch()
    ag = _FakeAg(fused_ok=False)
    outs, rewritten = adapter._up02_prepare_dispatch(ag, tensors, sizes)
    assert len(local_quant) == 1, "declined fusion must quantize locally"
    assert outs is not None and rewritten[0] is not tensors[0]
    assert rewritten[3] is not tensors[3], "placeholder must be replaced by a real scale"
    assert getattr(tensors[0], MARKER) is False


def test_float16_marker_never_reaches_the_plain_gather(local_quant):
    """tq.can_use admits float16; the fused consumer does not.  The pair must be rewritten."""
    tensors, sizes = _dispatch(dtype=torch.float16)
    ag = _FakeAg(fused_ok=True)
    outs, rewritten = adapter._up02_prepare_dispatch(ag, tensors, sizes)
    assert not ag.quant_gathered, "float16 must not enter the bf16-only fused prologue"
    assert len(local_quant) == 1, "float16 deferral must fall back to a real quantization"
    assert rewritten[0] is not tensors[0] and rewritten[3] is not tensors[3]
    assert getattr(tensors[0], MARKER) is False


def test_fused_exception_falls_back_without_leaking_the_placeholder(local_quant):
    tensors, sizes = _dispatch()

    class _Boom(_FakeAg):
        def gather_quant_prod(self, *args):
            raise RuntimeError("kernel unavailable")

    outs, rewritten = adapter._up02_prepare_dispatch(_Boom(), tensors, sizes)
    assert len(local_quant) == 1 and rewritten[3] is not tensors[3]
    assert getattr(tensors[0], MARKER) is False


def test_unmarked_dispatch_is_untouched(local_quant):
    tensors, sizes = _dispatch(mark=False)
    outs, rewritten = adapter._up02_prepare_dispatch(_FakeAg(), tensors, sizes)
    assert outs is None and rewritten == tensors and not local_quant


def test_marker_is_not_reconsumed(local_quant):
    tensors, sizes = _dispatch()
    ag = _FakeAg(fused_ok=True)
    adapter._up02_prepare_dispatch(ag, tensors, sizes)
    outs, rewritten = adapter._up02_prepare_dispatch(ag, tensors, sizes)
    assert outs is None and rewritten is tensors, "a retired marker must not defer again"
    assert not local_quant


@pytest.mark.parametrize(
    "payload, dim",
    [
        ("single", 0),          # not a tensor list: the scale never reaches the gather
        ("two", 0),             # wrong tensor count for the production dispatch
        ("four", 1),            # row dimension is not gathered
    ],
)
def test_guard_rejects_unconsumable_payloads(payload, dim):
    tensors, _ = _dispatch()
    input_ = {"single": tensors[0], "two": tensors[:2], "four": tensors}[payload]
    with pytest.raises(RuntimeError, match="deferred per-token FP8"):
        adapter._up02_guard_unconsumable(input_, _FakeAg(), dim)


def test_guard_rejects_marker_without_ipc_all_gather():
    tensors, _ = _dispatch()
    with pytest.raises(RuntimeError, match="deferred per-token FP8"):
        adapter._up02_guard_unconsumable(tensors, None, 0)


def test_guard_allows_the_production_dispatch():
    tensors, _ = _dispatch()
    adapter._up02_guard_unconsumable(tensors, _FakeAg(), 0)


def test_guard_ignores_unmarked_payloads():
    tensors, _ = _dispatch(mark=False)
    adapter._up02_guard_unconsumable(tensors[0], None, 1)
    adapter._up02_guard_unconsumable(tensors[:2], None, 0)
