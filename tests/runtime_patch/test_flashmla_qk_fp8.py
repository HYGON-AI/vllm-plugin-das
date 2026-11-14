# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU dispatch contracts; native accuracy is tested separately on gfx938."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def adapter(monkeypatch):
    path = Path(__file__).resolve().parents[2] / (
        "vllm_hcu/v1/attention/ops/flashmla_qk_fp8.py"
    )
    spec = importlib.util.spec_from_file_location("qk_fp8_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def public(*args):
        calls.append(("public", args))
        return "bf16_out", "bf16_max", "bf16_lse"

    def native(*args):
        calls.append(("native", args))
        return "fp8_out", object(), "fp8_lse"

    monkeypatch.setitem(
        sys.modules,
        "flash_mla.flash_mla_interface",
        SimpleNamespace(
            flash_mla_sparse_fwd=public,
            flash_mla_cuda=SimpleNamespace(sparse_prefill_fwd=native),
        ),
    )
    monkeypatch.delenv("FLASH_MLA_DSA_FP8_QK", raising=False)
    monkeypatch.delenv("FLASH_MLA_DISABLE_DSA_MLS_PREFILL", raising=False)
    return module, calls


def _inputs():
    return (
        torch.empty(2, 64, 576, dtype=torch.bfloat16),
        torch.empty(2048, 1, 576, dtype=torch.bfloat16),
        torch.arange(2048, dtype=torch.int32).view(1, 1, -1).repeat(2, 1, 1),
    )


def _enable(module, monkeypatch):
    monkeypatch.setenv("FLASH_MLA_DSA_FP8_QK", "1")
    monkeypatch.setattr(module, "_supports_qk_fp8", lambda *args: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


@pytest.mark.parametrize("flag", [None, "0", "true"])
def test_disabled_preserves_public_arguments(adapter, monkeypatch, flag):
    module, calls = adapter
    if flag is not None:
        monkeypatch.setenv("FLASH_MLA_DSA_FP8_QK", flag)
    q, kv, indices = _inputs()
    counts = torch.tensor([1, 2048], dtype=torch.int32)
    assert module.flash_mla_sparse_fwd(q, kv, indices, 0.125, topk_length=counts) == (
        "bf16_out",
        "bf16_max",
        "bf16_lse",
    )
    assert calls[0][0] == "public"
    assert calls[0][1][2] is indices
    assert calls[0][1][-1] is counts


@pytest.mark.parametrize("with_counts", [False, True])
def test_full_rows_use_native_and_hide_uninitialized_max(
    adapter, monkeypatch, with_counts
):
    module, calls = adapter
    _enable(module, monkeypatch)
    q, kv, indices = _inputs()
    original = indices.clone()
    counts = torch.full((2,), 2048, dtype=torch.int32) if with_counts else None
    assert module.flash_mla_sparse_fwd(q, kv, indices, 0.125, topk_length=counts) == (
        "fp8_out",
        None,
        "fp8_lse",
    )
    assert calls[0][0] == "native"
    assert calls[0][1][-2:] == (None, None)
    assert calls[0][1][2] is indices
    assert torch.equal(indices, original)


@pytest.mark.parametrize("bad_index", [-1, 2048, 999999])
@pytest.mark.parametrize("with_counts", [False, True])
def test_invalid_indices_never_reach_unchecked_kernel(
    adapter, monkeypatch, bad_index, with_counts
):
    module, calls = adapter
    _enable(module, monkeypatch)
    q, kv, indices = _inputs()
    indices[0, 0, 12] = bad_index
    counts = torch.full((2,), 2048, dtype=torch.int32) if with_counts else None
    module.flash_mla_sparse_fwd(q, kv, indices, 0.125, topk_length=counts)
    assert calls[0][0] == "public"
    # A non-None count also prevents the public frontend selecting QK-FP8.
    assert calls[0][1][-1] is not None
    assert torch.equal(calls[0][1][-1], torch.full((2,), 2048, dtype=torch.int32))


@pytest.mark.parametrize("count", [0, 1, 2047, 2049])
def test_count_semantics_are_preserved(adapter, monkeypatch, count):
    module, calls = adapter
    _enable(module, monkeypatch)
    q, kv, indices = _inputs()
    counts = torch.tensor([count, 2048], dtype=torch.int32)
    module.flash_mla_sparse_fwd(q, kv, indices, 0.125, topk_length=counts)
    assert calls[0][0] == "public"
    assert calls[0][1][-1] is counts


def test_capture_does_not_read_device_values(adapter, monkeypatch):
    module, calls = adapter
    _enable(module, monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        torch.Tensor, "item", lambda _: pytest.fail("capture synchronized")
    )
    q, kv, indices = _inputs()
    module.flash_mla_sparse_fwd(q, kv, indices, 0.125)
    assert calls[0][0] == "public"
    assert calls[0][1][-1] is not None


def test_data_is_rechecked_after_inplace_update(adapter, monkeypatch):
    module, calls = adapter
    _enable(module, monkeypatch)
    q, kv, indices = _inputs()
    module.flash_mla_sparse_fwd(q, kv, indices, 0.125)
    indices[0, 0, 0] = -1
    module.flash_mla_sparse_fwd(q, kv, indices, 0.125)
    assert [name for name, _ in calls] == ["native", "public"]


def test_explicit_entry_matches_documented_two_result_interface(adapter, monkeypatch):
    module, calls = adapter
    _enable(module, monkeypatch)
    q, kv, indices = _inputs()
    assert module.sparse_prefill_qk_fp8(q, kv, indices, 0.125) == ("fp8_out", "fp8_lse")
    assert calls[0][0] == "native"


def test_explicit_entry_rejects_padding(adapter, monkeypatch):
    module, calls = adapter
    _enable(module, monkeypatch)
    q, kv, indices = _inputs()
    indices[0, 0, 0] = -1
    with pytest.raises(ValueError, match="indices|topk_length"):
        module.sparse_prefill_qk_fp8(q, kv, indices, 0.125)
    assert not calls


def test_native_error_is_not_hidden_by_fallback(adapter, monkeypatch):
    module, calls = adapter
    _enable(module, monkeypatch)
    native = sys.modules["flash_mla.flash_mla_interface"].flash_mla_cuda

    def fail(*args):
        raise RuntimeError("native failure")

    monkeypatch.setattr(native, "sparse_prefill_fwd", fail)
    q, kv, indices = _inputs()
    with pytest.raises(RuntimeError, match="native failure"):
        module.flash_mla_sparse_fwd(q, kv, indices, 0.125)
    assert not calls


def test_counts_are_rechecked_after_inplace_update(adapter, monkeypatch):
    module, calls = adapter
    _enable(module, monkeypatch)
    q, kv, indices = _inputs()
    counts = torch.full((2,), 2048, dtype=torch.int32)
    module.flash_mla_sparse_fwd(q, kv, indices, 0.125, topk_length=counts)
    counts[0] = 17
    module.flash_mla_sparse_fwd(q, kv, indices, 0.125, topk_length=counts)
    assert [name for name, _ in calls] == ["native", "public"]


def test_existing_plugin_entry_uses_adapter(adapter, monkeypatch):
    module, calls = adapter
    _enable(module, monkeypatch)
    interface = sys.modules["flash_mla.flash_mla_interface"]
    for name in ("FlashMLASchedMeta", "flash_mla_with_kvcache", "get_mla_metadata"):
        monkeypatch.setattr(interface, name, lambda: None, raising=False)
    monkeypatch.setitem(
        sys.modules, "vllm.logger", SimpleNamespace(init_logger=lambda _: None)
    )
    monkeypatch.setitem(
        sys.modules, "vllm_hcu.platforms.hcu", SimpleNamespace(on_gfx938=lambda: True)
    )
    monkeypatch.setitem(
        sys.modules, "vllm_hcu.v1.attention.ops.flashmla_qk_fp8", module
    )
    path = Path(module.__file__).with_name("flashmla.py")
    spec = importlib.util.spec_from_file_location("hcu_flashmla_entry_under_test", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    assert entry.flash_mla_sparse_fwd is module.flash_mla_sparse_fwd
    q, kv, indices = _inputs()
    assert entry.flash_mla_sparse_fwd(q, kv, indices, 0.125)[1] is None
    assert calls[0][0] == "native"


class _DeviceTensor:
    def __init__(self, shape, dtype=torch.bfloat16, strides=None):
        self.shape, self.dtype = shape, dtype
        self.device, self.is_cuda = torch.device("cuda:0"), True
        self._strides = strides or tuple(torch.empty(shape, device="meta").stride())

    def dim(self):
        return len(self.shape)

    def stride(self, axis):
        return self._strides[axis]


@pytest.mark.parametrize(
    "heads,kv_len,resident,expected",
    [
        (64, 2047, "1", False),
        (64, 2048, "1", True),
        (64, 24575, "0", False),
        (64, 24576, "0", True),
        (128, 24575, "1", False),
        (128, 24576, "0", True),
        (32, 85000, "1", False),
    ],
)
def test_geometry_boundaries(adapter, monkeypatch, heads, kv_len, resident, expected):
    module, _ = adapter
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(gcnArchName="gfx938"),
    )
    monkeypatch.setenv("FLASH_MLA_ENABLE_MLS_RESIDENT_PREFILL", resident)
    q = _DeviceTensor((2, heads, 576))
    kv = _DeviceTensor((kv_len, 1, 576))
    indices = _DeviceTensor((2, 1, 2048), torch.int32)
    assert module._supports_qk_fp8(q, kv, indices, 512, None) is expected


@pytest.mark.parametrize(
    "case",
    [
        "arch",
        "disabled",
        "sink",
        "dtype",
        "dim",
        "heads_kv",
        "index_dtype",
        "index_shape",
        "device",
        "stride",
        "empty",
        "resident_stride",
        "d_v",
    ],
)
def test_unsupported_metadata(adapter, monkeypatch, case):
    module, _ = adapter
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(gcnArchName="gfx936" if case == "arch" else "gfx938"),
    )
    monkeypatch.setenv("FLASH_MLA_ENABLE_MLS_RESIDENT_PREFILL", "1")
    if case == "disabled":
        monkeypatch.setenv("FLASH_MLA_DISABLE_DSA_MLS_PREFILL", "0")
    q = _DeviceTensor((2, 64, 576))
    kv = _DeviceTensor((2048, 1, 576))
    indices = _DeviceTensor((2, 1, 2048), torch.int32)
    sink, d_v = None, 512
    if case == "sink":
        sink = object()
    if case == "dtype":
        q.dtype = torch.float16
    if case == "dim":
        q.shape = (2, 64, 512)
    if case == "heads_kv":
        kv.shape = (2048, 2, 576)
    if case == "index_dtype":
        indices.dtype = torch.int64
    if case == "index_shape":
        indices.shape = (2, 1, 1024)
    if case == "device":
        kv.device = torch.device("cuda:1")
    if case == "stride":
        q._strides = (64 * 576, 576, 2)
    if case == "empty":
        q.shape, indices.shape = (0, 64, 576), (0, 1, 2048)
    if case == "resident_stride":
        kv._strides = (8192, 576, 1)
    if case == "d_v":
        d_v = 256
    assert not module._supports_qk_fp8(q, kv, indices, d_v, sink)
