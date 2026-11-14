# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Native QK-FP8 accuracy and masked BF16 fallback on one gfx938 device."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = [
    pytest.mark.hcu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires one free gfx938 device"
    ),
]


@pytest.fixture
def kernels(monkeypatch):
    if "gfx938" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("QK-FP8 requires gfx938")
    pytest.importorskip("flash_mla")
    path = Path(__file__).resolve().parents[2] / "benchmarks/bench_flashmla_qk_fp8.py"
    spec = importlib.util.spec_from_file_location("qk_accuracy_helpers", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    monkeypatch.setenv("FLASH_MLA_DSA_FP8_QK", "1")
    monkeypatch.delenv("FLASH_MLA_DISABLE_DSA_MLS_PREFILL", raising=False)
    monkeypatch.setenv("FLASH_MLA_ENABLE_MLS_RESIDENT_PREFILL", "1")
    return helper, helper.load_adapter()


@pytest.mark.parametrize("heads,kv_len", [(64, 2048), (64, 24576), (128, 24576)])
@pytest.mark.parametrize("seed", [1, 42, 20261114])
def test_native_output_and_lse_against_fp32(kernels, heads, kv_len, seed):
    helper, adapter = kernels
    q, kv, indices, counts = helper.inputs(5, kv_len, heads, seed)
    expected, expected_lse = helper.reference(q, kv, indices, counts)
    out, lse = adapter.sparse_prefill_qk_fp8(q, kv, indices, 576**-0.5)
    error = helper.errors(out, lse, expected, expected_lse)
    assert error["finite"]
    assert error["max_abs"] < 0.05, error
    assert error["relative_l2"] < 0.05, error
    assert error["lse_max_abs"] < 0.05, error


@pytest.mark.parametrize(
    "case", ["short_counts", "negative", "out_of_bounds", "valid_after_count"]
)
def test_masked_fallback_keeps_semantics(kernels, case):
    helper, adapter = kernels
    q, kv, indices, counts = helper.inputs(5, 2048, 64)
    if case in ("short_counts", "valid_after_count"):
        counts[0] = 9
    if case == "negative":
        indices[0, 0, 1] = -1
    if case == "out_of_bounds":
        indices[0, 0, 1] = kv.shape[0]
    # None counts still require the adapter to protect native -1/OOB loads.
    if case in ("negative", "out_of_bounds"):
        counts = None
    expected, expected_lse = helper.reference(q, kv, indices, counts)
    out, maximum, lse = adapter.flash_mla_sparse_fwd(
        q, kv, indices, 576**-0.5, topk_length=counts
    )
    assert maximum is not None, "masked input incorrectly used QK-FP8"
    torch.testing.assert_close(out.float(), expected, atol=0.006, rtol=0.01)
    torch.testing.assert_close(lse, expected_lse, atol=2e-5, rtol=2e-5)


def test_fused_validation_handles_strides_and_mutation(kernels):
    _, adapter = kernels
    from vllm_hcu.v1.attention.ops.flashmla_qk_fp8_validation import invalid_rows

    storage = torch.zeros((5, 4096), dtype=torch.int32, device="cuda")
    indices = storage[:, :2048].view(5, 1, 2048)
    count_storage = torch.full((10,), 2048, dtype=torch.int32, device="cuda")
    counts = count_storage[::2]
    assert not invalid_rows(indices, counts, 2048).any().item()
    indices[1, 0, 0] = -1
    indices[2, 0, 7] = 2048
    counts[3] = 2047
    assert invalid_rows(indices, counts, 2048).cpu().tolist() == [0, 1, 1, 1, 0]
    assert adapter._has_invalid_rows(indices, counts, 2048)


@pytest.mark.parametrize("q_len", [5, 768])
@pytest.mark.parametrize("has_sink", [False, True])
def test_graph_capture_uses_bf16_without_host_read(
    kernels, q_len, has_sink, monkeypatch
):
    helper, adapter = kernels
    q, kv, indices, counts = helper.inputs(q_len, 2048, 64)
    sink = (
        torch.linspace(4, 11, 64, device=q.device, dtype=torch.float32)
        if has_sink
        else None
    )
    counts[0] = 17
    for _ in range(3):
        adapter.flash_mla_sparse_fwd(
            q, kv, indices, 576**-0.5, attn_sink=sink, topk_length=counts
        )
    torch.cuda.synchronize()
    monkeypatch.setattr(
        adapter,
        "_has_invalid_rows",
        lambda *args: pytest.fail("host validation during capture"),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, maximum, lse = adapter.flash_mla_sparse_fwd(
            q, kv, indices, 576**-0.5, attn_sink=sink, topk_length=counts
        )
    assert maximum is not None
    graph.replay()
    expected, expected_lse = helper.reference(
        q[:4], kv, indices[:4], counts[:4], attn_sink=sink
    )
    torch.testing.assert_close(output[:4].float(), expected, atol=0.006, rtol=0.01)
    torch.testing.assert_close(lse[:4], expected_lse, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "shape", [(16, 2048, 64), (128, 24576, 64), (256, 85000, 128), (512, 24576, 64)]
)
def test_adaptive_dispatch_output_and_skipped_scan(kernels, shape, monkeypatch):
    helper, adapter = kernels
    q, kv, indices, counts = helper.inputs(*shape)
    expected, expected_lse = helper.reference(q[:4], kv, indices[:4], counts[:4])
    prefer = adapter._prefer_qk_fp8(q, kv)
    if not prefer:
        monkeypatch.setattr(
            adapter,
            "_has_invalid_rows",
            lambda *args: pytest.fail("unnecessary validation"),
        )
    out, maximum, lse = adapter.flash_mla_sparse_fwd(
        q, kv, indices, 576**-0.5, topk_length=counts
    )
    assert (maximum is None) == prefer
    error = helper.errors(out[:4], lse[:4], expected, expected_lse)
    assert error["finite"] and error["relative_l2"] < 0.05, error
    assert error["max_abs"] < 0.05 and error["lse_max_abs"] < 0.05, error


@pytest.mark.parametrize("heads", [64, 128])
def test_fp8_preserves_attention_sink_and_lse(kernels, heads):
    helper, adapter = kernels
    q, kv, indices, counts = helper.inputs(768, 24576, heads)
    sink = torch.linspace(4, 11, heads, device="cuda", dtype=torch.float32)
    sink[0], sink[1] = float("inf"), -float("inf")
    expected, expected_lse = helper.reference(
        q[:4], kv, indices[:4], counts[:4], attn_sink=sink
    )
    output, maximum, lse = adapter.flash_mla_sparse_fwd(
        q, kv, indices, 576**-0.5, attn_sink=sink, topk_length=counts
    )
    assert maximum is None, "sink silently fell back to BF16"
    assert not output[:, 0].count_nonzero()
    error = helper.errors(output[:4], lse[:4], expected, expected_lse)
    assert error["finite"] and error["relative_l2"] < 0.05, error
    assert error["max_abs"] < 0.05 and error["lse_max_abs"] < 0.05, error


def test_hyv4_sink_backend_reaches_native_fp8(kernels, monkeypatch):
    helper, adapter = kernels
    from vllm_hcu.models.hy_v4 import hcu_sparse

    q, kv, indices, counts = helper.inputs(768, 24576, 64)
    impl = hcu_sparse.HYV4FlashMLASparseImpl.__new__(hcu_sparse.HYV4FlashMLASparseImpl)
    impl.num_heads = 64
    impl.prefill_padding = 64
    impl.softmax_scale = 576**-0.5
    impl.sinks = torch.linspace(4, 11, 64, device="cuda", dtype=torch.float32)
    impl._dcp_sinks = None
    calls = []

    def entry(*args, **kwargs):
        result = adapter.flash_mla_sparse_fwd(*args, **kwargs)
        calls.append(result[1])
        return result

    monkeypatch.setattr(hcu_sparse, "flash_mla_sparse_fwd", entry)
    out, lse = impl._bf16_flash_mla_kernel_with_lse(q, kv, indices[:, 0], counts)
    assert calls == [None], "actual HYV4 backend missed QK FP8"
    expected, pre_lse = helper.reference(
        q[:4], kv, indices[:4], counts[:4], attn_sink=impl.sinks
    )
    expected_lse = torch.logaddexp(pre_lse, impl.sinks[None, :])
    error = helper.errors(out[:4], lse[:4], expected, expected_lse)
    assert error["finite"] and error["relative_l2"] < 0.05, error
    assert error["max_abs"] < 0.05 and error["lse_max_abs"] < 0.05, error


@pytest.mark.parametrize("case", ["negative", "short_counts"])
def test_sink_masked_fallback_applies_sink_once(kernels, case):
    helper, adapter = kernels
    q, kv, indices, counts = helper.inputs(768, 24576, 64)
    sink = torch.linspace(4, 11, 64, device=q.device, dtype=torch.float32)
    if case == "negative":
        indices[0, 0, 7] = -1
    else:
        counts[0] = 17
    expected, lse_ref = helper.reference(
        q[:4], kv, indices[:4], counts[:4], attn_sink=sink
    )
    out, maximum, lse = adapter.flash_mla_sparse_fwd(
        q, kv, indices, 576**-0.5, attn_sink=sink, topk_length=counts
    )
    assert maximum is not None
    torch.testing.assert_close(out[:4].float(), expected, atol=0.006, rtol=0.01)
    torch.testing.assert_close(lse[:4], lse_ref, atol=2e-5, rtol=2e-5)
