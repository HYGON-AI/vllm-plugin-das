# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import ast
import copy
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


REPO = Path(__file__).resolve().parents[2]


def _load_fused_qkv_impl():
    """Load the lazy LightOp boundary without importing the full vLLM adapter."""
    source = (
        REPO / "vllm_hcu/model_executor/layers/attention_runtime.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = copy.deepcopy(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "fused_qkv_split_rmsnorm_rope_kv_store_impl"
        )
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "split_kv_cache": lambda *_args, **_kwargs: None,
        "logger": SimpleNamespace(warning_once=lambda *_args, **_kwargs: None),
    }
    exec(compile(module, "attention_runtime_contract", "exec"), namespace)
    return namespace[function.name]


def test_fused_qkv_runtime_uses_categorized_lightop_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A categorized-only LightOp package remains usable at the lazy boundary."""
    calls: list[tuple[object, ...]] = []

    def split_qkv(*args, **kwargs):
        calls.append((args, kwargs))
        return (
            torch.ones((1, 1, 2)),
            torch.ones((1, 1, 2)),
            torch.ones((1, 1, 2)),
        )

    lightop = ModuleType("lightop")
    lightop.__path__ = []  # type: ignore[attr-defined]
    attention = ModuleType("lightop.attention")
    attention.split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant = split_qkv
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.attention", attention)
    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        SimpleNamespace(
            get_forward_context=lambda: SimpleNamespace(
                slot_mapping={},
                no_compile_layers={"layer": SimpleNamespace(kv_cache=None)},
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_hcu.platforms.hcu",
        SimpleNamespace(get_hcu_flash_attn_mode=lambda: "other"),
    )

    result = _load_fused_qkv_impl()(
        torch.ones((1, 6)),
        torch.tensor([0]),
        "layer",
        "auto",
        torch.ones((1, 2)),
        torch.ones(2),
        torch.ones(2),
        1e-5,
        2,
        2,
        2,
        2,
        16,
    )

    assert len(calls) == 1
    assert all(tensor.shape == (1, 1, 2) for tensor in result)


def test_fused_qkv_runtime_rejects_top_level_lightop_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lazy boundary must not retry the removed top-level export."""
    lightop = ModuleType("lightop")
    lightop.__path__ = []  # type: ignore[attr-defined]
    lightop.split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant = (
        lambda *_args, **_kwargs: pytest.fail("legacy top-level kernel called")
    )
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.delitem(sys.modules, "lightop.attention", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "vllm.forward_context",
        SimpleNamespace(
            get_forward_context=lambda: SimpleNamespace(
                slot_mapping={},
                no_compile_layers={"layer": SimpleNamespace(kv_cache=None)},
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_hcu.platforms.hcu",
        SimpleNamespace(get_hcu_flash_attn_mode=lambda: "other"),
    )

    with pytest.raises(ImportError):
        _load_fused_qkv_impl()(
            torch.ones((1, 6)),
            torch.tensor([0]),
            "layer",
            "auto",
            torch.ones((1, 2)),
            torch.ones(2),
            torch.ones(2),
            1e-5,
            2,
            2,
            2,
            2,
            16,
        )


def _runtime():
    return importlib.import_module("vllm_hcu.v1.attention.ops.rocm_aiter_mla_sparse")


def test_lightop_mask_topk_route_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    name = "VLLM_HCU_USE_LIGHTOP_MASK_TOPK"
    monkeypatch.setenv(name, "1")

    assert name not in runtime.henvs.hcu_vllm_environment_variables
    assert not hasattr(runtime.henvs, name)
    assert not hasattr(runtime, "_lightop_sparse_mask_topk_ops")
    assert not hasattr(runtime, "_lightop_mask_topk_decode_metadata")
    assert not hasattr(runtime, "_lightop_mask_topk_decode")


@pytest.fixture
def fast_topk_runtime():
    runtime = _runtime()
    runtime._lightop_fast_topk_transform.cache_clear()
    yield runtime
    runtime._lightop_fast_topk_transform.cache_clear()


@pytest.mark.parametrize("is_gfx938", [False, True])
def test_sparse_mla_uses_categorized_mqa_abi_with_fp32_contiguous_weights(
    monkeypatch: pytest.MonkeyPatch,
    is_gfx938: bool,
) -> None:
    runtime = _runtime()
    calls: list[tuple[object, ...]] = []
    output = object()

    def mqa_logits(*args):
        calls.append(args)
        return output

    monkeypatch.setattr(runtime.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(runtime, "on_gfx938", lambda: is_gfx938)
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    monkeypatch.setattr(
        runtime,
        "lightop_attention",
        SimpleNamespace(mqa_logits=mqa_logits),
        raising=False,
    )
    monkeypatch.delattr(runtime, "lightop", raising=False)

    weights = torch.arange(8, dtype=torch.float16).reshape(2, 4).transpose(0, 1)
    scale = torch.ones(3)
    result = runtime.rocm_fp8_mqa_logits(
        torch.ones((4, 1, 2)),
        (torch.ones((3, 2)), scale),
        weights,
        torch.zeros(4, dtype=torch.int32),
        torch.full((4,), 3, dtype=torch.int32),
    )

    assert result is output
    assert len(calls) == 1
    assert len(calls[0]) == 6
    supplied_weights = calls[0][2]
    assert supplied_weights.dtype is torch.float32
    assert supplied_weights.is_contiguous()
    assert torch.equal(supplied_weights, weights.float().contiguous())
    assert calls[0][5] is (scale if is_gfx938 else None)


def test_sparse_mla_does_not_retry_legacy_namespace(monkeypatch):
    runtime = _runtime()
    legacy = SimpleNamespace(mqa_logits=lambda *_: pytest.fail("legacy called"))
    monkeypatch.setattr(runtime.current_platform, "is_rocm", lambda: True)
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    monkeypatch.setattr(runtime, "lightop_attention", SimpleNamespace())
    monkeypatch.setattr(runtime, "lightop", legacy, raising=False)
    with pytest.raises(AttributeError):
        runtime.rocm_fp8_mqa_logits(
            torch.ones((1, 1, 2)),
            (torch.ones((1, 2)), torch.ones(1)),
            torch.ones((1, 1)),
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
        )


def test_aiter_opus_paged_mqa_uses_native_page64_cache_and_page_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    expected = torch.full((4, 128), 3.0, dtype=torch.float32)

    def paged_mqa_logits(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(
        runtime.henvs,
        "VLLM_HCU_USE_AITER_OPUS_PAGED_MQA_LOGITS",
        True,
        raising=False,
    )
    monkeypatch.setattr(runtime.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(runtime, "on_gfx938", lambda: True)
    monkeypatch.setattr(
        runtime,
        "_aiter_opus_paged_mqa_logits_fn",
        lambda: paged_mqa_logits,
    )

    q = torch.zeros((1, 4, 32, 128), dtype=torch.float8_e4m3fn)
    cache = torch.empty((3, 64, 1, 132), dtype=torch.uint8)
    page_bytes = cache.view(3, -1)
    page_bytes[0, :64 * 128].fill_(1)
    page_bytes[0, 64 * 128:].fill_(11)
    page_bytes[1, :64 * 128].fill_(2)
    page_bytes[1, 64 * 128:].fill_(22)
    page_bytes[2, :64 * 128].fill_(3)
    page_bytes[2, 64 * 128:].fill_(33)
    weights = torch.ones((4, 32), dtype=torch.float32)
    context_lens = torch.tensor([100], dtype=torch.int32)
    # Static KV allocations can expose more pages than max_model_len needs.
    block_tables = torch.tensor([[1, 0, 2]], dtype=torch.int32)

    result = runtime.rocm_fp8_paged_mqa_logits(
        q,
        cache,
        weights,
        context_lens,
        block_tables,
        torch.empty(0),
        128,
    )

    assert result is expected
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is q
    assert args[1] is cache
    assert args[2] is weights
    assert args[3] is context_lens
    assert args[4] is block_tables
    assert args[5] == 128
    assert kwargs == {
        "out": None,
        "clean_logits": True,
        "kernelId": None,
    }


def test_aiter_opus_paged_mqa_uses_final_mtp_context_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    calls: list[tuple[object, ...]] = []
    expected = torch.full((4, 128), 5.0, dtype=torch.float32)

    def paged_mqa_logits(*args, **_kwargs):
        calls.append(args)
        return expected

    monkeypatch.setattr(
        runtime.henvs,
        "VLLM_HCU_USE_AITER_OPUS_PAGED_MQA_LOGITS",
        True,
        raising=False,
    )
    monkeypatch.setattr(runtime.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(runtime, "on_gfx938", lambda: True)
    monkeypatch.setattr(
        runtime,
        "_aiter_opus_paged_mqa_logits_fn",
        lambda: paged_mqa_logits,
    )

    context_lens = torch.tensor([[97, 98, 99, 100]], dtype=torch.int32)
    result = runtime.rocm_fp8_paged_mqa_logits(
        torch.zeros((1, 4, 32, 128), dtype=torch.float8_e4m3fn),
        torch.zeros((3, 64, 1, 132), dtype=torch.uint8),
        torch.ones((4, 32), dtype=torch.float32),
        context_lens,
        torch.tensor([[0, 1, 2]], dtype=torch.int32),
        torch.empty(0),
        128,
    )

    assert result is expected
    supplied_context_lens = calls[0][3]
    assert torch.equal(supplied_context_lens, torch.tensor([100], dtype=torch.int32))
    assert supplied_context_lens.is_contiguous()


def test_aiter_opus_paged_mqa_falls_back_for_unsupported_mtp_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        runtime.henvs,
        "VLLM_HCU_USE_AITER_OPUS_PAGED_MQA_LOGITS",
        True,
        raising=False,
    )
    monkeypatch.setattr(runtime.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(runtime, "on_gfx938", lambda: True)
    monkeypatch.setattr(
        runtime,
        "_aiter_opus_paged_mqa_logits_fn",
        lambda: pytest.fail("unsupported R=3 called AITER Opus"),
    )
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    expected = torch.ones((3, 64), dtype=torch.float32)
    monkeypatch.setattr(
        runtime,
        "lightop_attention",
        SimpleNamespace(paged_mqa_logits=lambda *_args: expected),
        raising=False,
    )

    result = runtime.rocm_fp8_paged_mqa_logits(
        torch.zeros((1, 3, 32, 128), dtype=torch.float8_e4m3fn),
        torch.zeros((1, 64, 1, 132), dtype=torch.uint8),
        torch.ones((3, 32), dtype=torch.float32),
        torch.tensor([64], dtype=torch.int32),
        torch.tensor([[0]], dtype=torch.int32),
        torch.empty(0),
        64,
    )

    assert result is expected


def test_aiter_opus_paged_mqa_falls_back_off_gfx938(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        runtime.henvs,
        "VLLM_HCU_USE_AITER_OPUS_PAGED_MQA_LOGITS",
        True,
        raising=False,
    )
    monkeypatch.setattr(runtime.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(runtime, "on_gfx938", lambda: False)
    monkeypatch.setattr(
        runtime,
        "_aiter_opus_paged_mqa_logits_fn",
        lambda: pytest.fail("AITER Opus probed on an unsupported device"),
    )
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    expected = torch.ones((4, 64), dtype=torch.float32)
    monkeypatch.setattr(
        runtime,
        "lightop_attention",
        SimpleNamespace(paged_mqa_logits=lambda *_args: expected),
        raising=False,
    )

    result = runtime.rocm_fp8_paged_mqa_logits(
        torch.zeros((1, 4, 32, 128), dtype=torch.float8_e4m3fn),
        torch.zeros((1, 64, 1, 132), dtype=torch.uint8),
        torch.ones((4, 32), dtype=torch.float32),
        torch.tensor([64], dtype=torch.int32),
        torch.tensor([[0]], dtype=torch.int32),
        torch.empty(0),
        64,
    )

    assert result is expected


def test_aiter_opus_paged_mqa_falls_back_for_padded_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        runtime.henvs,
        "VLLM_HCU_USE_AITER_OPUS_PAGED_MQA_LOGITS",
        True,
        raising=False,
    )
    monkeypatch.setattr(runtime.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(runtime, "on_gfx938", lambda: True)
    monkeypatch.setattr(
        runtime,
        "_aiter_opus_paged_mqa_logits_fn",
        lambda: pytest.fail("AITER Opus called for a padded decode batch"),
    )
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    expected = torch.ones((4, 64), dtype=torch.float32)
    monkeypatch.setattr(
        runtime,
        "lightop_attention",
        SimpleNamespace(paged_mqa_logits=lambda *_args: expected),
        raising=False,
    )

    result = runtime.rocm_fp8_paged_mqa_logits(
        torch.zeros((1, 4, 32, 128), dtype=torch.float8_e4m3fn),
        torch.zeros((1, 64, 1, 132), dtype=torch.uint8),
        torch.ones((4, 32), dtype=torch.float32),
        torch.tensor([64], dtype=torch.int32),
        torch.tensor([[0]], dtype=torch.int32),
        torch.empty(0),
        64,
        allow_aiter_opus=False,
    )

    assert result is expected


def test_chunked_sparse_mla_uses_new_abi_and_categorized_topk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    q_fp8 = torch.ones((4, 1, 2), dtype=torch.float16)
    k_fp8 = torch.ones((3, 2))
    weights = torch.arange(8, dtype=torch.float16).reshape(2, 4).transpose(0, 1)
    cu_seqlen_ks = torch.zeros(4, dtype=torch.int32)
    cu_seqlen_ke = torch.full((4,), 3, dtype=torch.int32)
    logits_buffer = torch.zeros(128 * 128)
    topk_indices_buffer = torch.empty((4, 1), dtype=torch.int32)
    marker = 37.5
    topk_marker = 17

    def mqa_logits(
        q: torch.Tensor,
        k: torch.Tensor,
        kernel_weights: torch.Tensor,
        ks: torch.Tensor,
        ke: torch.Tensor,
        kv_scale: torch.Tensor | None,
        clean_logit: bool,
        D_out: torch.Tensor,
    ) -> None:
        assert q.data_ptr() == q_fp8.data_ptr()
        assert torch.equal(q, q_fp8)
        assert k is k_fp8
        assert torch.equal(kernel_weights, weights.float().contiguous())
        assert kernel_weights.dtype is torch.float32
        assert kernel_weights.is_contiguous()
        assert ks.data_ptr() == cu_seqlen_ks.data_ptr()
        assert torch.equal(ks, cu_seqlen_ks)
        assert ke.data_ptr() == cu_seqlen_ke.data_ptr()
        assert torch.equal(ke, cu_seqlen_ke)
        assert kv_scale is None
        assert clean_logit is True
        assert D_out.shape == (128, 128)
        assert D_out.dtype is torch.float32
        assert D_out.data_ptr() == logits_buffer.data_ptr()
        D_out.fill_(marker)

    def top_k_prefill(
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        topk_indices: torch.Tensor,
        num_rows: int,
        row_stride: int,
        column_stride: int,
        topk_tokens: int,
    ) -> None:
        assert logits.shape == (4, 3)
        assert torch.all(logits == marker)
        assert torch.equal(row_starts, cu_seqlen_ks)
        assert torch.equal(row_ends, cu_seqlen_ke)
        assert topk_indices.data_ptr() == topk_indices_buffer.data_ptr()
        assert topk_indices.shape == topk_indices_buffer.shape
        assert num_rows == 4
        assert row_stride == 128
        assert column_stride == 1
        assert topk_tokens == 1
        topk_indices.fill_(topk_marker)

    monkeypatch.setattr(
        runtime,
        "lightop_attention",
        SimpleNamespace(
            mqa_logits=mqa_logits,
            top_k_per_row_prefill=top_k_prefill,
        ),
        raising=False,
    )
    monkeypatch.delattr(runtime, "op", raising=False)
    monkeypatch.setattr(runtime, "on_gfx938", lambda: False)
    monkeypatch.setattr(runtime, "get_logits_buffer", lambda _device: logits_buffer)
    monkeypatch.setattr(runtime.henvs, "VLLM_HCU_USE_LIGHTOP_TOPK", True)
    monkeypatch.setattr(runtime.henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)

    runtime.mqa_logits_inner_chunked(
        SimpleNamespace(
            token_start=0,
            token_end=4,
            cu_seqlen_ks=cu_seqlen_ks,
            cu_seqlen_ke=cu_seqlen_ke,
        ),
        q_fp8,
        k_fp8,
        weights,
        torch.ones(3),
        topk_indices_buffer,
        1,
    )

    assert torch.equal(
        topk_indices_buffer,
        torch.full_like(topk_indices_buffer, topk_marker),
    )


def test_sparse_mla_topk_helpers_use_categorized_attention_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    prefill_calls: list[tuple[object, ...]] = []
    decode_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        runtime,
        "lightop_attention",
        SimpleNamespace(
            top_k_per_row_prefill=lambda *args: prefill_calls.append(args),
            top_k_per_row_decode=lambda *args: decode_calls.append(args),
        ),
        raising=False,
    )
    monkeypatch.delattr(runtime, "op", raising=False)
    logits = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    topk = torch.empty((2, 2), dtype=torch.int32)

    runtime._lightop_topk_indices_prefill(
        logits,
        torch.tensor([0, 1]),
        torch.tensor([6, 5]),
        topk,
        2,
    )
    runtime._lightop_topk_indices_decode(
        logits,
        torch.tensor([6, 5]),
        1,
        topk,
        2,
    )

    assert len(prefill_calls) == 1
    assert len(decode_calls) == 1


def test_sparse_mla_decode_uses_fast_topk_transform_for_mtp3(
    fast_topk_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = fast_topk_runtime
    fused_calls: list[dict[str, object]] = []
    expected = torch.arange(2048, dtype=torch.int32).repeat(4, 1)

    def fast_topk_transform_fused(**kwargs):
        fused_calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        runtime,
        "lightop_attention",
        SimpleNamespace(
            fast_topk_transform_fused=fast_topk_transform_fused,
            top_k_per_row_decode=lambda *_args: pytest.fail(
                "legacy decode TopK called"
            ),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime.henvs,
        "VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM",
        True,
        raising=False,
    )
    logits = torch.arange(4 * 4096, dtype=torch.float32).reshape(4, 4096)
    topk = torch.full((4, 2048), -1, dtype=torch.int32)

    runtime._lightop_topk_indices_decode(
        logits,
        torch.tensor([2051], dtype=torch.int32),
        4,
        topk,
        2048,
    )

    assert len(fused_calls) == 1
    call = fused_calls[0]
    assert call["score"] is logits
    assert torch.equal(call["lengths"], torch.tensor([2048, 2049, 2050, 2051]))
    assert torch.equal(
        call["cu_seqlens_q"], torch.tensor([0, 1, 2, 3, 4])
    )
    assert call["topk"] == 2048
    assert call["row_starts"] is None
    page_table = call["page_table_size_1"]
    assert isinstance(page_table, torch.Tensor)
    assert page_table.shape == (4, 4096)
    assert page_table.is_contiguous()
    assert torch.equal(page_table[3], torch.arange(4096, dtype=torch.int32))
    assert torch.equal(topk, expected)


def test_sparse_mla_decode_falls_back_without_fast_topk(
    fast_topk_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = fast_topk_runtime

    def top_k_per_row_decode(*args):
        args[3].fill_(5)

    monkeypatch.setattr(
        runtime,
        "lightop_attention",
        SimpleNamespace(top_k_per_row_decode=top_k_per_row_decode),
        raising=False,
    )
    monkeypatch.setattr(
        runtime.henvs,
        "VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM",
        True,
        raising=False,
    )
    logits = torch.arange(2 * 4096, dtype=torch.float32).reshape(2, 4096)
    topk = torch.full((2, 2048), -1, dtype=torch.int32)

    runtime._lightop_topk_indices_decode(
        logits,
        torch.tensor([4096, 4095], dtype=torch.int32),
        1,
        topk,
        2048,
    )

    assert torch.equal(topk, torch.full((2, 2048), 5, dtype=torch.int32))


def test_sparse_mla_decode_keeps_direct_topk_when_fast_transform_disabled(
    fast_topk_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = fast_topk_runtime

    def top_k_per_row_decode(*args):
        args[3].fill_(6)

    monkeypatch.setattr(
        runtime,
        "lightop_attention",
        SimpleNamespace(
            fast_topk_transform_fused=lambda **_kwargs: pytest.fail(
                "opt-in fused transform called while disabled"
            ),
            top_k_per_row_decode=top_k_per_row_decode,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime.henvs,
        "VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM",
        False,
        raising=False,
    )
    topk = torch.full((1, 2048), -1, dtype=torch.int32)

    runtime._lightop_topk_indices_decode(
        torch.arange(4096, dtype=torch.float32).reshape(1, 4096),
        torch.tensor([4096], dtype=torch.int32),
        1,
        topk,
        2048,
    )

    assert torch.equal(topk, torch.full_like(topk, 6))


def test_sparse_mla_decode_keeps_direct_topk_for_unsupported_topk(
    fast_topk_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = fast_topk_runtime

    def top_k_per_row_decode(*args):
        args[3].fill_(4)

    monkeypatch.setattr(
        runtime,
        "lightop_attention",
        SimpleNamespace(
            fast_topk_transform_fused=lambda **_kwargs: pytest.fail(
                "fused transform only supports topk=2048"
            ),
            top_k_per_row_decode=top_k_per_row_decode,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime.henvs,
        "VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM",
        True,
        raising=False,
    )
    topk = torch.full((1, 2), -1, dtype=torch.int32)

    runtime._lightop_topk_indices_decode(
        torch.arange(16, dtype=torch.float32).reshape(1, 16),
        torch.tensor([16], dtype=torch.int32),
        1,
        topk,
        2,
    )

    assert torch.equal(topk, torch.full_like(topk, 4))


def test_sparse_mla_decode_reuses_unit_query_cu_seqlens(
    fast_topk_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = fast_topk_runtime
    cu_seqlens: list[torch.Tensor] = []

    def fast_topk_transform_fused(**kwargs):
        cu_seqlens.append(kwargs["cu_seqlens_q"])
        return torch.zeros((2, 2048), dtype=torch.int32)

    monkeypatch.setattr(
        runtime,
        "lightop_attention",
        SimpleNamespace(fast_topk_transform_fused=fast_topk_transform_fused),
        raising=False,
    )
    monkeypatch.setattr(
        runtime.henvs,
        "VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM",
        True,
        raising=False,
    )
    logits = torch.arange(2 * 4096, dtype=torch.float32).reshape(2, 4096)
    topk = torch.empty((2, 2048), dtype=torch.int32)

    for _ in range(2):
        runtime._lightop_topk_indices_decode(
            logits,
            torch.tensor([4096, 4095], dtype=torch.int32),
            1,
            topk,
            2048,
        )

    assert len(cu_seqlens) == 2
    assert cu_seqlens[0].data_ptr() == cu_seqlens[1].data_ptr()
    assert torch.equal(cu_seqlens[0], torch.tensor([0, 1, 2]))


def test_sparse_mla_reserves_identity_table_for_fused_decode(
    fast_topk_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = fast_topk_runtime
    calls: list[tuple[torch.device, int, int]] = []
    cu_seqlens_calls: list[tuple[torch.device, int]] = []
    monkeypatch.setattr(runtime.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(runtime, "on_gfx938", lambda: True)
    monkeypatch.setattr(runtime.henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        runtime.henvs, "VLLM_HCU_USE_LIGHTOP_SPARSE_MLA_TOPK", True
    )
    monkeypatch.setattr(
        runtime.henvs,
        "VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "lightop_attention",
        SimpleNamespace(fast_topk_transform_fused=lambda **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_lightop_identity_page_table",
        lambda device, rows, max_model_len: calls.append(
            (torch.device(device), rows, max_model_len)
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_lightop_unit_query_cu_seqlens",
        lambda device, rows: cu_seqlens_calls.append(
            (torch.device(device), rows)
        ),
    )
    hidden_states = torch.zeros((64, 1), dtype=torch.float32)
    q_fp8 = torch.zeros((64, 32, 128), dtype=torch.float8_e4m3fn)

    runtime._reserve_lightop_identity_page_table_for_profile(
        hidden_states,
        q_fp8,
        topk_tokens=2048,
        max_model_len=8192,
    )

    assert calls == [(torch.device("cpu"), 64, 8192)]
    assert cu_seqlens_calls == [(torch.device("cpu"), 64)]


def test_sparse_mla_skips_fused_profile_mapping_when_api_is_missing(
    fast_topk_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = fast_topk_runtime
    monkeypatch.setattr(runtime.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(runtime, "on_gfx938", lambda: True)
    monkeypatch.setattr(runtime.henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(
        runtime.henvs, "VLLM_HCU_USE_LIGHTOP_SPARSE_MLA_TOPK", True
    )
    monkeypatch.setattr(
        runtime.henvs,
        "VLLM_HCU_USE_LIGHTOP_FAST_TOPK_TRANSFORM",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "lightop_attention",
        SimpleNamespace(),
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_lightop_identity_page_table",
        lambda *_args: pytest.fail("unused fused mapping was reserved"),
    )

    runtime._reserve_lightop_identity_page_table_for_profile(
        torch.zeros((64, 1), dtype=torch.float32),
        torch.zeros((64, 32, 128), dtype=torch.float8_e4m3fn),
        topk_tokens=2048,
        max_model_len=8192,
    )
