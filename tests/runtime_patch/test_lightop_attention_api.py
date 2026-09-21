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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    fused_calls: list[dict[str, object]] = []
    expected = torch.tensor(
        [[8, 7], [9, 8], [10, 9], [11, 10]], dtype=torch.int32
    )

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
    resolver = getattr(runtime, "_lightop_fast_topk_transform", None)
    if resolver is not None:
        resolver.cache_clear()
    logits = torch.arange(64, dtype=torch.float32).reshape(4, 16)
    topk = torch.full((4, 2), -1, dtype=torch.int32)

    runtime._lightop_topk_indices_decode(
        logits,
        torch.tensor([12], dtype=torch.int32),
        4,
        topk,
        2,
    )

    assert len(fused_calls) == 1
    call = fused_calls[0]
    assert call["score"] is logits
    assert torch.equal(call["lengths"], torch.tensor([9, 10, 11, 12]))
    assert torch.equal(
        call["cu_seqlens_q"], torch.tensor([0, 1, 2, 3, 4])
    )
    assert call["topk"] == 2
    assert call["row_starts"] is None
    page_table = call["page_table_size_1"]
    assert isinstance(page_table, torch.Tensor)
    assert page_table.shape == (4, 16)
    assert page_table.is_contiguous()
    assert torch.equal(page_table[3], torch.arange(16, dtype=torch.int32))
    assert torch.equal(topk, expected)


def test_sparse_mla_decode_falls_back_without_fast_topk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()

    def top_k_per_row_decode(*args):
        args[3].fill_(5)

    monkeypatch.setattr(
        runtime,
        "lightop_attention",
        SimpleNamespace(top_k_per_row_decode=top_k_per_row_decode),
        raising=False,
    )
    resolver = getattr(runtime, "_lightop_fast_topk_transform", None)
    if resolver is not None:
        resolver.cache_clear()
    logits = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    topk = torch.full((2, 2), -1, dtype=torch.int32)

    runtime._lightop_topk_indices_decode(
        logits,
        torch.tensor([6, 5], dtype=torch.int32),
        1,
        topk,
        2,
    )

    assert torch.equal(topk, torch.full((2, 2), 5, dtype=torch.int32))


def _enable_sparse_mask_route(monkeypatch: pytest.MonkeyPatch):
    runtime = _runtime()
    monkeypatch.setattr(runtime.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(runtime, "on_gfx938", lambda: True)
    monkeypatch.setattr(runtime.henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(runtime.henvs, "VLLM_HCU_USE_LIGHTOP_MASK_TOPK", True)
    runtime._lightop_sparse_mask_topk_ops.cache_clear()
    return runtime


def test_sparse_mask_route_respects_opt_in_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _enable_sparse_mask_route(monkeypatch)
    monkeypatch.setattr(runtime.henvs, "VLLM_HCU_USE_LIGHTOP_MASK_TOPK", False)
    monkeypatch.setattr(
        runtime,
        "_lightop_sparse_mask_topk_ops",
        lambda: pytest.fail("LightOp APIs probed while route is disabled"),
    )

    q = torch.zeros((1, 1, 32, 128), dtype=torch.float8_e4m3fn)
    kv_cache = torch.zeros((1, 64, 1, 132), dtype=torch.uint8)
    weights = torch.zeros((1, 32), dtype=torch.float32)
    seq_lens = torch.tensor([12], dtype=torch.int32)
    block_table = torch.zeros((1, 1), dtype=torch.int32)

    assert runtime._lightop_mask_topk_decode(
        q, kv_cache, weights, seq_lens, block_table, 1, 1, 2048, 64, False
    ) is None


def test_sparse_mask_route_pairs_plain_producer_and_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _enable_sparse_mask_route(monkeypatch)
    calls: list[str] = []

    def producer(*args, **kwargs):
        calls.append("producer")
        assert args[0].shape == (2, 1, 32, 128)
        assert args[4].shape == (2, 1)
        return torch.zeros((2, 64)), torch.zeros((2, 4), dtype=torch.int16)

    def consumer(**kwargs):
        calls.append("consumer")
        assert kwargs["page_table_size_1"].is_contiguous()
        assert torch.equal(
            kwargs["page_table_size_1"][0],
            torch.arange(64, dtype=torch.int32),
        )
        assert torch.equal(
            kwargs["cu_seqlens_q"],
            torch.tensor([0, 1, 2], dtype=torch.int32),
        )
        return torch.full((2, 2048), 7, dtype=torch.int32)

    monkeypatch.setattr(
        runtime,
        "_lightop_sparse_mask_topk_ops",
        lambda: (producer, consumer),
    )
    q = torch.zeros((2, 1, 32, 128), dtype=torch.float8_e4m3fn)
    kv_cache = torch.zeros((1, 64, 1, 132), dtype=torch.uint8)
    weights = torch.zeros((2, 32), dtype=torch.float32)
    seq_lens = torch.tensor([12, 20], dtype=torch.int32)
    block_table = torch.zeros((2, 1), dtype=torch.int32)

    result = runtime._lightop_mask_topk_decode(
        q,
        kv_cache,
        weights,
        seq_lens,
        block_table,
        2,
        1,
        2048,
        64,
        False,
    )

    assert result is not None
    assert torch.equal(result, torch.full((2, 2048), 7, dtype=torch.int32))
    assert calls == ["producer", "consumer"]


def test_sparse_mask_route_reuses_identity_page_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _enable_sparse_mask_route(monkeypatch)
    page_tables: list[torch.Tensor] = []

    def producer(*args, **kwargs):
        del args, kwargs
        return torch.zeros((2, 192)), torch.zeros((2, 12), dtype=torch.int16)

    def consumer(**kwargs):
        page_tables.append(kwargs["page_table_size_1"])
        return torch.zeros((2, 2048), dtype=torch.int32)

    monkeypatch.setattr(
        runtime,
        "_lightop_sparse_mask_topk_ops",
        lambda: (producer, consumer),
    )
    q = torch.zeros((2, 1, 32, 128), dtype=torch.float8_e4m3fn)
    kv_cache = torch.zeros((3, 64, 1, 132), dtype=torch.uint8)
    weights = torch.zeros((2, 32), dtype=torch.float32)
    seq_lens = torch.tensor([12, 20], dtype=torch.int32)
    block_table = torch.zeros((2, 3), dtype=torch.int32)

    for _ in range(2):
        assert runtime._lightop_mask_topk_decode(
            q,
            kv_cache,
            weights,
            seq_lens,
            block_table,
            2,
            1,
            2048,
            192,
            False,
        ) is not None

    assert len(page_tables) == 2
    assert page_tables[0].data_ptr() == page_tables[1].data_ptr()
    assert page_tables[0].is_contiguous()
    torch.testing.assert_close(
        page_tables[0][1], torch.arange(192, dtype=torch.int32)
    )


def test_sparse_mask_route_reserves_identity_table_during_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _enable_sparse_mask_route(monkeypatch)
    arange_calls: list[int] = []
    original_arange = torch.arange

    def tracked_arange(*args, **kwargs):
        if args and args[0] == 320:
            arange_calls.append(args[0])
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", tracked_arange)
    monkeypatch.setattr(
        runtime,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )

    runtime.rocm_aiter_sparse_attn_indexer_native(
        hidden_states=torch.zeros((3, 1), dtype=torch.float32),
        k_cache_prefix="layer",
        kv_cache=torch.zeros((1, 64, 132), dtype=torch.uint8),
        q_fp8=torch.zeros((3, 32, 128), dtype=torch.float8_e4m3fn),
        k=torch.zeros((3, 128), dtype=torch.float32),
        weights=torch.ones((3, 32), dtype=torch.float32),
        quant_block_size=128,
        scale_fmt="e4m3",
        topk_tokens=2048,
        head_dim=128,
        max_model_len=320,
        total_seq_lens=320,
        topk_indices_buffer=torch.full((3, 2048), -1, dtype=torch.int32),
        skip_k_cache_insert=True,
    )

    assert arange_calls == [320]


def test_sparse_mask_route_uses_public_plain_producer_for_mtp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _enable_sparse_mask_route(monkeypatch)
    calls: list[str] = []

    def producer(*args, **kwargs):
        calls.append("producer")
        assert args[0].shape == (6, 1, 32, 128)
        assert args[4].shape == (6, 1)
        assert "group_size" not in kwargs
        return torch.zeros((6, 64)), torch.zeros((6, 4), dtype=torch.int16)

    def consumer(**kwargs):
        calls.append("consumer")
        assert kwargs["lengths"].shape == (6,)
        assert torch.equal(
            kwargs["cu_seqlens_q"],
            torch.arange(7, dtype=torch.int32),
        )
        return torch.zeros((6, 2048), dtype=torch.int32)

    monkeypatch.setattr(
        runtime,
        "_lightop_sparse_mask_topk_ops",
        lambda: (producer, consumer),
    )
    q = torch.zeros((2, 3, 32, 128), dtype=torch.float8_e4m3fn)
    kv_cache = torch.zeros((1, 64, 1, 132), dtype=torch.uint8)
    weights = torch.zeros((6, 32), dtype=torch.float32)
    seq_lens = torch.tensor([12, 20], dtype=torch.int32)
    block_table = torch.zeros((2, 1), dtype=torch.int32)

    result = runtime._lightop_mask_topk_decode(
        q,
        kv_cache,
        weights,
        seq_lens,
        block_table,
        2,
        3,
        2048,
        64,
        False,
    )

    assert result is not None
    assert result.shape == (6, 2048)
    assert calls == ["producer", "consumer"]


def test_sparse_mask_route_falls_back_for_unsupported_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _enable_sparse_mask_route(monkeypatch)
    calls = 0

    def ops():
        nonlocal calls
        calls += 1
        return (lambda *_args, **_kwargs: None,) * 2

    monkeypatch.setattr(runtime, "_lightop_sparse_mask_topk_ops", ops)
    q = torch.zeros((2, 1, 8, 128), dtype=torch.float8_e4m3fn)
    kv_cache = torch.zeros((1, 64, 1, 132), dtype=torch.uint8)
    weights = torch.zeros((2, 8), dtype=torch.float32)
    seq_lens = torch.tensor([12, 20], dtype=torch.int32)
    block_table = torch.zeros((2, 1), dtype=torch.int32)

    result = runtime._lightop_mask_topk_decode(
        q,
        kv_cache,
        weights,
        seq_lens,
        block_table,
        2,
        1,
        2048,
        64,
        False,
    )

    assert result is None
    assert calls == 0
