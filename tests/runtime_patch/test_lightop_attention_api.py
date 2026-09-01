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
    namespace = {"torch": torch, "split_kv_cache": lambda *_args, **_kwargs: None}
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


def _runtime():
    return importlib.import_module("vllm_hcu.v1.attention.ops.rocm_aiter_mla_sparse")


def test_sparse_mla_uses_categorized_mqa_abi_with_fp32_contiguous_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    calls: list[tuple[object, ...]] = []
    output = object()

    def mqa_logits(*args):
        calls.append(args)
        return output

    monkeypatch.setattr(runtime.current_platform, "is_rocm", lambda: True)
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
    result = runtime.rocm_fp8_mqa_logits(
        torch.ones((4, 1, 2)),
        (torch.ones((3, 2)), torch.ones(3)),
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
