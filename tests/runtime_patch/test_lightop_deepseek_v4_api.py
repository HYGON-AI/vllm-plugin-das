# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import __future__
import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch


SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm_hcu/model_executor/layers/deepseek_v4_attention.py"
)


def _extracted_method(name: str) -> Any:
    """Compile the production method body without importing its heavy module."""
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "DeepseekV4MultiHeadLatentAttentionWrapper"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace: dict[str, Any] = {"cast": lambda _type, value: value, "torch": torch}
    exec(
        compile(module, str(SOURCE_PATH), "exec", __future__.annotations.compiler_flag),
        namespace,
    )
    return namespace[name]


class _QrKv:
    def __init__(self, qr: object, kv: object) -> None:
        self.qr = qr
        self.kv = kv

    def split(self, sizes: list[int], dim: int) -> tuple[object, object]:
        assert sizes == [2, 2]
        assert dim == -1
        return self.qr, self.kv


def test_attention_impl_normalizes_only_qr_and_keeps_raw_kv_for_insert() -> None:
    """A two-output RMSNorm would replace raw KV before its sole owner sees it."""
    attention_impl = _extracted_method("attention_impl")
    raw_qr = object()
    raw_kv = object()
    normalized_qr = object()
    legacy_qr = object()
    legacy_kv = object()
    qnorm_inputs: list[object] = []
    projection_inputs: list[object] = []
    inserted: list[tuple[object, object]] = []

    class FakeSelf:
        q_lora_rank = 2
        head_dim = 2
        n_local_heads = 2
        indexer = None
        compressor = None

        def __init__(self) -> None:
            class QNorm:
                weight = SimpleNamespace(data=object())

                def __call__(_self, value: object) -> object:
                    qnorm_inputs.append(value)
                    return normalized_qr

            self.q_norm = QNorm()
            self.kv_norm = SimpleNamespace(weight=SimpleNamespace(data=object()))
            self.eps = 1e-6
            self.wq_b = lambda value: projection_inputs.append(value) or torch.zeros(
                (1, 2, 2)
            )
            self.mla_attn = lambda *args, **kwargs: None

        def attn_gemm_parallel_execute(self, hidden_states: object) -> tuple[Any, ...]:
            return _QrKv(raw_qr, raw_kv), None, None, None

        def _fused_qnorm_rope_kv_insert(
            self, q: object, kv: object, positions: object, metadata: object
        ) -> None:
            inserted.append((q, kv))

    # Keep the old helper callable so this test fails on its data flow, not
    # because the extracted method is missing a global.
    attention_impl.__globals__["fused_q_kv_rmsnorm"] = (
        lambda *args: (legacy_qr, legacy_kv)
    )
    attention_impl.__globals__["get_forward_context"] = lambda: SimpleNamespace(
        attn_metadata={}
    )
    attention_impl(FakeSelf(), object(), object(), torch.zeros((1, 2, 2)))

    assert qnorm_inputs == [raw_qr]
    assert projection_inputs == [normalized_qr]
    assert inserted[0][1] is raw_kv


def _install_attention_module(
    monkeypatch: pytest.MonkeyPatch, kernel: Any | None = None
) -> None:
    lightop = ModuleType("lightop")
    lightop.__path__ = []  # type: ignore[attr-defined]
    attention = ModuleType("lightop.attention")
    if kernel is not None:
        attention.fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32 = kernel
    lightop.attention = attention
    lightop.op = SimpleNamespace(
        fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert=lambda *args: None
    )
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.attention", attention)


def _fused_self() -> tuple[object, dict[str, object]]:
    cache_2d = object()
    kv_weight = object()
    slot_mapping = object()
    cos_sin_cache = object()

    class Cache:
        shape = (4, 2, 3)

        def view(self, first_dim: int, second_dim: int) -> object:
            assert (first_dim, second_dim) == (4, -1)
            return cache_2d

    metadata = SimpleNamespace(slot_mapping=slot_mapping, block_size=64)
    self = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(prefix="swa", kv_cache=Cache()),
        kv_norm=SimpleNamespace(weight=SimpleNamespace(data=kv_weight)),
        rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin_cache),
        eps=1e-6,
    )
    return self, {
        "cache_2d": cache_2d,
        "kv_weight": kv_weight,
        "metadata": metadata,
        "slot_mapping": slot_mapping,
        "cos_sin_cache": cos_sin_cache,
    }


def test_fused_insert_passes_raw_kv_and_all_nine_categorized_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrong kernel, normalization ownership, or argument order changes output data flow."""
    fused_insert = _extracted_method("_fused_qnorm_rope_kv_insert")
    calls: list[tuple[object, ...]] = []
    _install_attention_module(monkeypatch, lambda *args: calls.append(args))
    self, expected = _fused_self()
    q = object()
    raw_kv = object()

    class Positions:
        def to(self, dtype: torch.dtype) -> object:
            assert dtype is torch.int64
            return converted_positions

    converted_positions = object()
    fused_insert(self, q, raw_kv, Positions(), {"swa": expected["metadata"]})

    assert calls == [
        (
            q,
            raw_kv,
            expected["kv_weight"],
            expected["cache_2d"],
            expected["slot_mapping"],
            converted_positions,
            expected["cos_sin_cache"],
            1e-6,
            64,
        )
    ]


def test_fused_insert_requires_categorized_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fused_insert = _extracted_method("_fused_qnorm_rope_kv_insert")
    _install_attention_module(monkeypatch)
    self, expected = _fused_self()

    with pytest.raises(
        RuntimeError,
        match=(
            "DeepSeek V4 requires lightop\\.attention\\."
            "fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32; "
            "upgrade LightOp"
        ),
    ):
        fused_insert(
            self,
            object(),
            object(),
            torch.tensor([0]),
            {"swa": expected["metadata"]},
        )


def test_fused_insert_propagates_categorized_kernel_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fused_insert = _extracted_method("_fused_qnorm_rope_kv_insert")

    def failing_kernel(*args: object) -> None:
        raise RuntimeError("kernel failure")

    _install_attention_module(monkeypatch, failing_kernel)
    self, expected = _fused_self()
    with pytest.raises(RuntimeError, match="kernel failure"):
        fused_insert(
            self,
            object(),
            object(),
            torch.tensor([0]),
            {"swa": expected["metadata"]},
        )
