# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Portable ABI contracts for DeepSeek V4 owners added after PR #27."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_attention


REPO = Path(__file__).resolve().parents[2]
_KERNEL = "fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32"


def _install_lightop_attention(
    monkeypatch: pytest.MonkeyPatch,
    kernel: Any | None,
    *,
    legacy_kernel: Any | None = None,
) -> None:
    lightop = ModuleType("lightop")
    lightop.__path__ = []  # type: ignore[attr-defined]
    attention = ModuleType("lightop.attention")
    if kernel is not None:
        setattr(attention, _KERNEL, kernel)
    lightop.attention = attention  # type: ignore[attr-defined]
    lightop.op = SimpleNamespace(
        fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert=(
            legacy_kernel
            if legacy_kernel is not None
            else lambda *args: pytest.fail("selected obsolete lightop.op kernel")
        )
    )
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.attention", attention)


@pytest.fixture
def dspark_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load only the DSpark adapter, with its upstream imports replaced."""

    mhc = ModuleType("vllm.model_executor.layers.mhc")
    mhc.HCHeadOp = type("HCHeadOp", (), {})
    monkeypatch.setitem(sys.modules, mhc.__name__, mhc)

    dspark = ModuleType("vllm.models.deepseek_v4.nvidia.dspark")
    dspark.DSparkDeepseekV4Model = type("DSparkDeepseekV4Model", (nn.Module,), {})
    dspark.DSparkDeepseekV4ForCausalLM = type(
        "DSparkDeepseekV4ForCausalLM", (nn.Module,), {}
    )
    dspark._insert_context_kv = lambda *args: None
    deepseek_v4 = ModuleType("vllm.models.deepseek_v4")
    deepseek_v4.__path__ = []  # type: ignore[attr-defined]
    nvidia = ModuleType("vllm.models.deepseek_v4.nvidia")
    nvidia.__path__ = []  # type: ignore[attr-defined]
    nvidia.dspark = dspark  # type: ignore[attr-defined]
    deepseek_v4.nvidia = nvidia  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, deepseek_v4.__name__, deepseek_v4)
    monkeypatch.setitem(sys.modules, nvidia.__name__, nvidia)
    monkeypatch.setitem(sys.modules, dspark.__name__, dspark)

    module_name = "vllm_hcu.models._test_lightop_deepseek_v4_dspark"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO / "vllm_hcu/models/deepseek_v4_dspark.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_dspark_context_passes_raw_kv_and_kv_norm_weight(
    monkeypatch: pytest.MonkeyPatch,
    dspark_module: ModuleType,
) -> None:
    calls: list[tuple[object, ...]] = []
    _install_lightop_attention(monkeypatch, lambda *args: calls.append(args))
    raw_kv = torch.ones((2, 8), dtype=torch.float32)
    kv_weight = object()
    slot_mapping = torch.tensor([3, 99, 7, 99], dtype=torch.int64)[::2]

    class QrKv:
        def __getitem__(self, key: object) -> torch.Tensor:
            assert key == (..., slice(2, None))
            return raw_kv

    class KvNorm:
        weight = SimpleNamespace(data=kv_weight)

        def __call__(self, value: torch.Tensor) -> torch.Tensor:
            del value
            pytest.fail("DSpark pre-normalized KV")

    attn = SimpleNamespace(
        q_lora_rank=2,
        fused_wqa_wkv=lambda value: (QrKv(), None),
        kv_norm=KvNorm(),
        n_local_heads=1,
        head_dim=8,
        eps=1e-6,
        rotary_emb=SimpleNamespace(cos_sin_cache=object()),
        swa_cache_layer=SimpleNamespace(
            kv_cache=torch.zeros((2, 4, 8), dtype=torch.uint8),
            block_size=4,
        ),
    )
    model = object.__new__(dspark_module.DSparkDeepseekV4Model)
    model.layers = [SimpleNamespace(attn=attn)]

    dspark_module.DSparkDeepseekV4Model.precompute_and_store_context_kv(
        model,
        object(),
        torch.tensor([4, 5], dtype=torch.int32),
        [slot_mapping],
    )

    q, passed_raw_kv, passed_weight, _cache, slots, *_ = calls[0]
    assert q.shape == (2, 1, 8)
    assert passed_raw_kv is raw_kv
    assert passed_weight is kv_weight
    assert slots.dtype is torch.int32
    assert slots.is_contiguous()


def _core_fix_instance() -> tuple[
    object, object, torch.Tensor, torch.Tensor, object
]:
    class DeepseekV4Attention:
        def __init__(
            self,
            vllm_config,
            prefix,
            topk_indices_buffer=None,
            aux_stream_list=None,
        ):
            del vllm_config, prefix, topk_indices_buffer, aux_stream_list

        def attn_gemm_parallel_execute(self, hidden_states):
            return hidden_states

        def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
            del q, kv, positions, attn_metadata
            return "official"

    module = ModuleType(patch_deepseek_v4_attention.TARGET_MODULE)
    module.DeepseekV4Attention = DeepseekV4Attention  # type: ignore[attr-defined]
    module.execute_in_parallel = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    module.envs = SimpleNamespace(VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1)  # type: ignore[attr-defined]
    patch_deepseek_v4_attention.apply_to_module(module)

    instance = DeepseekV4Attention(
        SimpleNamespace(quant_config=None), "layer.attn"
    )
    kv_weight = object()
    instance.kv_norm = SimpleNamespace(weight=SimpleNamespace(data=kv_weight))
    instance.swa_cache_layer = SimpleNamespace(
        prefix="layer.swa",
        kv_cache=torch.zeros((2, 4, 8), dtype=torch.uint8),
    )
    instance.rotary_emb = SimpleNamespace(cos_sin_cache=object())
    instance.eps = 1e-6
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([2, 99, 4, 99], dtype=torch.int64)[::2],
        block_size=4,
    )
    return instance, kv_weight, torch.ones((2, 8)), torch.tensor([6, 7]), metadata


def test_core_fix_patch_passes_kv_norm_weight_and_int32_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    _install_lightop_attention(monkeypatch, lambda *args: calls.append(args))
    instance, kv_weight, raw_kv, positions, metadata = _core_fix_instance()
    q = torch.zeros((2, 1, 8))

    result = instance._fused_qnorm_rope_kv_insert(
        q, raw_kv, positions, {"layer.swa": metadata}
    )

    assert result is q
    assert calls[0][:3] == (q, raw_kv, kv_weight)
    assert calls[0][4].dtype is torch.int32
    assert calls[0][4].is_contiguous()
    assert calls[0][5].dtype is torch.int64


def test_core_fix_patch_requires_categorized_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_lightop_attention(
        monkeypatch,
        kernel=None,
        legacy_kernel=lambda *args: pytest.fail(
            "selected obsolete lightop.op kernel"
        ),
    )
    instance, _weight, raw_kv, positions, metadata = _core_fix_instance()

    with pytest.raises(RuntimeError, match="lightop\\.attention"):
        instance._fused_qnorm_rope_kv_insert(
            torch.zeros((2, 1, 8)), raw_kv, positions, {"layer.swa": metadata}
        )
