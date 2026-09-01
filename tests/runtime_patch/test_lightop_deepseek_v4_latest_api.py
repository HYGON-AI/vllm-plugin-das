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


def _core_fix_instance(
    official_inserted_kv: list[torch.Tensor] | None = None,
) -> tuple[
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

        def forward(self, positions, hidden_states, llama_4_scaling=None):
            del llama_4_scaling
            qr_kv, *_ = self.attn_gemm_parallel_execute(hidden_states)
            qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
            qr = self.q_norm(qr)
            kv = self.kv_norm(kv)
            out = torch.empty(
                (hidden_states.shape[0], self.padded_heads, self.head_dim),
                dtype=hidden_states.dtype,
            )
            self.attention_impl(
                hidden_states,
                qr,
                kv,
                None,
                None,
                None,
                positions,
                out,
            )
            return self._o_proj(out[:, : self.n_local_heads, :], positions)

        def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
            del positions, attn_metadata
            if official_inserted_kv is not None:
                official_inserted_kv.append(kv.clone())
            return q

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


def test_core_fix_forward_applies_kv_norm_exactly_once_in_uint8_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The patched upstream caller must pass raw KV into the KVNorm kernel."""

    normalized_by: list[str] = []
    inserted_kv: list[torch.Tensor] = []

    def lightop_insert(
        _q,
        kv,
        _kv_norm_weight,
        _cache,
        _slots,
        _positions,
        _cos_sin_cache,
        _eps,
        _block_size,
    ) -> None:
        inserted_kv.append(kv.clone())
        normalized_by.append("lightop-kernel")

    _install_lightop_attention(monkeypatch, lightop_insert)
    instance, _kv_weight, _raw_kv, positions, metadata = _core_fix_instance()
    instance.q_lora_rank = 2
    instance.head_dim = 4
    instance.n_local_heads = 1
    instance.padded_heads = 1
    raw_qr_kv = torch.tensor(
        [[1.0, 2.0, 10.0, 20.0, 30.0, 40.0],
         [3.0, 4.0, 50.0, 60.0, 70.0, 80.0]]
    )
    instance.attn_gemm_parallel_execute = lambda _hidden: (
        raw_qr_kv,
        None,
        None,
        None,
    )
    instance.q_norm = lambda qr: qr + 100

    class KvNorm:
        weight = instance.kv_norm.weight

        def __call__(self, kv):
            normalized_by.append("caller")
            return kv + 1000

    instance.kv_norm = KvNorm()

    def attention_impl(
        _hidden_states,
        q,
        kv,
        _kv_score,
        _indexer_kv_score,
        _indexer_weights,
        forwarded_positions,
        out,
    ) -> None:
        instance._fused_qnorm_rope_kv_insert(
            q, kv, forwarded_positions, {"layer.swa": metadata}
        )
        out.zero_()

    instance.attention_impl = attention_impl
    instance._o_proj = lambda out, _positions: out

    instance.forward(positions.to(torch.int64), torch.zeros((2, 6)))

    assert normalized_by == ["lightop-kernel"]
    torch.testing.assert_close(inserted_kv[0], raw_qr_kv[:, 2:])


def test_core_fix_forward_normalizes_kv_when_delegating_non_uint8_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The official cache implementation still receives normalized KV."""

    _install_lightop_attention(
        monkeypatch,
        lambda *_args: pytest.fail("selected uint8 LightOp insertion kernel"),
    )
    official_inserted_kv: list[torch.Tensor] = []
    instance, _kv_weight, _raw_kv, positions, metadata = _core_fix_instance(
        official_inserted_kv
    )
    instance.swa_cache_layer.kv_cache = torch.zeros(
        (2, 4, 4), dtype=torch.bfloat16
    )
    instance.q_lora_rank = 2
    instance.head_dim = 4
    instance.n_local_heads = 1
    instance.padded_heads = 1
    raw_qr_kv = torch.tensor(
        [
            [1.0, 2.0, 10.0, 20.0, 30.0, 40.0],
            [3.0, 4.0, 50.0, 60.0, 70.0, 80.0],
        ]
    )
    instance.attn_gemm_parallel_execute = lambda _hidden: (
        raw_qr_kv,
        None,
        None,
        None,
    )
    instance.q_norm = lambda qr: qr + 100
    kv_norm_inputs: list[torch.Tensor] = []

    class KvNorm:
        weight = instance.kv_norm.weight

        def __call__(self, kv):
            kv_norm_inputs.append(kv.clone())
            return kv + 1000

    instance.kv_norm = KvNorm()

    def attention_impl(
        _hidden_states,
        q,
        kv,
        _kv_score,
        _indexer_kv_score,
        _indexer_weights,
        forwarded_positions,
        out,
    ) -> None:
        instance._fused_qnorm_rope_kv_insert(
            q, kv, forwarded_positions, {"layer.swa": metadata}
        )
        out.zero_()

    instance.attention_impl = attention_impl
    instance._o_proj = lambda out, _positions: out

    instance.forward(positions.to(torch.int64), torch.zeros((2, 6)))

    assert len(kv_norm_inputs) == 1
    torch.testing.assert_close(kv_norm_inputs[0], raw_qr_kv[:, 2:])
    torch.testing.assert_close(official_inserted_kv[0], raw_qr_kv[:, 2:] + 1000)


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
