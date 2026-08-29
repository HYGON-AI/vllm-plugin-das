# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def dspark_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the HCU adapter while replacing only accelerator dependencies."""

    class StubModule(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            super().__init__()

    mhc = ModuleType("vllm.model_executor.layers.mhc")
    mhc.HCHeadOp = type("HCHeadOp", (), {})
    monkeypatch.setitem(sys.modules, mhc.__name__, mhc)

    dspark = ModuleType("vllm.models.deepseek_v4.nvidia.dspark")
    dspark.DSparkDeepseekV4Model = type(
        "DSparkDeepseekV4Model",
        (nn.Module,),
        {},
    )
    dspark.DSparkDeepseekV4ForCausalLM = type(
        "DSparkDeepseekV4ForCausalLM",
        (nn.Module,),
        {},
    )
    dspark.DeepseekV4DecoderLayer = object()
    dspark.make_deepseek_v4_expert_params_mapping = object()
    dspark._insert_context_kv = object()
    dspark.VocabParallelEmbedding = StubModule
    dspark.ReplicatedLinear = StubModule
    dspark.RMSNorm = StubModule
    dspark.DSparkMarkovHead = StubModule
    dspark.ParallelLMHead = StubModule
    dspark.LogitsProcessor = StubModule
    dspark.maybe_prefix = lambda prefix, name: f"{prefix}.{name}" if prefix else name
    dspark.get_current_vllm_config = lambda: SimpleNamespace()
    dspark._test_originals = {
        name: getattr(dspark, name)
        for name in (
            "DSparkDeepseekV4Model",
            "DSparkDeepseekV4ForCausalLM",
            "DeepseekV4DecoderLayer",
            "make_deepseek_v4_expert_params_mapping",
            "_insert_context_kv",
        )
    }
    deepseek_v4 = ModuleType("vllm.models.deepseek_v4")
    deepseek_v4.__path__ = []  # type: ignore[attr-defined]
    nvidia = ModuleType("vllm.models.deepseek_v4.nvidia")
    nvidia.__path__ = []  # type: ignore[attr-defined]
    nvidia.dspark = dspark
    deepseek_v4.nvidia = nvidia
    monkeypatch.setitem(sys.modules, deepseek_v4.__name__, deepseek_v4)
    monkeypatch.setitem(sys.modules, nvidia.__name__, nvidia)
    monkeypatch.setitem(sys.modules, dspark.__name__, dspark)

    module_name = "vllm_hcu.models._test_deepseek_v4_dspark"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO / "vllm_hcu/models/deepseek_v4_dspark.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_dspark_adapter_import_keeps_upstream_module_symbols(
    dspark_module: ModuleType,
) -> None:
    upstream = dspark_module._dspark

    for name, original in upstream._test_originals.items():
        assert getattr(upstream, name) is original


def test_dspark_adapter_construction_keeps_upstream_module_symbols(
    dspark_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AMDDeepseekV4DecoderLayer(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            super().__init__()
            self.ffn = SimpleNamespace(use_mega_moe=True)

    amd_model = ModuleType("vllm.models.deepseek_v4.amd.model")
    amd_model.DeepseekV4DecoderLayer = AMDDeepseekV4DecoderLayer
    amd = ModuleType("vllm.models.deepseek_v4.amd")
    amd.__path__ = []  # type: ignore[attr-defined]
    amd.model = amd_model
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(device_type="cpu")
    monkeypatch.setitem(sys.modules, amd.__name__, amd)
    monkeypatch.setitem(sys.modules, amd_model.__name__, amd_model)
    monkeypatch.setitem(sys.modules, platforms.__name__, platforms)

    config = SimpleNamespace(
        hidden_size=4,
        hc_mult=2,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        num_hidden_layers=2,
        dspark_target_layer_ids=(1,),
        n_mtp_layers=1,
        vocab_size=16,
        dspark_markov_rank=2,
        index_topk=2,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        model_config=SimpleNamespace(hf_config=config),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        quant_config=None,
    )
    dspark_module._dspark.get_current_vllm_config = lambda: vllm_config

    dspark_module.DSparkDeepseekV4ForCausalLM(
        vllm_config=vllm_config,
    )

    upstream = dspark_module._dspark
    for name, original in upstream._test_originals.items():
        assert getattr(upstream, name) is original


def test_register_model_points_dspark_draft_to_hcu_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_hcu.models as models

    registrations: list[tuple[str, str]] = []
    monkeypatch.setattr(
        models.ModelRegistry,
        "register_model",
        lambda name, target: registrations.append((name, target)),
    )

    models.register_model()

    assert (
        "DSparkDraftModel",
        "vllm_hcu.models.deepseek_v4_dspark:DSparkDeepseekV4ForCausalLM",
    ) in registrations


def test_dspark_context_insert_uses_only_non_pcp_lightop(
    dspark_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    def normal(*args: object) -> None:
        calls.append(args)

    def pcp(*args: object) -> None:
        raise AssertionError(f"PCP kernel must not be called: {args!r}")

    lightop = ModuleType("lightop")
    lightop.op = SimpleNamespace(
        fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert=normal,
        fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp=pcp,
    )
    monkeypatch.setitem(sys.modules, "lightop", lightop)

    cache = torch.zeros((2, 4, 8), dtype=torch.float32)
    cos_sin_cache = torch.arange(16, dtype=torch.float32)
    attn = SimpleNamespace(
        n_local_heads=2,
        head_dim=8,
        eps=1e-6,
        rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin_cache),
        swa_cache_layer=SimpleNamespace(kv_cache=cache, block_size=4),
    )
    kv = torch.ones((3, 8), dtype=torch.float32)
    positions = torch.tensor([0, 1, 2], dtype=torch.int32)
    slot_mapping = torch.tensor([4, 5, 6], dtype=torch.int64)

    dspark_module._insert_context_kv(attn, kv, positions, slot_mapping)

    assert len(calls) == 1
    dummy_q, passed_kv, cache_2d, passed_slots, passed_positions, rope, eps, block = (
        calls[0]
    )
    assert dummy_q.shape == (3, 2, 8)
    assert dummy_q.dtype == kv.dtype
    assert passed_kv is kv
    assert cache_2d.shape == (2, 32)
    assert passed_slots is slot_mapping
    assert passed_positions.dtype == torch.int64
    assert rope is cos_sin_cache
    assert eps == 1e-6
    assert block == 4
