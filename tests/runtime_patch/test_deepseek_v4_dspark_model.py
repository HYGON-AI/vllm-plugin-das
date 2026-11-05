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
    dspark.DSparkConfidenceHead = StubModule
    dspark.ParallelLMHead = StubModule
    dspark.LogitsProcessor = StubModule
    dspark.maybe_prefix = lambda prefix, name: f"{prefix}.{name}" if prefix else name
    dspark.get_current_vllm_config = lambda: SimpleNamespace()
    dspark._use_sequence_parallel = (
        lambda vllm_config: vllm_config.parallel_config.sequence_parallel
    )
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


@pytest.mark.parametrize(
    ("weight_block_size", "expected_pad"),
    ((None, False), ((128, 128), True)),
)
def test_dspark_adapter_construction_keeps_upstream_module_symbols(
    dspark_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    weight_block_size: tuple[int, int] | None,
    expected_pad: bool,
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
        enable_confidence_head=True,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        model_config=SimpleNamespace(hf_config=config),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        parallel_config=SimpleNamespace(sequence_parallel=False),
        quant_config=SimpleNamespace(weight_block_size=weight_block_size),
    )
    dspark_module._dspark.get_current_vllm_config = lambda: vllm_config

    model = dspark_module.DSparkDeepseekV4ForCausalLM(
        vllm_config=vllm_config,
    )

    assert model.quant_config is vllm_config.quant_config
    assert model.pad_shared_expert is expected_pad
    assert model.model.use_sequence_parallel is False
    assert model.model.confidence_head is not None

    upstream = dspark_module._dspark
    for name, original in upstream._test_originals.items():
        assert getattr(upstream, name) is original


@pytest.mark.parametrize(
    ("weight_block_size", "sequence_parallel", "expected_padding"),
    (
        ((128, 128), False, True),
        ((128, 128), True, False),
        (None, False, False),
    ),
)
def test_dspark_adapter_initializes_upstream_shared_expert_loader_state(
    dspark_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    weight_block_size: tuple[int, int] | None,
    sequence_parallel: bool,
    expected_padding: bool,
) -> None:
    class DraftModel(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            super().__init__()

    monkeypatch.setattr(dspark_module, "DSparkDeepseekV4Model", DraftModel)
    config = SimpleNamespace(vocab_size=16, hidden_size=4)
    quant_config = SimpleNamespace(weight_block_size=weight_block_size)
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        parallel_config=SimpleNamespace(sequence_parallel=sequence_parallel),
        quant_config=quant_config,
    )

    model = dspark_module.DSparkDeepseekV4ForCausalLM(
        vllm_config=vllm_config,
    )

    assert model.quant_config is quant_config
    assert model.pad_shared_expert is expected_padding


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

    lightop = ModuleType("lightop")
    lightop.__path__ = []  # type: ignore[attr-defined]
    attention = ModuleType("lightop.attention")
    attention.fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32 = (
        normal
    )
    lightop.attention = attention  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.attention", attention)

    cache = torch.zeros((2, 4, 8), dtype=torch.uint8)
    cos_sin_cache = torch.arange(16, dtype=torch.float32)
    kv_weight = object()
    attn = SimpleNamespace(
        n_local_heads=2,
        head_dim=8,
        eps=1e-6,
        kv_norm=SimpleNamespace(weight=SimpleNamespace(data=kv_weight)),
        rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin_cache),
        swa_cache_layer=SimpleNamespace(kv_cache=cache, block_size=4),
    )
    kv = torch.ones((3, 8), dtype=torch.float32)
    positions = torch.tensor([0, 1, 2], dtype=torch.int32)
    slot_mapping = torch.tensor([4, 99, 5, 99, 6, 99], dtype=torch.int64)[::2]

    dspark_module._insert_context_kv(attn, kv, positions, slot_mapping)

    assert len(calls) == 1
    (
        dummy_q,
        passed_kv,
        passed_weight,
        cache_2d,
        passed_slots,
        passed_positions,
        rope,
        eps,
        block,
    ) = calls[0]
    assert dummy_q.shape == (3, 2, 8)
    assert dummy_q.dtype == kv.dtype
    assert passed_kv is kv
    assert passed_weight is kv_weight
    assert cache_2d.shape == (2, 32)
    assert passed_slots.dtype is torch.int32
    assert passed_slots.is_contiguous()
    assert torch.equal(passed_slots, torch.tensor([4, 5, 6], dtype=torch.int32))
    assert passed_positions.dtype == torch.int64
    assert rope is cos_sin_cache
    assert eps == 1e-6
    assert block == 4


@pytest.mark.parametrize(
    "cache_dtype",
    [
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float8_e4m3fn, id="per-tensor-fp8"),
    ],
)
def test_dspark_context_insert_normalizes_non_uint8_cache_before_upstream(
    dspark_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    cache_dtype: torch.dtype,
) -> None:
    fallback_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        dspark_module._dspark,
        "_insert_context_kv",
        lambda *args: fallback_calls.append(args),
    )
    kv = torch.arange(24, dtype=torch.bfloat16).view(3, 8)
    normalized_kv = kv + 2
    norm_calls: list[torch.Tensor] = []

    def kv_norm(value: torch.Tensor) -> torch.Tensor:
        norm_calls.append(value)
        return normalized_kv

    attn = SimpleNamespace(
        n_local_heads=2,
        head_dim=8,
        eps=1e-6,
        kv_norm=kv_norm,
        rotary_emb=SimpleNamespace(
            cos_sin_cache=torch.arange(16, dtype=torch.float32)
        ),
        swa_cache_layer=SimpleNamespace(
            kv_cache=torch.empty((2, 4, 8), dtype=cache_dtype),
            block_size=4,
        ),
    )
    positions = torch.tensor([0, 1, 2], dtype=torch.int32)
    slot_mapping = torch.tensor([4, 5, 6], dtype=torch.int64)

    dspark_module._insert_context_kv(attn, kv, positions, slot_mapping)

    assert len(norm_calls) == 1
    assert norm_calls[0] is kv
    assert len(fallback_calls) == 1
    passed_attn, passed_kv, passed_positions, passed_slots = fallback_calls[0]
    assert passed_attn is attn
    assert passed_kv is normalized_kv
    assert passed_positions is positions
    assert passed_slots is slot_mapping


def test_dspark_eplb_registers_all_draft_layers(dspark_module, monkeypatch):
    from vllm.model_executor.models.interfaces import get_mixture_of_experts_model
    events = []

    class MoE:
        n_logical_experts = n_routed_experts = 256
        n_physical_experts = 320
        n_local_physical_experts = 40
        n_shared_experts = 1
        n_redundant_experts = 64
        def __init__(self, index):
            self.experts = SimpleNamespace(
                get_expert_weights=lambda: [index],
                set_eplb_state=lambda **kw: events.append((index, kw)),
                update_expert_map=lambda: events.append(("update", index)),
            )

    amd = ModuleType("vllm.models.deepseek_v4.amd.model")
    amd.DeepseekV4MoE = MoE
    monkeypatch.setitem(sys.modules, amd.__name__, amd)
    cls = dspark_module.DSparkDeepseekV4ForCausalLM
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(n_group=8)
    model.model = SimpleNamespace(layers=[
        SimpleNamespace(ffn=MoE(i)) for i in range(3)
    ])
    model.set_moe_parameters()
    assert get_mixture_of_experts_model(model) is model
    assert model.num_moe_layers == 3
    assert model.num_physical_experts == 320
    load, mapping, replicas = object(), object(), object()
    model.set_eplb_state(load, mapping, replicas)
    assert model.expert_weights == [[0], [1], [2]]
    for i, (index, kw) in enumerate(events):
        assert index == kw["moe_layer_idx"] == i
        assert kw["expert_load_view"] is load
        assert kw["logical_to_physical_map"] is mapping
        assert kw["logical_replica_count"] is replicas
    model.update_physical_experts_metadata(320, 40)
    assert events[-3:] == [("update", i) for i in range(3)]


@pytest.mark.parametrize("fail", [False, True])
def test_dspark_eplb_loader_uses_private_redundant_mapping(
    dspark_module, monkeypatch, fail,
):
    seen = []
    def mapping(*args, **kwargs):
        seen.append(kwargs)
        return []
    namespace = {"fused_moe_make_expert_params_mapping": mapping}
    exec("def load_weights(self, weights):\n"
         "    result = fused_moe_make_expert_params_mapping(self, num_experts=256)\n"
         "    if self.fail: raise RuntimeError('load failed')\n"
         "    return set()\n", namespace)
    upstream = dspark_module._dspark.DSparkDeepseekV4ForCausalLM
    monkeypatch.setattr(upstream, "load_weights", namespace["load_weights"], raising=False)
    cls = dspark_module.DSparkDeepseekV4ForCausalLM
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model._hcu_enable_eplb = True
    model.num_redundant_experts = 64
    model.fail = fail
    if fail:
        with pytest.raises(RuntimeError, match="load failed"):
            model.load_weights([1])
    else:
        assert model.load_weights([]) == set()
    assert seen == [{"num_experts": 256, "num_redundant_experts": 64}]
    assert namespace["load_weights"].__globals__["fused_moe_make_expert_params_mapping"] is mapping
    assert not model._loading_dspark_weights


@pytest.mark.parametrize("redundant", [16, 258])
@pytest.mark.parametrize("rank", [0, 1])
def test_dspark_loader_populates_every_local_expert_replica(
    dspark_module, monkeypatch, redundant, rank,
):
    import ast
    import re
    import __future__
    import vllm
    from vllm.model_executor.layers.fused_moe.layer import fused_moe_make_expert_params_mapping

    # Execute the installed upstream loader, rather than reproducing its loop.
    source = Path(vllm.__file__).parent / "models/deepseek_v4/nvidia/dspark.py"
    tree = ast.parse(source.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == "DSparkDeepseekV4ForCausalLM")
    selected = [n for n in owner.body if isinstance(n, ast.FunctionDef)
                and n.name in ("load_weights", "_remap_dspark_name")]
    namespace = dict(torch=torch, re=re, _EXPERT_SCALE_RE=re.compile(r"\.experts\.\d+\."),
                     fused_moe_make_expert_params_mapping=fused_moe_make_expert_params_mapping,
                     get_tensor_model_parallel_world_size=lambda: 1,
                     get_tensor_model_parallel_rank=lambda: 0,
                     logger=SimpleNamespace(info_once=lambda *args: None))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    upstream = dspark_module._dspark.DSparkDeepseekV4ForCausalLM
    monkeypatch.setattr(upstream, "load_weights", namespace["load_weights"], raising=False)
    monkeypatch.setattr(upstream, "_remap_dspark_name", namespace["_remap_dspark_name"], raising=False)
    cls = dspark_module.DSparkDeepseekV4ForCausalLM
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model._hcu_enable_eplb = True
    model.num_redundant_experts = redundant
    model.config = SimpleNamespace(n_routed_experts=256, num_attention_heads=64,
                                   expert_dtype="fp8")
    model.model = SimpleNamespace(layers=[SimpleNamespace(ffn=SimpleNamespace(use_mega_moe=False))],
                                  confidence_head=None)
    model.pad_shared_expert = False
    model.process_weights_after_loading = lambda: None
    local = (256 + redundant) // 2
    first = rank * local
    loaded = {(shard, kind): torch.full((local,), -1.)
              for shard in ("w1", "w2", "w3") for kind in ("weight", "scale")}

    def weight_loader(param, weight, name, *, expert_id, shard_id, return_success):
        if first <= expert_id < first + local:
            kind = "scale" if "scale" in name else "weight"
            loaded[shard_id, kind][expert_id - first] = weight
            return True
        return False

    params = {}
    for name in ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv"):
        param = nn.Parameter(torch.zeros(1), requires_grad=False)
        param.weight_loader = weight_loader
        params[f"model.layers.0.ffn.experts.routed_experts.{name}"] = param
    model.named_parameters = lambda *args, **kwargs: iter(params.items())
    weights = [(f"mtp.0.ffn.experts.{expert}.{shard}.{kind}", torch.tensor(float(expert)))
               for expert in range(256) for shard, kind in loaded]
    result = model.load_weights(iter(weights))
    assert set(params) <= result
    expected = torch.arange(first, first + local).remainder(256).float()
    for shard, actual in loaded.items():
        torch.testing.assert_close(actual, expected, msg=f"rank={rank}, shard={shard}")
