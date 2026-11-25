"""CPU contracts for standard MegaMoE integration and TP token ownership."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def adapter(monkeypatch):
    class Base:
        def __init__(self, moe):
            self.moe = moe
    for name, symbols in {
        "vllm.config": {"get_current_vllm_config": lambda: None},
        "vllm.distributed": {"get_ep_group": lambda: None, "get_tp_group": lambda: None},
        "vllm.model_executor.layers.fused_moe.fused_moe_method_base": {"FusedMoEMethodBase": Base},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(symbols)
        monkeypatch.setitem(sys.modules, name, module)
    path = ROOT / "vllm_hcu/model_executor/layers/fused_moe/deepgemm_megamoe.py"
    spec = importlib.util.spec_from_file_location("_standard_mega_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("rows", [0, 1, 3, 4, 5, 16, 64, 256])
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("clamp", [0., None])
def test_tp_slices_preserve_every_row_and_empty_ranks_participate(adapter, monkeypatch, rows, decode, clamp):
    mega = NS()
    package = ModuleType("deepgemm")
    package.mega = mega
    monkeypatch.setitem(sys.modules, "deepgemm", package)
    x = torch.arange(rows * 8, dtype=torch.bfloat16).reshape(rows, 8)
    ids = torch.zeros(rows, 2, dtype=torch.int64)
    weights = torch.ones(rows, 2)
    results, calls = [], []
    chunks = (rows + 3) // 4
    for rank in range(4):
        method = adapter.DeepGemmMegaMoEMethod(NS(experts_per_token=2, activation="swiglustep"), None)
        method._capacity, method._tp_size, method._tp_rank = 64, 4, rank
        method._weights = (object(), object())
        method._buffer = NS(decode=decode, topk_idx=torch.ones(64, 2, dtype=torch.int64),
                            topk_weights=torch.ones(64, 2))
        state = {}
        def preprocess(local, local_ids, local_weights, buffer, valid):
            assert local_ids.shape == local_weights.shape == (valid, 2)
            state["x"] = local.clone()
            if not decode:
                assert (buffer.topk_idx[valid:] == -1).all()
                assert (buffer.topk_weights[valid:] == 0).all()
        def run(output, *args, **kwargs):
            calls.append((rank, len(output)))
            assert kwargs == {"activation_clamp": 7. if clamp is None else 0., "fast_math": True, "tokens_all_ranks": -1}
            output.copy_(state["x"])
        mega.mega_moe_preprocess, mega.fp8_mega_moe = preprocess, run
        def gather(tensors, sizes):
            assert sizes == [chunks] * 4
            results.append(tensors[0].clone())
            return [torch.zeros(chunks * 4, 8, dtype=torch.bfloat16)]
        method._gather = NS(gather=gather)
        assert method.apply(NS(swiglu_limit=clamp), x, weights, ids).shape == x.shape
    assert len(calls) == 4
    torch.testing.assert_close(torch.cat(results)[:rows], x)
    if chunks * 4 > rows:
        assert (torch.cat(results)[rows:] == 0).all()


def test_source_capacity_rejected_before_collective(adapter):
    method = adapter.DeepGemmMegaMoEMethod(NS(experts_per_token=2), None)
    method._buffer, method._capacity, method._tp_size = object(), 1, 4
    package = ModuleType("deepgemm")
    package.mega = NS()
    previous = sys.modules.get("deepgemm")
    sys.modules["deepgemm"] = package
    try:
        with pytest.raises(ValueError, match="capacity"):
            method.apply(NS(), torch.zeros(5, 8, dtype=torch.bfloat16),
                         torch.ones(5, 2), torch.zeros(5, 2, dtype=torch.int64))
    finally:
        if previous is None:
            sys.modules.pop("deepgemm", None)
        else:
            sys.modules["deepgemm"] = previous


@pytest.mark.parametrize("experts,hidden,intermediate", [(352, 4096, 1536), (256, 4096, 2048)])
def test_weight_preparation_uses_public_api_and_releases_canonical_storage(
    adapter, monkeypatch, experts, hidden, intermediate,
):
    # Tiny canonical tensors; model geometry is independently passed to factory.
    layer = NS(w13_weight=torch.nn.Parameter(torch.zeros(1, 4, 4, dtype=torch.float8_e4m3fn), requires_grad=False),
               w2_weight=torch.nn.Parameter(torch.zeros(1, 4, 2, dtype=torch.float8_e4m3fn), requires_grad=False),
               w13_weight_scale=torch.ones(1, 2 * intermediate, 1),
               w2_weight_scale=torch.ones(1, hidden, 1), swiglu_limit=7.)
    packed = ((torch.zeros(1, 1, 1, 2, 4, 1, 1, dtype=torch.float8_e4m3fn), torch.ones(1)),
              (torch.zeros(1, 1, 1, 2, 4, 1, 1, dtype=torch.float8_e4m3fn), torch.ones(1)))
    mega = NS(transform_weights_for_mega_moe=Mock(return_value=packed),
              get_symm_buffer_for_mega_moe=Mock(return_value=object()))
    package = ModuleType("deepgemm")
    package.mega = mega
    monkeypatch.setitem(sys.modules, "deepgemm", package)
    ep = NS(world_size=8, rank_in_group=0, cpu_group=object())
    tp = NS(world_size=1, rank_in_group=0)
    monkeypatch.setattr(adapter, "get_ep_group", lambda: ep)
    monkeypatch.setattr(adapter, "get_tp_group", lambda: tp)
    monkeypatch.setattr(adapter.DeepGemmMegaMoEMethod, "get_fused_moe_quant_config", lambda *args: object())
    moe = NS(num_experts=experts, hidden_dim=hidden, intermediate_size_per_partition=intermediate,
             max_num_tokens=64, experts_per_token=8)
    method = adapter.DeepGemmMegaMoEMethod(moe, None)
    method.process_weights_after_loading(layer)
    assert layer.w13_weight.data_ptr() == packed[0][0].data_ptr()
    assert layer.w2_weight.data_ptr() == packed[1][0].data_ptr()
    assert mega.transform_weights_for_mega_moe.call_args.kwargs == {}
    assert mega.get_symm_buffer_for_mega_moe.call_args.args == (ep.cpu_group, experts, 64, 8, hidden, intermediate)
    assert mega.get_symm_buffer_for_mega_moe.call_args.kwargs == {
        "use_fp8_dispatch": True, "activation": "swiglu", "peer_memory_mode": "ipc"}


def test_old_production_entry_is_removed():
    files = [ROOT / "vllm_hcu/model_executor/layers/fused_moe/moe_runner.py",
             ROOT / "vllm_hcu/forward_context_runtime.py"]
    for path in files:
        source = path.read_text(encoding="utf-8")
        assert "HCU_PAIR32_MEGAMOE_DECODE" not in source
        assert "step5_pair32_megamoe" not in source


@pytest.mark.parametrize("activation,layer_limit,config_limit,expected", [
    ("swiglustep", None, None, 7.),
    ("swiglu_step", None, None, 7.),
    (NS(value="swiglustep"), None, None, 7.),
    ("silu", None, None, float("inf")),
    ("silu", None, 5., 5.),
    ("swiglustep", 9., 5., 9.),
    ("silu", 0., None, 0.),
])
def test_intrinsic_and_explicit_activation_clamps(adapter, activation, layer_limit, config_limit, expected):
    # SWIGLUSTEP is a distinct activation with its own limit even when
    # FusedMoEFactory leaves the separate optional swiglu_limit unset.
    moe = NS(activation=activation, swiglu_limit=config_limit)
    assert adapter.activation_clamp(moe, NS(swiglu_limit=layer_limit)) == expected


def test_swiglustep_limit_reaches_quant_metadata(adapter, monkeypatch):
    config = ModuleType("vllm.model_executor.layers.fused_moe.config")
    config.FusedMoEQuantConfig = NS(make=Mock(return_value=object()))
    monkeypatch.setitem(sys.modules, config.__name__, config)
    method = adapter.DeepGemmMegaMoEMethod(NS(activation="swiglustep", swiglu_limit=None), None)
    method.get_fused_moe_quant_config(NS(swiglu_limit=None, w13_weight_scale=object(), w2_weight_scale=object()))
    assert config.FusedMoEQuantConfig.make.call_args.kwargs["gemm1_clamp_limit"] == 7.


@pytest.mark.parametrize("model_type", ["step5", "deepseek_v4", "future_model"])
def test_selection_uses_quantization_contract_not_model_name(adapter, monkeypatch, model_type):
    strategy = NS(CHANNEL="channel", TOKEN="token")
    for name, fields in {
        "compressed_tensors.quantization": {"QuantizationStrategy": strategy},
        "vllm.model_executor.layers.quantization.compressed_tensors.utils": {
            "should_ignore_layer": lambda *args, **kwargs: False},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, module)
    config = NS(model_config=NS(hf_config=NS(model_type=model_type)),
                parallel_config=NS(enable_expert_parallel=True, enable_eplb=False, enable_dbo=False))
    monkeypatch.setattr(adapter, "get_current_vllm_config", lambda: config)
    weight = NS(strategy="channel", dynamic=False, symmetric=True)
    scheme = {"weights": weight,
              "input_activations": NS(strategy="token", dynamic=True, symmetric=True)}
    quant = NS(get_name=lambda: "compressed-tensors", ignore=[], packed_modules_mapping={},
               _add_fused_moe_to_target_scheme_map=lambda: None,
               get_scheme_dict=lambda *args: scheme, _is_fp8_w8a8=lambda *args: True)
    layer = NS(layer_name="layers.0.experts", expert_map_manager=NS(num_fused_shared_experts=0))
    moe = NS(has_bias=False, is_lora_enabled=False, pcp_size=1, is_sequence_parallel=False,
             in_dtype=torch.bfloat16, skip_final_all_reduce=False, activation="swiglustep")
    assert isinstance(adapter.select_method(layer, quant, moe), adapter.DeepGemmMegaMoEMethod)
    config.parallel_config.enable_eplb = True
    with pytest.raises(ValueError, match="EPLB"):
        adapter.select_method(layer, quant, moe)
