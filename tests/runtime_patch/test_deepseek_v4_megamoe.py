# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]


def load_file(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(monkeypatch):
    class Base:
        def __init__(self, moe):
            self.moe = moe
            self.moe_kernel = self.moe_quant_config = None

    deps = {
        "vllm.config": {"get_current_vllm_config": lambda: None},
        "vllm.distributed": {"get_ep_group": lambda: NS(device_group=object())},
        "vllm.forward_context": {"get_forward_context": lambda: NS(dp_metadata=None)},
        "vllm.model_executor.layers.fused_moe.fused_moe_method_base": {"FusedMoEMethodBase": Base},
    }
    for name, symbols in deps.items():
        mod = ModuleType(name)
        mod.__dict__.update(symbols)
        monkeypatch.setitem(sys.modules, name, mod)
    return load_file(monkeypatch, "test_megamoe_runtime",
                     "vllm_hcu/model_executor/layers/fused_moe/megamoe_runtime.py")


@pytest.fixture
def int4(monkeypatch):
    import triton
    import triton.language as tl
    mod = ModuleType("vllm.triton_utils")
    mod.triton, mod.tl = triton, tl
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return load_file(monkeypatch, "test_megamoe_int4",
                     "vllm_hcu/model_executor/layers/fused_moe/hcu_int4_megamoe.py")


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8])
def test_int4_preparation_retains_weights_and_scales(int4, dtype):
    e, h, i = 2, 256, 256
    w1 = torch.arange(e * 2 * i * h // 2).to(dtype).reshape(e, 2*i, h//2)
    w2 = torch.arange(e * h * i // 2).to(dtype).reshape(e, h, i//2)
    s1, s2 = torch.rand(e, 2*i, 1), torch.rand(e, h, 1)
    l1, l2 = int4.transform_int4_weights_for_mega_moe_normal(
        w1, w2, l1_scale=s1, l2_scale=s2, prepack=False)
    for weight, scale, prepared in ((w1, s1, l1), (w2, s2, l2)):
        values, scales = prepared["raw_int4"]
        assert values is weight  # No duplicate resident weights or shuffle.
        torch.testing.assert_close(scales, scale.squeeze(-1)*16)
        assert scales.dtype == torch.float32
    _, true_scales = int4._pack(w1, s1, 1.0, prepack=False)["raw_int4"]
    torch.testing.assert_close(true_scales, s1.squeeze(-1))


@pytest.mark.parametrize("factor", [0, -1, float("inf"), float("nan")])
def test_int4_rejects_invalid_scale(int4, factor):
    w = torch.zeros((1, 256, 128), dtype=torch.int8)
    with pytest.raises(ValueError, match="scale_multiplier"):
        int4.transform_int4_weights_for_mega_moe_normal(w, w, l1_scale=None,
                                                     l2_scale=None, scale_multiplier=factor)


def valid_config():
    config = NS(parallel_config=NS(enable_expert_parallel=True, enable_eplb=False,
                                   enable_dbo=False, tensor_parallel_size=1),
                model_config=NS(enforce_eager=True, hf_config=NS(model_type="deepseek_v4")))
    moe = NS(ep_size=8, moe_backend="deep_gemm_mega_moe", pcp_size=1, is_sequence_parallel=False,
             num_experts=256, num_local_experts=32, hidden_dim=4096,
             intermediate_size_per_partition=2048, num_fused_shared_experts=0,
             has_bias=False, is_lora_enabled=False, activation=NS(value="silu"),
             in_dtype=torch.bfloat16, skip_final_all_reduce=False,
             experts_per_token=6, max_num_tokens=16, max_capture_size=8, dp_size=1, dp_rank=0)
    layer = NS(scoring_func="sqrtsoftplus", expert_placement_strategy="linear", swiglu_limit=10.,
               expert_map_manager=NS(num_fused_shared_experts=0))
    return config, moe, layer


@pytest.mark.parametrize("field,value,match", [
    ("ep_size", 4, "EP8"),
    ("is_sequence_parallel", True, "sequence"),
    ("skip_final_all_reduce", True, "defer"),
    ("in_dtype", torch.float16, "bfloat16"),
    ("moe_backend", "triton", "deep_gemm_mega_moe"),
    ("moe_backend", "auto", "deep_gemm_mega_moe"),
])
def test_config_rejects_unsupported_modes(runtime, field, value, match):
    config, moe, layer = valid_config()
    runtime.validate_config(config, moe, layer)
    moe.moe_backend = "deep_gemm_mega_moe"
    runtime.validate_config(config, moe, layer)
    setattr(moe, field, value)
    with pytest.raises(ValueError, match=match):
        runtime.validate_config(config, moe, layer)


def test_config_rejects_graphs_and_round_robin(runtime):
    config, moe, layer = valid_config()
    config.model_config.enforce_eager = False
    with pytest.raises(ValueError, match="enforce-eager"):
        runtime.validate_config(config, moe, layer, "int8")
    config.model_config.enforce_eager = True
    layer.expert_placement_strategy = "round_robin"
    with pytest.raises(ValueError, match="linear expert"):
        runtime.validate_config(config, moe, layer)


def test_config_rejects_original_tp_before_replicated_routes_are_staged(runtime):
    config, moe, layer = valid_config()
    config.parallel_config.tensor_parallel_size = 8
    # EP has already rewritten the MoE-local TP topology; validation must use
    # the original ParallelConfig or 16 * top-6 routes would be staged 8 times.
    moe.tp_size = 1
    with pytest.raises(ValueError, match="tensor parallelism"):
        runtime.validate_config(config, moe, layer)


def test_fp8_idle_dp_rank_participates_and_clears_stale_routes(runtime, monkeypatch):
    _, moe, layer = valid_config()
    method = runtime.HcuMegaMoEMethod(moe, "fp8")
    buf = NS(topk_idx=torch.ones(16, 6, dtype=torch.int64),
             topk_weights=torch.ones(16, 6), x=torch.empty(16, 4096),
             x_sf=torch.empty(16))
    method._buffer = buf
    method._l1, method._l2 = object(), object()
    calls = []
    fake = ModuleType("megamoe")
    fake.fp8_w8a8_mega_moe = lambda *args, **kwargs: calls.append((args, kwargs))
    fake.cast_to_fp8_channelwise = lambda x: (x, torch.ones(x.shape[0]))
    monkeypatch.setitem(sys.modules, "megamoe", fake)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(runtime, "get_forward_context", lambda: NS(
        dp_metadata=NS(num_tokens_across_dp_cpu=torch.tensor([0, 7]))))
    x = torch.empty(0, 4096, dtype=torch.bfloat16)
    result = method.apply(layer, x, torch.empty(0, 6), torch.empty(0, 6, dtype=torch.int64))
    assert result.shape == x.shape
    assert len(calls) == 1
    assert calls[0][1]["capacity_num_tokens"] == 7
    assert calls[0][1]["activation_clamp"] == 10.
    assert (buf.topk_idx == -1).all() and (buf.topk_weights == 0).all()

    monkeypatch.setattr(runtime, "get_forward_context", lambda: NS(dp_metadata=None))
    method.apply(layer, x, torch.empty(0, 6), torch.empty(0, 6, dtype=torch.int64))
    assert len(calls) == 1  # All ranks empty: no collective required.


def test_fp8_stages_routes_and_scales(runtime, monkeypatch):
    _, moe, layer = valid_config()
    method = runtime.HcuMegaMoEMethod(moe, "fp8")
    buf = NS(topk_idx=torch.ones(16, 6, dtype=torch.int64),
             topk_weights=torch.ones(16, 6), x=torch.empty(16, 4096), x_sf=torch.empty(16))
    method._buffer, method._l1, method._l2 = buf, {}, {}
    fake = ModuleType("megamoe")
    fake.cast_to_fp8_channelwise = lambda x: (x.float(), torch.full((x.shape[0], 1), 2.))
    def run(y, *args, **kwargs):
        y.fill_(3)
    fake.fp8_w8a8_mega_moe = run
    monkeypatch.setitem(sys.modules, "megamoe", fake)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    x = torch.ones(2, 4096, dtype=torch.bfloat16)
    ids = torch.arange(12).reshape(2, 6)
    weights = torch.full((2, 6), .125)
    result = method.apply(layer, x, weights, ids)
    torch.testing.assert_close(buf.x[:2], x.float())
    torch.testing.assert_close(buf.topk_idx[:2], ids)
    torch.testing.assert_close(buf.topk_weights[:2], weights)
    assert (buf.x_sf[:2] == 2).all() and (result == 3).all()
    assert (buf.topk_idx[2:] == -1).all()
    with pytest.raises(ValueError, match="capacity"):
        method.apply(layer, torch.ones(17, 4096, dtype=torch.bfloat16),
                     torch.ones(17, 6), torch.ones(17, 6, dtype=torch.int64))


def test_chunk_limit_keeps_k1_strides_in_uint32(runtime):
    _, moe, _ = valid_config()
    moe.max_num_tokens = 32768
    limit = runtime.max_chunk_tokens(moe, "normal")
    rows = lambda t: runtime._k1_rows("normal", t, 8, 6, 32)
    assert rows(32768) * 4096 > 2**32 - 1  # The DSV4-Flash profile-run failure.
    assert rows(limit) * 4096 <= 2**32 - 1 < rows(limit + 1) * 4096
    assert 16384 < limit < 32768
    ll = runtime.max_chunk_tokens(moe, "ll")
    assert runtime._k1_rows("ll", ll, 8, 6, 32) * 4096 <= 2**32 - 1
    moe.max_num_tokens = 16
    assert runtime.max_chunk_tokens(moe, "normal") == 16


def test_fp8_large_batch_is_chunked_on_every_dp_rank(runtime, monkeypatch):
    _, moe, layer = valid_config()
    method = runtime.HcuMegaMoEMethod(moe, "fp8")
    method._max_chunk_tokens = 4
    buf = NS(topk_idx=torch.ones(16, 6, dtype=torch.int64),
             topk_weights=torch.ones(16, 6), x=torch.empty(16, 4096), x_sf=torch.empty(16))
    method._buffer, method._l1, method._l2 = buf, {}, {}
    calls = []
    fake = ModuleType("megamoe")
    fake.cast_to_fp8_channelwise = lambda x: (x.float(), torch.ones(x.shape[0], 1))
    def run(y, *args, **kwargs):
        m = y.shape[0]
        calls.append((m, kwargs["capacity_num_tokens"], buf.topk_idx[:, 0].clone()))
        y.copy_(buf.x[:m])
    fake.fp8_w8a8_mega_moe = run
    monkeypatch.setitem(sys.modules, "megamoe", fake)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(runtime, "get_forward_context", lambda: NS(
        dp_metadata=NS(num_tokens_across_dp_cpu=torch.tensor([6, 10]))))
    x = torch.arange(6, dtype=torch.bfloat16)[:, None].expand(6, 4096).contiguous()
    ids = torch.arange(6)[:, None].expand(6, 6).contiguous()
    result = method.apply(layer, x, torch.ones(6, 6), ids)
    torch.testing.assert_close(result, x)
    # Shared capacity 10 -> 4+4+2 calls; this rank has 4, 2, then 0 tokens.
    assert [(m, cap) for m, cap, _ in calls] == [(4, 4), (2, 4), (0, 2)]
    assert calls[1][2][:3].tolist() == [4, 5, -1]  # Stale routes cleared.
    assert (calls[2][2] == -1).all()


@pytest.mark.parametrize("inference", [False, True])
def test_post_load_reuses_buffer_and_invalidates_weight_cache(runtime, monkeypatch, inference):
    _, moe, _ = valid_config()
    import torch.nn as nn
    # Tiny CPU shapes exercise lifecycle without allocating a real model.
    def layer():
        return NS(w13_weight=nn.Parameter(torch.zeros(2, 8, 4, dtype=torch.float8_e4m3fn), requires_grad=False),
                  w2_weight=nn.Parameter(torch.zeros(2, 4, 4, dtype=torch.float8_e4m3fn), requires_grad=False),
                  w13_weight_scale=nn.Parameter(torch.ones(2, 8, 1), requires_grad=False),
                  w2_weight_scale=nn.Parameter(torch.ones(2, 4, 1), requires_grad=False))
    fake = ModuleType("megamoe")
    calls = []
    fake.flatten_pack5_weight_asm_normal = lambda w: w.clone()
    fake.flatten_pack5_weight = lambda w: w.clone()
    fake.cast_to_fp8_channelwise = lambda x: (x, torch.ones(x.shape[0]))
    fake.fp8_w8a8_mega_moe = lambda *args, **kwargs: None
    def factory(*args, **kwargs):
        calls.append(kwargs)
        return object()
    fake.get_symm_buffer_for_mega_moe = factory
    monkeypatch.setitem(sys.modules, "megamoe", fake)
    group = object()
    ep = NS(device_group=group)
    monkeypatch.setattr(runtime, "get_ep_group", lambda: ep)
    monkeypatch.setattr(runtime.HcuMegaMoEMethod, "get_fused_moe_quant_config", lambda *args: object())
    a, b = runtime.HcuMegaMoEMethod(moe, "fp8"), runtime.HcuMegaMoEMethod(moe, "fp8")
    with torch.inference_mode(inference):
        weights = layer()
    a.process_weights_after_loading(weights)
    original = a._l1
    addresses = [t.data_ptr() for packed in (a._l1, a._l2)
                 for t in packed["unified"]]
    # Real loaders write via .data, leaving Parameter version counters unchanged.
    version = None if inference else weights.w13_weight._version
    weights.w13_weight.data.copy_(torch.ones_like(weights.w13_weight))
    weights.w2_weight.data.copy_(torch.ones_like(weights.w2_weight))
    weights.w13_weight_scale.data.copy_(torch.full_like(weights.w13_weight_scale, 3))
    if not inference:
        assert weights.w13_weight._version == version
    a.process_weights_after_loading(weights)
    for packed in (a._l1, a._l2):
        assert (packed["unified"][0].float() == 1).all()
    assert (a._l1["unified"][1] == 3).all()
    assert addresses == [t.data_ptr() for packed in (a._l1, a._l2)
                         for t in packed["unified"]]
    a.process_weights_after_loading(weights)
    assert a._l1 is original
    # Explicit post-load refresh also covers ordinary in-place scale updates.
    with torch.inference_mode(inference), torch.no_grad():
        weights.w13_weight_scale.fill_(2)
    a.process_weights_after_loading(weights)
    assert a._l1 is original  # Graph-safe reload preserves captured addresses.
    assert (a._l1["unified"][1] == 2).all()
    b.process_weights_after_loading(layer())
    assert a._buffer is b._buffer and len(calls) == 1


def test_opt_in_patch_preserves_default_and_is_idempotent(monkeypatch):
    from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_megamoe as patch
    calls = []
    class RoutedExperts:
        def _get_quant_method(self, prefix, quant_config, moe_config):
            calls.append((prefix, quant_config, moe_config))
            return "existing"
    module = ModuleType(patch.TARGET_MODULE)
    module.RoutedExperts = RoutedExperts
    monkeypatch.delenv("VLLM_HCU_USE_MEGAMOE", raising=False)
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)
    instance = RoutedExperts()
    assert instance._get_quant_method("model.layers.0.experts", None, None) == "existing"
    assert len(calls) == 1
    fake = ModuleType("vllm_hcu.model_executor.layers.fused_moe.megamoe_runtime")
    fake.select_method = lambda *args: "mega"
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setenv("VLLM_HCU_USE_MEGAMOE", "1")
    # A stale legacy variable must not override explicit backend selection.
    for backend in ("auto", "triton", "aiter", "deep_gemm"):
        assert instance._get_quant_method(
            "model.layers.0.experts", None, NS(moe_backend=backend)) == "existing"
    assert len(calls) == 5
    # Only the CLI backend opts in, even if the old variable says disabled.
    monkeypatch.setenv("VLLM_HCU_USE_MEGAMOE", "0")
    mega = NS(moe_backend="deep_gemm_mega_moe")
    assert instance._get_quant_method("model.layers.0.experts", None, mega) == "mega"
    fake.select_method = lambda *args: None
    with pytest.raises(ValueError, match="DeepSeek-V4"):
        instance._get_quant_method("model.layers.0.experts", None, mega)
    assert instance._get_quant_method(
        "model.layers.0.experts", None, NS(moe_backend="auto")) == "existing"
    RoutedExperts._get_quant_method = lambda *args: None
    with pytest.raises(RuntimeError, match="stale"):
        patch.apply_to_module(module)


def test_runner_avoids_duplicate_collectives_for_megamoe():
    # Load the real small runner methods without importing accelerator ops.
    import ast
    path = REPO / "vllm_hcu/model_executor/layers/fused_moe/moe_runner.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MoERunner")
    names = {"_fused_output_is_reduced", "do_naive_dispatch_combine",
             "_maybe_reduce_shared_expert_output", "_maybe_reduce_final_output"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    reduced = []
    def reduce(x):
        reduced.append(x)
        return x * 8
    namespace = {"torch": torch, "tensor_model_parallel_all_reduce": reduce}
    test_cls = ast.ClassDef(name="Runner", bases=[], keywords=[], body=methods, decorator_list=[])
    code = ast.fix_missing_locations(ast.Module(body=[test_cls], type_ignores=[]))
    exec(compile(code, str(path), "exec"), namespace)
    runner = namespace["Runner"]()
    runner._quant_method = NS(moe_kernel=None, hcu_owns_dispatch=True, hcu_output_is_reduced=True)
    runner.moe_config = NS(dp_size=2, tp_size=1, ep_size=8, is_sequence_parallel=False,
                           skip_final_all_reduce=False)
    assert not runner.do_naive_dispatch_combine
    shared = runner._maybe_reduce_shared_expert_output(torch.ones(2, 4))
    final = runner._maybe_reduce_final_output(torch.ones(2, 4)+shared, None)
    assert len(reduced) == 1 and (final == 9).all()
    runner._quant_method = NS(moe_kernel=None, supports_internal_mk=False)
    assert runner.do_naive_dispatch_combine
    assert not runner._fused_output_is_reduced
    shared = runner._maybe_reduce_shared_expert_output(torch.ones(2, 4))
    assert (shared == 1).all()
    runner._maybe_reduce_final_output(shared, None)
    assert len(reduced) == 2


def test_config_rejects_fused_shared_experts(runtime):
    config, moe, layer = valid_config()
    layer.expert_map_manager.num_fused_shared_experts = 1
    with pytest.raises(ValueError, match="fused shared"):
        runtime.validate_config(config, moe, layer)


@pytest.mark.parametrize("layout", ["raw_int4", "normal_int4"])
def test_int4_adapter_reuses_scratch_across_layers_and_separates_streams(int4, monkeypatch, layout):
    fake = ModuleType("megamoe")
    calls = []
    def run(y, l1, l2, buf, **kwargs):
        calls.append((l1, l2, kwargs))
        return y
    fake.int8_mega_moe = run
    monkeypatch.setitem(sys.modules, "megamoe", fake)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    stream = NS(cuda_stream=1)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    def unpack(p, out):
        raw = p.to(torch.int16)
        values = torch.stack((raw >> 4, raw & 15), -1)
        out.copy_(((values ^ 8)-8).to(torch.int8).reshape(out.shape))
    monkeypatch.setattr(int4, "unpack_raw_int4_pack5_out", unpack)
    monkeypatch.setattr(int4, "unpack_signed_int4_out", unpack)
    scale = torch.ones(1, 8)
    l1 = {layout: (torch.full((1, 1, 8), 0x8F, dtype=torch.uint8), scale)}
    l2 = {layout: (torch.full((1, 1, 4), 0x17, dtype=torch.uint8), scale)}
    buf = NS(quant_mode="int8")
    y = torch.empty(2, 4)
    int4.int4_w4a8_mega_moe(y, l1, l2, buf, capacity_num_tokens=2)
    scratch = calls[-1][0]["normal"][0]
    torch.testing.assert_close(scratch, torch.tensor([[-8, -1]*8], dtype=torch.int8))
    int4.int4_w4a8_mega_moe(y, l1, l2, buf)
    assert calls[-1][0]["normal"][0] is scratch
    stream.cuda_stream = 2
    int4.int4_w4a8_mega_moe(y, l1, l2, buf)
    assert calls[-1][0]["normal"][0] is not scratch
    with pytest.raises(NotImplementedError, match="eager"):
        int4.int4_w4a8_mega_moe(y, l1, l2, buf, graph=True)
    with pytest.raises(NotImplementedError, match="normal"):
        int4.int4_w4a8_mega_moe(y, l1, l2, buf, megamoe_backend="ll")


def test_selects_slimquant_before_fallback_construction(runtime, monkeypatch):
    config, moe, layer = valid_config()
    layer.layer_name = "model.layers.60.ffn.experts"
    monkeypatch.setattr(runtime, "get_current_vllm_config", lambda: config)
    utils = ModuleType("vllm.model_executor.layers.quantization.compressed_tensors.utils")
    utils.should_ignore_layer = lambda *args, **kwargs: False
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    quant = NS(get_name=lambda: "slimquant_w4a8", ignore=[], packed_modules_mapping={})
    method = runtime.select_method(layer, quant, moe)
    assert method.quant_mode == "int8" and method.topk_indices_dtype == torch.int64
    utils.should_ignore_layer = lambda *args, **kwargs: True
    with pytest.raises(ValueError, match="ignored"):
        runtime.select_method(layer, quant, moe)


def test_selects_channel_fp8_and_rejects_mismatched_projections(runtime, monkeypatch):
    from enum import Enum
    class Strategy(Enum):
        CHANNEL = "channel"
        TOKEN = "token"
    quant_module = ModuleType("compressed_tensors.quantization")
    quant_module.QuantizationStrategy = Strategy
    monkeypatch.setitem(sys.modules, quant_module.__name__, quant_module)
    config, moe, layer = valid_config()
    layer.layer_name = "model.layers.0.ffn.experts"
    monkeypatch.setattr(runtime, "get_current_vllm_config", lambda: config)
    weight = NS(strategy=Strategy.CHANNEL, symmetric=True, dynamic=False)
    activation = NS(strategy=Strategy.TOKEN, dynamic=True, symmetric=True)
    scheme = {"weights": weight, "input_activations": activation}
    names = []
    def get_scheme(layer, name):
        names.append(name)
        return scheme
    utils = ModuleType("vllm.model_executor.layers.quantization.compressed_tensors.utils")
    utils.should_ignore_layer = lambda *args, **kwargs: False
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    quant = NS(get_name=lambda: "compressed-tensors", ignore=[], packed_modules_mapping={},
               _add_fused_moe_to_target_scheme_map=lambda: None,
               get_scheme_dict=get_scheme, _is_fp8_w8a8=lambda *args: True)
    method = runtime.select_method(layer, quant, moe)
    assert method.quant_mode == "fp8" and method.weight_quant is weight
    assert set(names) == {layer.layer_name+f".0.{p}_proj" for p in ("gate", "up", "down")}
    activation.dynamic = False
    with pytest.raises(ValueError, match="Channel-FP8"):
        runtime.select_method(layer, quant, moe)
    quant.get_scheme_dict = lambda layer, name: scheme if "gate" in name else None
    with pytest.raises(ValueError, match="matching"):
        runtime.select_method(layer, quant, moe)


def test_fp8_graph_has_fixed_workspace_and_updates_token_count(runtime, monkeypatch):
    _, moe, layer = valid_config()
    method = runtime.HcuMegaMoEMethod(moe, "fp8", graph_tokens=8)
    buf = NS(topk_idx=torch.ones(16, 6, dtype=torch.int64),
             topk_weights=torch.ones(16, 6), x=torch.empty(16, 4096),
             x_sf=torch.empty(16), cuda_graph_num_tokens=torch.tensor([99]))
    method._buffer, method._l1, method._l2 = buf, {}, {}
    fake = ModuleType("megamoe")
    fake.cast_to_fp8_channelwise = lambda x: (x.float(), torch.ones(x.shape[0]))
    calls = []
    def run(y, *args, **kwargs):
        calls.append((y.shape, kwargs, int(buf.cuda_graph_num_tokens[0])))
        y.fill_(4)
    fake.fp8_w8a8_mega_moe = run
    monkeypatch.setitem(sys.modules, "megamoe", fake)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(runtime, "get_forward_context", lambda: NS(
        dp_metadata=object(), batch_descriptor=NS(num_tokens=2)))
    x = torch.ones(2, 4096, dtype=torch.bfloat16)
    out = method.apply(layer, x, torch.ones(2, 6), torch.zeros(2, 6, dtype=torch.int64))
    assert out.shape == x.shape and (out == 4).all()
    shape, kwargs, tokens = calls[-1]
    assert shape == (8, 4096) and tokens == 2
    assert kwargs["graph"] and kwargs["capacity_num_tokens"] == 8
    assert kwargs["megamoe_backend"] == "ll"
    assert (buf.topk_idx[2:] == -1).all()
    buf.cuda_graph_num_tokens.fill_(99)
    method.apply(layer, x, torch.ones(2, 6), torch.zeros(2, 6, dtype=torch.int64))
    assert calls[-1][2] == 2


def test_backend_selection_and_pcp_guard(runtime, monkeypatch):
    config, moe, layer = valid_config()
    monkeypatch.setenv("MEGAMOE_HCU_NORMAL_LL_TOKEN_THRESHOLD", "4")
    fp8 = runtime.HcuMegaMoEMethod(moe, "fp8")
    assert fp8._backend(4) == "ll" and fp8._backend(5) == "normal"
    assert runtime.HcuMegaMoEMethod(moe, "int8")._backend(1) == "normal"
    assert runtime.HcuMegaMoEMethod(moe, "fp8", force_normal=True)._backend(1) == "normal"
    moe.pcp_size = 2
    runtime.validate_config(config, moe, layer)
    moe.dp_size = 2
    with pytest.raises(ValueError, match="PCP requires DP1"):
        runtime.validate_config(config, moe, layer)
    moe.dp_size = 1
    config.model_config.enforce_eager = False
    with pytest.raises(ValueError, match="PCP requires DP1"):
        runtime.validate_config(config, moe, layer)


def test_dp_requires_exact_coordinated_counts(runtime, monkeypatch):
    _, moe, layer = valid_config()
    moe.dp_size = 2
    method = runtime.HcuMegaMoEMethod(moe, "fp8")
    method._buffer = object()
    monkeypatch.setitem(sys.modules, "megamoe", ModuleType("megamoe"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    x = torch.ones(3, 4096, dtype=torch.bfloat16)
    weights, ids = torch.ones(3, 6), torch.zeros(3, 6, dtype=torch.int64)
    with pytest.raises(ValueError, match="coordinated token metadata"):
        method.apply(layer, x, weights, ids)
    monkeypatch.setattr(runtime, "get_forward_context", lambda: NS(
        dp_metadata=NS(num_tokens_across_dp_cpu=torch.tensor([2, 4]))))
    with pytest.raises(ValueError, match="disagree"):
        method.apply(layer, x, weights, ids)


def test_raw_int4_transform_retains_only_canonical_weights(int4):
    w1 = torch.zeros((2, 512, 128), dtype=torch.int8)
    w2 = torch.zeros((2, 256, 128), dtype=torch.int8)
    l1, l2 = int4.transform_int4_weights_for_mega_moe_normal(
        w1, w2, l1_scale=torch.ones(2, 512, 1),
        l2_scale=torch.ones(2, 256, 1), prepack=False,
    )
    assert l1["raw_int4"][0] is w1 and l2["raw_int4"][0] is w2
    assert (l1["raw_int4"][1] == 16).all()


def test_release_buffers_destroys_once_and_drops_scratch(runtime, monkeypatch):
    events = []
    class Buffer:
        def __init__(self):
            self._int4_unpack_workspaces = {"scratch": torch.ones(4)}
        def destroy(self):
            events.append("destroy")
    buf = Buffer()
    ep = NS(_hcu_megamoe_buffers={"int8": buf})
    monkeypatch.setattr(runtime, "get_ep_group", lambda: ep)
    runtime.release_buffers()
    runtime.release_buffers()
    assert events == ["destroy"]
    assert not ep._hcu_megamoe_buffers and not buf._int4_unpack_workspaces


def test_other_models_bypass_feature_even_when_global_switch_is_on(runtime, monkeypatch):
    config, moe, layer = valid_config()
    config.model_config.hf_config.model_type = "qwen3_moe"
    monkeypatch.setattr(runtime, "get_current_vllm_config", lambda: config)
    assert runtime.select_method(layer, None, moe) is None


def test_old_int8_factory_is_rejected_before_allocation(int4, monkeypatch):
    fake = ModuleType("megamoe")
    fake.int8_mega_moe = fake.mega_moe_pre_dispatch_int8 = lambda *a, **kw: None
    class SymmBuffer:
        def __init__(self, *, quant_mode):
            pass
    fake.SymmBuffer = SymmBuffer
    fake.get_symm_buffer_for_mega_moe = lambda group: None
    monkeypatch.setitem(sys.modules, "megamoe", fake)
    with pytest.raises(RuntimeError, match="get_symm_buffer_for_mega_moe"):
        int4.validate_megamoe_int8_runtime()


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8])
def test_raw_int4_pack5_triton_on_hcu(int4, dtype):
    try:
        torch.cuda.init()
    except RuntimeError as exc:
        pytest.skip(f"requires HCU to execute the unpack kernel: {exc}")
    e, n, k = 2, 256, 256
    raw = torch.arange(e*n*k//2).to(dtype).reshape(e, n, k//2)
    cpu = raw.to(torch.int16)
    values = ((torch.stack(((cpu >> 4) & 15, cpu & 15), -1) ^ 8)-8).to(torch.int8)
    reference = values.reshape(e, n//256, 16, 16, k//64, 4, 16)
    reference = reference.permute(0, 4, 1, 2, 5, 3, 6).contiguous().reshape(e, n*k)
    out = torch.empty((e, n*k), dtype=torch.int8, device="cuda")
    int4.unpack_raw_int4_pack5_out(raw.to("cuda"), out)
    torch.testing.assert_close(out.cpu(), reference)


@pytest.mark.parametrize("ll_capacity,backends", [(16, ["ll", "normal"]), (4, ["normal"])])
def test_graph_warmup_is_per_buffer_and_before_capture(runtime, monkeypatch, ll_capacity, backends):
    _, moe, layer = valid_config()
    method = runtime.HcuMegaMoEMethod(moe, "fp8", graph_tokens=8)
    method._l1, method._l2 = {}, {}
    buf = NS(x=torch.ones(16, 4096), x_sf=torch.ones(16),
             topk_idx=torch.ones(16, 6, dtype=torch.int64),
             topk_weights=torch.ones(16, 6), cuda_graph_num_tokens=torch.tensor([99]))
    buf.ll_scratch_capacity_tokens_per_rank = ll_capacity
    method._buffer = buf
    fake = ModuleType("megamoe")
    calls = []
    fake.fp8_w8a8_mega_moe = lambda *args, **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "megamoe", fake)
    method._warmup_graph(layer)
    method._warmup_graph(layer)
    assert [c["megamoe_backend"] for c in calls] == backends
    assert all(c["graph"] and c["capacity_num_tokens"] == 8 for c in calls)
    assert buf.cuda_graph_num_tokens.item() == 0
    assert (buf.topk_idx == -1).all()


def test_pcp_runner_keeps_rank_local_tokens_for_megamoe():
    import ast
    path = REPO / "vllm_hcu/model_executor/layers/fused_moe/moe_runner.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MoERunner")
    names = {"_maybe_dispatch", "_maybe_combine", "do_naive_dispatch_combine",
             "_pcp_routing_inputs"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    calls = []
    group = NS(all_gather=lambda x, **kw: calls.append("gather") or x,
               reduce_scatter=lambda x, **kw: calls.append("scatter") or x)
    from contextlib import contextmanager
    padding = torch.tensor([False, True])
    context = NS(is_padding=padding)
    namespace = {"torch": torch, "get_pcp_group": lambda: group,
                 "contextmanager": contextmanager,
                 "get_forward_context": lambda: context,
                 "is_forward_context_available": lambda: True}
    test_cls = ast.ClassDef(name="Runner", bases=[], keywords=[], body=methods, decorator_list=[])
    code = ast.fix_missing_locations(ast.Module(body=[test_cls], type_ignores=[]))
    exec(compile(code, str(path), "exec"), namespace)
    runner = namespace["Runner"]()
    runner.shared_experts = None
    runner._ep_fused_combine = False
    runner.moe_config = NS(pcp_size=2, dp_size=1, is_sequence_parallel=False,
                           moe_parallel_config=NS(use_all2all_kernels=False))
    runner._quant_method = NS(hcu_owns_dispatch=True)
    x, logits = torch.ones(2, 4), torch.ones(2, 256)
    assert runner._maybe_dispatch(x, logits) == (x, logits)
    assert runner._maybe_combine(None, x) is x
    ids = torch.tensor([17, 23])
    for _ in range(2):
        with runner._pcp_routing_inputs(ids, 2) as routing_ids:
            assert routing_ids is ids
            assert context.is_padding is padding
        assert context.is_padding is padding
    assert calls == []
    runner._quant_method = NS(supports_internal_mk=False)
    runner._maybe_dispatch(x, logits)
    runner._maybe_combine(None, x)
    with runner._pcp_routing_inputs(ids, 1) as routing_ids:
        torch.testing.assert_close(routing_ids, ids[:1])
        torch.testing.assert_close(context.is_padding, padding[:1])
    assert context.is_padding is padding
    assert calls == ["gather", "gather", "scatter", "gather", "gather"]


def test_offload_sleep_and_nonstandard_activation_rejected(runtime):
    config, moe, layer = valid_config()
    config.offload_config = NS(uva=NS(cpu_offload_gb=1), prefetch=NS(offload_group_size=0))
    with pytest.raises(ValueError, match="offloading"):
        runtime.validate_config(config, moe, layer)
    config.offload_config.uva.cpu_offload_gb = 0
    config.model_config.enable_sleep_mode = True
    with pytest.raises(ValueError, match="sleep mode"):
        runtime.validate_config(config, moe, layer)
    config.model_config.enable_sleep_mode = False
    layer.swiglu_alpha = 2.
    with pytest.raises(ValueError, match="parameterized"):
        runtime.validate_config(config, moe, layer)


@pytest.mark.parametrize("e,n,k", [(1, 256, 64), (2, 512, 256)])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8])
def test_raw_pack5_kernel_interpreter_matches_independent_layout(int4, e, n, k, dtype):
    # Execute the production Triton indexing on host pointers. Unlike a
    # mirrored Python implementation, this also exercises tl.load/store,
    # signed-byte shifts and every PACK5 block/expert boundary without a GPU.
    from triton.runtime.interpreter import InterpretedFunction
    raw = torch.arange(e*n*k//2).to(torch.uint8).view(dtype).reshape(e, n, k//2)
    out = torch.empty((e, n*k), dtype=torch.int8)
    kernel = InterpretedFunction(int4._unpack_raw_int4_pack5.fn)
    kernel[(int4.triton.cdiv(out.numel(), 2048),)](
        raw, out, out.numel(), n, k, 2048, num_warps=4)
    byte = raw.to(torch.int16)
    values = ((torch.stack(((byte >> 4) & 15, byte & 15), -1) ^ 8)-8).to(torch.int8)
    reference = values.reshape(e, n//256, 16, 16, k//64, 4, 16)
    reference = reference.permute(0, 4, 1, 2, 5, 3, 6).contiguous().reshape(e, n*k)
    torch.testing.assert_close(out, reference)


@pytest.mark.parametrize("backend", ["auto", "ll"])
def test_ll_selection_respects_native_scratch_and_stride(runtime, monkeypatch, backend):
    _, moe, _ = valid_config()
    monkeypatch.setenv("MEGAMOE_HCU_BACKEND", backend)
    method = runtime.HcuMegaMoEMethod(moe, "fp8")
    method._buffer = NS(ll_scratch_capacity_tokens_per_rank=4)
    assert method._ll_capacity() == 4
    if backend == "auto":
        assert method._backend(4) == "ll"
        assert method._backend(5) == "normal"
        # A small graph bucket still launches with the buffer's graph maximum.
        assert method._backend(2, launch_capacity=8) == "normal"
    monkeypatch.setattr(runtime, "max_chunk_tokens", lambda *args: 2)
    assert method._ll_capacity() == 2
    if backend == "auto":
        assert method._backend(3) == "normal"


@pytest.mark.parametrize("owns_dispatch,expected", [(True, False), (False, True)])
def test_ep_fused_combine_excludes_backend_owned_dispatch(owns_dispatch, expected):
    import ast
    path = REPO / "vllm_hcu/model_executor/layers/fused_moe/moe_runner.py"
    tree = ast.parse(path.read_text())
    assignment = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and target.attr == "_ep_fused_combine"
            for target in node.targets)
    )
    runner = NS(_quant_method=NS(hcu_owns_dispatch=owns_dispatch))
    namespace = dict(
        self=runner, _EP_FUSED_COMBINE=True, shared_experts=object(),
        moe_config=NS(dp_size=8, pcp_size=1, is_sequence_parallel=False,
                      skip_final_all_reduce=False),
        routed_scaling_factor=1., routed_input_transform=None,
        routed_output_transform=None,
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])),
                 str(path), "exec"), namespace)
    assert runner._ep_fused_combine is expected


@pytest.mark.parametrize("mode,match", [("normal", "stride"), ("ll", "LL scratch")])
def test_graph_warmup_rejects_unsafe_capacity_before_launch(runtime, monkeypatch, mode, match):
    _, moe, layer = valid_config()
    monkeypatch.setenv("MEGAMOE_HCU_BACKEND", mode)
    method = runtime.HcuMegaMoEMethod(moe, "fp8", graph_tokens=8)
    method._buffer = NS(ll_scratch_capacity_tokens_per_rank=4)
    if mode == "normal":
        method._max_chunk_tokens = 4
    monkeypatch.setitem(sys.modules, "megamoe", ModuleType("megamoe"))
    with pytest.raises(ValueError, match=match):
        method._warmup_graph(layer)


def test_int4_pack5_matches_expanded_int8_permutation_and_scales(int4):
    # Independent signed-nibble reference: use every possible byte, including
    # negative int8 bytes. Verify unpacking commutes with PACK5 permutation.
    e, h, i = 2, 256, 256
    w1 = torch.arange(e * 2 * i * h // 2).to(torch.uint8).reshape(e, 2*i, h//2)
    w2 = torch.arange(e * h * i // 2).to(torch.uint8).reshape(e, h, i//2)
    s1, s2 = torch.rand(e, 2*i, 1), torch.rand(e, h, 1)
    l1, l2 = int4.transform_int4_weights_for_mega_moe_normal(
        w1, w2, l1_scale=s1, l2_scale=s2)

    def expand(w):
        nibbles = torch.stack((w.to(torch.int16) >> 4, w.to(torch.int16) & 15), -1)
        return ((nibbles ^ 8) - 8).to(torch.int8).reshape(*w.shape[:-1], w.shape[-1]*2)

    for w, scale, packed in ((w1, s1, l1), (w2, s2, l2)):
        values, scales = packed["normal_int4"]
        raw = expand(w)
        _, n, k = raw.shape
        reference = raw.reshape(e, n//256, 16, 16, k//64, 4, 16)
        reference = reference.permute(0, 4, 1, 2, 5, 3, 6).contiguous().reshape(e, n*k)
        torch.testing.assert_close(expand(values), reference)
        torch.testing.assert_close(scales, scale.squeeze(-1)*16)
        assert values.numel() == w.numel()
        assert scales.dtype == torch.float32
    _, true_scales = int4._pack(w1, s1, 1.0)["normal_int4"]
    torch.testing.assert_close(true_scales, s1.squeeze(-1))


def test_signed_int4_triton_unpack_on_hcu(int4):
    try:
        torch.cuda.init()
    except RuntimeError as exc:
        pytest.skip(f"requires an HCU to execute the unpack kernel: {exc}")
    packed = torch.arange(256).to(torch.uint8).to("cuda")
    out = torch.empty(512, dtype=torch.int8, device="cuda")
    int4.unpack_signed_int4_out(packed, out)
    raw = torch.arange(256, dtype=torch.int16)
    reference = ((torch.stack((raw >> 4, raw & 15), -1) ^ 8)-8).to(torch.int8)
    torch.testing.assert_close(out.cpu(), reference.flatten())


@pytest.mark.parametrize("name", ["compressed-tensors", "slimquant_compressed_tensors_marlin"])
def test_fp8_ignored_experts_rejected_before_scheme_lookup(runtime, monkeypatch, name):
    config, moe, layer = valid_config()
    layer.layer_name = "model.layers.0.ffn.experts"
    monkeypatch.setattr(runtime, "get_current_vllm_config", lambda: config)
    ignore = ["model.layers.*.ffn.experts"]
    mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    utils = ModuleType("vllm.model_executor.layers.quantization.compressed_tensors.utils")
    def should_ignore(layer_name, **kwargs):
        assert layer_name == layer.layer_name
        assert kwargs == dict(ignore=ignore, fused_mapping=mapping, use_fnmatch=True)
        return True
    utils.should_ignore_layer = should_ignore
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    def unexpected(*args):
        pytest.fail("Ignored experts must be rejected before FP8 scheme lookup/construction")
    quant = NS(get_name=lambda: name, ignore=ignore, packed_modules_mapping=mapping,
               _add_fused_moe_to_target_scheme_map=unexpected, get_scheme_dict=unexpected)
    monkeypatch.setattr(runtime, "HcuMegaMoEMethod", unexpected)
    with pytest.raises(ValueError, match="ignored/unquantized"):
        runtime.select_method(layer, quant, moe)


def test_int4_native_layerwise_reload_releases_temporary_weights(runtime, int4, monkeypatch):
    import gc
    import logging
    import weakref
    import importlib

    # Execute the installed native reload helpers, stubbing only platform imports.
    reload_dir = next((Path(p) / "vllm/model_executor/model_loader/reload"
                       for p in sys.path
                       if (Path(p) / "vllm/model_executor/model_loader/reload/layerwise.py").is_file()), None)
    if reload_dir is None:
        pytest.skip("requires native vLLM reload helpers")
    package = ModuleType("test_native_megamoe_reload")
    package.__path__ = [str(reload_dir)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setattr(sys.modules["vllm.config"], "ModelConfig", object, raising=False)
    def weight_loader(param, loaded_weight):
        param.data.copy_(loaded_weight)
    deps = {
        "vllm.logger": {"init_logger": logging.getLogger},
        "vllm.model_executor.layers.attention": {"is_deferred_attention_layer": lambda layer: False},
        "vllm.model_executor.layers.quantization.base_config": {"QuantizeMethodBase": runtime.FusedMoEMethodBase},
        "vllm.model_executor.model_loader.weight_utils": {"default_weight_loader": weight_loader},
    }
    for name, symbols in deps.items():
        mod = ModuleType(name)
        mod.__dict__.update(symbols)
        monkeypatch.setitem(sys.modules, name, mod)
    native = importlib.import_module(package.__name__ + ".layerwise")
    monkeypatch.setattr(runtime, "__package__", package.__name__)
    monkeypatch.setitem(sys.modules, package.__name__ + ".hcu_int4_megamoe", int4)
    fake = ModuleType("megamoe")
    fake.get_symm_buffer_for_mega_moe = lambda *args, **kwargs: NS(quant_mode="int8")
    monkeypatch.setitem(sys.modules, "megamoe", fake)
    monkeypatch.setattr(int4, "validate_megamoe_int8_runtime", lambda: None)
    monkeypatch.setattr(runtime, "get_ep_group", lambda: NS(device_group=object()))
    monkeypatch.setattr(runtime.HcuMegaMoEMethod, "get_fused_moe_quant_config", lambda *args: None)
    _, moe, _ = valid_config()
    method = runtime.HcuMegaMoEMethod(moe, "int8")
    layer = torch.nn.Module()
    for name, shape in (("w13_weight", (2, 512, 128)), ("w2_weight", (2, 256, 128)),
                        ("w13_weight_scale", (2, 512, 1)), ("w2_weight_scale", (2, 256, 1))):
        param = torch.nn.Parameter(torch.ones(shape, dtype=torch.int8 if "scale" not in name else torch.float32),
                                   requires_grad=False)
        param.weight_loader = weight_loader
        layer.register_parameter(name, param)
    del param
    layer.quant_method = method
    native.record_metadata_for_reloading(layer)
    method.process_weights_after_loading(layer)
    canonical = dict(layer.named_parameters())
    temporary = []
    transform = int4.transform_int4_weights_for_mega_moe_normal
    def track(w1, w2, **kwargs):
        temporary.extend((weakref.ref(w1), weakref.ref(w2)))
        return transform(w1, w2, **kwargs)
    monkeypatch.setattr(int4, "transform_int4_weights_for_mega_moe_normal", track)
    for value in (2, 3):
        native.initialize_layerwise_reload(layer)
        for name in canonical:
            target = getattr(layer, name)
            target.weight_loader(target, torch.full_like(canonical[name], value))
        del target
        native.finalize_layerwise_reload(layer, None)
        gc.collect()
        assert all(ref() is None for ref in temporary)
        for name, param in canonical.items():
            assert getattr(layer, name) is param
            assert (param == value).all()
        assert method._l1 is None and method._l2 is None
        assert all((scale == value * 16).all() for scale in method._int4_scales)

    # Forward must resolve the restored canonical weights, not the temporaries.
    layer.swiglu_limit = None
    method._buffer.topk_idx = torch.zeros(16, 6, dtype=torch.int64)
    method._buffer.topk_weights = torch.zeros(16, 6)
    method._buffer.x = torch.empty(16, 4096)
    method._buffer.x_sf = torch.empty(16)
    fake.mega_moe_pre_dispatch_int8 = lambda *args: None
    seen = []
    def run(y, l1, l2, *args, **kwargs):
        assert l1["raw_int4"][0] is canonical["w13_weight"]
        assert l2["raw_int4"][0] is canonical["w2_weight"]
        seen.append(True)
    monkeypatch.setattr(int4, "int4_w4a8_mega_moe", run)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(runtime, "get_forward_context", lambda: NS(dp_metadata=None))
    method.apply(layer, torch.ones(1, 4096, dtype=torch.bfloat16),
                 torch.ones(1, 6), torch.zeros(1, 6, dtype=torch.int64))
    assert seen == [True]


@pytest.mark.parametrize("capturing", [True, False])
@pytest.mark.parametrize("threshold,expected", [(4, "normal"), (8, "ll")])
def test_dp_graph_uses_same_backend_for_uneven_local_buckets(runtime, monkeypatch, capturing, threshold, expected):
    monkeypatch.setenv("MEGAMOE_HCU_BACKEND", "auto")
    monkeypatch.setenv("MEGAMOE_HCU_NORMAL_LL_TOKEN_THRESHOLD", str(threshold))
    fake = ModuleType("megamoe")
    fake.cast_to_fp8_channelwise = lambda x: (x.float(), torch.ones(x.shape[0]))
    backends = []
    fake.fp8_w8a8_mega_moe = lambda *args, **kwargs: backends.append(kwargs["megamoe_backend"])
    monkeypatch.setitem(sys.modules, "megamoe", fake)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    for rank, tokens in enumerate((2, 6)):
        _, moe, layer = valid_config()
        moe.dp_size, moe.dp_rank = 2, rank
        method = runtime.HcuMegaMoEMethod(moe, "fp8", graph_tokens=8)
        method._buffer = NS(topk_idx=torch.ones(16, 6, dtype=torch.int64),
                            topk_weights=torch.ones(16, 6), x=torch.empty(16, 4096),
                            x_sf=torch.empty(16), cuda_graph_num_tokens=torch.tensor([0]),
                            ll_scratch_capacity_tokens_per_rank=16)
        method._l1, method._l2 = {}, {}
        context = NS(dp_metadata=NS(num_tokens_across_dp_cpu=torch.tensor([2, 6])),
                     batch_descriptor=NS(num_tokens=tokens),
                     cudagraph_runtime_mode=NS(name="FULL"))
        monkeypatch.setattr(runtime, "get_forward_context", lambda: context)
        method.apply(layer, torch.ones(tokens, 4096, dtype=torch.bfloat16),
                     torch.ones(tokens, 6), torch.zeros(tokens, 6, dtype=torch.int64))
    assert backends == [expected, expected]


@pytest.mark.parametrize("dp_size,backend,auto,rejected", [
    (2, "deepep_low_latency", False, True),
    (1, "deepep_low_latency", False, False),
    (2, "deepep_low_latency", True, False),
    (2, "deepep_high_throughput", False, False),
])
def test_config_rejects_dp_backend_without_token_metadata(runtime, dp_size, backend, auto, rejected):
    config, moe, layer = valid_config()
    moe.dp_size = dp_size
    config.parallel_config.all2all_backend = backend
    config.parallel_config._vllm_hcu_deepep_auto = auto
    if rejected:
        with pytest.raises(ValueError, match="deepep_low_latency"):
            runtime.validate_config(config, moe, layer)
    else:
        runtime.validate_config(config, moe, layer)


@pytest.mark.parametrize("missing", ["megamoe", "megamoe_native_dependency"])
def test_optional_megamoe_import_error(runtime, monkeypatch, missing):
    import builtins
    original_import = builtins.__import__
    imports = []
    def import_without_megamoe(name, *args, **kwargs):
        if name == "megamoe":
            imports.append(name)
            raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", import_without_megamoe)
    _, moe, _ = valid_config()
    method = runtime.HcuMegaMoEMethod(moe, "fp8")
    assert imports == []  # No import before explicit MegaMoE weight finalization.
    if missing == "megamoe":
        with pytest.raises(RuntimeError, match="optional standalone megamoe") as exc:
            method.process_weights_after_loading(None)
        assert "--moe-backend deep_gemm_mega_moe" in str(exc.value)
        assert "Build and install" in str(exc.value)
        assert "https://github.com/Sherlolo/DeepGEMM-megamoe-w8a8" in str(exc.value)
        assert isinstance(exc.value.__cause__, ModuleNotFoundError)
    else:
        # Preserve errors inside an installed package instead of claiming it is absent.
        with pytest.raises(ModuleNotFoundError) as exc:
            method.process_weights_after_loading(None)
        assert exc.value.name == missing
