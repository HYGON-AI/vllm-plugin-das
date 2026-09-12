# SPDX-License-Identifier: Apache-2.0
"""CPU numerical contracts at current SlimQuant's accelerator boundaries."""
from types import SimpleNamespace

import pytest
import torch

from tests.models.hy_v4.test_custom_w4a8 import module, manifest
from tests.models.hy_v4.test_weight_loading import checkpoint_model


def test_custom_format_swaps_nibbles_without_changing_signed_values():
    packed = torch.tensor([[0x98, 0xba, 0xdc, 0xfe, 0x10, 0x32, 0x54, 0x76]], dtype=torch.uint8)
    adapted = module("hyv4_w4a8_weights").to_aiter_packing(packed)
    torch.testing.assert_close(module("hyv4_w4a8_kernels").unpack_int4(adapted),
        torch.tensor([list(range(-8, 8))], dtype=torch.int8))
    assert packed.tolist() == [[152, 186, 220, 254, 16, 50, 84, 118]]


def copy_loader(param, value, *args, **kwargs):
    with torch.no_grad():
        param.copy_(value)
    return True


def test_linear_inherits_current_owner_and_loads_signed_int8(tmp_path, monkeypatch):
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import compressed_tensors_w8a8_int8 as scheme
    from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8LinearMethod
    from vllm.model_executor import parameter
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    # Kernel allocation probes accelerator capability. Keep real parameter
    # construction, the current SlimQuant method and current INT8 runtime.
    monkeypatch.setattr(scheme, "init_int8_linear_kernel", lambda **kwargs: SimpleNamespace())
    config = module("hyv4_w4a8").HYV4W4A8Config(str(manifest(tmp_path)))
    layer = object.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    method = config.get_quant_method(layer, "model.layers.1.mlp.shared_experts.gate_up_proj")
    assert isinstance(method, SlimQuantW4A8Int8LinearMethod)
    assert method.apply.__func__ is SlimQuantW4A8Int8LinearMethod.apply
    assert isinstance(config.get_quant_method(layer, "model.layers.0.self_attn.q_a_proj"), UnquantizedLinearMethod)
    method.create_weights(layer, 4, [2], 4, 2, torch.bfloat16, weight_loader=copy_loader)
    assert layer.weight.shape == (2, 4)
    layer.weight.weight_loader(layer.weight, torch.tensor([[0x87, 0x10], [0xf2, 0x43]], dtype=torch.uint8))
    layer.weight_scale.weight_loader(layer.weight_scale, torch.tensor([[.5], [.25]]))
    torch.testing.assert_close(layer.weight, torch.tensor([[7, -8, 0, 1], [2, -1, 3, 4]], dtype=torch.int8))
    # Current runtime executes INT8 GEMM; replace only categorized LightOp
    # device entry points with CPU arithmetic, not the SlimQuant apply path.
    import sys
    from types import ModuleType
    from vllm_hcu.model_executor.layers.quantization.int8_runtime import apply_int8_linear
    quant, gemm = ModuleType("lightop.quant"), ModuleType("lightop.gemm_ops")
    quant.per_token_quant_int8 = lambda x: (x.to(torch.int8), torch.ones((*x.shape[:-1], 1)))
    gemm.hipblaslt_w8a8_gemm = lambda q, w, xs, ws, m, n, k, layout, dtype: (True, ((q.float() @ w.float().T) * xs * ws.T).to(dtype))
    monkeypatch.setitem(sys.modules, "lightop.quant", quant)
    monkeypatch.setitem(sys.modules, "lightop.gemm_ops", gemm)
    layer.params_dtype = torch.bfloat16
    layer.scheme.apply_weights = lambda layer, x, bias: apply_int8_linear(x, layer.weight, layer.weight_scale, layer.params_dtype, bias=bias)
    actual = method.apply(layer, torch.tensor([[[1., 2., 3., 4.]]]), torch.tensor([1., -1.], dtype=torch.bfloat16))
    torch.testing.assert_close(actual, torch.tensor([[[-1.5, 5.25]]], dtype=torch.bfloat16))


@pytest.mark.parametrize("backend", ["aiter", "triton"])
def test_moe_load_converts_nibbles_and_scale_before_current_dispatch(tmp_path, monkeypatch, backend):
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8AiterMoEMethod
    from vllm_hcu.model_executor.layers.quantization import compressed_tensors_moe_runtime as runtime
    config = module("hyv4_w4a8").HYV4W4A8Config(str(manifest(tmp_path)))
    layer = object.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = SimpleNamespace(moe_backend=backend, experts_per_token=2,
                                      in_dtype=torch.float32)
    layer.expert_map_manager = SimpleNamespace(expert_map=None, expert_mask=None)
    method = config.get_quant_method(layer, "model.layers.1.mlp.experts.routed_experts")
    assert isinstance(method, SlimQuantW4A8Int8AiterMoEMethod)
    assert method.apply.__func__ is SlimQuantW4A8Int8AiterMoEMethod.apply
    assert method.process_weights_after_loading.__func__ is SlimQuantW4A8Int8AiterMoEMethod.process_weights_after_loading
    assert not method.supports_eplb
    method.create_weights(layer, 2, 2, 2, torch.bfloat16, weight_loader=copy_loader)
    packed13 = torch.tensor([[[0x11], [0x11], [0x22], [0x22]], [[0xff], [0xff], [0x11], [0x11]]], dtype=torch.uint8)
    packed2 = torch.tensor([[[0x11], [0x11]], [[0x11], [0x11]]], dtype=torch.uint8)
    for name, value in (("w13_weight", packed13), ("w2_weight", packed2),
                        ("w13_weight_scale", torch.full((2, 4, 1), .5)),
                        ("w2_weight_scale", torch.full((2, 2, 1), .25))):
        param = getattr(layer, name)
        param.weight_loader(param, value)
    torch.testing.assert_close(layer.w13_weight_scale, torch.full((2, 4, 1), 1/32))
    qconfig = method.get_fused_moe_quant_config(layer)
    torch.testing.assert_close(qconfig.w1_scale, torch.full((2, 4, 1), .5))
    # Keep current postprocessing, fallback layout and routed runtime real.
    # Only tuned device lookup and the final Triton device call are replaced.
    import sys
    from types import ModuleType
    aiter_moe = ModuleType("aiter.moe")
    aiter_moe.MoeQuantType = SimpleNamespace(W4A8=object())
    monkeypatch.setitem(sys.modules, "aiter.moe", aiter_moe)
    monkeypatch.setattr(runtime, "select_aiter_moe_config", lambda *args, **kwargs: None)
    method.process_weights_after_loading(layer)
    before = layer.w13_weight.clone()
    method.process_weights_after_loading(layer)
    torch.testing.assert_close(layer.w13_weight, before)
    def cpu_experts(x, w13, w2, weights, ids, **kwargs):
        assert kwargs["use_int8_w8a8"] and kwargs["per_channel_quant"]
        w13 = w13.float() * kwargs["w1_scale"]
        w2 = w2.float() * kwargs["w2_scale"]
        out = torch.zeros_like(x)
        for token in range(len(x)):
            for slot in range(ids.shape[1]):
                expert = ids[token, slot]
                gate, up = (w13[expert] @ x[token]).chunk(2)
                out[token] += weights[token, slot] * (w2[expert] @ (torch.nn.functional.silu(gate) * up))
        return out
    import importlib
    kernel = importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")
    monkeypatch.setattr(kernel, "fused_experts_impl", cpu_experts)
    x = torch.tensor([[1., 1.], [2., 2.]])
    actual = method.apply(layer, x, torch.tensor([[.25, .75], [.75, .25]]),
                          torch.tensor([[0, 1], [1, 0]]), None, None)
    # Expert 0 = silu(sum(x)/2)*sum(x)/2; expert 1 has
    # negative gate and half as large up projection.
    expected = torch.tensor([[.0819116, .0819116], [.7019927, .7019927]])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("native", [False, True])
def test_formats_remain_static_eplb_unsupported(tmp_path, native):
    from tests.accuracy.test_hyv4_native_format import native_index
    from tests.models.static_eplb_test_utils import GenericMoE, config_and_map
    from vllm_hcu.model_executor.layers.fused_moe.static_eplb import bind_static_eplb_plan
    config = (module("hyv4_native").HYV4NativeW4A8Config(str(native_index(tmp_path))) if native
              else module("hyv4_w4a8").HYV4W4A8Config(str(manifest(tmp_path))))
    model = GenericMoE(0)
    for layer in model.moe_layers:
        del layer.routed_experts.quant_method
        layer.routed_experts.quant_method = module("hyv4_w4a8").HYV4W4A8MoEMethod(config, SimpleNamespace(moe_backend="aiter"))
    current, _ = config_and_map(tmp_path)
    with pytest.raises(ValueError, match="EPLB|eplb"):
        bind_static_eplb_plan(current, model)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("draft", [False, True])
def test_real_target_and_mtp_packed_linear_loading(tmp_path, monkeypatch, checkpoint_model, native, draft):
    import json
    from tests.models.hy_v4.test_mtp import _minimal_draft, _mtp, _weights
    from tests.accuracy.test_hyv4_native_format import native_index
    from vllm.model_executor import parameter
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import compressed_tensors_w8a8_int8 as scheme
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(scheme, "init_int8_linear_kernel", lambda **kwargs: SimpleNamespace())
    name = "model.mtp_layers.0.eh_proj" if draft else "model.projection"
    if native:
        path = native_index(tmp_path)
        data = json.loads(path.read_text())
        data["parameters"][name + ".weight"] = {"kind": "quantized"}
        path.write_text(json.dumps(data))
        config = module("hyv4_native").HYV4NativeW4A8Config(str(path))
    else:
        config = module("hyv4_w4a8").HYV4W4A8Config(str(manifest(tmp_path, [name + ".weight"])))
    if draft:
        model = _minimal_draft(_mtp(), monkeypatch)
        config = _mtp()._remap_mtp_quant_exclusions(config, 2, 1)
        layer = RowParallelLinear(4, 2, bias=False, params_dtype=torch.bfloat16,
            quant_config=config, prefix="model.layers.2.eh_proj", disable_tp=True)
        model.model.layers["2"].eh_proj = layer
        weights = [(key, value) for key, value in _weights() if ".eh_proj." not in key]
    else:
        model = checkpoint_model
        weights = [(key, torch.ones_like(value)) for key, value in model.named_parameters()]
        layer = RowParallelLinear(4, 2, bias=False, params_dtype=torch.bfloat16,
            quant_config=config, prefix="model.projection", disable_tp=True)
        model.model.projection = layer
    model.quant_config = config
    packed = torch.tensor([[0x87, 0x10], [0xf2, 0x43]], dtype=torch.uint8)
    if native:
        weights = [(key + ".weight", value) for key, value in weights]
        quantized = [(name + ".weight.packed", packed),
                     (name + ".weight.scale", torch.tensor([[.5], [.25]]))]
    else:
        quantized = [(name + ".weight.int4_packed", packed),
                     (name + ".weight.scale", torch.tensor([.5, .25])),
                     (name + ".weight.input_scale", torch.ones(4))]
    assert model.load_weights(iter(weights + quantized)) == set(dict(model.named_parameters()))
    torch.testing.assert_close(layer.weight, torch.tensor([[7, -8, 0, 1], [2, -1, 3, 4]], dtype=torch.int8))
    torch.testing.assert_close(layer.weight_scale, torch.tensor([[.5], [.25]]))
    with pytest.raises(RuntimeError, match="Duplicate"):
        model.load_weights(iter(weights + quantized + [quantized[0]]))
    with pytest.raises(RuntimeError, match="Missing"):
        model.load_weights(iter(weights + quantized[1:]))


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("draft", [False, True])
def test_real_target_and_mtp_expert_packing_scales_and_ledger(tmp_path, monkeypatch, checkpoint_model, native, draft):
    from tests.models.hy_v4.test_mtp import _minimal_draft, _mtp, _weights
    from tests.models.static_eplb_test_utils import GenericMoE
    from tests.accuracy.test_hyv4_native_format import native_index
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    config = (module("hyv4_native").HYV4NativeW4A8Config(str(native_index(tmp_path))) if native
              else module("hyv4_w4a8").HYV4W4A8Config(str(manifest(tmp_path))))
    experts = GenericMoE(0).moe_layers[0]
    owner = experts.routed_experts
    del owner.quant_method
    method = module("hyv4_w4a8").HYV4W4A8MoEMethod(config, owner.moe_config)
    owner.quant_method = method
    method.create_weights(owner, 2, 2, 2, torch.bfloat16, weight_loader=owner.weight_loader)
    if draft:
        model = _minimal_draft(_mtp(), monkeypatch)
        inner = model
        mlp = model.model.layers["2"].mtp_block.mlp
        del mlp.gate_up_proj
        weights = [(name, value) for name, value in _weights() if ".mlp." not in name]
        prefix = "model.mtp_layers.0.mlp.experts"
    else:
        model = checkpoint_model
        inner = model.model
        inner.mlp = torch.nn.Module()
        mlp = inner.mlp
        weights = [(name, torch.ones_like(value)) for name, value in model.named_parameters()]
        prefix = "model.mlp.experts"
    mlp.experts = experts
    model.quant_config = config
    inner.config.num_experts = 3
    inner.num_redundant_experts = 1
    inner.get_expert_mapping = lambda: RoutedExperts.build_expert_params_mapping("gate_proj", "down_proj", "up_proj", 3)
    if native:
        weights = [(name + ".weight", value) for name, value in weights]
        pieces = [(f"{prefix}.{expert}.{projection}.weight.{suffix}", value)
                  for expert in range(3) for projection in ("gate_proj", "up_proj", "down_proj")
                  for suffix, value in (("packed", torch.full((2, 1), 0x87, dtype=torch.uint8)),
                                        ("scale", torch.full((2, 1), .5)))]
    else:
        pieces = [(f"{prefix}.{projection}.{suffix}", value)
                  for projection, rows in (("gate_up_proj", 4), ("down_proj", 2))
                  for suffix, value in (("int4_packed", torch.full((3, rows, 1), 0x87, dtype=torch.uint8)),
                                        ("scale", torch.full((3, rows), .5)))]
    assert model.load_weights(iter(weights + pieces)) == set(dict(model.named_parameters()))
    # Reload source bytes through the same wrappers: normalization must not
    # accumulate, retain a stale ledger, or lose the original bound owner.
    assert model.load_weights(iter(weights + pieces)) == set(dict(model.named_parameters()))
    assert owner.w13_weight.weight_loader.__self__ is owner
    assert owner.w13_weight_scale.weight_loader.__self__ is owner
    torch.testing.assert_close(owner.w13_weight, torch.full((2, 4, 1), 0x78, dtype=torch.int8))
    torch.testing.assert_close(owner.w13_weight_scale, torch.full((2, 4, 1), 1/32))
    torch.testing.assert_close(owner.w2_weight_scale, torch.full((2, 2, 1), 1/32))
    with pytest.raises(RuntimeError, match="Missing"):
        model.load_weights(iter(weights + pieces[1:]))
    with pytest.raises(RuntimeError, match="Duplicate"):
        model.load_weights(iter(weights + pieces + [pieces[0]]))
    assert not hasattr(inner, "_checkpoint_accounting")


def _scale_config(tmp_path, native):
    import json
    if not native:
        return module("hyv4_w4a8").HYV4W4A8Config(str(manifest(tmp_path, ["model.projection.weight"])))
    path = tmp_path / "hy4-checkpoint.index.json"
    path.write_text(json.dumps({"format": "hy4_w4a8_v1", "complete": True,
        "parameters": {"model.projection.weight": {"kind": "quantized"}}}))
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 2}))
    return module("hyv4_native").HYV4NativeW4A8Config(str(path))


def _scale_linear(tmp_path, monkeypatch, native, kind="row", rank=0):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import compressed_tensors_w8a8_int8 as scheme
    tp = 1 if kind == "row" else 2
    for owner in (parameter, linear):
        monkeypatch.setattr(owner, "get_tensor_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(owner, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(scheme, "init_int8_linear_kernel", lambda **kwargs: SimpleNamespace())
    config = _scale_config(tmp_path, native)
    kwargs = dict(bias=False, params_dtype=torch.bfloat16, quant_config=config, prefix="model.projection")
    if kind == "row":
        layer = linear.RowParallelLinear(4, 2, **kwargs)
    elif kind == "column":
        layer = linear.ColumnParallelLinear(4, 4, **kwargs)
    elif kind == "merged":
        layer = linear.MergedColumnParallelLinear(4, [4, 6], **kwargs)
    else:
        layer = linear.QKVParallelLinear(4, 2, 4, 1, **kwargs)
    layer.weight_scale.data.fill_(7)
    return config, layer


SCALE_ALIASES = [(False, "weight.scale"), (False, "weight_scale"),
                 (True, "weight.scale"), (True, "weight_scale"),
                 (True, "weight_scale.weight")]


@pytest.mark.parametrize("native,alias", SCALE_ALIASES)
@pytest.mark.parametrize("bad", ["nan", "inf", "zero", "negative", "fp16", "int32",
                                 "short", "long", "rank", "columns"])
def test_linear_scale_alias_rejects_before_real_owner_mutation(tmp_path, monkeypatch, native, alias, bad):
    config, layer = _scale_linear(tmp_path, monkeypatch, native)
    value = torch.full((2, 1), .5)
    if bad in ("nan", "inf", "zero", "negative"):
        value[0, 0] = {"nan": float("nan"), "inf": float("inf"), "zero": 0, "negative": -1}[bad]
    elif bad in ("fp16", "int32"):
        value = value.to(torch.float16 if bad == "fp16" else torch.int32)
    elif bad == "short":
        value = torch.ones(1, 1)
    elif bad == "long":
        value = torch.ones(3, 1)
    elif bad == "rank":
        value = torch.ones(1, 2, 1)
    else:
        value = torch.ones(2, 2)
    if not native and alias == "weight.scale" and value.ndim == 2 and value.shape[1] == 1:
        value = value.squeeze(-1)
    with pytest.raises(ValueError, match="HYV4.*scale"):
        for name, scale in config.adapt_weights([("model.projection." + alias, value)]):
            assert name == "model.projection.weight_scale"
            layer.weight_scale.weight_loader(layer.weight_scale, scale)
    torch.testing.assert_close(layer.weight_scale, torch.full((2, 1), 7.))


@pytest.mark.parametrize("native,alias", SCALE_ALIASES)
def test_linear_scale_alias_valid_real_owner_control(tmp_path, monkeypatch, native, alias):
    config, layer = _scale_linear(tmp_path, monkeypatch, native)
    value = torch.tensor([[.5], [.25]])
    if not native and alias == "weight.scale":
        value = value.squeeze(-1)
    for _, scale in config.adapt_weights([("model.projection." + alias, value)]):
        layer.weight_scale.weight_loader(layer.weight_scale, scale)
    torch.testing.assert_close(layer.weight_scale, torch.tensor([[.5], [.25]]))


@pytest.mark.parametrize("kind,shard,rows,want", [
    ("column", None, 4, [3, 4]),
    ("merged", 0, 4, [3, 4, 7, 7, 7]),
    ("merged", 1, 6, [7, 7, 4, 5, 6]),
    ("merged", (0, 1), 10, [3, 4, 8, 9, 10]),
    ("merged", None, 10, [3, 4, 8, 9, 10]),
    ("qkv", "q", 8, [5, 6, 7, 8, 7, 7, 7, 7]),
    ("qkv", "k", 2, [7, 7, 7, 7, 1, 2, 7, 7]),
    ("qkv", "v", 2, [7, 7, 7, 7, 7, 7, 1, 2]),
    ("qkv", None, 12, [5, 6, 7, 8, 9, 10, 11, 12]),
])
@pytest.mark.parametrize("delta", [0, 1, -1])
def test_linear_scale_extent_respects_current_tp_and_fused_projection(tmp_path, monkeypatch, kind, shard, rows, want, delta):
    _, layer = _scale_linear(tmp_path, monkeypatch, False, kind=kind, rank=1)
    value = torch.arange(1, rows + delta + 1, dtype=torch.float32)[:, None]
    args = () if kind == "column" else (shard,)
    if delta:
        before = layer.weight_scale.clone()
        with pytest.raises(ValueError, match="HYV4.*scale"):
            layer.weight_scale.weight_loader(layer.weight_scale, value, *args)
        torch.testing.assert_close(layer.weight_scale, before)
    else:
        layer.weight_scale.weight_loader(layer.weight_scale, value, *args)
        torch.testing.assert_close(layer.weight_scale[:, 0], torch.tensor(want, dtype=torch.float32))


def _scale_experts(tmp_path, native, tp=1, rank=0, padded=False):
    from tests.models.static_eplb_test_utils import GenericMoE
    owner = GenericMoE(0).moe_layers[0].routed_experts
    owner.moe_config.tp_rank = rank
    owner.moe_config.tp_size = tp
    owner.moe_config.moe_parallel_config.tp_size = tp
    owner.moe_config.hidden_dim_unpadded = 4
    owner.moe_config.intermediate_size_per_partition_unpadded = 2
    del owner.quant_method
    config = _scale_config(tmp_path, native)
    owner.quant_method = module("hyv4_w4a8").HYV4W4A8MoEMethod(config, owner.moe_config)
    owner.quant_method.create_weights(owner, 2, 6 if padded else 4,
        4 if padded else 2, torch.bfloat16, weight_loader=owner.weight_loader)
    return config, owner


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("shard", ["w1", "w3", "w2"])
@pytest.mark.parametrize("bad", ["short", "long", "columns", "rank"])
def test_expert_scale_extent_rejects_before_real_owner_mutation(tmp_path, native, tp, shard, bad):
    _, owner = _scale_experts(tmp_path, native, tp=tp)
    name = "w2_weight_scale" if shard == "w2" else "w13_weight_scale"
    param = getattr(owner, name)
    rows = 4 if shard == "w2" else 2 * tp
    value = torch.full((rows, 1), .5)
    if bad == "short":
        value = value[:-1]
    elif bad == "long":
        value = torch.full((rows + 1, 1), .5)
    elif bad == "columns":
        value = value.expand(rows, 2)
    else:
        value = value.unsqueeze(0)
    before = param.clone()
    with pytest.raises(ValueError, match="HYV4.*scale"):
        param.weight_loader(param, value, name, shard, 0, return_success=True)
    torch.testing.assert_close(param, before)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("tp,rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("shard", ["w1", "w3", "w2"])
def test_expert_scale_valid_tp_and_declared_padding(tmp_path, native, tp, rank, padded, shard):
    _, owner = _scale_experts(tmp_path, native, tp, rank, padded)
    name = "w2_weight_scale" if shard == "w2" else "w13_weight_scale"
    param = getattr(owner, name)
    rows = 4 if shard == "w2" else 2 * tp
    source = torch.arange(1, rows + 1, dtype=torch.float32)[:, None]
    param.weight_loader(param, source, name, shard, 0, return_success=True)
    expected = torch.ones_like(param)
    start = 0 if shard != "w3" else (4 if padded else 2)
    expected[0, start:start + (4 if shard == "w2" else 2)] = (
        source if shard == "w2" else source[rank * 2:rank * 2 + 2]) / 16
    torch.testing.assert_close(param, expected)
    # Explicit padding does not authorize arbitrary short or padded source
    # channel counts; the serialized projection has its declared logical size.
    if padded:
        before = param.clone()
        with pytest.raises(ValueError, match="HYV4.*scale"):
            param.weight_loader(param, torch.ones(rows + 1, 1), name, shard, 0)
        torch.testing.assert_close(param, before)


@pytest.mark.parametrize("parameter,shard", [("w13_weight_scale", "w2"),
                                            ("w2_weight_scale", "w1")])
def test_expert_scale_wrong_projection_fails_before_owner(tmp_path, parameter, shard):
    _, owner = _scale_experts(tmp_path, True)
    param = getattr(owner, parameter)
    before = param.clone()
    with pytest.raises(ValueError, match="HYV4.*scale"):
        param.weight_loader(param, torch.ones(4, 1), parameter, shard, 0)
    torch.testing.assert_close(param, before)


@pytest.mark.parametrize("field,shard", [("hidden_dim_unpadded", "w2"),
    ("intermediate_size_per_partition_unpadded", "w1")])
@pytest.mark.parametrize("invalid", [None, 0, 9])
def test_expert_scale_invalid_declared_extent_is_not_allocation_fallback(tmp_path, field, shard, invalid):
    _, owner = _scale_experts(tmp_path, True)
    setattr(owner.moe_config, field, invalid)
    name = "w2_weight_scale" if shard == "w2" else "w13_weight_scale"
    param = getattr(owner, name)
    before = param.clone()
    with pytest.raises(ValueError, match="HYV4.*scale"):
        param.weight_loader(param, torch.ones(4 if shard == "w2" else 2, 1), name, shard, 0)
    torch.testing.assert_close(param, before)
