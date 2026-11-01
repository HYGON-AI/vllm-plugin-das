"""Real target CT/PLE/Parameter/loader contracts; no device initialization."""

# Explicitly exercise the independent source package, not the installed wheel.
# ruff: noqa: E402

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from weakref import WeakKeyDictionary

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("QNT009_WHEEL_ROOT") or str(ROOT / "adapter_extension/src"))

from hcu_vllm_das.w4a8 import config as api
from hcu_vllm_das.w4a8 import lifecycle, runtime
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.oracle.int8 import Int8MoeBackend
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    monkeypatch.setattr(lifecycle, "_fork_error", None)
    monkeypatch.setattr(lifecycle, "_workers", WeakKeyDictionary())
    monkeypatch.setattr(lifecycle, "_alias_configs", [])
    monkeypatch.setattr(lifecycle, "_frozen_rows", None)

    def forbidden(*a, **kw):
        pytest.fail("CPU contract attempted a GPU query")

    for name in ("get_device_properties", "current_device", "init", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    yield
    assert not torch.cuda.is_initialized()


def metadata():
    return dict(
        quant_method="slimquant_w4a8",
        activation_scheme="dynamic",
        ignore=[r"re:.*\.ple\..*", r"re:.*\.o_proj$"],
        mixed_precision=dict(
            experts=dict(
                format="pack-int4-nibble",
                weight_bits=4,
                scope="per_channel",
                modules=[
                    "model.layers.*.mlp.experts.gate_up_proj",
                    "model.layers.*.mlp.experts.down_proj",
                ],
            ),
            w8a8_include=dict(
                format="int8-per-channel",
                weight_bits=8,
                activation_bits=8,
                activation_scheme="dynamic_per_token",
                scope="per_channel",
                modules=["model.layers.*.self_attn.q_proj"],
            ),
        ),
    )


def moe(tp_rank=0, tp_size=4):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )

    parallel = replace(
        FusedMoEParallelConfig.make_no_parallel(), tp_rank=tp_rank, tp_size=tp_size
    )
    return FusedMoEConfig(
        num_experts=2,
        num_local_experts=2,
        num_logical_experts=2,
        experts_per_token=2,
        hidden_dim=16,
        intermediate_size=8 * tp_size,
        moe_parallel_config=parallel,
        activation=MoEActivation.SILU,
        device="cpu",
        in_dtype=torch.bfloat16,
        routing_method=RoutingMethodType.Default,
    )


def layer_and_method(monkeypatch, rank=0, padded=False):
    import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 as official

    monkeypatch.setattr(
        official,
        "select_int8_moe_backend",
        lambda **kw: (Int8MoeBackend.TRITON, TritonExperts),
    )
    native = (
        runtime.NativePlan(runtime.Shape(2, 32, 64, 2, 2, torch.bfloat16), (), None)
        if padded
        else None
    )
    method = api.SlimQuantW4A8MoEMethod(moe(rank), native)
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config, layer.quant_method = method.moe, method
    layer.quant_config = api.SlimQuantW4A8Config.from_config(metadata())
    layer.layer_name = "model.layers.0.mlp.experts"
    layer.ckpt_gate_proj_name, layer.ckpt_up_proj_name = "gate_proj", "up_proj"
    layer.ckpt_down_proj_name = "down_proj"
    layer.lora_base_layer_prefix = ""
    layer.is_fused_checkpoint_transposed = False
    layer._fused_shared_expert_quantizer = None
    layer.expert_map_manager = NS(
        num_fused_shared_experts=0, map_global_to_local=lambda x: x
    )
    h, i = method.maybe_roundup_sizes(
        16, 8, torch.bfloat16, method.moe.moe_parallel_config
    )
    layer.moe_config.intermediate_size_per_partition = i
    layer.moe_config.hidden_dim = h
    method.create_weights(
        layer, 2, h, i, torch.bfloat16, weight_loader=layer.weight_loader
    )
    return layer, method


def test_all_256_bytes_signed_order_and_x16():
    packed = torch.arange(256, dtype=torch.uint8).view(torch.int8).reshape(1, 1, 256)
    weights, scales = runtime.int8_weights_and_scales(
        (packed, packed), (torch.tensor([0.25]), torch.tensor([0.125]))
    )
    reference = []
    for byte in range(256):
        for nibble in (byte // 16, byte % 16):
            reference.append(nibble - 16 if nibble >= 8 else nibble)
    assert weights[0].flatten().tolist() == reference
    assert scales[0].item() == 4 and scales[1].item() == 2
    assert packed.view(torch.uint8).flatten().tolist() == list(range(256))


def test_real_ct_schemes_ple_and_mapper(monkeypatch):
    q = api.SlimQuantW4A8Config.from_config(metadata())
    assert isinstance(q, api.CompressedTensorsConfig)
    assert q.quant_format == "pack-int4-nibble"
    assert (
        q.get_scheme_dict(None, "model.layers.0.mlp.experts.gate_up_proj")[
            "weights"
        ].num_bits
        == 4
    )
    assert (
        q.get_scheme_dict(None, "model.layers.0.self_attn.q_proj")["weights"].num_bits
        == 8
    )
    assert q.get_scheme_dict(None, "model.layers.0.self_attn.o_proj") is None
    assert isinstance(
        Qwen4ExpPLEEmbeddingMethod.from_quant_config(q, "model.layers.2.ple.embedding"),
        Qwen4ExpPLEUnquantizedEmbeddingMethod,
    )
    q.apply_vllm_mapper(WeightsMapper(orig_to_new_prefix={"model.": "language_model."}))
    assert (
        q.get_scheme_dict(None, "language_model.layers.0.self_attn.q_proj") is not None
    )
    assert q.get_scheme_dict(None, "model.layers.0.self_attn.q_proj") is None


@pytest.mark.parametrize("rank", range(4))
def test_real_fused_checkpoint_loader_tp_bytes_and_scales(monkeypatch, rank):
    layer, method = layer_and_method(monkeypatch, rank)
    gate_up = (
        torch.arange(2 * 64 * 8, dtype=torch.int64).to(torch.int8).reshape(2, 64, 8)
    )
    down = (
        torch.arange(2 * 16 * 16, dtype=torch.int64).to(torch.int8).reshape(2, 16, 16)
    )
    gate_scale = torch.arange(128, dtype=torch.float32).reshape(2, 64, 1) / 100 + 0.01
    down_scale = torch.arange(32, dtype=torch.float32).reshape(2, 16, 1) / 50 + 0.02
    named = [
        ("gate_up_proj", gate_up),
        ("down_proj", down),
        ("gate_up_proj.weight_scale", gate_scale),
        ("down_proj.weight_scale", down_scale),
    ]
    mapper = layer.quant_config.get_checkpoint_weight_mapper()
    mapped = [
        (
            mapper.map_name(layer.layer_name + "." + n).removeprefix(
                layer.layer_name + "."
            ),
            t,
        )
        for n, t in named
    ]
    loaded = set(layer.load_weights(mapped))
    assert loaded == {"w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"}
    expected13 = torch.cat(
        (
            gate_up[:, rank * 8 : (rank + 1) * 8],
            gate_up[:, 32 + rank * 8 : 32 + (rank + 1) * 8],
        ),
        1,
    )
    assert torch.equal(layer.w13_weight, expected13)
    assert torch.equal(layer.w2_weight, down[:, :, rank * 4 : (rank + 1) * 4])
    expected_scale = torch.cat(
        (
            gate_scale[:, rank * 8 : (rank + 1) * 8],
            gate_scale[:, 32 + rank * 8 : 32 + (rank + 1) * 8],
        ),
        1,
    )
    assert torch.equal(layer.w13_weight_scale, expected_scale)
    assert torch.equal(layer.w2_weight_scale, down_scale)
    assert layer.w13_weight.weight_loader.__func__ is RoutedExperts.weight_loader
    monkeypatch.setattr(lifecycle, "require_selection", lambda: None)
    monkeypatch.setattr(
        method.delegate, "process_weights_after_loading", lambda layer: None
    )
    monkeypatch.setattr(
        method.delegate, "get_fused_moe_quant_config", lambda layer: None
    )
    method.process_weights_after_loading(layer)
    assert torch.equal(layer.w13_weight, runtime.unpack(expected13))
    assert torch.equal(layer.w13_weight_scale, expected_scale * 16)
    assert torch.equal(layer.w2_weight_scale, down_scale * 16)
    with pytest.raises(runtime.W4A8Error, match="already"):
        method.process_weights_after_loading(layer)


def provider(bad_m=None, mixed_m=None):
    calls = []

    def lookup(**kw):
        calls.append(kw)
        m = kw["M"]
        return m != bad_m, NS(
            quant_type="w4a8",
            solution_type=("triton" if m == mixed_m else "moe_c"),
            need_shuffle=m != mixed_m,
            need_shuffle_scale=False,
            config={"MODE": 1, "BLOCK_SIZE_M": 16},
        )

    return NS(get_aiter_moe_config=lookup, MoeQuantType=NS(W4A8="w4a8"), calls=calls)


def test_all_m_checked_before_layout_selection():
    p = provider()
    shape = runtime.Shape(512, 2560, 192, 10, 513, torch.bfloat16)
    plan = runtime.all_m_plan(shape, p)
    assert plan is not None
    assert [c["M"] for c in p.calls] == list(range(1, 514))
    assert p.calls[-1]["N1"] == 384 and p.calls[-1]["K"] == 2560
    assert runtime.all_m_plan(shape, provider(bad_m=513)) is None
    assert runtime.all_m_plan(shape, provider(mixed_m=513)) is None


def test_vendor_mutating_shuffle_cannot_touch_canonical():
    p = provider()

    def shuffle(a, b, config, block_shape):
        a.add_(1)
        b.sub_(1)
        return a, b

    p.aiter_moe_shfl_weight = shuffle
    plan = runtime.all_m_plan(runtime.Shape(4, 128, 64, 2, 2, torch.bfloat16), p)
    a = torch.zeros((4, 128, 64), dtype=torch.int8)
    b = a.clone()
    converted = plan.pack((a, b))
    assert a.count_nonzero() == b.count_nonzero() == 0
    assert converted[0].unique().item() == 1 and converted[1].unique().item() == -1


def test_real_checkpoint_bf16_ple_protocol():
    path = Path("/model/Qwen3.8-Flash-Next-w4a8-slimquant/config.json")
    if not path.exists():
        pytest.skip("local checkpoint inventory unavailable")
    raw = json.loads(path.read_text())
    text = raw["text_config"]
    assert (
        text["hidden_size"],
        text["moe_intermediate_size"],
        text["num_experts"],
        text["num_experts_per_tok"],
    ) == (2560, 640, 512, 10)
    q = api.SlimQuantW4A8Config.from_config(raw["compression_config"])
    result = Qwen4ExpPLEEmbeddingMethod.from_quant_config(
        q, "model.language_model.layers.2.ple.embedding"
    )
    assert isinstance(result, Qwen4ExpPLEUnquantizedEmbeddingMethod)


@pytest.mark.parametrize(
    "field,value",
    [("format", "pack-quantized"), ("scope", "per_group"), ("weight_bits", 8)],
)
def test_wrong_packed_abi_fails_without_allocation(field, value):
    raw = metadata()
    raw["mixed_precision"]["experts"][field] = value
    with pytest.raises(ValueError):
        api.SlimQuantW4A8Config.from_config(raw)


def test_int8_ple_remains_explicitly_unimplemented():
    raw = metadata()
    raw["mixed_precision"]["ngram_embedding"] = {"format": "int8", "weight_bits": 8}
    with pytest.raises(NotImplementedError, match="INT8 PLE"):
        api.SlimQuantW4A8Config.from_config(raw)


@pytest.mark.parametrize("rank", range(4))
def test_official_tp_loader_keeps_padded_tails_zero_without_shard_shift(
    monkeypatch, rank
):
    layer, _ = layer_and_method(monkeypatch, rank, padded=True)
    gate_up = torch.arange(2 * 64 * 8).to(torch.int8).reshape(2, 64, 8) % 127
    down = torch.arange(2 * 16 * 16).to(torch.int8).reshape(2, 16, 16)
    list(layer.load_weights([("gate_up_proj", gate_up), ("down_proj", down)]))
    assert torch.equal(
        layer.w13_weight[:, :8, :8], gate_up[:, rank * 8 : (rank + 1) * 8]
    )
    assert torch.equal(
        layer.w13_weight[:, 64:72, :8], gate_up[:, 32 + rank * 8 : 40 + rank * 8]
    )
    assert layer.w13_weight[:, :, 8:].count_nonzero() == 0
    assert layer.w13_weight[:, 8:64].count_nonzero() == 0
    assert layer.w13_weight[:, 72:].count_nonzero() == 0
    assert torch.equal(
        layer.w2_weight[:, :16, :4], down[:, :, rank * 4 : (rank + 1) * 4]
    )
    assert layer.w2_weight[:, :, 4:].count_nonzero() == 0
    assert layer.w2_weight[:, 16:].count_nonzero() == 0


def test_prequantized_dense_tuple_preserves_supplied_scale(monkeypatch):
    from vllm.model_executor.kernels.linear.scaled_mm.triton import (
        TritonInt8ScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        triton_scaled_mm as mm,
    )

    kernel = TritonInt8ScaledMMLinearKernel.__new__(TritonInt8ScaledMMLinearKernel)
    layer = NS(
        scheme=NS(kernel=kernel),
        weight=torch.tensor([[1, 2], [3, 4]], dtype=torch.int8),
        weight_scale=torch.tensor([[0.25], [0.5]]),
    )
    calls = []

    def scaled(a, b, *, scale_a, scale_b, out_dtype, bias):
        calls.append((a, scale_a))
        result = (a.float() @ b.float()) * scale_a * scale_b.t()
        return (result + bias).to(out_dtype)

    monkeypatch.setattr(mm, "triton_scaled_mm", scaled)
    method = api.SlimQuantW4A8LinearMethod(None)
    x = torch.full((2, 2), float("nan"), dtype=torch.bfloat16)
    q = torch.tensor([[2, -1], [-3, 4]], dtype=torch.int8)
    scale = torch.tensor([[0.125], [0.75]])
    bias = torch.tensor([0.25, -0.5], dtype=torch.bfloat16)
    out = method.apply(layer, x, bias, (q, scale))
    reference = (
        (q.float() @ layer.weight.float()) * scale * layer.weight_scale.t() + bias
    ).bfloat16()
    assert torch.equal(out, reference) and calls[0][0].data_ptr() == q.data_ptr()
    assert calls[0][1].data_ptr() == scale.data_ptr()


def test_missing_native_module_and_unsupported_routes_never_pack(monkeypatch):
    m = moe()
    m.moe_parallel_config.use_ep = True
    assert runtime.prepare_native(m) is None


def test_unverified_vendor_triton_w4_is_not_a_native_route():
    p = provider(mixed_m=1)
    assert (
        runtime.all_m_plan(runtime.Shape(4, 128, 64, 2, 1, torch.bfloat16), p) is None
    )


def test_fork_adopts_only_unselected_registration(monkeypatch):
    monkeypatch.setattr(lifecycle, "_pid", -1)
    monkeypatch.setattr(lifecycle, "_selected", {})
    monkeypatch.setattr(lifecycle, "_frozen", None)
    lifecycle._check_process()
    assert lifecycle._pid == os.getpid()
    monkeypatch.setattr(lifecycle, "_pid", -1)
    lifecycle._selected["layer"] = NS(native=None)
    with pytest.raises(RuntimeError, match="inherited.*across fork"):
        lifecycle._check_process()


def test_native_rejects_wrong_router_abi_before_call():
    plan = runtime.all_m_plan(
        runtime.Shape(4, 128, 64, 2, 1, torch.bfloat16), provider()
    )
    assert api.SlimQuantW4A8MoEMethod(moe(), plan).topk_indices_dtype == torch.int32
    x = torch.zeros((1, 128), dtype=torch.bfloat16)
    with pytest.raises(runtime.W4A8Error, match="geometry/dtype"):
        plan.run(
            x,
            x,
            x,
            (x, x),
            torch.ones((1, 2)),
            torch.zeros((1, 2), dtype=torch.int64),
            global_experts=4,
        )


def test_fallback_kernel_setup_failure_precedes_canonical_weight_mutation(monkeypatch):
    layer, method = layer_and_method(monkeypatch)
    tensors = {
        name: (value.data_ptr(), value.detach().clone())
        for name, value in layer.named_parameters()
    }
    monkeypatch.setattr(lifecycle, "require_selection", lambda: None)

    def fail(staged):
        assert staged.w13_weight.shape[-1] == 16
        raise RuntimeError("kernel ABI unavailable")

    monkeypatch.setattr(method.delegate, "process_weights_after_loading", fail)
    with pytest.raises(RuntimeError, match="ABI unavailable"):
        method.process_weights_after_loading(layer)
    for name, (pointer, original) in tensors.items():
        assert getattr(layer, name).data_ptr() == pointer
        assert torch.equal(getattr(layer, name), original)
    assert not method._processed


def test_dense_scheme_and_ignored_layer_use_official_ct_methods(monkeypatch):
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsW8A8Int8,
    )

    monkeypatch.setattr(lifecycle, "require_selection", lambda: None)
    q = api.SlimQuantW4A8Config.from_config(metadata())
    layer = ColumnParallelLinear.__new__(ColumnParallelLinear)
    torch.nn.Module.__init__(layer)
    method = q.get_quant_method(layer, "model.layers.0.self_attn.q_proj")
    assert isinstance(method, api.CompressedTensorsLinearMethod)
    assert isinstance(layer.scheme, CompressedTensorsW8A8Int8)
    assert isinstance(
        q.get_quant_method(layer, "model.layers.0.self_attn.o_proj"),
        UnquantizedLinearMethod,
    )


def test_real_adapter_callback_freezes_normal_and_aot_keys_before_env_cache(
    monkeypatch,
):
    import hcu_adapter.model_load as model_load
    from hcu_adapter.bridges import vllm_model_load as model_load_host
    import vllm.envs as envs
    import vllm.model_executor.layers.quantization as registry
    from vllm.compilation.caching import aot_compile_hash_factors
    from vllm.config import (
        DeviceConfig,
        ParallelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.config.utils import hash_factors

    monkeypatch.delenv("VLLM_HCU_DAS_W4A8_NATIVE", raising=False)
    for name, value in dict(
        _installed=False,
        _envs=None,
        _setting=None,
        _selected={},
        _frozen=None,
        _pid=os.getpid(),
    ).items():
        monkeypatch.setattr(lifecycle, name, value)
    monkeypatch.setattr(
        envs, "environment_variables", envs.environment_variables.copy()
    )
    # Other lifecycle tests can leave both declarations in this module. Reset
    # the callback and mutation-guard registries together, as the lifecycle
    # fixture does, before registering this test's actual atomic pair.
    importlib.reload(model_load)
    importlib.reload(model_load_host)
    monkeypatch.setattr(
        registry, "QUANTIZATION_METHODS", registry.QUANTIZATION_METHODS.copy()
    )
    monkeypatch.setattr(registry, "_CUSTOMIZED_METHOD_TO_QUANT_CONFIG", {})
    config = VllmConfig(
        device_config=DeviceConfig("cpu"),
        parallel_config=ParallelConfig(
            worker_cls=model_load_host.POST_MODEL_LOAD_WORKER,
            distributed_executor_backend="uni",
        ),
    )
    config.model_config = NS(
        quantization="slimquant_w4a8", compute_hash=lambda: "w4-cpu"
    )

    def keys():
        return hash_factors(envs.compile_factors()), aot_compile_hash_factors(config)

    try:
        assert lifecycle.install() and not lifecycle.install()
        assert (
            registry._CUSTOMIZED_METHOD_TO_QUANT_CONFIG["slimquant_w4a8"]
            is api.SlimQuantW4A8Config
        )
        early = keys()

        class Selected:
            native = None
            _processed = True

        method = Selected()
        with set_current_vllm_config(config):
            lifecycle.record("test.experts", method)
        selected = keys()
        # Callback runs after official model construction context has exited.
        model_load_host.dispatch_post_model_load(CPUWorker(config), load_dummy_weights=False)
        ready = keys()
        assert early != selected != ready
        assert json.loads(lifecycle.factor())["phase"] == "frozen"
        envs.enable_envs_cache()
        assert keys() == ready
        with pytest.raises(RuntimeError, match="frozen"):
            lifecycle.require_selection(config)
        monkeypatch.setenv("VLLM_HCU_DAS_W4A8_NATIVE", "1")
        assert "invalid" in json.loads(lifecycle.factor()) and keys() != ready
    finally:
        envs.disable_envs_cache()


class CPUWorker:
    def __init__(self, config):
        self.vllm_config = config
        self.model_runner = NS(
            get_model=lambda: self.model, get_draft_model=lambda: None
        )
        self.model = torch.nn.Module()


def test_native_fp16_waits_for_dtype_aware_vendor_config():
    assert (
        runtime.all_m_plan(
            runtime.Shape(512, 3072, 192, 10, 512, torch.float16), provider()
        )
        is None
    )


def test_logical_expert_scheme_and_glob_ignored_are_officially_visible():
    raw = metadata()
    raw["ignore"].append("model.layers.*.mlp.experts.*.down_proj")
    q = api.SlimQuantW4A8Config.from_config(raw)
    assert (
        q.get_scheme_dict(None, "model.layers.0.mlp.experts.42.gate_proj")[
            "weights"
        ].num_bits
        == 4
    )
    assert q.get_scheme_dict(None, "model.layers.0.mlp.experts.42.down_proj") is None
