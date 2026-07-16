# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe import aiter_runtime
from vllm_hcu.model_executor.layers.quantization import (
    compressed_tensors_moe_runtime,
    int8_runtime,
    lightop_fp8_runtime,
)
from vllm_hcu.patch.worker.op_opt import (
    patch_activation,
    patch_aiter_ops,
    patch_compressed_tensors,
    patch_compressed_tensors_moe_w8a8_fp8,
    patch_compressed_tensors_moe_wna16,
    patch_compressed_tensors_scheme,
    patch_compressed_tensors_w8a8_fp8,
    patch_compressed_tensors_w8a8_int8,
    patch_deep_gemm,
    patch_input_quant_fp8,
    patch_layers_utils,
    patch_scaled_mm_linear_kernel,
    patch_w8a8_utils,
)


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _package(name: str, **attributes: object) -> ModuleType:
    module = _module(name, **attributes)
    module.__package__ = name
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _install_fake_vllm_envs(
    monkeypatch: pytest.MonkeyPatch,
    **attributes: object,
) -> ModuleType:
    """Install a complete fake package edge for ``import vllm.envs``.

    Adding only the child to ``sys.modules`` still makes Python import the real
    parent package.  That both defeats isolation and can leave a partially
    initialized ``vllm`` behind for later tests when the child is deliberately
    minimal.
    """

    envs = _module("vllm.envs", **attributes)
    vllm = _package("vllm", envs=envs)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)
    return envs


def _install_fake_vllm_torch_utils(
    monkeypatch: pytest.MonkeyPatch,
) -> object:
    """Provide only the custom-op registration dependency under test."""

    def direct_register_custom_op(**kwargs):
        return kwargs

    torch_utils = _module(
        "vllm.utils.torch_utils",
        direct_register_custom_op=direct_register_custom_op,
    )
    utils = _package("vllm.utils", torch_utils=torch_utils)
    vllm = _package("vllm", utils=utils)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.utils", utils)
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils)
    return direct_register_custom_op


def test_aiter_gfx93x_capability_extends_upstream(monkeypatch: pytest.MonkeyPatch):
    fake_hcu = _module("vllm_hcu.platforms.hcu", on_gfx93x=lambda: True)
    monkeypatch.setitem(sys.modules, "vllm_hcu.platforms.hcu", fake_hcu)
    platform = SimpleNamespace(is_rocm=lambda: True)
    assert aiter_runtime.is_aiter_found_and_supported(
        lambda: False, platform, True
    )
    assert not aiter_runtime.is_aiter_found_and_supported(
        lambda: False, platform, False
    )


class _QuantType(enum.IntEnum):
    No = 0
    Other = 1


def _install_fake_aiter(
    monkeypatch: pytest.MonkeyPatch, **attributes: object
) -> ModuleType:
    module = _module("aiter", QuantType=_QuantType, **attributes)
    module.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aiter", module)
    return module


def test_aiter_gelu_tanh_and_feature_off_delegation(
    monkeypatch: pytest.MonkeyPatch,
):
    class ActivationType:
        GeluTanh = object()

    _install_fake_aiter(monkeypatch, ActivationType=ActivationType)
    fused_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fused_moe(*args, **kwargs):
        fused_calls.append((args, kwargs))
        return "gelu_tanh"

    monkeypatch.setitem(
        sys.modules,
        "aiter.fused_moe",
        _module("aiter.fused_moe", fused_moe=fused_moe),
    )
    _install_fake_vllm_envs(
        monkeypatch,
        VLLM_ROCM_USE_AITER=False,
        VLLM_ROCM_USE_AITER_MOE=False,
    )
    calls: list[tuple[object, ...]] = []

    def original(*args):
        calls.append(args)
        return "upstream"

    x = torch.ones(1, 2)
    w1 = torch.ones(1, 4, 2)
    w2 = torch.ones(1, 2, 2)
    topk_weight = torch.ones(1, 1)
    topk_ids = torch.zeros(1, 1, dtype=torch.int64)
    assert (
        aiter_runtime.fused_moe_impl(
            original, x, w1, w2, topk_weight, topk_ids, activation_method=0
        )
        == "upstream"
    )
    assert len(calls) == 1
    assert (
        aiter_runtime.fused_moe_impl(
            original, x, w1, w2, topk_weight, topk_ids, activation_method=3
        )
        == "gelu_tanh"
    )
    assert fused_calls[0][0][6] == "gelu_tanh"
    assert aiter_runtime.get_gelu_tanh_activation_type() is ActivationType.GeluTanh


def test_aiter_activation_string_mapping(monkeypatch: pytest.MonkeyPatch):
    sentinel = object()

    class ActivationType:
        GeluTanh = sentinel

    _install_fake_aiter(monkeypatch, ActivationType=ActivationType)
    original = lambda value: {"silu": "silu"}.get(value)
    assert (
        aiter_runtime.get_aiter_activation_type(original, "gelu_tanh")
        is sentinel
    )
    assert (
        aiter_runtime.get_aiter_activation_type(
            original, "GELU_PYTORCH_TANH"
        )
        is sentinel
    )
    assert aiter_runtime.get_aiter_activation_type(original, "silu") == "silu"


def test_aiter_w16a16_asm_selection(monkeypatch: pytest.MonkeyPatch):
    _install_fake_aiter(monkeypatch)
    _install_fake_vllm_envs(
        monkeypatch,
        VLLM_ROCM_USE_AITER=True,
        VLLM_ROCM_USE_AITER_MOE=True,
    )
    asm_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def asm(*args, **kwargs):
        asm_calls.append((args, kwargs))
        return "asm"

    monkeypatch.setitem(
        sys.modules,
        "aiter.fused_moe_asm_wna16",
        _module("aiter.fused_moe_asm_wna16", fused_experts_asm_impl=asm),
    )
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_W16A16_MOE_SHUFFLE", False)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_AITER_MOE_CONFIG", False)
    x = torch.ones(2, 4)
    w1 = torch.ones(3, 8, 4)
    w2 = torch.ones(3, 4, 4)
    topk_weight = torch.ones(2, 2)
    topk_ids = torch.zeros(2, 2, dtype=torch.int64)

    def upstream(*args):
        raise AssertionError("ASM selection must not delegate")

    assert (
        aiter_runtime.fused_moe_impl(
            upstream, x, w1, w2, topk_weight, topk_ids, activation_method=3
        )
        == "asm"
    )
    assert asm_calls[0][1] == {
        "activation": "gelu_tanh",
        "global_num_experts": 3,
        "expert_map": None,
        "use_shuffle": 0,
    }


def test_aiter_solution_lookup_success_and_failures(monkeypatch: pytest.MonkeyPatch):
    class MoeQuantType:
        W16A16 = "w16a16"

    class MoeSolutionType:
        ASM = "asm"

    def get_config(**kwargs):
        return True, SimpleNamespace(
            solution_type=MoeSolutionType.ASM,
            config={"SOL_ID1": 4, "SOL_ID2": 9},
        )

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            MoeSolutionType=MoeSolutionType,
            get_aiter_moe_config=get_config,
        ),
    )
    aiter_runtime.get_w16a16_moe_solution_id.cache_clear()
    assert (
        aiter_runtime.get_w16a16_moe_solution_id(
            1, 2, 3, 4, 5, 6, torch.bfloat16, "silu", 1
        )
        == "4+9"
    )


@pytest.mark.parametrize("extended", [False, True])
def test_aiter_topk_supports_old_and_new_abi(
    monkeypatch: pytest.MonkeyPatch, extended: bool
):
    calls: list[tuple[object, ...]] = []
    if extended:

        def topk(a, b, c, d, e, num_shared_experts=0, shared_expert_scoring_func=""):
            calls.append(
                (a, b, c, d, e, num_shared_experts, shared_expert_scoring_func)
            )

    else:

        def topk(a, b, c, d, e):
            calls.append((a, b, c, d, e))

    _install_fake_aiter(monkeypatch, topk_softmax=topk)
    tensors = [torch.empty(1) for _ in range(4)]
    aiter_runtime.topk_softmax_impl(*tensors, True, 2, "sigmoid")
    assert len(calls[0]) == (7 if extended else 5)


def test_scaled_mm_prequantized_input_bypasses_quantizer():
    calls: list[tuple[object, ...]] = []

    class FP8ScaledMMLinearKernel:
        def apply_weights(self, layer, x, bias=None):
            calls.append(("original", layer, x, bias))
            return "original"

    module = _module(
        patch_scaled_mm_linear_kernel.TARGET_MODULE,
        FP8ScaledMMLinearKernel=FP8ScaledMMLinearKernel,
    )
    patch_scaled_mm_linear_kernel.apply_to_module(module)
    kernel = FP8ScaledMMLinearKernel()
    kernel.config = SimpleNamespace(out_dtype=None)
    weight = torch.ones(3, 5)
    weight_scale = torch.ones(5)
    kernel._get_layer_params = lambda layer: (weight, weight_scale, None, None)
    kernel.apply_scaled_mm = lambda **kwargs: calls.append(("scaled", kwargs)) or kwargs
    x = torch.ones(2, 3)
    x_q = torch.ones(2, 3, dtype=torch.int8)
    x_scale = torch.ones(2, 1)
    result = kernel.apply_weights(object(), x, x_and_scale_quanted=(x_q, x_scale))
    assert calls[0][0] == "scaled"
    assert result["A"] is x_q and result["As"] is x_scale
    assert kernel.supports_quanted_inputs() is True
    assert kernel.apply_weights(object(), x) == "original"


def test_clamp_swiglu_enforces_rocm_custom_op():
    sentinel = object()

    class CustomOp:
        def __init__(self, *, enforce_enable=False, compile_native=False):
            self.base_args = (enforce_enable, compile_native)
            self._forward_method = "dispatched"

    class SiluAndMulWithClamp(CustomOp):
        def __init__(self, swiglu_limit: float, *, compile_native: bool = True):
            super().__init__(compile_native=compile_native)
            self.swiglu_limit = float(swiglu_limit)

        def forward_native(self, x):
            return x

    platform = SimpleNamespace(
        is_rocm=lambda: True,
        is_cuda_alike=lambda: False,
        is_xpu=lambda: False,
        is_cpu=lambda: False,
    )
    module = _module(
        patch_activation.TARGET_MODULE,
        SiluAndMulWithClamp=SiluAndMulWithClamp,
        current_platform=platform,
        torch=SimpleNamespace(
            ops=SimpleNamespace(
                _C=SimpleNamespace(silu_and_mul_with_clamp=sentinel)
            )
        ),
    )
    patch_activation.apply_to_module(module)
    instance = SiluAndMulWithClamp(7.0, compile_native=False)
    assert instance.base_args == (True, False)
    assert instance.op is sentinel


def test_compressed_linear_only_forwards_supported_prequantized_input():
    class CompressedTensorsLinearMethod:
        def apply(self, layer, x, bias=None):
            return layer.scheme.apply_weights(layer, x, bias=bias)

    module = _module(
        patch_compressed_tensors.TARGET_MODULE,
        CompressedTensorsLinearMethod=CompressedTensorsLinearMethod,
    )
    patch_compressed_tensors.apply_to_module(module)
    calls: list[dict[str, object]] = []
    scheme = SimpleNamespace(
        supports_quanted_inputs=lambda: True,
        apply_weights=lambda layer, x, **kwargs: calls.append(kwargs) or "quantized",
    )
    method = CompressedTensorsLinearMethod()
    pair = (torch.ones(1, 2, dtype=torch.int8), torch.ones(1, 1))
    assert (
        method.apply(
            SimpleNamespace(scheme=scheme),
            torch.ones(1, 2),
            x_and_scale_quanted=pair,
        )
        == "quantized"
    )
    assert calls == [{"bias": None, "x_and_scale_quanted": pair}]
    assert method.supports_quanted_inputs() is True


def test_compressed_scheme_inactive_anchor_is_corrected_at_runtime():
    class CompressedTensorsScheme:
        def process_weights_after_loading(self, layer):
            raise NotImplementedError()

    module = _module(
        patch_compressed_tensors_scheme.TARGET_MODULE,
        CompressedTensorsScheme=CompressedTensorsScheme,
    )
    assert patch_compressed_tensors_scheme.apply_to_module(module) is True
    assert CompressedTensorsScheme().supports_quanted_inputs() is False
    assert patch_compressed_tensors_scheme.apply_to_module(module) is False


def test_slimquant_w4a8_moe_quant_config_uses_int4_weight_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    calls: list[dict[str, object]] = []

    class FusedMoEQuantConfig:
        @staticmethod
        def make(*args, **kwargs):
            calls.append({"args": args, **kwargs})
            return SimpleNamespace(kind="w4a8_quant_config")

    monkeypatch.setattr(
        slimquant_w4a8, "FusedMoEQuantConfig", FusedMoEQuantConfig
    )
    method = object.__new__(slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod)
    layer = SimpleNamespace(
        w13_weight_scale=torch.ones(2, 4, 1),
        w2_weight_scale=torch.ones(2, 2, 1),
        w13_input_scale=None,
        w2_input_scale=None,
    )

    quant_config = method.get_fused_moe_quant_config(layer)

    assert quant_config.kind == "w4a8_quant_config"
    assert method.moe_quant_config is quant_config
    assert len(calls) == 1
    call = calls[0]
    assert call["args"] == (torch.int8,)
    assert call["w1_scale"] is layer.w13_weight_scale
    assert call["w2_scale"] is layer.w2_weight_scale
    assert call["a1_scale"] is None
    assert call["a2_scale"] is None
    assert call["per_act_token_quant"] is True
    assert call["per_out_ch_quant"] is False
    assert call["block_shape"] is None
    assert call["weight_dtype"] == "int4"


def test_slimquant_w4a8_moe_method_is_a_direct_fused_moe_method():
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    method = slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod(
        quant_config=object(),
        moe=object(),
    )

    assert type(method) is slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod
    assert isinstance(method, slimquant_w4a8.FusedMoEMethodBase)


def test_slimquant_w4a8_apply_uses_triton_channelwise_raw_packed_weights(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    calls: list[dict[str, object]] = []

    def fake_fused_experts_impl(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        hidden_states = args[0]
        return hidden_states + 1

    monkeypatch.setattr(
        slimquant_w4a8.aiter_triton_fused_moe,
        "fused_experts_impl",
        fake_fused_experts_impl,
    )

    method = object.__new__(slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod)
    w13_weight = torch.arange(2 * 4 * 2, dtype=torch.int8).reshape(2, 4, 2)
    w2_weight = torch.arange(2 * 2 * 2, dtype=torch.int8).reshape(2, 2, 2)
    w13_parameter = torch.nn.Parameter(w13_weight.clone(), requires_grad=False)
    w2_parameter = torch.nn.Parameter(w2_weight.clone(), requires_grad=False)
    layer = SimpleNamespace(
        w13_weight=w13_parameter,
        w2_weight=w2_parameter,
        w13_weight_scale=torch.nn.Parameter(
            torch.ones(2, 4, 1), requires_grad=False
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.ones(2, 2, 1), requires_grad=False
        ),
    )

    method.process_weights_after_loading(layer)

    assert layer.w13_weight is w13_parameter
    assert layer.w2_weight is w2_parameter
    torch.testing.assert_close(layer.w13_weight.data, w13_weight)
    torch.testing.assert_close(layer.w2_weight.data, w2_weight)

    x = torch.zeros(3, 2, dtype=torch.float16)
    topk_weights = torch.ones(3, 1, dtype=torch.float16)
    topk_ids = torch.zeros(3, 1, dtype=torch.int32)
    result = method.apply(layer, x, topk_weights, topk_ids, None)

    torch.testing.assert_close(result, x + 1)
    assert len(calls) == 1
    call = calls[0]
    assert call["args"][:4] == (
        x,
        layer.w13_weight,
        layer.w2_weight,
        topk_weights,
    )
    assert call["args"][4] is topk_ids
    kwargs = call["kwargs"]
    assert kwargs["output_dtype"] is x.dtype
    assert kwargs["use_int4_w4a8"] is True
    assert kwargs["per_channel_quant"] is True
    assert kwargs["global_num_experts"] == 2
    assert kwargs["expert_map"] is None
    assert kwargs["w1_scale"] is layer.w13_weight_scale
    assert kwargs["w2_scale"] is layer.w2_weight_scale
    assert kwargs["activation"] == "silu"
    assert kwargs["is_gated"] is True


def test_slimquant_w4a8_apply_fails_closed_for_unsupported_inputs(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8

    method = object.__new__(slimquant_w4a8.SlimQuantW4A8Int8AiterMoEMethod)
    layer = SimpleNamespace()
    x = torch.zeros(3, 2, dtype=torch.float16)
    topk_weights = torch.ones(3, 1, dtype=torch.float16)
    topk_ids = torch.zeros(3, 1, dtype=torch.int32)

    with pytest.raises(ValueError, match="rank-2 hidden states"):
        method.apply(layer, x.unsqueeze(0), topk_weights, topk_ids, None)
    with pytest.raises(ValueError, match="shared_experts_input"):
        method.apply(layer, x, topk_weights, topk_ids, x)
    with pytest.raises(ValueError, match="same shape"):
        method.apply(layer, x, topk_weights[:, :0], topk_ids, None)
    with pytest.raises(ValueError, match="same token count"):
        method.apply(layer, x, topk_weights[:2], topk_ids[:2], None)

    import_error = ImportError("missing aiter Triton fused MoE")
    monkeypatch.setattr(slimquant_w4a8, "aiter_triton_fused_moe", None)
    monkeypatch.setattr(slimquant_w4a8, "_AITER_TRITON_IMPORT_ERROR", import_error)
    with pytest.raises(RuntimeError, match="requires aiter.ops.triton.fused_moe"):
        method.apply(layer, x, topk_weights, topk_ids, None)


def _fake_moe_fp8_module():
    class CompressedTensorsW8A8Fp8MoEMethod:
        def __init__(self, weight_quant, input_quant, moe, layer_name=None):
            self.weight_quant = weight_quant
            self.input_quant = input_quant
            self.moe = moe
            self.layer_name = layer_name

        def process_weights_after_loading(self, layer):
            layer.upstream_processed = True

        def apply(
            self,
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts_input,
        ):
            return (
                "upstream",
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts_input,
            )

    return _module(
        patch_compressed_tensors_moe_w8a8_fp8.TARGET_MODULE,
        CompressedTensorsW8A8Fp8MoEMethod=CompressedTensorsW8A8Fp8MoEMethod,
    )


def _fp8_moe_layer() -> SimpleNamespace:
    return SimpleNamespace(
        activation=SimpleNamespace(value="silu"),
        apply_router_weight_on_input=False,
        w13_weight=torch.ones(3, 8, 4),
        w2_weight=torch.ones(3, 4, 4),
        w13_weight_scale=torch.ones(3, 8, 1),
        w2_weight_scale=torch.ones(3, 4, 1),
        w13_input_scale=None,
        w2_input_scale=None,
        global_num_experts=3,
        expert_map=None,
        layer_name="model.layers.0.mlp.experts",
    )


def test_moe_fp8_feature_off_delegates_exactly(monkeypatch: pytest.MonkeyPatch):
    module = _fake_moe_fp8_module()
    monkeypatch.setattr(
        patch_compressed_tensors_moe_w8a8_fp8,
        "_aiter_requested",
        lambda: False,
    )
    assert patch_compressed_tensors_moe_w8a8_fp8.apply_to_module(module) is True
    assert patch_compressed_tensors_moe_w8a8_fp8.apply_to_module(module) is False
    method = module.CompressedTensorsW8A8Fp8MoEMethod(
        object(), object(), SimpleNamespace(disable_inplace=False)
    )
    layer = _fp8_moe_layer()
    x = torch.ones(2, 4)
    weights = torch.ones(2, 2)
    ids = torch.zeros(2, 2, dtype=torch.int64)
    result = method.apply(layer, x, weights, ids, None)
    assert result == ("upstream", layer, x, weights, ids, None)
    with pytest.raises(RuntimeError, match="require the HCU AITER"):
        method.apply(layer, x, weights, ids, None, torch.ones_like(x), None)


def test_moe_fp8_aiter_config_is_layer_aware_and_cached(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict[str, object]] = []

    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"

    def get_config(**kwargs):
        calls.append(kwargs)
        return True, SimpleNamespace(solution_type="ASM", serial=len(calls))

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
        ),
    )
    method = SimpleNamespace(_hcu_aiter_moe_config_cache={})
    x = torch.ones(2, 4)
    ids = torch.zeros(2, 2, dtype=torch.int64)
    first_layer = _fp8_moe_layer()
    first = compressed_tensors_moe_runtime.get_aiter_w8a8_runtime_config(
        method, first_layer, x, ids
    )
    again = compressed_tensors_moe_runtime.get_aiter_w8a8_runtime_config(
        method, first_layer, x, ids
    )
    second = compressed_tensors_moe_runtime.get_aiter_w8a8_runtime_config(
        method, _fp8_moe_layer(), x, ids
    )
    assert first is again and first is not second
    assert len(calls) == 2
    assert calls[0]["quant_type"] == "fp8_w8a8"
    assert calls[0]["M"] == 2 and calls[0]["top_k"] == 2


def test_moe_fp8_moe_c_shuffle_invalidates_on_weight_change(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    def shuffle_w1(weight):
        calls.append("w1")
        return weight + 1

    def shuffle_w2(weight):
        calls.append("w2")
        return weight + 2

    monkeypatch.setitem(sys.modules, "aiter.ops", _module("aiter.ops"))
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.shuffle",
        _module(
            "aiter.ops.shuffle",
            moe_layout_shuffle_gemm1=shuffle_w1,
            moe_layout_shuffle_gemm2=shuffle_w2,
        ),
    )
    layer = _fp8_moe_layer()
    first = compressed_tensors_moe_runtime.get_aiter_weights_for_solution(
        layer, "MOE_C"
    )
    again = compressed_tensors_moe_runtime.get_aiter_weights_for_solution(
        layer, "MOE_C"
    )
    assert first[0] is again[0] and calls == ["w1", "w2"]
    layer.w13_weight.add_(1)
    refreshed = compressed_tensors_moe_runtime.get_aiter_weights_for_solution(
        layer, "MOE_C"
    )
    assert refreshed[0] is not first[0]
    assert calls == ["w1", "w2", "w1", "w2"]


def test_moe_fp8_prequantized_inputs_are_paired(monkeypatch: pytest.MonkeyPatch):
    module = _fake_moe_fp8_module()
    monkeypatch.setattr(
        patch_compressed_tensors_moe_w8a8_fp8,
        "_aiter_requested",
        lambda: True,
    )
    patch_compressed_tensors_moe_w8a8_fp8.apply_to_module(module)
    method = module.CompressedTensorsW8A8Fp8MoEMethod(
        object(), object(), SimpleNamespace(disable_inplace=False)
    )
    x = torch.ones(2, 4)
    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match="i_q and i_s together",
    ):
        method.apply(
            _fp8_moe_layer(),
            x,
            torch.ones(2, 2),
            torch.zeros(2, 2, dtype=torch.int64),
            None,
            torch.ones_like(x),
            None,
        )


def test_moe_fp8_aiter_path_and_shared_expert_guard(
    monkeypatch: pytest.MonkeyPatch,
):
    kernel_calls: list[dict[str, object]] = []

    class MoeQuantType:
        FP8_W8A8 = "fp8_w8a8"

    def get_config(**kwargs):
        return True, SimpleNamespace(solution_type="ASM")

    def aiter_moe(**kwargs):
        kernel_calls.append(kwargs)
        return torch.full((2, 4), 7.0)

    monkeypatch.setitem(
        sys.modules,
        "aiter.moe",
        _module(
            "aiter.moe",
            MoeQuantType=MoeQuantType,
            get_aiter_moe_config=get_config,
            aiter_moe=aiter_moe,
        ),
    )
    method = SimpleNamespace(
        moe=SimpleNamespace(disable_inplace=True),
        _hcu_aiter_moe_config_cache={},
    )
    layer = _fp8_moe_layer()
    x = torch.ones(2, 4)
    weights = torch.ones(2, 2)
    ids = torch.zeros(2, 2, dtype=torch.int64)
    output = compressed_tensors_moe_runtime.apply_aiter_w8a8_fp8_moe(
        method, layer, x, weights, ids, None
    )
    torch.testing.assert_close(output, torch.full((2, 4), 7.0))
    assert kernel_calls[0]["hidden_states"] is x
    assert kernel_calls[0]["inplace"] is False
    assert kernel_calls[0]["topk_ids"].dtype is torch.int32
    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match="shared_experts_input",
    ):
        compressed_tensors_moe_runtime.apply_aiter_w8a8_fp8_moe(
            method, layer, x, weights, ids, x
        )


def test_moe_fp8_dpsk_backend_postprocesses_hcu_experts():
    module = _fake_moe_fp8_module()
    patch_compressed_tensors_moe_w8a8_fp8.apply_to_module(module)
    method = module.CompressedTensorsW8A8Fp8MoEMethod(
        object(), object(), SimpleNamespace(disable_inplace=False)
    )
    processed: list[object] = []
    experts = SimpleNamespace(
        process_weights_after_loading=lambda layer: processed.append(layer)
    )
    method.fp8_backend = SimpleNamespace(value="DPSK_DEEPGEMM")
    method.moe_kernel = SimpleNamespace(
        fused_experts=SimpleNamespace(experts=experts)
    )
    layer = _fp8_moe_layer()
    method.process_weights_after_loading(layer)
    assert layer.upstream_processed is True
    assert processed == [layer]


def _fake_moe_wna16_module():
    attrs_calls: list[tuple[torch.Tensor, dict[str, object]]] = []
    config_calls: list[dict[str, object]] = []

    def set_weight_attrs(weight, attrs):
        attrs_calls.append((weight, dict(attrs)))

    def config_builder(**kwargs):
        config_calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    class CompressedTensorsWNA16MoEMethod:
        def create_weights(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        ):
            del params_dtype, extra_weight_attrs
            layer.register_parameter(
                "w13_weight_packed",
                torch.nn.Parameter(
                    torch.empty(num_experts, hidden_size, 1), requires_grad=False
                ),
            )
            layer.register_parameter(
                "w2_weight_packed",
                torch.nn.Parameter(
                    torch.empty(
                        num_experts, intermediate_size_per_partition, 1
                    ),
                    requires_grad=False,
                ),
            )
            layer.register_parameter(
                "w13_weight_scale",
                torch.nn.Parameter(torch.ones(1), requires_grad=False),
            )
            layer.register_parameter(
                "w2_weight_scale",
                torch.nn.Parameter(torch.ones(1), requires_grad=False),
            )
            return "upstream-create"

        def get_fused_moe_quant_config(self, layer):
            return ("upstream-config", layer)

    module = _module(
        patch_compressed_tensors_moe_wna16.TARGET_MODULE,
        CompressedTensorsWNA16MoEMethod=CompressedTensorsWNA16MoEMethod,
        set_weight_attrs=set_weight_attrs,
        int4_w4a16_moe_quant_config=config_builder,
    )
    return module, attrs_calls, config_calls


def _wna16_method(module, *, gated: bool, num_bits: int = 4):
    method = module.CompressedTensorsWNA16MoEMethod()
    method.num_bits = num_bits
    method.group_size = 4
    method.strategy = "group"
    method.moe = SimpleNamespace(is_act_and_mul=gated)
    return method


def test_moe_wna16_feature_off_delegates_exactly(
    monkeypatch: pytest.MonkeyPatch,
):
    module, _, config_calls = _fake_moe_wna16_module()
    monkeypatch.setattr(
        patch_compressed_tensors_moe_wna16,
        "_aiter_requested",
        lambda: False,
    )
    assert patch_compressed_tensors_moe_wna16.apply_to_module(module) is True
    assert patch_compressed_tensors_moe_wna16.apply_to_module(module) is False
    method = _wna16_method(module, gated=True)
    layer = torch.nn.Module()
    assert method.create_weights(layer, 2, 16, 24, torch.bfloat16) == (
        "upstream-create"
    )
    assert not hasattr(layer, "w13_qzeros") and not hasattr(layer, "w2_qzeros")
    assert method.get_fused_moe_quant_config(layer) == ("upstream-config", layer)
    assert config_calls == []


@pytest.mark.parametrize("gated,shards", [(False, 1), (True, 2)])
def test_moe_wna16_allocates_correct_initialized_qzeros(
    monkeypatch: pytest.MonkeyPatch,
    gated: bool,
    shards: int,
):
    module, attrs_calls, _ = _fake_moe_wna16_module()
    monkeypatch.setattr(
        patch_compressed_tensors_moe_wna16,
        "_aiter_requested",
        lambda: True,
    )
    patch_compressed_tensors_moe_wna16.apply_to_module(module)
    method = _wna16_method(module, gated=gated)
    layer = torch.nn.Module()
    method.create_weights(layer, 2, 16, 24, torch.bfloat16, loader="test")
    # AITER/vLLM pack two output-channel zero points per byte, while the
    # K/group axis remains unpacked.
    assert layer.w13_qzeros.shape == (2, shards * 12, 4)
    assert layer.w2_qzeros.shape == (2, 8, 6)
    assert layer.w13_qzeros.dtype is torch.uint8
    assert torch.all(layer.w13_qzeros == 0x88)
    assert torch.all(layer.w2_qzeros == 0x88)
    assert attrs_calls[-2][1] == {
        "loader": "test",
        "is_transposed": True,
        "quant_method": "group",
    }


def test_moe_wna16_quant_config_requires_registered_qzeros(
    monkeypatch: pytest.MonkeyPatch,
):
    module, _, config_calls = _fake_moe_wna16_module()
    monkeypatch.setattr(
        patch_compressed_tensors_moe_wna16,
        "_aiter_requested",
        lambda: True,
    )
    patch_compressed_tensors_moe_wna16.apply_to_module(module)
    method = _wna16_method(module, gated=True)
    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match="w13_weight_scale",
    ):
        method.get_fused_moe_quant_config(torch.nn.Module())

    layer = torch.nn.Module()
    method.create_weights(layer, 2, 16, 24, torch.bfloat16)
    config = method.get_fused_moe_quant_config(layer)
    assert config.w1_zp is layer.w13_qzeros
    assert config.w2_zp is layer.w2_qzeros
    assert config.block_shape == [0, 4]
    assert len(config_calls) == 1

    invalid = _wna16_method(module, gated=True, num_bits=8)
    invalid_layer = torch.nn.Module()
    with pytest.raises(
        compressed_tensors_moe_runtime.HcuCompressedTensorsMoeError,
        match="requires 4-bit",
    ):
        invalid.create_weights(invalid_layer, 2, 16, 24, torch.bfloat16)


def _fake_fp8_scheme_module():
    channel = object()

    class CompressedTensorsW8A8Fp8:
        def process_weights_after_loading(self, layer):
            layer.weight = torch.nn.Parameter(layer.weight.t().contiguous())

        def apply_weights(self, layer, x, bias=None):
            return self.fp8_linear.apply_weights(layer, x, bias)

    return _module(
        patch_compressed_tensors_w8a8_fp8.TARGET_MODULE,
        CompressedTensorsW8A8Fp8=CompressedTensorsW8A8Fp8,
        QuantizationStrategy=SimpleNamespace(CHANNEL=channel),
    ), channel


def test_fp8_channel_weight_layout_requires_hcu_kernel(monkeypatch: pytest.MonkeyPatch):
    module, channel = _fake_fp8_scheme_module()
    monkeypatch.setattr(
        patch_compressed_tensors_w8a8_fp8,
        "_custom_quantization_enabled",
        lambda: True,
    )
    patch_compressed_tensors_w8a8_fp8.apply_to_module(module)
    scheme = module.CompressedTensorsW8A8Fp8()
    scheme.strategy = channel
    scheme.fp8_linear = object()
    with pytest.raises(RuntimeError, match="adapter is not active"):
        scheme.process_weights_after_loading(SimpleNamespace(weight=torch.ones(2, 3)))


def test_fp8_adapter_resolves_hcu_flag_lazily(monkeypatch: pytest.MonkeyPatch):
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM", False)
    assert patch_compressed_tensors_w8a8_fp8._custom_quantization_enabled() is False
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM", True)
    assert patch_compressed_tensors_w8a8_fp8._custom_quantization_enabled() is True


def test_fp8_scheme_forwards_prequantized_input(monkeypatch: pytest.MonkeyPatch):
    module, channel = _fake_fp8_scheme_module()
    monkeypatch.setattr(
        patch_compressed_tensors_w8a8_fp8,
        "_custom_quantization_enabled",
        lambda: True,
    )
    patch_compressed_tensors_w8a8_fp8.apply_to_module(module)
    calls: list[tuple[object, ...]] = []

    class Kernel:
        _hcu_fp8_patch_applied = True

        def supports_quanted_inputs(self):
            return True

        def apply_weights(self, *args, **kwargs):
            calls.append((*args, kwargs))
            return "fp8"

    scheme = module.CompressedTensorsW8A8Fp8()
    scheme.strategy = channel
    scheme.fp8_linear = Kernel()
    layer = SimpleNamespace(weight=torch.nn.Parameter(torch.arange(6.0).view(2, 3)))
    original = layer.weight.detach().clone()
    scheme.process_weights_after_loading(layer)
    torch.testing.assert_close(layer.weight, original)
    pair = (torch.ones(1, 3, dtype=torch.int8), torch.ones(1, 1))
    assert (
        scheme.apply_weights(layer, torch.ones(1, 3), x_and_scale_quanted=pair)
        == "fp8"
    )
    assert calls[0][-1]["x_and_scale_quanted"] is pair


def test_int8_hcu_owned_kernel_validates_and_computes_shapes(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_hcu.platforms import envs as henvs

    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_FUSED_SILU_MUL_QUANT", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_FUSED_RMS_QUANT", False)

    def gemm(a, b, scale_a, scale_b, m, n, k, flag, out_dtype):
        assert (m, n, k, flag) == (2, 2, 3, "NT")
        output = (a.float() * scale_a) @ (b.float() * scale_b).t()
        return True, output.to(out_dtype)

    monkeypatch.setitem(
        sys.modules,
        "lmslim",
        _module("lmslim", quant_ops=SimpleNamespace(hipblaslt_w8a8_gemm=gemm)),
    )
    x = torch.ones(1, 2, 3, dtype=torch.bfloat16)
    x_q = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.int8)
    x_scale = torch.tensor([[[0.5], [0.25]]], dtype=torch.float32)
    weight = torch.tensor([[1, 0, -1], [2, 1, 0]], dtype=torch.int8)
    weight_scale = torch.tensor([[0.5], [0.25]], dtype=torch.float32)
    result = int8_runtime.apply_int8_linear(
        x,
        weight,
        weight_scale,
        torch.bfloat16,
        x_and_scale_quanted=(x_q, x_scale),
    )
    expected = (x_q.reshape(2, 3).float() * x_scale.reshape(2, 1)) @ (
        weight.float() * weight_scale
    ).t()
    torch.testing.assert_close(result.reshape(2, 2).float(), expected)
    with pytest.raises(
        int8_runtime.HcuInt8LinearError,
        match="quantized input/scale shapes",
    ):
        int8_runtime.apply_int8_linear(
            torch.ones(1, 2, 6, dtype=torch.bfloat16),
            weight,
            weight_scale,
            torch.bfloat16,
            x_and_scale_quanted=(x_q, x_scale),
        )
    with pytest.raises(int8_runtime.HcuInt8LinearError, match="symmetric"):
        int8_runtime.apply_int8_linear(
            x,
            weight,
            weight_scale,
            torch.bfloat16,
            input_zero_point=torch.ones(1, dtype=torch.int8),
            x_and_scale_quanted=(x_q, x_scale),
        )


def _fake_int8_scheme_module():
    class CompressedTensorsW8A8Int8:
        def process_weights_after_loading(self, layer):
            layer.weight = torch.nn.Parameter(
                layer.weight.t().contiguous(), requires_grad=False
            )

        def apply_weights(self, layer, x, bias):
            return ("upstream", layer, x, bias)

    return _module(
        patch_compressed_tensors_w8a8_int8.TARGET_MODULE,
        CompressedTensorsW8A8Int8=CompressedTensorsW8A8Int8,
    )


def test_int8_scheme_layout_and_feature_off_delegation(monkeypatch: pytest.MonkeyPatch):
    module = _fake_int8_scheme_module()
    monkeypatch.setattr(
        patch_compressed_tensors_w8a8_int8,
        "_custom_quantization_enabled",
        lambda: False,
    )
    patch_compressed_tensors_w8a8_int8.apply_to_module(module)
    scheme = module.CompressedTensorsW8A8Int8()
    layer = SimpleNamespace(weight=torch.nn.Parameter(torch.ones(2, 3)))
    assert scheme.apply_weights(layer, "x", None)[0] == "upstream"
    scheme.process_weights_after_loading(layer)
    assert layer.weight.shape == (3, 2)

    feature_module = _fake_int8_scheme_module()
    monkeypatch.setattr(
        patch_compressed_tensors_w8a8_int8,
        "_custom_quantization_enabled",
        lambda: True,
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        int8_runtime,
        "apply_int8_linear",
        lambda **kwargs: calls.append(kwargs) or "hcu",
    )
    patch_compressed_tensors_w8a8_int8.apply_to_module(feature_module)
    feature_scheme = feature_module.CompressedTensorsW8A8Int8()
    feature_layer = SimpleNamespace(
        weight=torch.nn.Parameter(torch.ones(2, 3)),
        weight_scale=torch.ones(2, 1),
        params_dtype=torch.bfloat16,
        input_scale=None,
        input_zero_point=None,
        azp_adj=None,
    )
    feature_scheme.process_weights_after_loading(feature_layer)
    assert feature_layer.weight.shape == (2, 3)
    assert feature_layer.weight.is_contiguous()
    assert feature_scheme.apply_weights(feature_layer, "x", None) == "hcu"
    assert calls[0]["weight"] is feature_layer.weight


def test_int8_weight_layout_rolls_back_if_upstream_processing_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    class CompressedTensorsW8A8Int8:
        def process_weights_after_loading(self, layer):
            raise RuntimeError("packing failed")

        def apply_weights(self, layer, x, bias):
            return None

    module = _module(
        patch_compressed_tensors_w8a8_int8.TARGET_MODULE,
        CompressedTensorsW8A8Int8=CompressedTensorsW8A8Int8,
    )
    monkeypatch.setattr(
        patch_compressed_tensors_w8a8_int8,
        "_custom_quantization_enabled",
        lambda: True,
    )
    patch_compressed_tensors_w8a8_int8.apply_to_module(module)
    layer = SimpleNamespace(
        weight=torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))
    )
    original_parameter = layer.weight
    original_value = layer.weight.detach().clone()
    with pytest.raises(RuntimeError, match="packing failed"):
        module.CompressedTensorsW8A8Int8().process_weights_after_loading(layer)
    assert layer.weight is original_parameter
    assert layer.weight.shape == (2, 3)
    torch.testing.assert_close(layer.weight, original_value)


def test_lightop_fp8_registration_is_single_owner_and_latched():
    lightop_fp8_runtime._reset_for_tests()
    calls: list[dict[str, object]] = []
    lightop_fp8_runtime.ensure_registered(
        torch.float8_e4m3fn,
        lambda **kwargs: calls.append(kwargs),
    )
    lightop_fp8_runtime.ensure_registered(
        torch.float8_e4m3fn,
        lambda **kwargs: calls.append(kwargs),
    )
    assert len(calls) == 1
    assert calls[0]["op_name"] == "lightop_per_token_quant_fp8"

    lightop_fp8_runtime._reset_for_tests()
    with pytest.raises(lightop_fp8_runtime.HcuLightOpRegistrationError):
        lightop_fp8_runtime.ensure_registered(
            torch.float8_e4m3fn,
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("duplicate")),
        )
    with pytest.raises(
        lightop_fp8_runtime.HcuLightOpRegistrationError,
        match="previously failed",
    ):
        lightop_fp8_runtime.ensure_registered(
            torch.float8_e4m3fn, lambda **kwargs: None
        )
    lightop_fp8_runtime._reset_for_tests()


def test_lightop_fp8_adapter_has_no_import_time_registration():
    lightop_fp8_runtime._reset_for_tests()
    module, _ = _fake_input_quant_module()
    patch_input_quant_fp8.apply_to_module(module)
    assert lightop_fp8_runtime._REGISTERED is False
    assert lightop_fp8_runtime._REGISTRATION_ERROR is None


def _fake_input_quant_module():
    per_token = object()

    class QuantFP8:
        def forward_cuda(self, x, scale=None, scale_ub=None, use_triton=False):
            return ("cuda", x, scale, scale_ub, use_triton)

        def forward_native(self, x, scale=None, scale_ub=None, use_triton=False):
            return ("native", x, scale, scale_ub, use_triton)

    return _module(
        patch_input_quant_fp8.TARGET_MODULE,
        QuantFP8=QuantFP8,
        GroupShape=SimpleNamespace(PER_TOKEN=per_token),
        _FP8_DTYPE=torch.float8_e4m3fn,
    ), per_token


@pytest.mark.parametrize("method_name", ["forward_cuda", "forward_native"])
def test_quant_fp8_eligibility_and_feature_off(
    monkeypatch: pytest.MonkeyPatch, method_name: str
):
    _install_fake_vllm_torch_utils(monkeypatch)
    module, per_token = _fake_input_quant_module()
    patch_input_quant_fp8.apply_to_module(module)
    instance = module.QuantFP8()
    instance.group_shape = per_token
    instance.num_token_padding = None
    x = torch.ones(2, 4)
    monkeypatch.setattr(patch_input_quant_fp8, "_lightop_requested", lambda: True)
    monkeypatch.setattr(
        lightop_fp8_runtime,
        "quantize",
        lambda value, dtype, register: ("lightop", value, dtype),
    )
    method = getattr(instance, method_name)
    assert method(x)[0] == "lightop"
    assert method(x.t())[0] == method_name.removeprefix("forward_")
    monkeypatch.setattr(patch_input_quant_fp8, "_lightop_requested", lambda: False)
    assert method(x)[0] == method_name.removeprefix("forward_")


def test_weight8bit_marlin2_layout_2d_3d_and_validation():
    module = _module(patch_w8a8_utils.TARGET_MODULE)
    patch_w8a8_utils.apply_to_module(module)
    weight = torch.arange(16 * 64, dtype=torch.int32).to(torch.int8).view(16, 64)
    result = module.weight8bit_nt_kpack2_marlin2(weight)
    reference = (
        weight.reshape(1, 16, 1, 4, 16)
        .permute(2, 0, 3, 1, 4)
        .contiguous()
        .reshape(1, 1024)
    )
    torch.testing.assert_close(result, reference)
    experts = torch.stack((weight, weight + 1))
    result_3d = module.weight8bit_nt_kpack2_marlin2(experts)
    assert result_3d.shape == (2, 1, 1024)
    with pytest.raises(ValueError, match="rank 2 or 3"):
        module.weight8bit_nt_kpack2_marlin2(torch.ones(16, dtype=torch.int8))


@pytest.mark.parametrize("is_rocm", [False, True])
def test_unquantized_gemm_dispatch_only_changes_rocm(is_rocm: bool):
    default = lambda *args: "default"
    rocm = lambda *args: "rocm"

    def dispatch_unquantized_gemm():
        return rocm if is_rocm else "other"

    module = _module(
        patch_layers_utils.TARGET_MODULE,
        current_platform=SimpleNamespace(is_rocm=lambda: is_rocm),
        default_unquantized_gemm=default,
        dispatch_unquantized_gemm=dispatch_unquantized_gemm,
    )
    patch_layers_utils.apply_to_module(module)
    assert module.dispatch_unquantized_gemm() is (default if is_rocm else "other")


def test_tf32_hc_prenorm_cpu_fallback_and_backend_delegation():
    calls: list[str] = []

    def lazy_init():
        calls.append("lazy")

    def original(x, fn, out, sqrsum, num_split):
        calls.append("backend")
        return "backend"

    module = _module(
        patch_deep_gemm.TARGET_MODULE,
        torch=torch,
        _lazy_init=lazy_init,
        _tf32_hc_prenorm_gemm_impl=None,
        tf32_hc_prenorm_gemm=original,
    )
    patch_deep_gemm.apply_to_module(module)
    x = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
    fn = torch.tensor([[2.0, 1.0], [0.5, -2.0], [1.0, 1.0]])
    out = torch.empty(2, 2, 3)
    sqrsum = torch.empty(2, 2)
    result = module.tf32_hc_prenorm_gemm(x, fn, out, sqrsum, 2)
    assert result is out
    torch.testing.assert_close(out[0], x @ fn.t())
    torch.testing.assert_close(out[1], torch.zeros_like(out[1]))
    torch.testing.assert_close(sqrsum[0], x.square().sum(-1))
    torch.testing.assert_close(sqrsum[1], torch.zeros_like(sqrsum[1]))

    module._tf32_hc_prenorm_gemm_impl = object()
    assert module.tf32_hc_prenorm_gemm(x, fn, out, sqrsum, 2) == "backend"
    assert calls[-1] == "backend"
