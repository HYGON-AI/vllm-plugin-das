# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.platforms import envs as henvs


def _load_autoawq_module():
    spec = importlib.util.find_spec(
        "vllm_hcu.model_executor.layers.quantization.lightop_autoawq"
    )
    assert spec is not None, "LightOp AutoAWQ backend is not implemented"
    return importlib.import_module(
        "vllm_hcu.model_executor.layers.quantization.lightop_autoawq"
    )


def test_awq_conversion_uses_checkpoint_nibble_order_and_raw_zero_points() -> None:
    autoawq = _load_autoawq_module()
    repeated_nibbles = [
        0,
        286331153,
        572662306,
        858993459,
        1145324612,
        1431655765,
        1717986918,
        2004318071,
    ]
    qweight = torch.tensor(repeated_nibbles, dtype=torch.int32).reshape(8, 1)
    qzeros = torch.tensor([-2042464975], dtype=torch.int32).reshape(1, 1)
    scales = torch.tensor(
        [[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]],
        dtype=torch.float16,
    )

    weight_trans, scales_zeros = autoawq.convert_awq_to_lightop_layout(
        qweight, qzeros, scales, group_size=8
    )

    assert weight_trans.shape == (8, 1)
    assert torch.equal(
        weight_trans[:, 0],
        torch.full((8,), 1966171168, dtype=torch.int32),
    )
    metadata = scales_zeros.flatten().view(torch.float16).reshape(8, 1, 2)
    assert torch.equal(
        metadata[:, 0, 0],
        torch.tensor([65, 66, 67, 68, 69, 70, 71, 72], dtype=torch.float16),
    )
    assert torch.equal(metadata[:, 0, 1], scales[0])


@pytest.mark.parametrize(
    ("k", "n", "expected"),
    ((8192, 10240, True), (4096, 8192, True), (4096, 4096, False)),
)
def test_awq_shape_support_is_limited_to_lightop_tuned_pairs(
    k: int, n: int, expected: bool
) -> None:
    autoawq = _load_autoawq_module()
    assert autoawq.is_lightop_awq_shape_supported(k, n) is expected


class _Delegate:
    def __init__(self) -> None:
        self.created = 0
        self.processed = 0
        self.applied = 0

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ) -> None:
        self.created += 1
        output_size_per_partition = sum(output_partition_sizes)
        layer.register_parameter(
            "qweight",
            torch.nn.Parameter(
                torch.zeros(
                    input_size_per_partition,
                    output_size_per_partition // 8,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "qzeros",
            torch.nn.Parameter(
                torch.zeros(
                    input_size_per_partition // 128,
                    output_size_per_partition // 8,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "scales",
            torch.nn.Parameter(
                torch.ones(
                    input_size_per_partition // 128,
                    output_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            ),
        )

    def process_weights_after_loading(self, layer) -> None:
        self.processed += 1

    def apply(self, layer, x, bias=None):
        self.applied += 1
        return x.new_full(x.shape[:-1] + (3,), 17)


def _make_method(autoawq, delegate, *, dtype=torch.float16):
    quant_config = SimpleNamespace(
        weight_bits=4,
        group_size=128,
        zero_point=True,
        pack_factor=8,
    )
    method = autoawq.LightOpAutoAWQLinearMethod(delegate, quant_config)
    layer = torch.nn.Module()
    method.create_weights(layer, 128, [8], 128, 8, dtype)
    return method, layer


def test_hybrid_awq_replaces_standard_weights_once_and_preserves_apply_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autoawq = _load_autoawq_module()
    delegate = _Delegate()
    method, layer = _make_method(autoawq, delegate)
    gemm_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def repack(weight_trans, n, k):
        assert (n, k) == (8, 128)
        return weight_trans + 9

    def gemm(inputs, weight, scales_zeros):
        gemm_calls.append((inputs, weight, scales_zeros))
        return torch.arange(
            inputs.shape[0] * scales_zeros.shape[0], dtype=inputs.dtype
        ).reshape(inputs.shape[0], scales_zeros.shape[0])

    monkeypatch.setattr(
        autoawq, "is_lightop_awq_shape_supported", lambda k, n: True
    )
    monkeypatch.setattr(
        autoawq, "_resolve_lightop_awq_ops", lambda: (repack, gemm)
    )
    method.process_weights_after_loading(layer)

    assert delegate.processed == 0
    assert not hasattr(layer, "qzeros")
    assert not hasattr(layer, "scales")
    assert hasattr(layer, "scales_zeros")
    inputs = torch.ones((2, 2, 128), dtype=torch.float16)
    bias = torch.arange(8, dtype=torch.float16)
    output = method.apply(layer, inputs, bias)
    expected = torch.arange(32, dtype=torch.float16).reshape(2, 2, 8) + bias
    assert torch.equal(output, expected)
    assert len(gemm_calls) == 1
    assert gemm_calls[0][0].shape == (4, 128)
    assert delegate.applied == 0


def test_hybrid_awq_delegates_without_mutating_unsupported_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autoawq = _load_autoawq_module()
    delegate = _Delegate()
    method, layer = _make_method(autoawq, delegate, dtype=torch.bfloat16)
    qweight = layer.qweight
    qzeros = layer.qzeros
    scales = layer.scales
    monkeypatch.setattr(
        autoawq,
        "_resolve_lightop_awq_ops",
        lambda: pytest.fail("LightOp must not be resolved for BF16"),
    )

    method.process_weights_after_loading(layer)
    output = method.apply(layer, torch.ones((2, 128), dtype=torch.bfloat16))

    assert delegate.processed == 1
    assert delegate.applied == 1
    assert layer.qweight is qweight
    assert layer.qzeros is qzeros
    assert layer.scales is scales
    assert torch.equal(output, torch.full((2, 3), 17, dtype=torch.bfloat16))


def test_lightop_awq_resolver_rejects_incompatible_public_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autoawq = _load_autoawq_module()
    lightop = ModuleType("lightop")
    lightop.__path__ = []
    gemm_ops = ModuleType("lightop.gemm_ops")
    gemm_ops.awq_gemm_marlin_weight_repack = lambda weight: weight
    gemm_ops.gemm_awq_w4a16_marlin = lambda a, b, scales_zeros: a
    lightop.gemm_ops = gemm_ops
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.gemm_ops", gemm_ops)

    with pytest.raises(ImportError, match="incompatible signature"):
        autoawq._resolve_lightop_awq_ops()


def test_autoawq_patch_treats_public_import_oserror_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autoawq = _load_autoawq_module()
    patch = _load_autoawq_patch_module()

    def incompatible_binary():
        raise OSError("unresolved vendor symbol")

    monkeypatch.setattr(autoawq, "_resolve_lightop_awq_ops", incompatible_binary)

    assert patch._public_lightop_awq_available() is False


def test_hybrid_awq_repack_runtime_failure_is_atomic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autoawq = _load_autoawq_module()
    delegate = _Delegate()
    method, layer = _make_method(autoawq, delegate)
    original = (layer.qweight, layer.qzeros, layer.scales)
    method._lightop_active = True
    method._lightop_gemm = lambda *_args: pytest.fail(
        "stale LightOp state must be cleared"
    )

    def incompatible_repack(*_args):
        raise RuntimeError("vendor ABI mismatch")

    monkeypatch.setattr(
        autoawq, "is_lightop_awq_shape_supported", lambda _k, _n: True
    )
    monkeypatch.setattr(
        autoawq,
        "_resolve_lightop_awq_ops",
        lambda: (incompatible_repack, lambda a, b, scales_zeros: a),
    )

    method.process_weights_after_loading(layer)

    assert delegate.processed == 1
    assert layer.qweight is original[0]
    assert layer.qzeros is original[1]
    assert layer.scales is original[2]
    assert not hasattr(layer, "scales_zeros")
    assert method._lightop_active is False
    assert method._lightop_gemm is None
    output = method.apply(layer, torch.ones((1, 128), dtype=torch.float16))
    assert torch.equal(output, torch.full((1, 3), 17, dtype=torch.float16))


def test_hybrid_awq_empty_input_does_not_call_lightop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autoawq = _load_autoawq_module()
    delegate = _Delegate()
    method, layer = _make_method(autoawq, delegate)
    monkeypatch.setattr(
        autoawq, "is_lightop_awq_shape_supported", lambda _k, _n: True
    )
    monkeypatch.setattr(
        autoawq,
        "_resolve_lightop_awq_ops",
        lambda: (
            lambda weight, _n, _k: weight,
            lambda *_args: pytest.fail("M=0 must not call LightOp"),
        ),
    )
    method.process_weights_after_loading(layer)

    output = method.apply(
        layer,
        torch.empty((2, 0, 128), dtype=torch.float16),
        torch.arange(8, dtype=torch.float16),
    )

    assert output.shape == (2, 0, 8)
    assert output.dtype == torch.float16


def test_hybrid_awq_reprocesses_after_vllm_restores_loader_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autoawq = _load_autoawq_module()
    import vllm.model_executor.parameter as parameter

    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter, "get_tensor_model_parallel_world_size", lambda: 1
    )
    from vllm.model_executor.layers.quantization.auto_awq import (
        AutoAWQConfig,
        AutoAWQLinearMethod,
    )
    from vllm.model_executor.model_loader.reload.layerwise import (
        get_layerwise_info,
        record_metadata_for_reloading,
    )
    from vllm.model_executor.model_loader.reload.meta import (
        materialize_layer,
        restore_layer_on_meta,
    )
    from vllm.model_executor.model_loader.reload.utils import (
        get_layer_params_buffers,
    )

    config = AutoAWQConfig(4, 128, True, False)
    method = autoawq.LightOpAutoAWQLinearMethod(
        AutoAWQLinearMethod(config), config
    )
    layer = torch.nn.Module()

    def weight_loader(param, loaded_weight):
        param.data.copy_(loaded_weight)

    method.create_weights(
        layer,
        128,
        [8],
        128,
        8,
        torch.float16,
        weight_loader=weight_loader,
    )
    record_metadata_for_reloading(layer)
    layer.qweight.data.zero_()
    layer.qzeros.data.zero_()
    layer.scales.data.fill_(1)
    repack_calls = 0

    def repack(weight, _n, _k):
        nonlocal repack_calls
        repack_calls += 1
        return weight

    monkeypatch.setattr(
        autoawq, "is_lightop_awq_shape_supported", lambda _k, _n: True
    )
    monkeypatch.setattr(
        autoawq,
        "_resolve_lightop_awq_ops",
        lambda: (repack, lambda a, b, scales_zeros: a),
    )
    method.process_weights_after_loading(layer)
    assert repack_calls == 1
    assert not hasattr(layer, "qzeros")

    info = get_layerwise_info(layer)
    info.kernel_tensors = get_layer_params_buffers(layer)
    restore_layer_on_meta(layer, info)
    materialize_layer(layer, info)
    assert hasattr(layer.qweight, "weight_loader")
    assert hasattr(layer.qzeros, "weight_loader")
    assert hasattr(layer.scales, "weight_loader")
    layer.qweight.data.zero_()
    layer.qzeros.data.zero_()
    layer.scales.data.fill_(1)

    method.process_weights_after_loading(layer)

    assert repack_calls == 2
    assert hasattr(layer, "scales_zeros")
    assert not hasattr(layer, "qzeros")
    assert not hasattr(layer, "scales")


@pytest.mark.parametrize(
    ("feature", "master", "expected"),
    ((None, None, False), ("1", None, True), ("1", "0", False)),
)
def test_lightop_awq_policy_is_opt_in_and_obeys_master(
    monkeypatch: pytest.MonkeyPatch,
    feature: str | None,
    master: str | None,
    expected: bool,
) -> None:
    for name, value in (
        ("VLLM_HCU_USE_LIGHTOP_AWQ", feature),
        ("VLLM_HCU_USE_CUSTOM_OPS", master),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    policy = getattr(henvs, "lightop_awq_enabled", None)
    assert callable(policy), "LightOp AWQ policy is not implemented"
    assert policy() is expected


def _load_autoawq_patch_module():
    spec = importlib.util.find_spec(
        "vllm_hcu.patch.worker.op_opt.patch_auto_awq"
    )
    assert spec is not None, "LightOp AutoAWQ selector patch is not implemented"
    return importlib.import_module(
        "vllm_hcu.patch.worker.op_opt.patch_auto_awq"
    )


def _make_autoawq_target(*, method=None):
    target = ModuleType(
        "vllm.model_executor.layers.quantization.auto_awq"
    )

    class AutoAWQLinearMethod:
        pass

    class AutoAWQMarlinLinearMethod:
        pass

    class AutoAWQConfig:
        weight_bits = 4
        group_size = 128
        zero_point = True

        def __init__(self):
            self.selected = method or AutoAWQMarlinLinearMethod()

        def get_quant_method(self, layer, prefix):
            return self.selected

    target.AutoAWQConfig = AutoAWQConfig
    target.AutoAWQLinearMethod = AutoAWQLinearMethod
    target.AutoAWQMarlinLinearMethod = AutoAWQMarlinLinearMethod
    return target, AutoAWQConfig


def test_autoawq_patch_wraps_supported_dense_method_when_public_api_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = _load_autoawq_patch_module()
    autoawq = _load_autoawq_module()
    target, config_class = _make_autoawq_target()
    monkeypatch.setenv("VLLM_HCU_USE_LIGHTOP_AWQ", "1")
    monkeypatch.delenv("VLLM_HCU_USE_CUSTOM_OPS", raising=False)
    monkeypatch.setattr(patch, "_public_lightop_awq_available", lambda: True)

    assert patch.apply_to_module(target) is True
    config = config_class()
    selected = config.get_quant_method(object(), "model.layers.0.mlp")

    assert isinstance(selected, autoawq.LightOpAutoAWQLinearMethod)
    assert selected.delegate is config.selected


@pytest.mark.parametrize(
    "unsupported", ("disabled", "group", "zero_point", "missing_public_api")
)
def test_autoawq_patch_returns_original_method_for_unsupported_configuration(
    monkeypatch: pytest.MonkeyPatch,
    unsupported: str,
) -> None:
    patch = _load_autoawq_patch_module()
    target, config_class = _make_autoawq_target()
    monkeypatch.setenv(
        "VLLM_HCU_USE_LIGHTOP_AWQ", "0" if unsupported == "disabled" else "1"
    )
    monkeypatch.delenv("VLLM_HCU_USE_CUSTOM_OPS", raising=False)
    monkeypatch.setattr(
        patch,
        "_public_lightop_awq_available",
        lambda: unsupported != "missing_public_api",
    )
    assert patch.apply_to_module(target) is True
    config = config_class()
    if unsupported == "group":
        config.group_size = 64
    elif unsupported == "zero_point":
        config.zero_point = False

    selected = config.get_quant_method(object(), "model.layers.0.mlp")

    assert selected is config.selected


def test_autoawq_patch_is_idempotent_and_rejects_signature_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch = _load_autoawq_patch_module()
    target, config_class = _make_autoawq_target()
    monkeypatch.setenv("VLLM_HCU_USE_LIGHTOP_AWQ", "1")
    assert patch.apply_to_module(target) is True
    assert patch.apply_to_module(target) is False
    assert config_class._vllm_hcu_original_get_quant_method is not None

    target, config_class = _make_autoawq_target()

    def incompatible(self, layer):
        return None

    config_class.get_quant_method = incompatible
    with pytest.raises(RuntimeError, match="incompatible signature"):
        patch.apply_to_module(target)
