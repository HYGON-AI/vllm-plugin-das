# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU/fake product tests for the Channel-FP8 target Triton route."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.runtime_compat import scaled_mm


_TARGET_MODULE = "vllm.model_executor.kernels.linear.scaled_mm.pytorch"
_TRITON_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "triton_scaled_mm"
)
_LIGHTOP_MODULE = "lightop.gemm_ops"
_FP8_DTYPE = getattr(torch, "float8_e4m3fnuz", None)
_LIGHTOP_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)

pytestmark = pytest.mark.skipif(
    _FP8_DTYPE is None or _LIGHTOP_FP8_DTYPE is None,
    reason="the target vLLM Channel-FP8 contract requires torch FP8 dtypes",
)


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _install_fake_module_tree(
    monkeypatch: pytest.MonkeyPatch,
    leaves: dict[str, ModuleType],
) -> None:
    package_names: set[str] = set()
    for leaf_name in leaves:
        parts = leaf_name.split(".")
        package_names.update(
            ".".join(parts[:index]) for index in range(1, len(parts))
        )

    modules: dict[str, ModuleType] = {
        name: _package(name) for name in package_names
    }
    modules.update(leaves)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name, module in sorted(
        modules.items(), key=lambda item: item[0].count(".")
    ):
        if "." not in name:
            continue
        parent_name, child_name = name.rsplit(".", 1)
        setattr(modules[parent_name], child_name, module)


@pytest.fixture()
def fake_product(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    original_calls: list[dict[str, object]] = []
    triton_calls: list[dict[str, object]] = []
    lightop_calls: list[dict[str, object]] = []
    state: dict[str, object] = {}

    # Most existing product tests exercise the retained Triton route. Tests
    # for the default/custom route delete or override this value explicitly.
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM", "0")

    def original_get_output_padding(self):
        return "target-padding"

    def original_apply_scaled_mm(self, **kwargs):
        original_calls.append(kwargs)
        raise AssertionError("the target legacy scaled-mm route must not run")

    class TorchFP8ScaledMMLinearKernel:
        get_output_padding = original_get_output_padding
        apply_scaled_mm = original_apply_scaled_mm

    class ChannelWiseTorchFP8ScaledMMLinearKernel(
        TorchFP8ScaledMMLinearKernel
    ):
        pass

    def default_behavior(input, weight, scale_a, scale_b, out_dtype, bias):
        del scale_a, scale_b, bias
        values = torch.arange(
            input.shape[0] * weight.shape[1], dtype=torch.float32
        )
        return values.reshape(input.shape[0], weight.shape[1]).to(out_dtype)

    state["behavior"] = default_behavior

    def triton_scaled_mm(
        input,
        weight,
        scale_a,
        scale_b,
        out_dtype,
        bias=None,
    ):
        call = {
            "input": input,
            "weight": weight,
            "scale_a": scale_a,
            "scale_b": scale_b,
            "out_dtype": out_dtype,
            "bias": bias,
        }
        triton_calls.append(call)
        behavior = state["behavior"]
        result = behavior(
            input,
            weight,
            scale_a,
            scale_b,
            out_dtype,
            bias,
        )
        call["result"] = result
        return result

    target = ModuleType(_TARGET_MODULE)
    target.TorchFP8ScaledMMLinearKernel = TorchFP8ScaledMMLinearKernel
    target.ChannelWiseTorchFP8ScaledMMLinearKernel = (
        ChannelWiseTorchFP8ScaledMMLinearKernel
    )
    triton = ModuleType(_TRITON_MODULE)
    triton.triton_scaled_mm = triton_scaled_mm
    lightop = ModuleType(_LIGHTOP_MODULE)

    def hipblaslt_w8a8_channelwise_gemm(
        a,
        b,
        scale_a,
        scale_b,
        m,
        n,
        k,
        transpose_flag,
        out_dtype,
        bias=None,
    ):
        values = torch.arange(m * n, dtype=torch.float32).reshape(1, m, n)
        result = values.to(out_dtype)
        lightop_calls.append(
            {
                "a": a,
                "b": b,
                "scale_a": scale_a,
                "scale_b": scale_b,
                "m": m,
                "n": n,
                "k": k,
                "transpose_flag": transpose_flag,
                "out_dtype": out_dtype,
                "bias": bias,
                "result": result,
            }
        )
        return True, result

    lightop.hipblaslt_w8a8_channelwise_gemm = (
        hipblaslt_w8a8_channelwise_gemm
    )
    _install_fake_module_tree(
        monkeypatch,
        {
            _TARGET_MODULE: target,
            _TRITON_MODULE: triton,
            _LIGHTOP_MODULE: lightop,
        },
    )
    # Ordinary product tests exercise adapter ownership and eager contracts.
    # Keep the process-global torch.library namespace untouched; the strict
    # custom-op compile contract is verified in an isolated subprocess below.
    monkeypatch.setattr(
        scaled_mm,
        "_ensure_channel_fp8_custom_op",
        lambda backend: backend,
    )

    return SimpleNamespace(
        target=target,
        triton=triton,
        base_class=TorchFP8ScaledMMLinearKernel,
        channel_class=ChannelWiseTorchFP8ScaledMMLinearKernel,
        original_get_output_padding=original_get_output_padding,
        original_apply_scaled_mm=original_apply_scaled_mm,
        original_calls=original_calls,
        triton_calls=triton_calls,
        lightop_calls=lightop_calls,
        state=state,
    )


def _column_major_weight(k: int, n: int, *, dtype=None) -> torch.Tensor:
    dtype = _FP8_DTYPE if dtype is None else dtype
    return torch.zeros((n, k), dtype=dtype).t()


def _valid_call(
    *,
    m: int = 6,
    k: int = 4,
    n: int = 5,
    output_shape: tuple[int, ...] | list[int] | None = None,
) -> dict[str, object]:
    if output_shape is None:
        output_shape = (m, n)
    return {
        "A": torch.zeros((m, k), dtype=_FP8_DTYPE),
        "B": _column_major_weight(k, n),
        "As": torch.ones((m, 1), dtype=torch.float32),
        "Bs": torch.ones((n, 1), dtype=torch.float32),
        "out_dtype": torch.bfloat16,
        "bias": torch.arange(n, dtype=torch.bfloat16),
        "output_shape": output_shape,
    }


def _install_and_make_kernel(fake_product: SimpleNamespace):
    scaled_mm.install_fp8_scaled_mm_compat(fake_product.target)
    return fake_product.channel_class()


@pytest.fixture(autouse=True)
def _reset_python_custom_op_state():
    scaled_mm._reset_channel_fp8_custom_op_for_tests()
    yield
    scaled_mm._reset_channel_fp8_custom_op_for_tests()


def test_custom_op_registration_is_idempotent_and_backend_owned(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    def registered(*args):
        return args

    def register():
        calls.append("register")
        return registered

    def backend(*args, **kwargs):
        del args, kwargs

    monkeypatch.setattr(scaled_mm, "_register_channel_fp8_custom_op", register)
    assert scaled_mm._ensure_channel_fp8_custom_op(backend) is registered
    assert scaled_mm._ensure_channel_fp8_custom_op(backend) is registered
    assert calls == ["register"]
    assert scaled_mm._CUSTOM_OP_BACKEND is backend
    assert scaled_mm._CUSTOM_OP_REGISTRATION_ERROR is None

    with pytest.raises(RuntimeError, match="different backend"):
        scaled_mm._ensure_channel_fp8_custom_op(lambda: None)


def test_custom_op_registration_uses_vllm_dispatcher_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict[str, object]] = []

    def custom_op(*args):
        return args

    def direct_register_custom_op(**kwargs):
        calls.append(kwargs)

    torch_utils = ModuleType("vllm.utils.torch_utils")
    torch_utils.direct_register_custom_op = direct_register_custom_op
    _install_fake_module_tree(
        monkeypatch,
        {"vllm.utils.torch_utils": torch_utils},
    )
    monkeypatch.setattr(
        scaled_mm,
        "_resolve_channel_fp8_custom_op",
        lambda: custom_op,
    )

    assert scaled_mm._register_channel_fp8_custom_op() is custom_op
    assert calls == [
        {
            "op_name": scaled_mm._CUSTOM_OP_NAME,
            "op_func": scaled_mm._channel_fp8_target_triton_impl,
            "mutates_args": [],
            "fake_impl": scaled_mm._channel_fp8_target_triton_fake,
        }
    ]


def test_custom_op_registration_failure_is_latched(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    def fail_registration():
        calls.append("register")
        raise RuntimeError("deterministic registration failure")

    monkeypatch.setattr(
        scaled_mm,
        "_register_channel_fp8_custom_op",
        fail_registration,
    )
    def backend():
        return None
    with pytest.raises(RuntimeError, match="failed to register"):
        scaled_mm._ensure_channel_fp8_custom_op(backend)
    with pytest.raises(RuntimeError, match="previously failed"):
        scaled_mm._ensure_channel_fp8_custom_op(backend)
    assert calls == ["register"]
    assert scaled_mm._CUSTOM_OP is None
    assert scaled_mm._CUSTOM_OP_BACKEND is None
    assert "deterministic registration failure" in (
        scaled_mm._CUSTOM_OP_REGISTRATION_ERROR or ""
    )


def test_non_callable_backend_fails_without_mutating_registration_state():
    with pytest.raises(TypeError, match="backend must be callable"):
        scaled_mm._ensure_channel_fp8_custom_op(object())
    assert scaled_mm._CUSTOM_OP is None
    assert scaled_mm._CUSTOM_OP_BACKEND is None
    assert scaled_mm._CUSTOM_OP_REGISTRATION_ERROR is None


def test_custom_op_real_and_fake_contracts_delegate_without_algorithm_fork(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict[str, object]] = []

    def backend(input, weight, **kwargs):
        calls.append({"input": input, "weight": weight, **kwargs})
        return torch.zeros(
            (input.shape[0], weight.shape[1]),
            dtype=kwargs["out_dtype"],
            device=input.device,
        )

    monkeypatch.setattr(scaled_mm, "_CUSTOM_OP_BACKEND", backend)
    A = torch.zeros((3, 4), dtype=_FP8_DTYPE)
    B = _column_major_weight(4, 5)
    As = torch.ones((3, 1), dtype=torch.float32)
    Bs = torch.ones((5, 1), dtype=torch.float32)
    bias = torch.ones((5,), dtype=torch.bfloat16)

    real = scaled_mm._channel_fp8_target_triton_impl(
        A, B, As, Bs, torch.bfloat16, bias
    )
    fake = scaled_mm._channel_fp8_target_triton_fake(
        A, B, As, Bs, torch.float16, None
    )

    assert tuple(real.shape) == (3, 5)
    assert real.dtype == torch.bfloat16
    assert calls == [
        {
            "input": A,
            "weight": B,
            "scale_a": As,
            "scale_b": Bs,
            "out_dtype": torch.bfloat16,
            "bias": bias,
        }
    ]
    assert tuple(fake.shape) == (3, 5)
    assert fake.dtype == torch.float16
    assert fake.device == A.device
    assert fake.is_contiguous()
    assert fake.stride() == (5, 1)


@pytest.mark.parametrize("weight_scale_shape", [(), (1,), (1, 1)])
def test_custom_op_real_impl_accepts_target_per_tensor_weight_scale(
    monkeypatch: pytest.MonkeyPatch,
    weight_scale_shape: tuple[int, ...],
):
    calls: list[torch.Tensor] = []

    def backend(input, weight, **kwargs):
        calls.append(kwargs["scale_b"])
        return torch.zeros(
            (input.shape[0], weight.shape[1]),
            dtype=kwargs["out_dtype"],
            device=input.device,
        )

    monkeypatch.setattr(scaled_mm, "_CUSTOM_OP_BACKEND", backend)
    A = torch.zeros((3, 4), dtype=_FP8_DTYPE)
    B = _column_major_weight(4, 5)
    As = torch.ones((3, 1), dtype=torch.float32)
    Bs = torch.ones(weight_scale_shape, dtype=torch.float32)

    output = scaled_mm._channel_fp8_target_triton_impl(
        A, B, As, Bs, torch.bfloat16
    )

    assert tuple(output.shape) == (3, 5)
    assert calls == [Bs]


@pytest.mark.parametrize("scale_name", ["activation", "weight"])
def test_custom_op_real_impl_rejects_invalid_scale_relations(
    monkeypatch: pytest.MonkeyPatch,
    scale_name: str,
):
    calls: list[str] = []

    def backend(input, weight, **kwargs):
        calls.append("backend")
        return torch.zeros(
            (input.shape[0], weight.shape[1]),
            dtype=kwargs["out_dtype"],
            device=input.device,
        )

    monkeypatch.setattr(scaled_mm, "_CUSTOM_OP_BACKEND", backend)
    A = torch.zeros((3, 4), dtype=_FP8_DTYPE)
    B = _column_major_weight(4, 5)
    As = torch.ones((4, 1), dtype=torch.float32)
    Bs = torch.ones((5, 1), dtype=torch.float32)
    if scale_name == "weight":
        As = torch.ones((3, 1), dtype=torch.float32)
        Bs = torch.ones((6, 1), dtype=torch.float32)

    with pytest.raises(ValueError, match=scale_name):
        scaled_mm._channel_fp8_target_triton_impl(
            A, B, As, Bs, torch.bfloat16
        )
    assert calls == []


@pytest.mark.parametrize(
    "result_kind", ["non-tensor", "shape", "dtype", "device"]
)
def test_custom_op_real_impl_rejects_incompatible_backend_output(
    monkeypatch: pytest.MonkeyPatch,
    result_kind: str,
):
    def backend(input, weight, **kwargs):
        if result_kind == "non-tensor":
            return object()
        shape = (
            (input.shape[0], weight.shape[1] + 1)
            if result_kind == "shape"
            else (input.shape[0], weight.shape[1])
        )
        dtype = torch.float32 if result_kind == "dtype" else kwargs["out_dtype"]
        device = "meta" if result_kind == "device" else input.device
        return torch.zeros(shape, dtype=dtype, device=device)

    monkeypatch.setattr(scaled_mm, "_CUSTOM_OP_BACKEND", backend)
    A = torch.zeros((3, 4), dtype=_FP8_DTYPE)
    B = _column_major_weight(4, 5)
    As = torch.ones((3, 1), dtype=torch.float32)
    Bs = torch.ones((5, 1), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="incompatible output|return a tensor"):
        scaled_mm._channel_fp8_target_triton_impl(
            A, B, As, Bs, torch.bfloat16
        )


def test_install_accepts_only_the_exact_target_module(
    fake_product: SimpleNamespace,
):
    wrong_name = ModuleType("not.the.reviewed.target")
    wrong_name.ChannelWiseTorchFP8ScaledMMLinearKernel = (
        fake_product.channel_class
    )

    for candidate in (object(), wrong_name):
        with pytest.raises(RuntimeError, match="requires the exact vLLM module"):
            scaled_mm.install_fp8_scaled_mm_compat(candidate)

    assert not hasattr(fake_product.channel_class, "_hcu_fp8_patch_applied")
    scaled_mm.install_fp8_scaled_mm_compat(fake_product.target)
    assert fake_product.channel_class._hcu_fp8_patch_applied is True
    assert fake_product.channel_class._hcu_fp8_backend == "target-triton"


def test_install_without_argument_resolves_the_exact_target_module(
    fake_product: SimpleNamespace,
):
    scaled_mm.install_fp8_scaled_mm_compat()
    assert fake_product.channel_class._hcu_fp8_patch_applied is True


def test_install_is_idempotent_and_preserves_first_wrapper_ownership(
    fake_product: SimpleNamespace,
):
    scaled_mm.install_fp8_scaled_mm_compat(fake_product.target)
    first_padding = fake_product.channel_class.get_output_padding
    first_apply = fake_product.channel_class.apply_scaled_mm
    saved_padding = fake_product.channel_class._hcu_original_get_output_padding
    saved_apply = fake_product.channel_class._hcu_original_apply_scaled_mm

    scaled_mm.install_fp8_scaled_mm_compat(fake_product.target)
    scaled_mm.install_fp8_scaled_mm_compat()

    assert fake_product.channel_class.get_output_padding is first_padding
    assert fake_product.channel_class.apply_scaled_mm is first_apply
    assert (
        fake_product.channel_class._hcu_original_get_output_padding
        is saved_padding
        is fake_product.original_get_output_padding
    )
    assert (
        fake_product.channel_class._hcu_original_apply_scaled_mm
        is saved_apply
        is fake_product.original_apply_scaled_mm
    )


def test_existing_marker_without_reviewed_wrapper_identity_fails_closed(
    fake_product: SimpleNamespace,
):
    fake_product.channel_class._hcu_fp8_patch_applied = True
    fake_product.channel_class._hcu_fp8_backend = "target-triton"
    with pytest.raises(RuntimeError, match="without the reviewed HCU wrapper"):
        scaled_mm.install_fp8_scaled_mm_compat(fake_product.target)


def test_only_channelwise_kernel_loses_output_padding(
    fake_product: SimpleNamespace,
):
    base_padding = fake_product.base_class.get_output_padding
    base_apply = fake_product.base_class.apply_scaled_mm
    scaled_mm.install_fp8_scaled_mm_compat(fake_product.target)

    assert fake_product.base_class.get_output_padding is base_padding
    assert fake_product.base_class.apply_scaled_mm is base_apply
    assert fake_product.base_class().get_output_padding() == "target-padding"
    assert fake_product.channel_class().get_output_padding() is None
    assert "_hcu_rocm_no_output_padding" not in fake_product.base_class.__dict__
    assert fake_product.channel_class._hcu_rocm_no_output_padding is True


@pytest.mark.parametrize(
    "output_shape",
    [
        pytest.param((6, 5), id="2d"),
        pytest.param([2, 3, 5], id="3d"),
    ],
)
def test_channelwise_route_calls_target_triton_and_reshapes_output(
    fake_product: SimpleNamespace,
    output_shape: tuple[int, ...] | list[int],
):
    kernel = _install_and_make_kernel(fake_product)
    kwargs = _valid_call(output_shape=output_shape)

    output = kernel.apply_scaled_mm(**kwargs)

    assert tuple(output.shape) == tuple(output_shape)
    assert len(fake_product.triton_calls) == 1
    call = fake_product.triton_calls[0]
    assert call["input"] is kwargs["A"]
    assert call["weight"] is kwargs["B"]
    assert call["scale_a"] is kwargs["As"]
    assert call["scale_b"] is kwargs["Bs"]
    assert call["out_dtype"] is torch.bfloat16
    assert call["bias"] is kwargs["bias"]
    assert output.untyped_storage().data_ptr() == call[
        "result"
    ].untyped_storage().data_ptr()
    assert kwargs["B"].stride() == (1, kwargs["A"].shape[1])
    assert fake_product.original_calls == []


@pytest.mark.parametrize("value", [None, "1", "true"])
def test_channelwise_route_uses_lightop_by_default_or_when_enabled(
    fake_product: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
):
    if value is None:
        monkeypatch.delenv(
            "VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM",
            raising=False,
        )
    else:
        monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM", value)
    kernel = _install_and_make_kernel(fake_product)
    kwargs = _valid_call(output_shape=(2, 3, 5))
    kwargs["A"] = kwargs["A"].to(_LIGHTOP_FP8_DTYPE)
    kwargs["B"] = _column_major_weight(4, 5, dtype=_LIGHTOP_FP8_DTYPE)

    output = kernel.apply_scaled_mm(**kwargs)

    assert tuple(output.shape) == (2, 3, 5)
    assert fake_product.triton_calls == []
    assert len(fake_product.lightop_calls) == 1
    call = fake_product.lightop_calls[0]
    assert call["a"] is kwargs["A"]
    assert tuple(call["b"].shape) == (
        kwargs["B"].shape[1],
        kwargs["B"].shape[0],
    )
    assert call["b"].is_contiguous()
    assert call["b"].untyped_storage().data_ptr() == (
        kwargs["B"].untyped_storage().data_ptr()
    )
    assert call["scale_a"] is kwargs["As"]
    assert call["scale_b"] is kwargs["Bs"]
    assert (call["m"], call["n"], call["k"]) == (6, 5, 4)
    assert call["transpose_flag"] == "NT"
    assert call["out_dtype"] is torch.bfloat16
    assert call["bias"] is kwargs["bias"]
    assert output.untyped_storage().data_ptr() == (
        call["result"].untyped_storage().data_ptr()
    )
    assert fake_product.channel_class._hcu_fp8_backend == "lightop"


def test_lightop_backend_resolution_returns_stable_adapter(
    fake_product: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM", "1")

    first_name, first_backend = scaled_mm._resolve_scaled_mm_backend()
    second_name, second_backend = scaled_mm._resolve_scaled_mm_backend()

    assert first_name == second_name == "lightop"
    assert first_backend is second_backend


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("fp8-fnuz", "float8_e4m3fn"),
        ("scale-f16", "float32 scales"),
        ("output-f32", "float16 or bfloat16"),
        ("bias-mismatch", "bias dtype"),
    ],
)
def test_lightop_route_rejects_unsupported_dtypes_before_backend(
    fake_product: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    match: str,
):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM", "1")
    kernel = _install_and_make_kernel(fake_product)
    kwargs = _valid_call()
    kwargs["A"] = kwargs["A"].to(_LIGHTOP_FP8_DTYPE)
    kwargs["B"] = _column_major_weight(4, 5, dtype=_LIGHTOP_FP8_DTYPE)

    if case == "fp8-fnuz":
        kwargs["A"] = kwargs["A"].to(_FP8_DTYPE)
        kwargs["B"] = _column_major_weight(4, 5, dtype=_FP8_DTYPE)
    elif case == "scale-f16":
        kwargs["As"] = kwargs["As"].to(torch.float16)
        kwargs["Bs"] = kwargs["Bs"].to(torch.float16)
    elif case == "output-f32":
        kwargs["out_dtype"] = torch.float32
        kwargs["bias"] = kwargs["bias"].to(torch.float32)
    elif case == "bias-mismatch":
        kwargs["bias"] = kwargs["bias"].to(torch.float16)
    else:  # pragma: no cover - protects the parameter table itself
        raise AssertionError(f"unknown test case: {case}")

    with pytest.raises(ValueError, match=match):
        kernel.apply_scaled_mm(**kwargs)
    assert fake_product.lightop_calls == []


def test_channelwise_route_preserves_target_per_tensor_weight_scale(
    fake_product: SimpleNamespace,
):
    kernel = _install_and_make_kernel(fake_product)
    kwargs = _valid_call()
    kwargs["Bs"] = torch.ones((1, 1), dtype=torch.float32)

    output = kernel.apply_scaled_mm(**kwargs)

    assert tuple(output.shape) == (6, 5)
    assert len(fake_product.triton_calls) == 1
    assert fake_product.triton_calls[0]["scale_b"] is kwargs["Bs"]


@pytest.mark.parametrize("custom_quantization_gemm", ["0", "1"])
def test_channelwise_custom_op_keeps_full_profile_range_in_one_graph(
    custom_quantization_gemm: str,
):
    """Exercise production plumbing with an isolated test dispatch key."""

    probe = textwrap.dedent(
        r"""
        from __future__ import annotations

        import os
        import sys
        from types import ModuleType

        import torch
        from sympy.logic.boolalg import Boolean
        from torch._inductor import standalone_compile
        from torch.library import Library, infer_schema
        from vllm.compilation.backends import split_graph

        from vllm_hcu.runtime_compat import scaled_mm


        TARGET = "vllm.model_executor.kernels.linear.scaled_mm.pytorch"
        TRITON = (
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "triton_scaled_mm"
        )
        LIBRARIES = []
        REAL_CALLS = []
        USE_LIGHTOP = (
            os.environ["VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM"] == "1"
        )


        def package(name):
            module = ModuleType(name)
            module.__package__ = name
            module.__path__ = []
            return module


        def install_tree(leaves):
            package_names = set()
            for leaf_name in leaves:
                parts = leaf_name.split(".")
                package_names.update(
                    ".".join(parts[:index])
                    for index in range(1, len(parts))
                )
            modules = {name: package(name) for name in package_names}
            modules.update(leaves)
            for name, module in modules.items():
                sys.modules[name] = module
            for name, module in sorted(
                modules.items(), key=lambda item: item[0].count(".")
            ):
                if "." not in name:
                    continue
                parent_name, child_name = name.rsplit(".", 1)
                setattr(modules[parent_name], child_name, module)


        def direct_register_custom_op(
            op_name,
            op_func,
            mutates_args=None,
            fake_impl=None,
            **_kwargs,
        ):
            library = Library("vllm", "FRAGMENT")
            schema = infer_schema(op_func, mutates_args=mutates_args or [])
            library.define(op_name + schema)
            library.impl(
                op_name,
                op_func,
                dispatch_key="CompositeExplicitAutograd",
            )
            if fake_impl is not None:
                library._register_fake(op_name, fake_impl)
            LIBRARIES.append(library)


        class ChannelWiseTorchFP8ScaledMMLinearKernel:
            def get_output_padding(self):
                return "target-padding"

            def apply_scaled_mm(self, **_kwargs):
                raise AssertionError("legacy route must not execute")


        def triton_scaled_mm(
            input,
            weight,
            scale_a,
            scale_b,
            out_dtype,
            bias=None,
        ):
            del scale_a, scale_b, bias
            m = input.shape[0]
            if m <= 32:
                bucket = 32
            elif m <= 64:
                bucket = 64
            elif m <= 128:
                bucket = 128
            else:
                bucket = 256
            REAL_CALLS.append(("target-triton", m, bucket))
            return torch.zeros(
                (m, weight.shape[1]),
                dtype=out_dtype,
                device=input.device,
            )


        def hipblaslt_w8a8_channelwise_gemm(
            a,
            b,
            scale_a,
            scale_b,
            m,
            n,
            k,
            transpose_flag,
            out_dtype,
            bias,
        ):
            del scale_a, scale_b, bias
            assert transpose_flag == "NT"
            assert tuple(a.shape) == (m, k)
            assert tuple(b.shape) == (n, k)
            if m <= 32:
                bucket = 32
            elif m <= 64:
                bucket = 64
            elif m <= 128:
                bucket = 128
            else:
                bucket = 256
            REAL_CALLS.append(("lightop", m, bucket))
            return True, torch.zeros(
                (1, m, n),
                dtype=out_dtype,
                device=a.device,
            )


        target = ModuleType(TARGET)
        target.ChannelWiseTorchFP8ScaledMMLinearKernel = (
            ChannelWiseTorchFP8ScaledMMLinearKernel
        )
        triton = ModuleType(TRITON)
        triton.triton_scaled_mm = triton_scaled_mm
        lightop_gemm_ops = ModuleType("lightop.gemm_ops")
        lightop_gemm_ops.hipblaslt_w8a8_channelwise_gemm = (
            hipblaslt_w8a8_channelwise_gemm
        )
        torch_utils = ModuleType("vllm.utils.torch_utils")
        torch_utils.direct_register_custom_op = direct_register_custom_op
        install_tree(
            {
                TARGET: target,
                TRITON: triton,
                "lightop.gemm_ops": lightop_gemm_ops,
                "vllm.utils.torch_utils": torch_utils,
            }
        )

        scaled_mm.install_fp8_scaled_mm_compat(target)
        expected_backend = "lightop" if USE_LIGHTOP else "target-triton"
        assert (
            ChannelWiseTorchFP8ScaledMMLinearKernel._hcu_fp8_backend
            == expected_backend
        )
        kernel = ChannelWiseTorchFP8ScaledMMLinearKernel()
        graphs = []
        captured_graphs = []


        def product(A, B, As, Bs, bias):
            output = kernel.apply_scaled_mm(
                A=A,
                B=B,
                As=As,
                Bs=Bs,
                out_dtype=torch.bfloat16,
                bias=bias,
                output_shape=[A.shape[0], B.shape[1]],
            )
            # Use a real vLLM splitting op after the adapter.  This exercises
            # the production failure boundary where scalar graph outputs from
            # the producer become inputs to a standalone-compiled subgraph.
            return torch.ops.aten.sigmoid.default(output)


        def counting_backend(graph_module, example_inputs):
            graphs.append(str(graph_module.graph))
            captured_graphs.append((graph_module, list(example_inputs)))
            return graph_module.forward


        compiled = torch.compile(
            product,
            backend=counting_backend,
            fullgraph=True,
            dynamic=True,
        )
        fp8 = torch.float8_e4m3fn if USE_LIGHTOP else torch.float8_e4m3fnuz
        B = torch.zeros((5, 4), dtype=fp8).t()
        Bs = torch.ones((5, 1), dtype=torch.float32)
        bias = torch.ones((5,), dtype=torch.bfloat16)
        for m in (2, 33, 65, 129):
            A = torch.zeros((m, 4), dtype=fp8)
            As = torch.ones((m, 1), dtype=torch.float32)
            torch._dynamo.mark_dynamic(A, 0, min=1, max=10240)
            torch._dynamo.mark_dynamic(As, 0, min=1, max=10240)
            output = compiled(A, B, As, Bs, bias)
            assert tuple(output.shape) == (m, 5)
            assert output.dtype == torch.bfloat16

        assert len(graphs) == 1, graphs
        assert len(captured_graphs) == 1, captured_graphs
        assert "hcu_channel_fp8_target_triton_scaled_mm" in graphs[0]
        assert REAL_CALLS == [
            (expected_backend, 2, 32),
            (expected_backend, 33, 64),
            (expected_backend, 65, 128),
            (expected_backend, 129, 256),
        ]

        captured_graph, _ = captured_graphs[0]
        _, split_items = split_graph(captured_graph, ["aten::sigmoid"])
        producers = [
            item
            for item in split_items
            if not item.is_splitting_graph
            and "hcu_channel_fp8_target_triton_scaled_mm"
            in str(item.graph.graph)
        ]
        assert len(producers) == 1, [str(item.graph.graph) for item in split_items]
        producer = producers[0].graph

        symbolic_boolean_nodes = []
        for item in split_items:
            for node in item.graph.graph.nodes:
                value = node.meta.get("example_value")
                if isinstance(value, (torch.SymBool, Boolean)):
                    symbolic_boolean_nodes.append(
                        (item.submod_name, node.name, repr(value))
                    )
        assert symbolic_boolean_nodes == [], symbolic_boolean_nodes

        concrete_inputs = []
        for node in producer.graph.nodes:
            if node.op != "placeholder":
                continue
            value = node.meta.get("example_value")
            assert value is not None, node
            if isinstance(value, torch.Tensor):
                concrete_inputs.append(
                    torch.empty_strided(
                        tuple(int(dim) for dim in value.shape),
                        tuple(int(stride) for stride in value.stride()),
                        dtype=value.dtype,
                        device="cpu",
                    )
                )
            elif isinstance(value, torch.SymInt):
                concrete_inputs.append(int(value))
            elif isinstance(value, (bool, int, float)):
                concrete_inputs.append(value)
            else:
                raise AssertionError(
                    f"unsupported producer placeholder {node.name}: "
                    f"{type(value)!r}"
                )

        compiled_artifact = standalone_compile(
            producer,
            concrete_inputs,
            dynamic_shapes="from_example_inputs",
        )
        assert compiled_artifact is not None
        """
    )
    child_env = dict(os.environ)
    child_env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "VLLM_PLUGINS": "__disabled__",
            "VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM": custom_quantization_gemm,
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[2],
        env=child_env,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("a-rank", ValueError, "requires 2D A and B"),
        ("b-rank", ValueError, "requires 2D A and B"),
        ("zero-dimension", ValueError, "positive dimensions"),
        ("k-mismatch", ValueError, r"A=\[M,K\], B=\[K,N\]"),
        ("a-layout", ValueError, "A must be contiguous"),
        ("b-layout", ValueError, r"column-major \[K,N\]"),
        ("operand-dtype", ValueError, "share device and dtype"),
        ("not-fp8", ValueError, "requires FP8 A and B"),
        ("output-rank", ValueError, "output_shape must have at least 2 dims"),
        ("output-tokens", ValueError, "output_shape does not match"),
        ("output-channels", ValueError, "output_shape does not match"),
        ("output-dtype", TypeError, "output dtype must be floating point"),
    ],
)
def test_layout_and_shape_errors_fail_closed_before_triton(
    fake_product: SimpleNamespace,
    case: str,
    error: type[Exception],
    match: str,
):
    kernel = _install_and_make_kernel(fake_product)
    kwargs = _valid_call()
    m, k = kwargs["A"].shape
    n = kwargs["B"].shape[1]

    if case == "a-rank":
        kwargs["A"] = kwargs["A"].unsqueeze(0)
    elif case == "b-rank":
        kwargs["B"] = kwargs["B"].unsqueeze(0)
    elif case == "zero-dimension":
        kwargs["A"] = torch.empty((0, k), dtype=_FP8_DTYPE)
        kwargs["As"] = torch.empty((0, 1), dtype=torch.float32)
        kwargs["output_shape"] = (0, n)
    elif case == "k-mismatch":
        kwargs["B"] = _column_major_weight(k + 1, n)
    elif case == "a-layout":
        kwargs["A"] = torch.zeros((k, m), dtype=_FP8_DTYPE).t()
    elif case == "b-layout":
        kwargs["B"] = torch.zeros((k, n), dtype=_FP8_DTYPE)
    elif case == "operand-dtype":
        kwargs["B"] = _column_major_weight(k, n, dtype=torch.int8)
    elif case == "not-fp8":
        kwargs["A"] = torch.zeros((m, k), dtype=torch.float16)
        kwargs["B"] = _column_major_weight(k, n, dtype=torch.float16)
    elif case == "output-rank":
        kwargs["output_shape"] = (m * n,)
    elif case == "output-tokens":
        kwargs["output_shape"] = (m + 1, n)
    elif case == "output-channels":
        kwargs["output_shape"] = (m, n + 1)
    elif case == "output-dtype":
        kwargs["out_dtype"] = torch.int32
    else:  # pragma: no cover - protects the parameter table itself
        raise AssertionError(f"unknown test case: {case}")

    with pytest.raises(error, match=match):
        kernel.apply_scaled_mm(**kwargs)
    assert fake_product.triton_calls == []
    assert fake_product.original_calls == []


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("as-not-tensor", TypeError, "scales must be tensors"),
        ("bs-not-tensor", TypeError, "scales must be tensors"),
        ("as-shape", ValueError, "activation scale"),
        ("bs-shape", ValueError, "weight scale"),
        ("scale-dtype-mismatch", ValueError, "share a floating dtype"),
        ("scale-not-floating", ValueError, "share a floating dtype"),
    ],
)
def test_scale_errors_fail_closed_before_triton(
    fake_product: SimpleNamespace,
    case: str,
    error: type[Exception],
    match: str,
):
    kernel = _install_and_make_kernel(fake_product)
    kwargs = _valid_call()
    m = kwargs["A"].shape[0]
    n = kwargs["B"].shape[1]

    if case == "as-not-tensor":
        kwargs["As"] = 1.0
    elif case == "bs-not-tensor":
        kwargs["Bs"] = 1.0
    elif case == "as-shape":
        kwargs["As"] = torch.ones((m, 2), dtype=torch.float32)
    elif case == "bs-shape":
        kwargs["Bs"] = torch.ones((n + 1,), dtype=torch.float32)
    elif case == "scale-dtype-mismatch":
        kwargs["Bs"] = torch.ones((n, 1), dtype=torch.float64)
    elif case == "scale-not-floating":
        kwargs["As"] = torch.ones((m, 1), dtype=torch.int32)
        kwargs["Bs"] = torch.ones((n, 1), dtype=torch.int32)
    else:  # pragma: no cover - protects the parameter table itself
        raise AssertionError(f"unknown test case: {case}")

    with pytest.raises(error, match=match):
        kernel.apply_scaled_mm(**kwargs)
    assert fake_product.triton_calls == []
    assert fake_product.original_calls == []


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("not-tensor", TypeError),
        ("wrong-shape", ValueError),
        ("not-floating", ValueError),
    ],
)
def test_bias_errors_fail_closed_before_triton(
    fake_product: SimpleNamespace,
    case: str,
    error: type[Exception],
):
    kernel = _install_and_make_kernel(fake_product)
    kwargs = _valid_call()
    n = kwargs["B"].shape[1]

    if case == "not-tensor":
        kwargs["bias"] = 0.0
    elif case == "wrong-shape":
        kwargs["bias"] = torch.ones((n, 1), dtype=torch.float32)
    elif case == "not-floating":
        kwargs["bias"] = torch.ones((n,), dtype=torch.int32)
    else:  # pragma: no cover - protects the parameter table itself
        raise AssertionError(f"unknown test case: {case}")

    with pytest.raises(error, match="bias"):
        kernel.apply_scaled_mm(**kwargs)
    assert fake_product.triton_calls == []
    assert fake_product.original_calls == []


@pytest.mark.parametrize("bad_result", ["non-tensor", "shape", "dtype"])
def test_incompatible_triton_outputs_fail_closed_without_fallback(
    fake_product: SimpleNamespace,
    bad_result: str,
):
    kernel = _install_and_make_kernel(fake_product)
    kwargs = _valid_call()

    def bad_behavior(input, weight, scale_a, scale_b, out_dtype, bias):
        del scale_a, scale_b, bias
        if bad_result == "non-tensor":
            return object()
        if bad_result == "shape":
            return torch.empty(
                (input.shape[0], weight.shape[1] + 1), dtype=out_dtype
            )
        return torch.empty(
            (input.shape[0], weight.shape[1]), dtype=torch.float32
        )

    fake_product.state["behavior"] = bad_behavior
    with pytest.raises(RuntimeError, match="selected Channel-FP8 backend"):
        kernel.apply_scaled_mm(**kwargs)
    assert len(fake_product.triton_calls) == 1
    assert fake_product.original_calls == []


def test_triton_exception_propagates_without_legacy_fallback(
    fake_product: SimpleNamespace,
):
    kernel = _install_and_make_kernel(fake_product)
    failure = RuntimeError("deterministic target Triton failure")

    def raise_failure(input, weight, scale_a, scale_b, out_dtype, bias):
        del input, weight, scale_a, scale_b, out_dtype, bias
        raise failure

    fake_product.state["behavior"] = raise_failure
    with pytest.raises(RuntimeError, match="deterministic target Triton failure") as exc:
        kernel.apply_scaled_mm(**_valid_call())
    assert exc.value is failure
    assert len(fake_product.triton_calls) == 1
    assert fake_product.original_calls == []


def test_target_triton_signature_drift_fails_before_class_mutation(
    fake_product: SimpleNamespace,
):
    def drifted_triton_scaled_mm(
        input,
        weight,
        unexpected_scale,
        scale_b,
        out_dtype,
        bias=None,
    ):
        del input, weight, unexpected_scale, scale_b, out_dtype, bias

    fake_product.triton.triton_scaled_mm = drifted_triton_scaled_mm
    with pytest.raises(RuntimeError, match="signature drifted"):
        scaled_mm.install_fp8_scaled_mm_compat(fake_product.target)
    assert (
        fake_product.channel_class.get_output_padding
        is fake_product.original_get_output_padding
    )
    assert (
        fake_product.channel_class.apply_scaled_mm
        is fake_product.original_apply_scaled_mm
    )
    assert not hasattr(fake_product.channel_class, "_hcu_fp8_patch_applied")
