#!/usr/bin/env python3
"""Dedicated CPU source replay for the adapter-owned CT W8A8 INT8 empty-row method.

Written before the module moved, so the contract survives the move: the class is
imported from `hcu_adapter.dsv41_int8_linear` directly, driven by a real
compressed-tensors parse of a real checkpoint schema, real Column/Merged/Row
Linear constructors, real public weight loaders and the real native INT8
postprocess. Quantization and GEMM are explicit doubles that record their calls,
so "empty never reaches the kernel" is observable rather than asserted.

The plugin's historical module path must resolve to the same class object; that
is what lets the pre-existing replays keep passing without edits.

No pytest collection, GPU execution, install, service or network is involved.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hcu_adapter import (
    ct_w4a8_registration,
    dsv41_activation,
    dsv41_channel_fp8,
    dsv41_dspark_registration,
    dsv41_int8_linear as owned,
)
from vllm import _custom_ops as ops
from vllm.model_executor import parameter
from vllm.model_executor.kernels import linear as kernel_factory
from vllm.model_executor.kernels.linear.scaled_mm.aiter import AiterInt8ScaledMMLinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonInt8ScaledMMLinearKernel
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization import base_config
from vllm.model_executor.layers.quantization.compressed_tensors import compressed_tensors as ct
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_int8 as int8,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)

K, N = 16, 6
ALIGNED = 16
CALLS = []
KERNEL = [TritonInt8ScaledMMLinearKernel]
RESULTS = []
OWNED = "hcu_adapter.dsv41_int8_linear"
FACADE = "hcu_vllm_das.quantization.dsv41_int8_linear"


def assert_that(value, message="assertion failed"):
    if not value:
        raise AssertionError(message)


def case(name, callback):
    try:
        callback()
        RESULTS.append({"case": name, "status": "passed"})
    except Exception as error:
        RESULTS.append({"case": name, "status": "failed", "error": f"{type(error).__name__}: {error}"})


def refused(callback, text=None):
    try:
        callback()
    except (ValueError, RuntimeError, AssertionError, TypeError) as error:
        if text is not None:
            assert_that(text.lower() in str(error).lower(), f"{text!r} not in {error!r}")
        return
    raise AssertionError("malformed or unsupported input was accepted")


def ct_description(strategy="channel", static=False, symmetric=True):
    """A real compressed-tensors checkpoint schema, not a hand-built scheme."""
    weight = {"num_bits": 8, "type": "int", "symmetric": True, "dynamic": False, "strategy": strategy}
    act = {
        "num_bits": 8, "type": "int", "symmetric": symmetric, "dynamic": not static,
        "strategy": "tensor" if static else "token",
    }
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "linear": {"targets": ["Linear"], "format": "int-quantized", "weights": weight, "input_activations": act},
        },
        "ignore": [r"re:.*\.ignored$"],
    }


class _OwnedCT(ct.CompressedTensorsConfig):
    """Real CT parse and scheme construction; only the INT8 selector is ours.

    The adapter method is deliberately reached through a plain CT config here, so
    this replay proves the module's own contract instead of re-proving the
    plugin's CT W4A8 routing that the other replay already covers.
    """

    def get_quant_method(self, layer, prefix):
        method = super().get_quant_method(layer, prefix)
        if (
            isinstance(layer, linear.LinearBase)
            and type(method) is CompressedTensorsLinearMethod
            and type(getattr(layer, "scheme", None)) is CompressedTensorsW8A8Int8
        ):
            return owned.HcuDSV41Int8LinearMethod(self)
        return method


def config(**kwargs):
    return _OwnedCT.from_config(ct_description(**kwargs))


def select_kernel(**kwargs):
    layer_config = kernel_factory.Int8ScaledMMLinearLayerConfig(
        is_channelwise=kwargs["is_channelwise"],
        is_static_input_scheme=kwargs["is_static_input_scheme"],
        input_symmetric=kwargs["input_symmetric"],
        is_fused_module=kwargs.get("is_fused_module", False),
    )
    return KERNEL[0](layer_config, ["weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj"])


def quantize(x, scale=None, zp=None, symmetric=True):
    rows = x.numel() // x.shape[-1]
    if rows == 0:
        raise RuntimeError("empty input reached native quantization")
    CALLS.append("quant")
    data = x.reshape(rows, x.shape[-1]).to(torch.int8)
    scales = torch.ones((rows, 1), dtype=torch.float32) if scale is None else scale
    zero = None if symmetric else (torch.zeros((rows, 1), dtype=torch.int32) if zp is None else zp)
    return data, scales, zero


def gemm(x, w, xs, ws, out_dtype, bias=None):
    if x.shape[0] == 0:
        raise RuntimeError("empty input reached native GEMM")
    CALLS.append("gemm")
    out = (x.float() @ w.float()) * xs * ws.reshape(1, -1)
    return (out if bias is None else out + bias).to(out_dtype)


def build(kind="column", cfg=None, k=K, n=N, finalize=True, **kwargs):
    cfg = config() if cfg is None else cfg
    common = dict(
        quant_config=cfg, params_dtype=torch.bfloat16, bias=False,
        prefix="model.layers.0.wkv", disable_tp=True,
    )
    common.update(kwargs)
    if kind == "merged":
        obj = linear.MergedColumnParallelLinear(k, [n // 2, n - n // 2], **common)
    elif kind == "row":
        obj = linear.RowParallelLinear(k, n, **common)
    else:
        obj = linear.ColumnParallelLinear(k, n, **common)
    # Real CT layout: the loader hands (N, K) rows and native finalization transposes.
    weight = torch.arange(n * k, dtype=torch.int32).remainder(17).reshape(n, k).to(torch.int8)
    scales = torch.arange(1, n + 1, dtype=torch.float32).reshape(n, 1)
    if kind == "merged":
        for shard, width in enumerate(obj.output_partition_sizes):
            start = sum(obj.output_partition_sizes[:shard])
            obj.weight.weight_loader(obj.weight, weight[start:start + width], shard)
            shard_scale = scales[start:start + width] if obj.scheme.strategy == "channel" else torch.tensor(float(shard + 1))
            obj.weight_scale.weight_loader(obj.weight_scale, shard_scale, shard)
    else:
        obj.weight.weight_loader(obj.weight, weight)
        obj.weight_scale.weight_loader(
            obj.weight_scale, scales if obj.scheme.strategy == "channel" else torch.tensor(2.0)
        )
    if obj.input_scale is not None:
        obj.input_scale.weight_loader(obj.input_scale, torch.tensor([0.5]))
    if obj.input_zero_point is not None:
        obj.input_zero_point.weight_loader(obj.input_zero_point, torch.tensor([1], dtype=torch.int8))
    if finalize:
        obj.quant_method.process_weights_after_loading(obj)
    return obj, weight, scales


def empty_case(kind, shape, dtype, cfg=None):
    obj, _, _ = build(kind, cfg=cfg)
    before = len(CALLS)
    output, returned_bias = obj(torch.empty(shape, dtype=dtype))
    assert_that(output.shape == (*shape[:-1], N), output.shape)
    assert_that(output.dtype == dtype and output.device.type == "cpu")
    assert_that(output.numel() == 0 and returned_bias is None)
    assert_that(len(CALLS) == before, "empty input reached the native doubles")


def malformed(field, value, text=None):
    obj, _, _ = build()
    with patch.object(obj, field, value):
        refused(lambda: obj.quant_method.apply(obj, torch.empty((0, K), dtype=torch.bfloat16)), text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original = (
        ct.CompressedTensorsLinearMethod.apply,
        ct.CompressedTensorsLinearMethod.create_weights,
        ct.CompressedTensorsLinearMethod.process_weights_after_loading,
        int8.CompressedTensorsW8A8Int8.process_weights_after_loading,
        int8.CompressedTensorsW8A8Int8.apply_weights,
        kernel_factory.init_int8_linear_kernel,
        kernel_factory.choose_scaled_mm_linear_kernel,
        base_config.resolve_quant_method,
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(ct.CompressedTensorsConfig, "_check_scheme_supported", return_value=True))
        for cls in (TritonInt8ScaledMMLinearKernel, AiterInt8ScaledMMLinearKernel):
            stack.enter_context(patch.object(cls, "is_supported", return_value=(True, None)))
        stack.enter_context(patch.object(int8, "init_int8_linear_kernel", side_effect=select_kernel))
        stack.enter_context(patch.object(parameter, "get_tensor_model_parallel_rank", return_value=0))
        stack.enter_context(patch.object(parameter, "get_tensor_model_parallel_world_size", return_value=1))
        stack.enter_context(patch.object(linear, "get_tensor_model_parallel_rank", return_value=0))
        stack.enter_context(patch.object(linear, "get_tensor_model_parallel_world_size", return_value=1))
        stack.enter_context(patch.object(ops, "scaled_int8_quant", side_effect=quantize))
        stack.enter_context(patch.object(torch.ops.vllm, "int8_scaled_mm_func", side_effect=gemm))

        def surface():
            assert_that(owned.__name__ == OWNED)
            assert_that(inspect.getsourcefile(owned.HcuDSV41Int8LinearMethod) == inspect.getsourcefile(owned))
            assert_that(issubclass(owned.HcuDSV41Int8LinearMethod, CompressedTensorsLinearMethod))
            assert_that(owned.HcuDSV41Int8LinearMethod.__name__ in linear.WEIGHT_LOADER_V2_SUPPORTED)
            assert_that(owned.HcuDSV41Int8LinearMethod.supports_pre_processed_weights is False)
            assert_that(dataclasses.is_dataclass(owned._FinalizedINT8))
            fields = tuple(item.name for item in dataclasses.fields(owned._FinalizedINT8))
            assert_that(fields == ("scheme", "kernel", "weight", "shape", "stride", "data_ptr", "device", "dtype"), fields)
            with_ = owned._FinalizedINT8.__dataclass_params__
            assert_that(with_.frozen is True)
        case("module_surface_and_weight_loader_v2_registration", surface)

        def facade():
            import sys
            from hcu_vllm_das.quantization import dsv41_int8_linear as plugin_module
            assert_that(sys.modules[FACADE] is owned, "plugin path is not the adapter module")
            assert_that(plugin_module is owned)
            assert_that(plugin_module.HcuDSV41Int8LinearMethod is owned.HcuDSV41Int8LinearMethod)
            import hcu_vllm_das.dsv41_activation as plugin_activation
            import hcu_vllm_das.ct_w4a8_activation as plugin_ct_activation
            import hcu_vllm_das.dsv41_dspark_registration as plugin_registration
            from hcu_vllm_das.quantization import dsv41_channel_fp8 as plugin_channel
            for plugin_side, adapter_side in (
                (plugin_activation, dsv41_activation),
                (plugin_ct_activation, ct_w4a8_registration),
                (plugin_registration, dsv41_dspark_registration),
                (plugin_channel, dsv41_channel_fp8),
            ):
                assert_that(plugin_side is adapter_side, plugin_side.__name__)
        case("plugin_facades_resolve_to_the_same_adapter_modules", facade)

        def registration_seam():
            previous = dsv41_dspark_registration._plugin_reference
            dsv41_dspark_registration._plugin_reference = None
            try:
                refused(lambda: dsv41_dspark_registration._plugin(), "not bound")
                dsv41_dspark_registration.set_plugin_reference(("pkg.module", "Draft"))
                assert_that(dsv41_dspark_registration._plugin() == ("pkg.module", "Draft"))
                dsv41_dspark_registration.set_plugin_reference(("pkg.module", "Draft"))
                refused(
                    lambda: dsv41_dspark_registration.set_plugin_reference(("other.module", "Draft")),
                    "another owner",
                )
                refused(lambda: dsv41_dspark_registration.set_plugin_reference("pkg.module:Draft"), "pair")
            finally:
                dsv41_dspark_registration._plugin_reference = previous
        case("draft_model_reference_seam_refuses_missing_and_second_owner", registration_seam)

        def direction():
            root = Path(owned.__file__).resolve().parent
            modules = {
                "dsv41_int8_linear.py", "dsv41_channel_fp8.py", "dsv41_activation.py",
                "ct_w4a8_registration.py", "ct_w4a8_packed_loader.py",
                "dsv41_dspark_registration.py", "dsv41_draft_weight_alias.py",
                "channel_fp8_empty_linear.py",
            }
            for name in sorted(modules):
                text = (root / name).read_text()
                assert_that("hcu_vllm_das" not in text, f"{name} names the plugin package")
            # The direction rule is package-wide, not a module list:
            # `fp4_indexer` used to import the plugin's graph module, and that
            # edge now resolves through the adapter-owned contract. Scan every
            # adapter module so no new edge can hide outside the list above.
            named = sorted(str(p.relative_to(root)) for p in root.rglob("*.py")
                           if "hcu_vllm_das" in p.read_text())
            assert_that(not named, f"adapter modules name the plugin package: {named}")
            assert_that("draft_config" not in (root / "dsv41_draft_weight_alias.py").read_text())
        case("adapter_modules_never_reference_the_plugin_package", direction)

        def routing():
            cfg = config()
            obj, _, _ = build(cfg=cfg)
            assert_that(type(obj.scheme) is CompressedTensorsW8A8Int8)
            assert_that(type(obj.quant_method) is owned.HcuDSV41Int8LinearMethod)
            assert_that(obj.scheme.kernel.input_quant_key() is None)
            ignored = linear.ColumnParallelLinear(K, N, quant_config=cfg, prefix="model.ignored", disable_tp=True)
            assert_that(type(ignored.quant_method) is linear.UnquantizedLinearMethod)
            assert_that(config().target_scheme_map is not cfg.target_scheme_map)
        case("real_ct_parse_selects_the_owned_method", routing)

        def loading():
            for kind in ("column", "merged", "row"):
                obj, weight, scales = build(kind)
                assert_that(torch.equal(obj.weight, weight.t()))
                assert_that(obj.weight.stride() == weight.t().stride())
                assert_that(torch.equal(obj.weight_scale, scales))
                assert_that(obj.weight.shape == (K, N) and obj.weight.dtype == torch.int8)
                assert_that(obj._hcu_int8_empty_state == "ready")
                assert_that(obj._hcu_int8_empty_witness.weight is obj.weight)
        case("real_loaders_and_native_finalization_on_column_merged_row", loading)

        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for shape in ((0, K), (2, 0, K), (0, 3, K)):
                case(f"empty_{dtype}_{shape}", lambda shape=shape, dtype=dtype: empty_case("column", shape, dtype))
        for kind in ("merged", "row"):
            case("empty_" + kind, lambda kind=kind: empty_case(kind, (0, K), torch.bfloat16))
        for strategy in ("tensor", "channel"):
            for static, symmetric in ((False, True), (False, False), (True, True), (True, False)):
                case(
                    f"metadata_{strategy}_static{static}_symmetric{symmetric}",
                    lambda strategy=strategy, static=static, symmetric=symmetric: empty_case(
                        "column", (0, K), torch.bfloat16, config(strategy=strategy, static=static, symmetric=symmetric)
                    ),
                )
        case("fused_per_tensor_expansion", lambda: empty_case("merged", (0, K), torch.bfloat16, config(strategy="tensor")))

        def nonempty():
            obj, weight, scale = build()
            x = torch.ones((2, K), dtype=torch.bfloat16)
            before = len(CALLS)
            out, _ = obj(x)
            assert_that(torch.equal(out, ((x.float() @ weight.float().t()) * scale.t()).to(x.dtype)))
            assert_that(CALLS[before:] == ["quant", "gemm"])
            error = RuntimeError("native GEMM failure identity")
            with patch.object(torch.ops.vllm, "int8_scaled_mm_func", side_effect=error):
                try:
                    obj(x)
                except RuntimeError as caught:
                    assert_that(caught is error)
                else:
                    raise AssertionError("native exception swallowed")
        case("nonempty_delegates_to_native_quant_and_gemm", nonempty)

        def biased():
            obj, _, _ = build(bias=True, skip_bias_add=True)
            out, bias = obj(torch.empty((0, K), dtype=torch.bfloat16))
            assert_that(out.shape == (0, N) and bias is obj.bias)
            result = obj.quant_method.apply(
                obj, torch.empty((0, K), dtype=torch.bfloat16), torch.ones(N, dtype=torch.bfloat16)
            )
            assert_that(result.shape == (0, N) and result.dtype == torch.bfloat16)
            bad = [torch.ones(N + 1), torch.ones((1, N)), torch.ones(N, dtype=torch.int8),
                   torch.ones(N, dtype=torch.float16), torch.ones(N, device="meta")]
            for tensor in bad:
                refused(lambda tensor=tensor: obj.quant_method.apply(
                    obj, torch.empty((0, K), dtype=torch.bfloat16), tensor))
        case("valid_bias_contract_and_bias_rejections", biased)

        def bad_input(shape, dtype=torch.bfloat16, device="cpu"):
            obj, _, _ = build()
            refused(lambda: obj.quant_method.apply(obj, torch.empty(shape, dtype=dtype, device=device)))
        for name, shape, dtype, device in (
            ("K", (0, K + 1), torch.bfloat16, "cpu"),
            ("zero_K", (0, 0), torch.bfloat16, "cpu"),
            ("rank1", (0,), torch.bfloat16, "cpu"),
            ("dtype", (0, K), torch.int8, "cpu"),
            ("device", (0, K), torch.bfloat16, "meta"),
        ):
            case("reject_input_" + name, lambda shape=shape, dtype=dtype, device=device: bad_input(shape, dtype, device))
        case("reject_unprocessed_rectangular", lambda: refused(lambda: build(finalize=False)[0](torch.empty((0, K), dtype=torch.bfloat16))))
        case("reject_unprocessed_square", lambda: refused(lambda: build(k=N, finalize=False)[0](torch.empty((0, N), dtype=torch.bfloat16))))
        case("reject_changed_local_width", lambda: malformed("output_partition_sizes", [N + 1], "partitions"))

        def scale_guards():
            malformed("weight_scale", torch.nn.Parameter(torch.ones((N, 2)), requires_grad=False))
            malformed("weight_scale", torch.nn.Parameter(torch.ones((N, 1), dtype=torch.float16), requires_grad=False))
            malformed("weight_scale", torch.nn.Parameter(torch.ones((N, 1), device="meta"), requires_grad=False))
            malformed("weight_scale", torch.nn.Parameter(torch.ones((N, 1)).t(), requires_grad=False))
        case("reject_scale_shape_dtype_device_layout", scale_guards)

        def layout_guard():
            obj, _, _ = build()
            with patch.object(owned, "is_weak_contiguous", return_value=False):
                refused(lambda: obj(torch.empty((0, K), dtype=torch.bfloat16)), "weak-contiguous")
        case("reject_weight_layout", layout_guard)

        def witness():
            obj, _, _ = build()
            with patch.object(obj, "weight", torch.nn.Parameter(torch.empty((K, N + 1), dtype=torch.int8), requires_grad=False)):
                refused(lambda: obj(torch.empty((0, K), dtype=torch.bfloat16)), "witness")
            obj, _, _ = build()
            with patch.object(obj, "scheme", copy.copy(obj.scheme)):
                refused(lambda: obj(torch.empty((0, K), dtype=torch.bfloat16)), "witness")
            obj, _, _ = build()
            with patch.object(obj, "_hcu_int8_empty_state", "failed"):
                refused(lambda: obj(torch.empty((0, K), dtype=torch.bfloat16)), "postprocess")
        case("reject_postprocess_witness_mismatch", witness)

        def repeat():
            obj, _, _ = build(k=N)
            previous = obj.weight.clone()
            refused(lambda: obj.quant_method.process_weights_after_loading(obj), "postprocess")
            assert_that(torch.equal(obj.weight, previous))
            assert_that(obj._hcu_int8_empty_state == "failed")
            refused(lambda: obj(torch.empty((0, N), dtype=torch.bfloat16)))
        case("repeat_postprocess_refused_before_second_transpose", repeat)

        def failed():
            obj, _, _ = build(finalize=False)
            error = RuntimeError("native postprocess failed")
            with patch.object(TritonInt8ScaledMMLinearKernel, "process_weights_after_loading", side_effect=error):
                try:
                    obj.quant_method.process_weights_after_loading(obj)
                except RuntimeError as caught:
                    assert_that(caught is error)
                else:
                    raise AssertionError("native failure swallowed")
            refused(lambda: obj(torch.empty((0, K), dtype=torch.bfloat16)))
            refused(lambda: obj.quant_method.process_weights_after_loading(obj))
        case("failed_postprocess_propagates_and_invalidates", failed)

        def aux_invalid(static, symmetric, field, tensor):
            obj, _, _ = build(cfg=config(static=static, symmetric=symmetric))
            with patch.object(obj, field, tensor):
                refused(lambda: obj(torch.empty((0, K), dtype=torch.bfloat16)))
        for name, static, symmetric, field, tensor in (
            ("dynamic_scale", False, True, "input_scale", torch.nn.Parameter(torch.ones(1), requires_grad=False)),
            ("static_scale", True, True, "input_scale", torch.nn.Parameter(torch.ones(2), requires_grad=False)),
            ("asym_zp", True, False, "input_zero_point", torch.nn.Parameter(torch.ones(1, dtype=torch.int8), requires_grad=False)),
            ("asym_adjustment", False, False, "azp_adj", torch.nn.Parameter(torch.ones((1, N), dtype=torch.float32), requires_grad=False)),
        ):
            case(
                "reject_aux_" + name,
                lambda static=static, symmetric=symmetric, field=field, tensor=tensor: aux_invalid(
                    static, symmetric, field, tensor
                ),
            )

        def quantized():
            obj, _, _ = build()
            for rows in (0, 2):
                qa = QuantizedActivation(
                    torch.empty((rows, K), dtype=torch.int8),
                    torch.ones((rows, 1)), torch.bfloat16, torch.Size((rows, K)), None,
                )
                refused(lambda qa=qa: obj.quant_method.apply(obj, qa), "QuantizedActivation")
        case("reject_quantized_activation", quantized)

        def aiter():
            previous = KERNEL[0]
            KERNEL[0] = AiterInt8ScaledMMLinearKernel
            reference_finalize = TritonInt8ScaledMMLinearKernel.process_weights_after_loading
            try:
                with patch.object(AiterInt8ScaledMMLinearKernel, "process_weights_after_loading", reference_finalize):
                    obj, _, _ = build(k=ALIGNED, n=ALIGNED, cfg=config(strategy="channel"))
                    assert_that(obj._hcu_int8_empty_state == "ready")
                    before = len(CALLS)
                    out, _ = obj(torch.empty((0, ALIGNED), dtype=torch.bfloat16))
                    assert_that(out.shape == (0, ALIGNED) and len(CALLS) == before)
                    # fp32 sits outside the AITER FP16/BF16 contract.
                    refused(lambda: obj(torch.empty((0, ALIGNED), dtype=torch.float32)), "aiter")
                    # N must be AITER-aligned; the kernel still finalizes, the
                    # adapter guard is what refuses the empty fast path.
                    unaligned, _, _ = build(k=ALIGNED, n=ALIGNED + 1, cfg=config(strategy="channel"))
                    refused(lambda: unaligned(torch.empty((0, ALIGNED), dtype=torch.bfloat16)), "aiter")
            finally:
                KERNEL[0] = previous
            # Static input, asymmetric activation and per-tensor weights are
            # refused by the AITER kernel during construction, so the adapter's
            # matching guard is defence in depth and cannot be reached through
            # the public constructor. Pin that ordering instead of faking a layer.
            for channelwise, static, symmetric in ((True, True, True), (True, False, False), (False, False, True)):
                layer_config = kernel_factory.Int8ScaledMMLinearLayerConfig(
                    is_channelwise=channelwise, is_static_input_scheme=static,
                    input_symmetric=symmetric, is_fused_module=False,
                )
                assert_that(AiterInt8ScaledMMLinearKernel.can_implement(layer_config)[0] is False)
        case("native_aiter_alignment_and_contract", aiter)

        def identity():
            current = (
                ct.CompressedTensorsLinearMethod.apply,
                ct.CompressedTensorsLinearMethod.create_weights,
                ct.CompressedTensorsLinearMethod.process_weights_after_loading,
                int8.CompressedTensorsW8A8Int8.process_weights_after_loading,
                int8.CompressedTensorsW8A8Int8.apply_weights,
                kernel_factory.init_int8_linear_kernel,
                kernel_factory.choose_scaled_mm_linear_kernel,
                base_config.resolve_quant_method,
            )
            assert_that(current == original)
            obj, _, _ = build()
            assert_that(type(obj.quant_method).__init__ is CompressedTensorsLinearMethod.__init__)
            assert_that(type(obj.quant_method).create_weights is CompressedTensorsLinearMethod.create_weights)
        case("official_factories_and_inherited_creation_are_untouched", identity)

    assert_that(not torch.cuda.is_initialized())
    assert_that(all(os.environ.get(name) == "" for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")))
    root = Path(__file__).resolve().parents[2]
    sha = lambda path: subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    report = {
        "level": "CPU source integration; hardware admission/quantization/GEMM doubled",
        "plugin": sha(root),
        "core": sha(root.parent / "vllm"),
        "adapter": sha(root.parent / "adapter-das"),
        "module": OWNED,
        "facade": FACADE,
        "module_sha256": hashlib.sha256(Path(owned.__file__).read_bytes()).hexdigest(),
        "cuda_initialized": torch.cuda.is_initialized(),
        "cases": RESULTS,
        "passed": sum(item["status"] == "passed" for item in RESULTS),
        "total": len(RESULTS),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
