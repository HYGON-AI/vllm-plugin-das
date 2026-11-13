#!/usr/bin/env python3
"""Fixed CPU source replay, written before the independent INT8 implementation.

Real config/CT parser, linear constructors, parameter loaders and native INT8
postprocessing; hardware admission, quantizer and GEMM are controlled doubles.
No pytest collection, GPU execution, install, service or network is involved.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hcu_adapter.extensions import ExtensionContext
from hcu_vllm_das import ct_w4a8_activation as activation
from hcu_vllm_das.quantization import ct_w4a8 as candidate
from vllm import _custom_ops as ops
from vllm.model_executor import parameter
from vllm.model_executor.kernels import linear as kernel_factory
from vllm.model_executor.kernels.linear.scaled_mm.aiter import AiterInt8ScaledMMLinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassInt8ScaledMMLinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonInt8ScaledMMLinearKernel
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization import base_config
from vllm.model_executor.layers.quantization.compressed_tensors import compressed_tensors as ct
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import compressed_tensors_w8a8_int8 as int8
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import CompressedTensorsW8A8Fp8
from vllm.model_executor.layers.quantization.compressed_tensors.utils import find_matched_target

K, N = 16, 6
RESULTS = []
CALLS = []
KERNEL = TritonInt8ScaledMMLinearKernel


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
            assert_that(text.lower() in str(error).lower(), str(error))
        return
    raise AssertionError("malformed/unsupported input was accepted")


def description(strategy="channel", static=False, symmetric=True):
    weight = {"num_bits": 8, "type": "int", "symmetric": True, "dynamic": False, "strategy": strategy}
    act = {"num_bits": 8, "type": "int", "symmetric": symmetric, "dynamic": not static, "strategy": "tensor" if static else "token"}
    return {
        "quant_method": "compressed-tensors", "format": "pack-quantized",
        "config_groups": {
            "expert": {"targets": [r"re:.*\.experts\..*"], "format": "pack-quantized", "weights": dict(weight, num_bits=4), "input_activations": dict(act, dynamic=True, strategy="token", symmetric=True)},
            "fp8": {"targets": [r"re:.*\.fp8$"], "format": "float-quantized", "weights": dict(weight, type="float"), "input_activations": dict(act, type="float", dynamic=True, strategy="token", symmetric=True)},
            "linear": {"targets": ["Linear"], "format": "int-quantized", "weights": weight, "input_activations": act},
        },
        "ignore": [r"re:.*\.ignored$"],
    }


def config(**kwargs):
    return candidate.HcuDSV41CTW4A8Config.from_config(description(**kwargs))


def select_kernel(**kwargs):
    c = kernel_factory.Int8ScaledMMLinearLayerConfig(
        is_channelwise=kwargs["is_channelwise"], is_static_input_scheme=kwargs["is_static_input_scheme"],
        input_symmetric=kwargs["input_symmetric"], is_fused_module=kwargs.get("is_fused_module", False),
    )
    return KERNEL(c, ["weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj"])


def quantize(x, scale=None, zp=None, symmetric=True):
    m = x.numel() // x.shape[-1]
    if m == 0:
        raise RuntimeError("empty input reached native quantization")
    CALLS.append("quant")
    q = x.reshape(m, x.shape[-1]).to(torch.int8)
    s = torch.ones((m, 1), dtype=torch.float32) if scale is None else scale
    z = None if symmetric else (torch.zeros((m, 1), dtype=torch.int32) if zp is None else zp)
    return q, s, z


def gemm(x, w, xs, ws, out_dtype, bias=None):
    if x.shape[0] == 0:
        raise RuntimeError("empty input reached native GEMM")
    CALLS.append("gemm")
    out = (x.float() @ w.float()) * xs * ws.reshape(1, -1)
    return (out if bias is None else out + bias).to(out_dtype)


def build(kind="column", cfg=None, k=K, n=N, finalize=True, **kwargs):
    cfg = config() if cfg is None else cfg
    common = dict(quant_config=cfg, params_dtype=torch.bfloat16, bias=False, prefix="model.layers.0.engram.wkv", disable_tp=True)
    common.update(kwargs)
    if kind == "merged":
        obj = linear.MergedColumnParallelLinear(k, [n // 2, n - n // 2], **common)
    elif kind == "row":
        obj = linear.RowParallelLinear(k, n, **common)
    else:
        obj = linear.ColumnParallelLinear(k, n, **common)
    weight = torch.arange(n * k, dtype=torch.int32).remainder(17).reshape(n, k).to(torch.int8)
    scales = torch.arange(1, n + 1, dtype=torch.float32).reshape(n, 1)
    # Real public parameter loaders; merged loading must use projection IDs.
    if kind == "merged":
        for shard, width in enumerate(obj.output_partition_sizes):
            start = sum(obj.output_partition_sizes[:shard])
            obj.weight.weight_loader(obj.weight, weight[start:start + width], shard)
            scale = scales[start:start + width] if obj.scheme.strategy == "channel" else torch.tensor(float(shard + 1))
            obj.weight_scale.weight_loader(obj.weight_scale, scale, shard)
    else:
        obj.weight.weight_loader(obj.weight, weight)
        obj.weight_scale.weight_loader(obj.weight_scale, scales if obj.scheme.strategy == "channel" else torch.tensor(2.0))
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
    assert_that(output.shape == (*shape[:-1], N) and output.dtype == dtype and output.device.type == "cpu")
    assert_that(output.numel() == 0 and returned_bias is None and len(CALLS) == before)


def malformed(field, value, text=None):
    obj, _, _ = build()
    with patch.object(obj, field, value):
        refused(lambda: obj.quant_method.apply(obj, torch.empty((0, K), dtype=torch.bfloat16)), text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original = (ct.CompressedTensorsLinearMethod.apply, ct.CompressedTensorsLinearMethod.create_weights,
                ct.CompressedTensorsLinearMethod.process_weights_after_loading, int8.CompressedTensorsW8A8Int8.process_weights_after_loading,
                int8.CompressedTensorsW8A8Int8.apply_weights, kernel_factory.init_int8_linear_kernel,
                kernel_factory.choose_scaled_mm_linear_kernel, base_config.resolve_quant_method)
    with ExitStack() as stack:
        stack.enter_context(patch.object(candidate, "_is_hcu", return_value=True))
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
        activation.activate_extension(ExtensionContext("0.1.0", "0.33.0"))

        def routing():
            cfg = config()
            obj, _, _ = build(cfg=cfg)
            assert_that(type(obj.scheme) is int8.CompressedTensorsW8A8Int8)
            assert_that(type(obj.quant_method).__name__ == "HcuDSV41Int8LinearMethod")
            assert_that(type(obj.quant_method).__name__ in linear.WEIGHT_LOADER_V2_SUPPORTED)
            assert_that(not obj.quant_method.supports_pre_processed_weights)
            assert_that(obj.scheme.kernel.input_quant_key() is None)
            ignored = linear.ColumnParallelLinear(K, N, quant_config=cfg, prefix="model.ignored", disable_tp=True)
            assert_that(type(ignored.quant_method) is linear.UnquantizedLinearMethod)
            # Avoid constructing an FP8 kernel: parser/get_quant_method is real.
            layer = linear.ColumnParallelLinear(K, N, quant_config=None, disable_tp=True)
            with patch("vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8.get_current_vllm_config", return_value=SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))):
                fp8_method = cfg.get_quant_method(layer, "model.fp8")
            assert_that(type(layer.scheme) is CompressedTensorsW8A8Fp8)
            assert_that(type(fp8_method).__name__ == "HcuDSV41ChannelFP8LinearMethod")
            raw = copy.deepcopy(description())
            raw["config_groups"]["linear"]["format"] = "pack-quantized"
            packed_cfg = candidate.HcuDSV41CTW4A8Config.from_config(raw)
            assert_that(packed_cfg.get_scheme(layer, "model.packed").__class__.__name__ == "CompressedTensorsWNA8Int")
            assert_that(find_matched_target("model.experts.0.gate_proj", layer, cfg.target_scheme_map) == r"re:.*\.experts\..*")
            assert_that(config().target_scheme_map is not cfg.target_scheme_map)
        case("mixed_real_config_scheme_and_owned_routing", routing)

        def loading():
            for kind in ("column", "merged", "row"):
                obj, weight, scales = build(kind)
                assert_that(torch.equal(obj.weight, weight.t()) and obj.weight.stride() == weight.t().stride())
                assert_that(torch.equal(obj.weight_scale, scales))
                assert_that(obj.weight.shape == (K, N) and obj.weight.dtype == torch.int8)
        case("real_v2_direct_merged_row_loading_and_native_finalization", loading)

        def rank_loading():
            with patch.object(parameter, "get_tensor_model_parallel_rank", return_value=1), patch.object(parameter, "get_tensor_model_parallel_world_size", return_value=2):
                cfg = config()
                obj = linear.ColumnParallelLinear(K, N * 2, quant_config=cfg, tp_rank=1, tp_size=2, bias=False)
                raw = torch.arange(N * 2 * K).remainder(17).reshape(N * 2, K).to(torch.int8)
                obj.weight.weight_loader(obj.weight, raw)
                obj.weight_scale.weight_loader(obj.weight_scale, torch.ones((N * 2, 1)))
                assert_that(torch.equal(obj.weight, raw[N:]))
                obj.quant_method.process_weights_after_loading(obj)
                assert_that(torch.equal(obj.weight, raw[N:].t()))
                replicated, weight, _ = build()
                assert_that(replicated.tp_rank == 0 and replicated.tp_size == 1 and torch.equal(replicated.weight, weight.t()))
        case("real_nonzero_tp_rank_and_disable_tp_loading", rank_loading)

        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for shape in ((0, K), (2, 0, K), (0, 3, K)):
                case(f"empty_tensor_{dtype}_{shape}", lambda shape=shape, dtype=dtype: empty_case("column", shape, dtype))
        for kind in ("merged", "row"):
            case("empty_" + kind, lambda kind=kind: empty_case(kind, (0, K), torch.bfloat16))
        for strategy in ("tensor", "channel"):
            for static, symmetric in ((False, True), (False, False), (True, True), (True, False)):
                name = f"metadata_{strategy}_static{static}_symmetric{symmetric}"
                case(name, lambda strategy=strategy, static=static, symmetric=symmetric: empty_case("column", (0, K), torch.bfloat16, config(strategy=strategy, static=static, symmetric=symmetric)))
        case("fused_per_tensor_expansion", lambda: empty_case("merged", (0, K), torch.bfloat16, config(strategy="tensor")))

        def aiter():
            global KERNEL
            KERNEL = AiterInt8ScaledMMLinearKernel
            try:
                obj, _, _ = build(k=16, n=16)
                before = len(CALLS)
                out, _ = obj(torch.empty((0, 16), dtype=torch.bfloat16))
                assert_that(out.shape == (0, 16) and len(CALLS) == before)
            finally:
                KERNEL = TritonInt8ScaledMMLinearKernel
        case("native_aiter_postprocess_empty", aiter)

        def collectives():
            obj, _, _ = build(gather_output=True)
            obj.tp_size = 2
            with patch.object(linear, "tensor_model_parallel_all_gather", side_effect=lambda x: torch.cat((x, x), dim=-1)) as gather:
                out, _ = obj(torch.empty((0, K), dtype=torch.bfloat16))
                assert_that(out.shape == (0, 2 * N) and gather.call_count == 1)
            row, _, _ = build("row")
            row.tp_size = 2
            with patch.object(linear, "tensor_model_parallel_all_reduce", side_effect=lambda x: x) as reduce:
                out, _ = row(torch.empty((0, K), dtype=torch.bfloat16))
                assert_that(out.shape == (0, N) and reduce.call_count == 1)
        case("outer_column_gather_and_row_reduce_remain", collectives)

        def nonempty():
            obj, weight, scale = build()
            x = torch.ones((2, K), dtype=torch.bfloat16)
            before = len(CALLS)
            out, _ = obj(x)
            expected = (x.float() @ weight.float().t()) * scale.t()
            assert_that(torch.equal(out, expected.to(x.dtype)) and CALLS[before:] == ["quant", "gemm"])
            error = RuntimeError("native GEMM failure identity")
            with patch.object(torch.ops.vllm, "int8_scaled_mm_func", side_effect=error):
                try:
                    obj(x)
                except RuntimeError as caught:
                    assert_that(caught is error)
                else:
                    raise AssertionError("native exception swallowed")
        case("nonempty_native_execution_and_exception_identity", nonempty)

        def biased():
            obj, _, _ = build(bias=True, skip_bias_add=True)
            out, bias = obj(torch.empty((0, K), dtype=torch.bfloat16))
            assert_that(out.shape == (0, N) and bias is obj.bias)
            result = obj.quant_method.apply(obj, torch.empty((0, K), dtype=torch.bfloat16), torch.ones(N, dtype=torch.bfloat16))
            assert_that(result.shape == (0, N))
        case("valid_bias_and_return_bias_contract", biased)

        def bad_input(shape, dtype=torch.bfloat16, device="cpu"):
            obj, _, _ = build()
            refused(lambda: obj.quant_method.apply(obj, torch.empty(shape, dtype=dtype, device=device)))
        for name, shape, dtype, device in (("K", (0, K + 1), torch.bfloat16, "cpu"), ("zero_K", (0, 0), torch.bfloat16, "cpu"), ("rank1", (0,), torch.bfloat16, "cpu"), ("dtype", (0, K), torch.int8, "cpu"), ("device", (0, K), torch.bfloat16, "meta")):
            case("reject_input_" + name, lambda shape=shape, dtype=dtype, device=device: bad_input(shape, dtype, device))
        case("reject_unprocessed_rectangular", lambda: refused(lambda: build(finalize=False)[0](torch.empty((0, K), dtype=torch.bfloat16))))
        case("reject_unprocessed_square", lambda: refused(lambda: build(k=N, finalize=False)[0](torch.empty((0, N), dtype=torch.bfloat16))))
        case("reject_changed_local_width", lambda: malformed("output_partition_sizes", [N + 1]))
        case("reject_weight_dtype", lambda: malformed("weight", torch.nn.Parameter(torch.empty((K, N)), requires_grad=False)))
        case("reject_scale_shape", lambda: malformed("weight_scale", torch.nn.Parameter(torch.ones((N, 2)), requires_grad=False)))
        case("reject_scale_dtype", lambda: malformed("weight_scale", torch.nn.Parameter(torch.ones((N, 1), dtype=torch.float16), requires_grad=False)))
        case("reject_scale_device", lambda: malformed("weight_scale", torch.nn.Parameter(torch.ones((N, 1), device="meta"), requires_grad=False)))

        def qa_rejected():
            obj, _, _ = build()
            for rows in (0, 2):
                qa = QuantizedActivation(torch.empty((rows, K), dtype=torch.int8), torch.ones((rows, 1)), torch.bfloat16, torch.Size((rows, K)), None)
                refused(lambda: obj.quant_method.apply(obj, qa), "QuantizedActivation")
        case("reject_all_quantized_activations_including_None_key", qa_rejected)

        def repeat():
            obj, _, _ = build(k=N)
            previous = obj.weight.clone()
            refused(lambda: obj.quant_method.process_weights_after_loading(obj), "postprocess")
            assert_that(torch.equal(obj.weight, previous))
            refused(lambda: obj(torch.empty((0, N), dtype=torch.bfloat16)))
        case("repeat_postprocess_refused_before_second_transpose_and_invalidates", repeat)

        def failed_process():
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
        case("failed_postprocess_invalidates_and_cannot_retry", failed_process)

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
            case("reject_aux_" + name, lambda static=static, symmetric=symmetric, field=field, tensor=tensor: aux_invalid(static, symmetric, field, tensor))
        def bad_bias():
            obj, _, _ = build()
            for bias in (torch.ones(N + 1), torch.ones((1, N)), torch.ones(N, dtype=torch.int8), torch.ones(N, dtype=torch.float16), torch.ones(N, device="meta")):
                refused(lambda: obj.quant_method.apply(obj, torch.empty((0, K), dtype=torch.bfloat16), bias))
        case("reject_bias_shape_dtype_device", bad_bias)

        def unknown_kernel():
            obj, _, _ = build()
            class ForeignKernel:
                input_quant_key = lambda self: None
                def apply_weights(self, layer, x, bias):
                    raise RuntimeError("foreign kernel empty is unsupported")
            obj.scheme.kernel = ForeignKernel()
            refused(lambda: obj(torch.empty((0, K), dtype=torch.bfloat16)))
        case("reject_changed_or_unknown_kernel", unknown_kernel)

        def identity():
            current = (ct.CompressedTensorsLinearMethod.apply, ct.CompressedTensorsLinearMethod.create_weights,
                       ct.CompressedTensorsLinearMethod.process_weights_after_loading, int8.CompressedTensorsW8A8Int8.process_weights_after_loading,
                       int8.CompressedTensorsW8A8Int8.apply_weights, kernel_factory.init_int8_linear_kernel,
                       kernel_factory.choose_scaled_mm_linear_kernel, base_config.resolve_quant_method)
            assert_that(current == original)
            obj, _, _ = build()
            assert_that(type(obj.quant_method).__init__ is ct.CompressedTensorsLinearMethod.__init__)
            assert_that(type(obj.quant_method).create_weights is ct.CompressedTensorsLinearMethod.create_weights)
        case("official_factory_selector_CT_methods_and_inherited_creation_identity", identity)
    assert_that(not torch.cuda.is_initialized())
    assert_that(all(os.environ.get(name) == "" for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")))
    root = Path(__file__).resolve().parents[2]
    sha = lambda path: subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    report = {"level": "CPU source integration; hardware admission/quantization/GEMM doubled", "plugin": sha(root), "core": sha(root.parent / "vllm"),
              "core_source": inspect.getsourcefile(int8.CompressedTensorsW8A8Int8), "cuda_initialized": torch.cuda.is_initialized(),
              "cases": RESULTS, "passed": sum(item["status"] == "passed" for item in RESULTS), "total": len(RESULTS)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
