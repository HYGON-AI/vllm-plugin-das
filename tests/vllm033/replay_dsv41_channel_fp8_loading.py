#!/usr/bin/env python3
"""Repeatable CPU integration replay; hardware admission and GEMM are doubles.

Written before the empty-row adapter. Uses real CT parameter creation, merged
projection loading, V41 target mapper/loader and CT weight finalization. Outputs
JSON with every case; no model construction, GPU, install or network downloads.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hcu_adapter.extensions import ExtensionContext
from hcu_vllm_das import dsv41_activation as activation
from hcu_vllm_das.quantization import dsv41_channel_fp8 as candidate
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.compressed_tensors import compressed_tensors as ct
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import compressed_tensors_w8a8_fp8 as scheme
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import FP8ScaledMMLinearKernel
from vllm.models.deepseek_v41.amd import model as target

K, N = 16, 5
SCHEMA = {"quant_method": "fp8", "is_per_channel": True, "activation_scheme": "dynamic"}
RESULTS = []


class CpuKernel:
    """Hardware selector/quantizer/GEMM double; apply_weights itself is native."""
    _get_layer_params = FP8ScaledMMLinearKernel._get_layer_params
    apply_weights = FP8ScaledMMLinearKernel.apply_weights

    def __init__(self, **kwargs):
        self.config = SimpleNamespace(out_dtype=kwargs["out_dtype"])
        self.key = kwargs["activation_quant_key"]
        self.fp8_dtype = torch.float8_e4m3fn
        self.layer_param_names = ("weight", "weight_scale", "input_scale", "input_scale_ub")
        self.calls = []
        self.failure = None

    def input_quant_key(self):
        return self.key

    def quant_fp8(self, x, scale, upper):
        if x.shape[0] == 0:
            raise RuntimeError("empty input reached quantization")
        self.calls.append("quant")
        return x.to(self.fp8_dtype), torch.ones((x.shape[0], 1), dtype=torch.float32)

    def apply_scaled_mm(self, *, A, B, out_dtype, As, Bs, bias, output_shape):
        if A.shape[0] == 0:
            raise RuntimeError("empty input reached GEMM")
        if self.failure is not None:
            raise self.failure
        self.calls.append("gemm")
        result = (A.float() @ B.float()) * As * Bs.t()
        if bias is not None:
            result += bias
        return result.to(out_dtype).view(output_shape)

    def process_weights_after_loading(self, layer):
        return None


class TinyTarget(target.DeepseekV4Model):
    """Small module tree, native loader; experts absent in this fixture."""
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        self.quant_config = config
        self.use_sequence_parallel = False
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.fused_wqa_wkv = linear.MergedColumnParallelLinear(
            K, [2, 3], bias=False, quant_config=config, params_dtype=torch.bfloat16,
            prefix="layers.0.attn.fused_wqa_wkv", disable_tp=True,
        )
        block.attn.proj = linear.ColumnParallelLinear(
            K, N, bias=False, quant_config=config, params_dtype=torch.bfloat16,
            prefix="layers.0.attn.proj", disable_tp=True,
        )
        self.layers = torch.nn.ModuleList([block])

    def get_expert_mapping(self):
        return []


def case(name, callback):
    try:
        callback()
        RESULTS.append({"case": name, "status": "passed"})
    except Exception as error:
        RESULTS.append({"case": name, "status": "failed", "error": f"{type(error).__name__}: {error}"})


def expect_error(callback, message=None, expected=(ValueError, RuntimeError, AssertionError)):
    try:
        callback()
    except expected as error:
        if message is not None:
            assert message in str(error), str(error)
        return
    raise AssertionError("malformed input was accepted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original_methods = (ct.CompressedTensorsLinearMethod.apply, FP8ScaledMMLinearKernel.apply_weights)
    with ExitStack() as stack:
        stack.enter_context(patch.object(candidate, "_is_hcu", return_value=True))
        stack.enter_context(patch.object(ct.CompressedTensorsConfig, "_check_scheme_supported", return_value=True))
        stack.enter_context(patch.object(scheme, "get_current_vllm_config", return_value=SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))))
        stack.enter_context(patch.object(scheme, "init_fp8_linear_kernel", side_effect=lambda **kwargs: CpuKernel(**kwargs)))
        stack.enter_context(patch.object(parameter, "get_tensor_model_parallel_rank", return_value=0))
        stack.enter_context(patch.object(parameter, "get_tensor_model_parallel_world_size", return_value=1))
        activation.activate_extension(ExtensionContext("0.1.0", "0.33.0"))
        config = candidate.HcuDSV41ChannelFP8Config.from_config(SCHEMA)
        config.packed_modules_mapping = {"fused_wqa_wkv": ["wq_a", "wkv"]}
        tiny = TinyTarget(config)
        merged = tiny.layers[0].attn.fused_wqa_wkv
        direct = tiny.layers[0].attn.proj
        weights = torch.arange(N * K, dtype=torch.float32).reshape(N, K).remainder(5).to(torch.float8_e4m3fn)
        scales = torch.arange(1, N + 1, dtype=torch.float32).reshape(N, 1)

        def load(suffix):
            raw = [("layers.0.attn.wq_a.weight", weights[:2]),
                   ("layers.0.attn.wkv.weight", weights[2:]),
                   (f"layers.0.attn.wq_a.{suffix}", scales[:2]),
                   (f"layers.0.attn.wkv.{suffix}", scales[2:]),
                   ("layers.0.attn.proj.weight", weights),
                   (f"layers.0.attn.proj.{suffix}", scales)]
            mapper = target._make_deepseek_v4_weights_mapper("fp8")
            mapped = [(name.removeprefix("model."), tensor) for name, tensor in mapper.apply(iter(raw))]
            loaded = tiny.load_weights(iter(mapped))
            expected = {name for name, _ in tiny.named_parameters()}
            assert loaded == expected, (loaded, expected)
            for obj in (merged, direct):
                assert torch.equal(obj.weight.float(), weights.float())
                assert torch.equal(obj.weight_scale, scales)
                assert obj.weight.weight_loader.__func__ is obj.weight_loader_v2.__func__
                assert obj.weight_scale.weight_loader.__func__ is obj.weight_loader_v2.__func__
        for suffix in ("scale", "weight_scale", "weight_scale_inv"):
            case("native_target_" + suffix, lambda suffix=suffix: load(suffix))

        def method_registration():
            assert type(merged.quant_method).__name__ == "HcuDSV41ChannelFP8LinearMethod"
            assert type(merged.quant_method).__name__ in linear.WEIGHT_LOADER_V2_SUPPORTED
        case("plugin_method_and_public_v2_registration", method_registration)

        for obj in (merged, direct):
            obj.quant_method.process_weights_after_loading(obj)
        def finalized():
            assert torch.equal(merged.weight.float(), weights.t().float())
            assert torch.equal(merged.weight_scale, scales)
            assert merged.processed_weight_dims == (0, 1)
            assert dict(merged.named_parameters()).keys() == {"weight", "weight_scale"}
        case("native_ct_finalization", finalized)

        for dtype in (torch.bfloat16, torch.float16):
            for shape in ((0, K), (2, 0, K), (0, 3, K)):
                def empty(dtype=dtype, shape=shape):
                    x = torch.empty(shape, dtype=dtype)
                    out, returned_bias = merged(x)
                    assert out.shape == (*shape[:-1], N) and out.dtype == torch.get_default_dtype()
                    assert out.device == x.device and out.numel() == 0 and returned_bias is None
                    assert not merged.scheme.fp8_linear.calls
                case(f"empty_{dtype}_{shape}", empty)
        kernel = merged.scheme.fp8_linear
        qa = QuantizedActivation(torch.empty((0, K), dtype=torch.float8_e4m3fn),
                                 torch.empty((0, 1)), torch.bfloat16,
                                 torch.Size((2, 0, K)), kernel.input_quant_key())
        def quantized_empty():
            result = merged.quant_method.apply(merged, qa)
            assert result.shape == (2, 0, N) and result.dtype == kernel.config.out_dtype
            assert not kernel.calls
        case("empty_native_quantized_activation", quantized_empty)
        def inferred_dtype():
            with patch.object(kernel.config, "out_dtype", None):
                for dtype in (torch.bfloat16, torch.float16):
                    out = merged.quant_method.apply(merged, torch.empty((0, K), dtype=dtype))
                    assert out.dtype == dtype
                assert merged.quant_method.apply(merged, qa).dtype == qa.orig_dtype
        case("empty_original_dtype_when_unspecified", inferred_dtype)
        def valid_bias():
            out = merged.quant_method.apply(merged, torch.empty((0, K)), torch.zeros(N))
            assert out.shape == (0, N)
        case("empty_valid_bias", valid_bias)
        case("empty_wrong_K_rejected", lambda: expect_error(lambda: merged.quant_method.apply(merged, torch.empty((0, K + 1))), "K"))
        case("empty_wrong_bias_rejected", lambda: expect_error(lambda: merged.quant_method.apply(merged, torch.empty((0, K)), torch.empty(N + 1)), "bias"))
        case("empty_wrong_device_rejected", lambda: expect_error(lambda: merged.quant_method.apply(merged, torch.empty((0, K), device="meta")), "device"))
        def bad_layout():
            with patch.object(merged, "processed_weight_dims", (1, 0)):
                expect_error(lambda: merged.quant_method.apply(merged, torch.empty((0, K))), "processed")
        case("empty_unprocessed_layout_rejected", bad_layout)
        def bad_quantized(field, value, message):
            values = vars(qa).copy()
            values[field] = value
            expect_error(lambda: merged.quant_method.apply(merged, QuantizedActivation(**values)), message)
        case("empty_quantized_shape_rejected", lambda: bad_quantized("data", torch.empty((0, K + 1), dtype=qa.data.dtype), "K"))
        case("empty_quantized_dtype_rejected", lambda: bad_quantized("data", torch.empty((0, K), dtype=torch.int8), "dtype"))
        case("empty_quantized_scale_rejected", lambda: bad_quantized("scale", torch.empty((0, 2)), "scale"))
        def bad_key():
            mismatched = QuantizedActivation(qa.data, qa.scale, qa.orig_dtype, qa.orig_shape, None)
            expect_error(lambda: merged.quant_method.apply(merged, mismatched))
        case("empty_mismatched_quant_key_rejected", bad_key)
        def native_nonempty():
            x = torch.ones((2, K), dtype=torch.bfloat16)
            out, _ = merged(x)
            expected = ((x.float() @ weights.float().t()) * scales.t()).to(kernel.config.out_dtype)
            assert torch.equal(out, expected)
            assert kernel.calls[-2:] == ["quant", "gemm"]
        case("nonempty_native_ct_path", native_nonempty)
        def native_error():
            error = RuntimeError("native GEMM failure must propagate")
            kernel.failure = error
            try:
                merged(torch.ones((1, K), dtype=torch.bfloat16))
            except RuntimeError as caught:
                assert caught is error
            else:
                raise AssertionError("native execution error swallowed")
            finally:
                kernel.failure = None
        case("nonempty_error_propagates", native_error)
        case("official_method_identity", lambda: assert_identity(original_methods))
    assert not torch.cuda.is_initialized()
    root = Path(__file__).resolve().parents[2]
    def sha(path):
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    report = {"level": "CPU source integration; hardware admission, quantizer and GEMM doubled",
              "core_source": inspect.getsourcefile(target.DeepseekV4Model),
              "plugin": sha(root), "core": sha(root.parent / "vllm"),
              "adapter": sha(root.parent / "adapter-das"),
              "installed_vllm_metadata": metadata.version("vllm"),
              "cuda_initialized": torch.cuda.is_initialized(), "cases": RESULTS,
              "passed": sum(item["status"] == "passed" for item in RESULTS), "total": len(RESULTS)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] == report["total"] else 1


def assert_identity(original):
    assert original == (ct.CompressedTensorsLinearMethod.apply, FP8ScaledMMLinearKernel.apply_weights)


if __name__ == "__main__":
    raise SystemExit(main())
