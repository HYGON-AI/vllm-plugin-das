"""CPU-only dispatch, fallback and alias contracts for decode producers."""
from contextlib import contextmanager
import __future__
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class Tensor:
    def __init__(self, shape, dtype="f32", device="cuda:0", contiguous=True):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device
        self.is_cuda = device.startswith("cuda")
        self.contiguous = contiguous

    def is_contiguous(self):
        return self.contiguous

    def clone(self):
        return Tensor(self.shape, self.dtype, self.device)

    def float(self):
        return self.to("f32")

    def to(self, dtype):
        return Tensor(self.shape, dtype, self.device)

    def __add__(self, other):
        if isinstance(other, Tensor):
            assert self.dtype == other.dtype == "f32"
        return Tensor(self.shape, self.dtype, self.device)


@contextmanager
def runtime(child="1", master="1"):
    torch = ModuleType("torch")
    torch.Tensor = Tensor
    torch.float32, torch.bfloat16, torch.float8_e4m3fn = "f32", "bf16", "fp8"
    torch.empty_like = lambda x, dtype=None: Tensor(x.shape, dtype or x.dtype, x.device)
    torch.empty = lambda shape, device, dtype: Tensor(shape, dtype, device)
    calls, registered = [], {}
    torch.ops = SimpleNamespace(vllm=SimpleNamespace())

    def norm(x, weight, eps):
        calls.append(("library_norm", x, weight))
        return Tensor(x.shape, "f32", x.device)

    torch.ops.vllm.hcu_step5_gemma_rms_norm = norm
    ir = SimpleNamespace(ops=SimpleNamespace(rms_norm=lambda x, w, eps: norm(x, w, eps)))
    vllm = ModuleType("vllm")
    vllm.ir = ir
    utils = ModuleType("vllm.utils.torch_utils")

    def register(**kwargs):
        registered[kwargs["op_name"]] = kwargs
        setattr(torch.ops.vllm, kwargs["op_name"], kwargs["op_func"])

    utils.direct_register_custom_op = register
    spec = importlib.util.spec_from_file_location("producer_envs", ROOT / "vllm_hcu/platforms/envs.py")
    envs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(envs)
    platforms = ModuleType("vllm_hcu.platforms")
    platforms.envs = envs
    kernel = ModuleType("vllm_hcu.ops.step5.decode_producers_kernel")

    def producer(x, weight, eps, addition=None, quant=True, warps=8):
        calls.append(("fused", x, addition, quant, warps))
        result = (Tensor(x.shape, "f32", x.device), Tensor(x.shape, "bf16", x.device))
        if quant:
            result += (Tensor(x.shape, "fp8", x.device), Tensor((x.shape[0], 1), "f32", x.device))
        return result

    kernel.producer = producer
    dense = ModuleType("vllm_hcu.ops.step5.fp8_dense")

    def quantize(y, dynamic_small_m=False):
        calls.append(("fallback_quant", y.dtype, dynamic_small_m))
        return Tensor(y.shape, "fp8", y.device), Tensor((y.shape[0], 1), "f32", y.device)

    dense.quant_per_token_fp8 = quantize
    modules = {"torch": torch, "vllm": vllm,
               "vllm.utils": ModuleType("vllm.utils"), "vllm.utils.torch_utils": utils,
               "vllm_hcu": ModuleType("vllm_hcu"), "vllm_hcu.platforms": platforms,
               "vllm_hcu.ops": ModuleType("vllm_hcu.ops"),
               "vllm_hcu.ops.step5": ModuleType("vllm_hcu.ops.step5"),
               kernel.__name__: kernel, dense.__name__: dense}
    values = {"VLLM_HCU_STEP5_DECODE_PRODUCERS": child,
              "VLLM_HCU_USE_CUSTOM_OPS": master, "VLLM_HCU_USE_CUSTOM_GEMMA_RMS_NORM": "1"}
    with patch.dict(os.environ, values), patch.dict(sys.modules, modules):
        name = "vllm_hcu.ops.step5.decode_producers_runtime"
        path = ROOT / "vllm_hcu/ops/step5/decode_producers_runtime.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec",
                     flags=__future__.annotations.compiler_flag), module.__dict__)
        yield module, calls, registered


class DecodeProducerContracts(unittest.TestCase):
    def test_decode_input_and_post_dispatch_without_aliasing_inputs(self):
        with runtime() as (module, calls, registered):
            weight = Tensor((4096,))
            for rows in (0, 1, 16, 64):
                x, pending = Tensor((rows, 4096)), Tensor((rows, 4096), "bf16")
                result = module._input(x, pending, weight, 1e-5)
                self.assertEqual([v.dtype for v in result], ["f32", "bf16", "fp8", "f32"])
                self.assertIsNot(result[0], x)
                self.assertEqual(result[3].shape, (rows, 1))
                post = module._post(x, pending, weight, 1e-5)
                self.assertEqual([v.dtype for v in post], ["f32", "bf16"])
                self.assertIsNot(post[0], x)
            self.assertTrue(all(c[0] == "fused" for c in calls))
            self.assertTrue(all(r["mutates_args"] == [] for r in registered.values()))

    def test_large_rows_keep_bf16_boundary_and_dynamic_quant_fallback(self):
        with runtime() as (module, calls, _):
            x, weight = Tensor((512, 4096)), Tensor((4096,))
            result = module._input(x, None, weight, 1e-5)
            self.assertIsNot(result[0], x)
            self.assertEqual(calls[-1], ("fallback_quant", "bf16", True))
            self.assertFalse(any(c[0] == "fused" for c in calls))

    def test_runtime_master_disable_overrides_cached_child_opt_in(self):
        with runtime() as (module, calls, _):
            os.environ["VLLM_HCU_USE_CUSTOM_OPS"] = "0"
            module._input(Tensor((64, 4096)), None, Tensor((4096,)), 1e-5)
            self.assertFalse(any(c[0] == "fused" for c in calls))
            self.assertEqual(calls[-1], ("fallback_quant", "bf16", True))

    def test_disabled_child_keeps_original_math_path(self):
        with runtime(child="0") as (module, calls, _):
            x, delta, weight = Tensor((16, 4096)), Tensor((16, 4096), "bf16"), Tensor((4096,))
            result = module._post(x, delta, weight, 1e-5)
            self.assertEqual(result[0].dtype, "f32")
            self.assertEqual(result[1].dtype, "bf16")
            self.assertFalse(any(c[0] == "fused" for c in calls))

    def test_unsupported_layouts_are_not_fused(self):
        with runtime() as (module, _, _):
            weight = Tensor((4096,))
            for x in (Tensor((65, 4096)), Tensor((16, 2048)),
                      Tensor((16, 4096), "bf16"), Tensor((16, 4096), device="cpu"),
                      Tensor((16, 4096), contiguous=False)):
                self.assertFalse(module._can_fuse(x, None, weight))
            x = Tensor((16, 4096))
            self.assertFalse(module._can_fuse(x, Tensor(x.shape, contiguous=False), weight))
            self.assertFalse(module._can_fuse(x, None, Tensor((4096,), "bf16")))

    def test_fake_functions_support_symbolic_rows_without_dispatch(self):
        class Rows:
            def __le__(self, other):
                raise AssertionError("Fake implementation must not inspect symbolic M")

        with runtime() as (module, calls, _):
            rows = Rows()
            x, weight = Tensor((rows, 4096)), Tensor((4096,))
            result = module._input_fake(x, None, weight, 1e-5)
            self.assertEqual(result[0].shape, x.shape)
            self.assertEqual(result[3].shape, (rows, 1))
            self.assertIsNot(result[0], x)
            self.assertEqual(len(module._post_fake(x, x, weight, 1e-5)), 2)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
