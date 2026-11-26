"""Portable actual-source cache-format policy; no device math or GPU imports."""

import ast
import dataclasses
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = (Path(__file__).resolve().parents[2] / "adapter_extension/src"
          / "hcu_vllm_das/models/hy_v4_flash")


class Missing:
    def __call__(self, *args, **kwargs):
        raise AssertionError("Missing capability must never be called")


def writer(*args):
    raise AssertionError("Policy selection must not execute the native writer")


class Tensor:
    def __init__(self, shape, dtype, device="cuda:0"):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.ndim = len(shape)
        self.is_cuda = True

    def stride(self, dim):
        if dim == 2 or dim == -1:
            return 1
        return 656 if dim == 1 else self.shape[1] * 656

    def is_contiguous(self):
        return True

    def view(self, *args):
        return self

    def unsqueeze(self, dim):
        return self

    def clone(self):
        return Tensor(self.shape, self.dtype, self.device)

    def contiguous(self):
        return self

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Tensor((min(self.shape[0], key.stop), *self.shape[1:]), self.dtype)
        return self


def actual_method(name, namespace, *, base=None):
    tree = ast.parse((SOURCE / "sparse_backend.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "FlashSparseImpl")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    body = [ast.parse("from __future__ import annotations").body[0]]
    if base is None:
        body.append(method)
    else:
        namespace["FixtureBase"] = base
        cls.name, cls.bases, cls.body = "Fixture", [ast.Name(id="FixtureBase", ctx=ast.Load())], [method]
        body.append(cls)
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(SOURCE / "sparse_backend.py"), "exec"), namespace)
    return namespace[name] if base is None else namespace["Fixture"]


class CacheFormatPolicy(unittest.TestCase):
    def setUp(self):
        package = ModuleType("flash_format_fixture")
        package.__path__ = []
        runtime = ModuleType("flash_format_fixture.runtime")
        runtime.is_hcu_platform = lambda: True
        base = ModuleType("hcu_adapter._base")
        base.Missing = Missing
        self.patch = patch.dict(sys.modules, {
            "flash_format_fixture": package,
            "flash_format_fixture.runtime": runtime,
            "hcu_adapter._base": base,
        })
        self.patch.start()
        self.addCleanup(self.patch.stop)
        name = "flash_format_fixture.flashmla_ops"
        spec = importlib.util.spec_from_file_location(name, SOURCE / "flashmla_ops.py")
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[name] = self.module
        self.addCleanup(sys.modules.pop, name, None)
        spec.loader.exec_module(self.module)

    def select(self, arch="gfx936", *, eligible=True, formats=None, native_writer=writer,
               native_reader=writer, metadata=writer):
        function = getattr(self.module, "select_fp8_ds_mla_format", None)
        self.assertTrue(callable(function), "Missing immutable FP8 format selection")
        ops = SimpleNamespace(get_sparse_kv_cache_formats=formats,
                              with_kvcache=native_reader, get_metadata=metadata)
        return function(arch, eligible=eligible, ops=ops, writer=native_writer)

    def test_e5_descriptor_is_immutable_and_writer_reader_agree(self):
        codec = self.select(formats=lambda: ("e4m3", "e5m2"))
        self.assertEqual(codec.name, "e5m2")
        self.assertEqual(codec.writer_dtype, "fp8_e5m2")
        self.assertEqual(codec.reader_kwargs, {"kv_cache_format": "e5m2"})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            codec.name = "e4m3"
        kwargs = codec.reader_kwargs
        kwargs["kv_cache_format"] = "e4m3"
        self.assertEqual(codec.reader_kwargs["kv_cache_format"], "e5m2")

    def test_old_frontend_and_old_binary_keep_explicit_legacy_e4(self):
        for getter in (None, lambda: ("e4m3",), Missing()):
            with self.subTest(getter=getter):
                codec = self.select(formats=getter)
                self.assertEqual(codec.name, "e4m3")
                self.assertEqual(codec.writer_dtype, "fp8_e4m3")
                self.assertEqual(codec.reader_kwargs, {})
                self.assertIn("legacy", codec.reason)

    def test_gfx938_and_unknown_do_not_probe_new_capability(self):
        def exploding():
            raise AssertionError("Old architecture path must not probe E5")
        for arch in ("gfx938", "gfx942", None):
            with self.subTest(arch=arch):
                codec = self.select(arch, formats=exploding, native_writer=Missing())
                self.assertEqual(codec.name, "e4m3")
                self.assertEqual(codec.reader_kwargs, {})

    def test_unsupported_domain_keeps_e4_without_capability_probe(self):
        codec = self.select(eligible=False, formats=Missing())
        self.assertEqual(codec.name, "e4m3")
        self.assertIn("domain", codec.reason)

    def test_e5_requires_real_public_writer(self):
        for unavailable in (None, Missing(), 17):
            with self.subTest(writer=unavailable):
                codec = self.select(formats=lambda: ("e4m3", "e5m2"),
                                    native_writer=unavailable)
                self.assertEqual(codec.name, "e4m3")
                self.assertIn("writer", codec.reason)

    def test_e5_does_not_admit_missing_reader_or_metadata(self):
        for kwargs in ({"native_reader": Missing()}, {"metadata": Missing()}):
            with self.subTest(kwargs=kwargs):
                codec = self.select(formats=lambda: ("e4m3", "e5m2"), **kwargs)
                self.assertEqual(codec.name, "e4m3")
                self.assertIn("reader", codec.reason)

    def test_malformed_capability_is_an_error(self):
        for malformed in ("e5m2", ["e4m3", "e5m2"], (), ("e5m2",),
                          ("e4m3", "bad"), ("e4m3", "e4m3"), (17,)):
            with self.subTest(formats=malformed):
                with self.assertRaises(RuntimeError):
                    self.select(formats=lambda: malformed)
        with self.assertRaises(RuntimeError):
            self.select(formats=17)

    def test_reader_capability_is_resolved_once_at_selection(self):
        calls = []
        def getter():
            calls.append(True)
            return ("e4m3", "e5m2")
        codec = self.select(formats=getter)
        self.assertEqual(codec.reader_kwargs, {"kv_cache_format": "e5m2"})
        self.assertEqual(codec.writer_dtype, "fp8_e5m2")
        self.assertEqual(len(calls), 1)

    def test_get_ops_keeps_capability_with_its_own_reader(self):
        def decode(*args, **kwargs):
            return None
        getter = lambda: ("e4m3", "e5m2")
        module = SimpleNamespace(flash_mla_sparse_fwd=decode,
                                 flash_mla_with_kvcache=decode,
                                 get_mla_metadata=decode,
                                 get_sparse_kv_cache_formats=getter)
        self.module.import_module = lambda source: module
        ops = self.module.get_flashmla_ops()
        self.assertIs(getattr(ops, "get_sparse_kv_cache_formats", None), getter)
        self.assertIs(ops.with_kvcache, decode)
        self.assertIs(ops.get_metadata, decode)

    def test_old_get_ops_signature_remains_compatible(self):
        ops = self.module.FlashMLAOps(writer, writer, writer, "legacy", False)
        self.assertIsNone(getattr(ops, "get_sparse_kv_cache_formats", None))

    def test_writer_uses_the_same_descriptor_public_dtype(self):
        codec = self.select(formats=lambda: ("e4m3", "e5m2"))
        tree = ast.parse((SOURCE / "fp8_ds_mla_cache.py").read_text(encoding="utf-8"))
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("_validate_inputs", "write_fp8_ds_mla_cache")]
        calls = []
        class Scatter:
            def __getitem__(self, grid):
                return lambda *args, **kwargs: calls.append(("scatter", args))
        torch = SimpleNamespace(bfloat16="bf16", uint8="u8", int64="i64",
            empty=lambda shape, dtype, device: Tensor(shape, dtype),
            arange=lambda n, dtype, device: Tensor((n,), dtype))
        namespace = {"__package__": "flash_format_fixture", "torch": torch,
                     "_scatter_656": Scatter()}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), "<actual-writer>", "exec"), namespace)
        public = ModuleType("lightop.kvcache")
        public.fused_quantize_and_store_mla_kv_cache = lambda *args: calls.append(("native", args))
        with patch.dict(sys.modules, {"lightop.kvcache": public}):
            namespace["write_fp8_ds_mla_cache"](
                Tensor((5, 512), "bf16"), Tensor((5, 1, 64), "bf16"),
                Tensor((2, 64, 656), "u8"), Tensor((3,), "i64"), codec=codec)
        self.assertEqual(calls[0][0], "native")
        self.assertEqual(calls[0][1][4], "fp8_e5m2")
        self.assertEqual(calls[0][1][0].shape, (3, 512))
        self.assertEqual(calls[1][0], "scatter")

    def test_e5_never_reuses_core_e4_scheduler_even_with_padded_heads(self):
        codec = self.select(formats=lambda: ("e4m3", "e5m2"))
        calls, created = [], []
        def metadata():
            result = object()
            created.append(result)
            return result, None
        def decode(**kwargs):
            calls.append(kwargs)
            return Tensor((1, 7, 64, 512), "bf16"), Tensor((1, 64, 7), "f32")
        namespace = {"torch": SimpleNamespace(bfloat16="bf16", uint8="u8", int32="i32"),
                     "F": SimpleNamespace(pad=lambda q, padding: Tensor(
                         (1, q.shape[1], q.shape[2] + padding[3], 576), q.dtype))}
        function = actual_method("_fp8_flash_mla_kernel", namespace)
        owner = SimpleNamespace(_hcu_ops=SimpleNamespace(get_metadata=metadata, with_kvcache=decode),
            _hcu_fp8_cache_format=codec, _hcu_window_device="cuda:0",
            _use_window_fp8_prefix=lambda *args: False,
            _use_fp8_small_heads=lambda *args: False,
            fp8_decode_padded_heads=64, _sinks_for_query=lambda *args: None,
            softmax_scale=.0625)
        old = object()
        kernel = SimpleNamespace(scheduler_metadata=old, dummy_block_table=object(), cache_lens=object())
        for heads in (4, 8):
            function(owner, Tensor((1, 7, heads, 576), "bf16"),
                     Tensor((2, 64, 656), "u8"), Tensor((1, 7, 2048), "i32"), kernel)
        self.assertEqual(len(created), 2)
        self.assertIsNot(created[0], created[1])
        for call in calls:
            self.assertEqual(call.get("kv_cache_format"), "e5m2")
            self.assertIsNot(call["tile_scheduler_metadata"], old)
            self.assertIsNone(call["block_table"])
            self.assertIsNone(call["cache_seqlens"])

    def test_e5_separate_prefill_is_rejected_before_core_gather(self):
        codec = self.select(formats=lambda: ("e4m3", "e5m2"))
        calls = []
        class Base:
            def forward_mqa(self, *args):
                calls.append("core-gather")
                return "core"
        cls = actual_method("forward_mqa", {}, base=Base)
        owner = cls()
        owner._hcu_fp8_cache_format = codec
        owner._checked_hisparse_call = lambda *args: None
        with self.assertRaises(NotImplementedError):
            owner.forward_mqa(None, None, SimpleNamespace(fp8_use_mixed_batch=False), None)
        self.assertFalse(calls)

    def test_e4_omits_new_keyword_and_retains_core_scheduler(self):
        codec = self.select("gfx938")
        old = object()
        calls = []
        def metadata():
            raise AssertionError("Qualified legacy scheduler must stay unchanged")
        def decode(**kwargs):
            self.assertNotIn("kv_cache_format", kwargs)
            self.assertIs(kwargs["tile_scheduler_metadata"], old)
            calls.append(kwargs)
            return Tensor((1, 7, 64, 512), "bf16"), Tensor((1, 64, 7), "f32")
        namespace = {"torch": SimpleNamespace(bfloat16="bf16", uint8="u8", int32="i32"),
                     "F": SimpleNamespace(pad=lambda q, padding: Tensor(
                         (1, q.shape[1], q.shape[2] + padding[3], 576), q.dtype))}
        function = actual_method("_fp8_flash_mla_kernel", namespace)
        owner = SimpleNamespace(_hcu_ops=SimpleNamespace(get_metadata=metadata, with_kvcache=decode),
            _hcu_fp8_cache_format=codec, _use_window_fp8_prefix=lambda *args: False,
            _use_fp8_small_heads=lambda *args: False, fp8_decode_padded_heads=64,
            _sinks_for_query=lambda *args: None, softmax_scale=.0625)
        kernel = SimpleNamespace(scheduler_metadata=old, dummy_block_table=object(), cache_lens=object())
        function(owner, Tensor((1, 7, 16, 576), "bf16"), Tensor((2, 64, 656), "u8"),
                 Tensor((1, 7, 2048), "i32"), kernel)
        self.assertEqual(len(calls), 1)

    def test_e5_rejects_runtime_domain_drift_without_native_fallback(self):
        codec = self.select(formats=lambda: ("e4m3", "e5m2"))
        namespace = {"torch": SimpleNamespace(bfloat16="bf16", uint8="u8", int32="i32")}
        function = actual_method("_fp8_flash_mla_kernel", namespace)
        owner = SimpleNamespace(_hcu_ops=object(), _hcu_fp8_cache_format=codec,
                                _hcu_window_device="cuda:0")
        normal = (Tensor((1, 7, 4, 576), "bf16"), Tensor((2, 64, 656), "u8"),
                  Tensor((1, 7, 2048), "i32"))
        for index, bad in ((0, Tensor((1, 7, 16, 576), "bf16")),
                           (0, Tensor((1, 7, 4, 576), "f16")),
                           (1, Tensor((2, 64, 584), "u8")),
                           (2, Tensor((1, 7, 1024), "i32")),
                           (2, Tensor((1, 7, 2048), "i64"))):
            with self.subTest(index=index, shape=bad.shape):
                inputs = list(normal)
                inputs[index] = bad
                with self.assertRaises(ValueError):
                    function(owner, *inputs, None)

    def test_actual_worker_ctor_policy_and_hisparse_domain(self):
        tree = ast.parse((SOURCE / "sparse_backend.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "FlashSparseImpl")
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("__init__", "_install_hisparse_hook")]
        runner_name = "flash_format_fixture.fullgraph_runner"
        spec = importlib.util.spec_from_file_location(runner_name, SOURCE / "fullgraph_runner.py")
        runner = importlib.util.module_from_spec(spec)
        sys.modules[runner_name] = runner
        self.addCleanup(sys.modules.pop, runner_name, None)
        spec.loader.exec_module(runner)
        index_module = ModuleType("vllm.v1.attention.backends.mla.index_group")
        class HiSparse:
            def cache(self, index):
                return object()
        index_module.HiSparseMLAIndexGroup = HiSparse
        public = ModuleType("lightop.kvcache")
        public.fused_quantize_and_store_mla_kv_cache = writer
        for arch, cu, tp, heads, hisparse, fusion, expected in (
            ("gfx936", 80, 8, 4, False, True, "e5m2"),
            ("gfx936", 80, 8, 8, False, False, "e5m2"),
            ("gfx936", 80, 8, 16, False, True, "e4m3"),
            ("gfx936", 80, 4, 4, False, True, "e4m3"),
            ("gfx936", 80, 8, 4, True, True, "e4m3"),
            ("gfx938", 64, 4, 8, False, True, "e4m3"),
            ("gfx942", 80, 8, 4, False, True, "e4m3"),
        ):
            with self.subTest(arch=arch, tp=tp, heads=heads, hisparse=hisparse):
                class Base:
                    def __init__(self):
                        self.num_heads = heads
                        self.q_head_size = 576
                        self.kv_lora_rank = 512
                        self.kv_cache_dtype = "fp8_ds_mla"
                        self.index_group = HiSparse() if hisparse else object()
                        self.index_group_index = 0
                parallel = SimpleNamespace(tensor_parallel_size=tp,
                    prefill_context_parallel_size=1, decode_context_parallel_size=1)
                torch = SimpleNamespace(cuda=SimpleNamespace(
                    is_available=lambda: True, current_device=lambda: 0,
                    get_device_properties=lambda device: SimpleNamespace(
                        gcnArchName=arch, multi_processor_count=cu)),
                    device=lambda kind, index: "cuda:0")
                namespace = {"__package__": "flash_format_fixture", "__name__": "actual_ctor",
                    "FixtureBase": Base, "torch": torch,
                    "get_current_vllm_config": lambda: SimpleNamespace(parallel_config=parallel),
                    "get_flashmla_ops": lambda: SimpleNamespace(
                        get_sparse_kv_cache_formats=lambda: ("e4m3", "e5m2"),
                        with_kvcache=writer, get_metadata=writer),
                    "is_hcu_platform": lambda: True,
                    "fp8_fusions_enabled": lambda: fusion,
                    "install_hisparse_prewrite_hook": lambda *args: None}
                derived = ast.ClassDef(name="Fixture", bases=[ast.Name(id="FixtureBase", ctx=ast.Load())],
                                       keywords=[], body=methods, decorator_list=[])
                module = ast.fix_missing_locations(ast.Module(body=[derived], type_ignores=[]))
                exec(compile(module, "<actual-constructor>", "exec"), namespace)
                with patch.dict(sys.modules, {
                    "vllm.v1.attention.backends.mla.index_group": index_module,
                    "lightop.kvcache": public,
                }):
                    owner = namespace["Fixture"]()
                self.assertEqual(owner._hcu_fp8_cache_format.name, expected)
                if hisparse:
                    self.assertIsNone(owner._hcu_small_head_device)
                if not fusion:
                    self.assertIsNone(owner._hcu_small_head_device)


if __name__ == "__main__":
    unittest.main(verbosity=2)
