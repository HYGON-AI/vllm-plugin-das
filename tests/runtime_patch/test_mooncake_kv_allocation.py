# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""CPU contracts for KV allocation scope, using a v0.25.1 allocator fixture."""
import ast
import importlib.util
from pathlib import Path
import sys
import threading
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
NS = types.SimpleNamespace

# Minimal raw allocator fixture; integration against the installed MRV2 was
# validated separately on HCU. No vLLM or accelerator import is needed here.
ALLOCATOR_FIXTURE = """\
def _allocate_kv_cache(
    kv_cache_config: KVCacheConfig, shared_layers: dict[str, str], device: torch.device
):
    kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
    packed_backing: torch.Tensor | None = None
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        if kv_cache_tensor.block_stride > 0:
            # Allocate once; all packed tensors alias the same backing.
            if packed_backing is None:
                packed_backing = torch.zeros(
                    kv_cache_tensor.size, dtype=torch.int8, device=device
                )
            tensor = packed_backing
        else:
            tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=device)
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = tensor

    layer_names = set()
    for group in kv_cache_config.kv_cache_groups:
        for layer_name in group.layer_names:
            layer_names.add(layer_name)
    assert layer_names == (kv_cache_raw_tensors.keys() | shared_layers.keys()), (
        "Some layers are not correctly initialized"
    )
    return kv_cache_raw_tensors
"""


def load_module():
    name = "_test_mooncake_kv_scope"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "vllm_hcu/v1/mooncake_kv_allocation.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_method_class(path, class_name, methods, namespace, base=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    cls.bases = [] if base is None else [ast.Name(id=base, ctx=ast.Load())]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[class_name]


class AllocationSpy:
    def __init__(self):
        self.local = threading.local()
        self.events = []
        self.depth = 0

    @property
    def depth(self):
        return getattr(self.local, "depth", 0)

    @depth.setter
    def depth(self, value):
        self.local.depth = value

    @contextmanager
    def use(self):
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1

    def zeros(self, size, *, dtype=None, device=None):
        tensor = NS(size=size, dtype=dtype, device=device, fabric=bool(self.depth))
        self.events.append((size, tensor.fabric))
        return tensor

    def auxiliary(self, name, *args, **kwargs):
        return self.zeros(name)


class KVAllocationScopeTests(unittest.TestCase):
    def setUp(self):
        self.helper = load_module()
        self.spy = AllocationSpy()
        self.torch = NS(zeros=self.spy.zeros, int8="int8")
        self.attn = types.ModuleType("vllm.v1.worker.gpu.attn_utils")
        self.attn.__dict__["torch"] = self.torch
        source = ALLOCATOR_FIXTURE
        exec("from __future__ import annotations\n" + source, self.attn.__dict__)
        gpu = types.ModuleType("vllm.v1.worker.gpu")
        gpu.attn_utils = self.attn
        self.modules = patch.dict(sys.modules, {
            "vllm.v1.worker.gpu": gpu, self.attn.__name__: self.attn})
        self.modules.start()
        self.runner = NS(device="cuda:7")

    def tearDown(self):
        self.modules.stop()

    def config(self):
        # Target + MTP share packed backing; index data/scale may share another
        # allocation. Verify aliases without reimplementing allocation logic.
        tensors = [NS(size=4096, block_stride=128, shared_by=["target"]),
                   NS(size=4096, block_stride=128, shared_by=["mtp"]),
                   NS(size=1024, block_stride=0, shared_by=["index", "scale"])]
        return NS(kv_cache_tensors=tensors,
                  kv_cache_groups=[NS(layer_names=["target", "mtp", "index", "scale"],
                                      kv_cache_spec=NS(block_size=64))])

    def scope(self, runner=None, use_v2=True):
        return self.helper.kv_cache_allocation_scope(
            runner or self.runner, lambda: self.spy, use_v2_model_runner=use_v2)

    def test_mrv2_packed_allocator_preserves_aliases_and_excludes_aux(self):
        original = self.attn._allocate_kv_cache
        with self.scope():
            self.spy.auxiliary("block_table")
            buffers = self.attn._allocate_kv_cache(self.config(), {}, "cuda:7")
            self.spy.auxiliary("pcp_workspace")
        self.assertIs(buffers["target"], buffers["mtp"])
        self.assertIs(buffers["index"], buffers["scale"])
        self.assertIsNot(buffers["target"], buffers["index"])
        self.assertEqual(self.spy.events,
                         [("block_table", False), (4096, True), (1024, True),
                          ("pcp_workspace", False)])
        self.assertIs(self.attn._allocate_kv_cache.__wrapped__, original)
        wrapper = self.attn._allocate_kv_cache
        with self.scope():
            self.attn._allocate_kv_cache(self.config(), {}, "cuda:7")
        self.assertIs(self.attn._allocate_kv_cache, wrapper)
        ordinary = self.attn._allocate_kv_cache(self.config(), {}, "cuda:7")
        self.assertFalse(ordinary["target"].fabric)


    def test_mrv1_actual_allocator_and_uniform_layout(self):
        namespace = {"torch": self.torch}
        Runner = load_method_class(ROOT / "vllm_hcu/v1/hcu_model_runner.py",
                                   "GPUModelRunner", {"_allocate_kv_cache_tensors"}, namespace)
        # Descriptor behavior of the inherited static alternate allocator.
        def uniform(config, attn_groups, cache_dtype, device, kernel_block_sizes):
            return self.spy.zeros(8192, dtype="int8", device=device)
        Runner.allocate_uniform_kv_caches = staticmethod(uniform)
        runner = Runner()
        runner.device = "cuda:7"
        runner.runner_only_attn_layers = set()
        other = Runner()
        other.device = "cuda:7"
        other.runner_only_attn_layers = set()
        with self.scope(runner, use_v2=False):
            buffers = runner._allocate_kv_cache_tensors(self.config())
            unrelated = other._allocate_kv_cache_tensors(self.config())
            uniform_buffer = runner.allocate_uniform_kv_caches(self.config(), [], "fp8", "cuda:7", [64])
            unrelated_uniform = other.allocate_uniform_kv_caches(self.config(), [], "fp8", "cuda:7", [64])
        self.assertIs(buffers["target"], buffers["mtp"])
        self.assertTrue(buffers["target"].fabric)
        self.assertFalse(unrelated["target"].fabric)
        self.assertTrue(uniform_buffer.fabric)
        self.assertFalse(unrelated_uniform.fabric)
        self.assertIsInstance(inspect_static(Runner, "allocate_uniform_kv_caches"), staticmethod)

    def test_unrelated_thread_and_nested_scope_do_not_inherit_pool(self):
        outputs = []
        errors = []
        def background():
            try:
                outputs.append(self.attn._allocate_kv_cache(self.config(), {}, "cuda:7"))
            except Exception as exc:
                errors.append(exc)
        with self.scope():
            thread = threading.Thread(target=background)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.attn._allocate_kv_cache(self.config(), {}, "cuda:7")
            with self.scope(NS(device="cuda:3")):
                inner = self.attn._allocate_kv_cache(self.config(), {}, "cuda:3")
            outer = self.attn._allocate_kv_cache(self.config(), {}, "cuda:7")
        self.assertEqual(errors, [])
        self.assertFalse(outputs[0]["target"].fabric)
        self.assertTrue(inner["target"].fabric)
        self.assertTrue(outer["target"].fabric)
        self.assertIsNone(self.helper._ACTIVE_SCOPE.get())

    def test_allocator_failure_restores_context_without_masking_error(self):
        def fail(size, **kwargs):
            raise MemoryError("allocation failed")
        self.torch.zeros = fail
        with self.assertRaisesRegex(MemoryError, "allocation failed"):
            with self.scope():
                self.attn._allocate_kv_cache(self.config(), {}, "cuda:7")
        self.assertEqual(self.spy.depth, 0)
        self.assertIsNone(self.helper._ACTIVE_SCOPE.get())

    def test_wrong_device_missing_hook_and_bypass_fail_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, "device mismatch"):
            with self.scope():
                self.attn._allocate_kv_cache(self.config(), {}, "cuda:3")
        self.assertEqual(self.spy.events, [])
        with self.assertRaisesRegex(RuntimeError, "hook was not reached"):
            with self.scope():
                pass
        del self.attn._allocate_kv_cache
        with self.assertRaisesRegex(RuntimeError, "entry point is missing"):
            with self.scope():
                pass
        self.assertIsNone(self.helper._ACTIVE_SCOPE.get())


def inspect_static(owner, name):
    import inspect
    return inspect.getattr_static(owner, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
