# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""CPU-only regression tests; no service startup or device initialization."""
import ast
import gc
import importlib.util
import os
from pathlib import Path
import runpy
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT
HELPER = SOURCE / "vllm_hcu/v1/mooncake_fabric_pool.py"
SWITCH = "VLLM_HCU_MOONCAKE_FABRIC_MEM_POOL"
NAME = "vllm_hcu.v1.mooncake_fabric_pool"


def load_helper():
    spec = importlib.util.spec_from_file_location(NAME, HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(connector="MooncakeConnector", role="kv_producer", **kwargs):
    return types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(kv_connector=connector, kv_role=role),
        model_config=types.SimpleNamespace(**kwargs),
    )


class FakeBase:
    def __init__(self, **kwargs):
        self.vllm_config = kwargs["vllm_config"]
        self.device = "cuda:7"
        self.calls = []

    def _maybe_get_memory_pool_context(self, tag):
        self.calls.append(tag)
        return ("original", tag)

    def shutdown(self):
        self.calls.append("base_shutdown")


def load_worker(enabled):
    # Execute the actual changed worker methods without loading inference
    # dependencies. This exercises inherited dispatch for both model runners.
    tree = ast.parse((SOURCE / "vllm_hcu/v1/worker.py").read_text())
    worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HcuGPUWorker")
    selected = {"__init__", "_maybe_get_memory_pool_context",
                "_get_mooncake_fabric_pool", "shutdown"}
    worker.body = [n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name in selected]
    worker.bases = [ast.Name(id="FakeBase", ctx=ast.Load())]
    module = ast.Module(body=[worker], type_ignores=[])
    ns = {"FakeBase": FakeBase, "VllmConfig": object,
          "henvs": types.SimpleNamespace(**{SWITCH: enabled})}
    exec(compile(ast.fix_missing_locations(module), "worker-under-test", "exec"), ns)
    return ns["HcuGPUWorker"]


class FabricPoolTests(unittest.TestCase):
    def setUp(self):
        self.module = load_helper()
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_switch_defaults_off_and_is_explicit(self):
        envs = runpy.run_path(str(SOURCE / "vllm_hcu/platforms/envs.py"))
        read = envs["__getattr__"]
        self.assertFalse(read(SWITCH))
        for value, expected in [("0", False), ("False", False), ("1", True), ("true", True)]:
            os.environ[SWITCH] = value
            self.assertEqual(read(SWITCH), expected)

    def test_config_accepts_mooncake_roles_and_preserves_transport(self):
        for role in ["kv_producer", "kv_consumer", "kv_both"]:
            c = config(role=role)
            c.kv_transfer_config.kv_connector_extra_config = {"mooncake_protocol": "rdma"}
            self.module.validate_fabric_pool_config(c)
            self.assertEqual(c.kv_transfer_config.kv_connector_extra_config, {"mooncake_protocol": "rdma"})

    def test_rejects_incompatible_configs(self):
        for c in [config("NixlConnector"), config(role="bad"),
                  config(enable_sleep_mode=True), config(enable_cumem_allocator=True)]:
            with self.assertRaises(ValueError):
                self.module.validate_fabric_pool_config(c)

    def test_rejects_expandable_segments_without_mutating_config(self):
        for name in ["PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF"]:
            with patch.dict(os.environ, {name: "max_split_size_mb:128,expandable_segments:True"}):
                with self.assertRaises(ValueError):
                    self.module.validate_fabric_pool_config(config())
                self.assertIn("expandable_segments:True", os.environ[name])

    def worker_modules(self, helper):
        runtime = types.ModuleType("vllm_hcu.patch.runtime_state")
        runtime.set_process_role = lambda role: None
        worker_patch = types.ModuleType("vllm_hcu.patch.worker")
        worker_patch.apply_worker_patches = lambda c: None
        return patch.dict(sys.modules, {
            NAME: helper,
            "vllm_hcu.patch.runtime_state": runtime,
            "vllm_hcu.patch.worker": worker_patch,
        })

    def test_default_path_never_imports_mooncake_or_constructs_pool(self):
        # A missing optional dependency must not break normal/RDMA execution.
        with self.worker_modules(None), patch.dict(sys.modules, {"mooncake": None}):
            worker = load_worker(False)(config(), 0, 0, "test")
            for tag in ["weights", "kv_cache", "other"]:
                self.assertEqual(worker._maybe_get_memory_pool_context(tag), ("original", tag))
            worker.shutdown()
            self.assertEqual(worker.calls, ["weights", "kv_cache", "other", "base_shutdown"])

    def test_enabled_worker_reuses_pool_and_shuts_down_parent_first(self):
        events = []
        class Pool:
            def __init__(self, device): events.append(("create", device))
            def use(self): return "fabric-context"
            def close(self): events.append(("close", worker.calls[-1]))
        helper = types.SimpleNamespace(validate_fabric_pool_config=self.module.validate_fabric_pool_config,
                                       MooncakeFabricPool=Pool)
        with self.worker_modules(helper):
            worker = load_worker(True)(config(), 0, 0, "test")
            self.assertEqual(worker._maybe_get_memory_pool_context("weights"), ("original", "weights"))
            scope_helper = types.ModuleType("vllm_hcu.v1.mooncake_kv_allocation")
            selected = []
            scope_helper.kv_cache_allocation_scope = lambda *a, **kw: selected.append((a, kw)) or "selection-only"
            worker.model_runner = object()
            worker.use_v2_model_runner = True
            with patch.dict(sys.modules, {scope_helper.__name__: scope_helper}):
                self.assertEqual(worker._maybe_get_memory_pool_context("kv_cache"), "selection-only")
            self.assertEqual(events, [])  # Selecting a worker allocates nothing.
            self.assertIs(selected[0][0][0], worker.model_runner)
            self.assertTrue(selected[0][1]["use_v2_model_runner"])
            for _ in range(2):
                self.assertEqual(worker._get_mooncake_fabric_pool().use(), "fabric-context")
            worker.shutdown()
            self.assertIsNone(worker._mooncake_fabric_pool)
        self.assertEqual(events, [("create", "cuda:7"), ("close", "base_shutdown")])

    @contextmanager
    def fake_runtime(self, backend=1):
        events = []
        current = [None]
        class Device:
            def __init__(self, name):
                self.type, index = str(name).split(":")
                self.index = int(index)
            def __str__(self): return f"{self.type}:{self.index}"
        @contextmanager
        def device(d):
            previous = current[0]
            current[0] = d.index
            try: yield
            finally: current[0] = previous
        @contextmanager
        def use_pool(pool, device):
            events.append(("enter", current[0], device.index))
            try: yield
            finally: events.append(("exit", current[0]))
        class Allocator:
            def allocator(self): return "native-allocator"
            def __del__(self): events.append("allocator_free")
        class Pool:
            def __init__(self, allocator, use_on_oom):
                events.append(("pool_create", allocator, use_on_oom))
                self.id = (1, 2)
            def __del__(self): events.append("pool_free")
        def get_allocator(d):
            events.append(("allocator", d.index, current[0]))
            return Allocator()
        def probe(index):
            events.append(("probe", index, current[0]))
            return backend
        torch = types.ModuleType("torch")
        torch.device = Device
        torch.cuda = types.SimpleNamespace(device=device, MemPool=Pool, use_mem_pool=use_pool,
            synchronize=lambda d: events.append(("sync", d.index)))
        allocator_module = types.ModuleType("mooncake.allocator")
        allocator_module.MemoryBackend = types.SimpleNamespace(USE_CUMEMCREATE=1)
        allocator_module.NVLinkAllocator = types.SimpleNamespace(
            _get_so_path=lambda: "/fake/allocator.so", get_allocator=get_allocator)
        with patch.dict(sys.modules, {"torch": torch, "mooncake.allocator": allocator_module}), \
             patch.object(self.module.ctypes, "CDLL", return_value=types.SimpleNamespace(mc_allocator_probe=probe)), \
             patch.object(self.module.atexit, "register"), patch.object(self.module.atexit, "unregister"):
            yield events, current

    def test_pool_binds_worker_device_restores_context_and_retains_owner(self):
        with self.fake_runtime() as (events, current):
            owner = self.module.MooncakeFabricPool("cuda:7")
            self.assertIn(("probe", 7, 7), events)
            self.assertIn(("allocator", 7, 7), events)
            self.assertIn(("pool_create", "native-allocator", False), events)
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                with owner.use():
                    self.assertEqual(current[0], 7)
                    raise RuntimeError("allocation failed")
            self.assertIsNone(current[0])
            self.assertIn(("exit", 7), events)
            gc.collect()
            self.assertNotIn("pool_free", events)
            owner.close()
            self.assertLess(events.index("pool_free"), events.index("allocator_free"))
            owner.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                with owner.use(): pass

    def test_no_fallback_for_unsupported_or_unknown_backends(self):
        for backend in [0, -1, -2, 42]:
            with self.fake_runtime(backend) as (events, current):
                with self.assertRaisesRegex(RuntimeError, "refusing ordinary-memory fallback"):
                    self.module.MooncakeFabricPool("cuda:3")
                self.assertEqual(events, [("probe", 3, 3)])
                self.assertIsNone(current[0])

    def test_enabled_missing_mooncake_is_actionable_error(self):
        with self.fake_runtime(), patch.dict(sys.modules, {"mooncake.allocator": None}):
            with self.assertRaisesRegex(RuntimeError, "requires a Mooncake wheel"):
                self.module.MooncakeFabricPool("cuda:0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
