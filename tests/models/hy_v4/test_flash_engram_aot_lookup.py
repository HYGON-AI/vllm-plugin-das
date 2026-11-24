# SPDX-License-Identifier: Apache-2.0
"""Real Torch CPU tests of the MR493 FlashEmbedding AST; NOT HCU/Inductor.

Default: stdlib source contracts only. Opt in with
HCU_HYV4_ENGRAM_NATIVE_CPU=1. The parent starts a fresh -I -S child with all
GPU selectors=-1 and imports real Torch from the interpreter's purelib only.
No production plugin/Core imports, pinning, real UVA, HCU kernels, downloads,
or global installs. TP=1, logging, weight-loader/attrs, row-lookup and UVA are
explicit stubs; nn.Module/Parameter/state_dict and Dynamo AOT are real.

HCU_HYV4_ENGRAM_SOURCE may select an isolated engram.py snapshot.
HCU_HYV4_ENGRAM_LEGACY_SOURCE optionally verifies/uses the exact old source.
The standalone legacy fixture below is the old _lookup body (remote whole-file
SHA256 059fc060277aba3509253ceba9687435f2d03e01d2eab7229d33665f769fb3f3).
The AOT backend is an explicitly serializable FX interpreter, NOT Inductor;
serialization itself is intentionally unsupported and is not tested.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import ModuleType, SimpleNamespace as NS
import unittest

SOURCE_ENV = "HCU_HYV4_ENGRAM_SOURCE"
LEGACY_ENV = "HCU_HYV4_ENGRAM_LEGACY_SOURCE"
NATIVE_ENV = "HCU_HYV4_ENGRAM_NATIVE_CPU"
CHILD_ENV = "_HCU_HYV4_ENGRAM_CPU_CHILD"
GPU_MASKS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
             "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "NVIDIA_VISIBLE_DEVICES")
LEGACY_LOOKUP = "def _lookup(self, parameter, indices, out, *, scale=False):\n    view_name = '_uva_scale' if scale else '_uva_weight'\n    view = getattr(self, view_name)\n    if view is None:\n        view = get_accelerator_view_from_cpu_tensor(parameter) if self.cpu_offload else parameter\n        setattr(self, view_name, view)\n    row_bytes = parameter.element_size() * (1 if scale else self.dim)\n    _lookup_ple_embedding_from_pinned_kernel[indices.numel(),](view.view(torch.uint8), indices, out.view(torch.uint8), row_bytes, self.vocab_start, self.vocab_end, BLOCK_D=triton.next_power_of_2(row_bytes))\n"


def _source():
    default = (Path(__file__).resolve().parents[3]
               / "vllm_hcu/models/hy_v4_flash/nvidia/engram.py")
    return Path(os.environ.get(SOURCE_ENV, str(default)))


def _embedding_ast(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == "FlashEmbedding")


class PortableSourceContracts(unittest.TestCase):
    def test_extracts_actual_class_and_required_methods_without_plugin_import(self):
        cls = _embedding_ast(_source())
        self.assertEqual([ast.unparse(b) for b in cls.bases], ["nn.Module"])
        methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        self.assertTrue({"__init__", "_lookup", "_load_shard", "load_weight", "forward"}
                        <= methods)

    def test_frozen_legacy_lookup_matches_supplied_original_when_available(self):
        fixture = ast.parse(LEGACY_LOOKUP).body[0]
        self.assertEqual(fixture.name, "_lookup")
        if not os.environ.get(LEGACY_ENV):
            self.skipTest("optional exact legacy source not supplied; frozen fixture available")
        old = next(n for n in _embedding_ast(Path(os.environ[LEGACY_ENV])).body
                   if isinstance(n, ast.FunctionDef) and n.name == "_lookup")
        self.assertEqual(ast.dump(old, include_attributes=False),
                         ast.dump(fixture, include_attributes=False))


@unittest.skipUnless(os.environ.get(NATIVE_ENV) == "1",
                     "opt-in real Torch isolated CPU lane; not HCU/Inductor")
class NativeIsolatedCPUContracts(unittest.TestCase):
    def test_real_torch_contracts_in_fresh_masked_child(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                   VLLM_PLUGINS="__disabled__", HCU_ADAPTER_HOST="none",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        env.update({k: "-1" for k in GPU_MASKS})
        env[CHILD_ENV] = "1"
        env[SOURCE_ENV] = str(_source().resolve())
        # -S avoids site .pth hooks; do not mutate installed Python packages.
        bootstrap = ("import runpy,sys,sysconfig;"
                     "sys.path.insert(0,sysconfig.get_paths()['purelib']);"
                     "sys.argv=[sys.argv[1],'--native-cpu-child'];"
                     "runpy.run_path(sys.argv[0],run_name='__main__')")
        run = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", bootstrap,
                              str(Path(__file__).resolve())], env=env,
                             capture_output=True, text=True, timeout=180)
        self.assertEqual(run.returncode, 0, run.stdout + "\n" + run.stderr)
        records = [l.split("=", 1)[1] for l in run.stdout.splitlines()
                   if l.startswith("ENGRAM_LOOKUP_CPU_RESULT=")]
        self.assertEqual(len(records), 1, run.stdout)
        result = json.loads(records[0])
        self.assertTrue(result["passed"], result)
        self.assertFalse(result["cuda_initialized"])
        self.assertGreaterEqual(result["tests_run"], 6)
        self.assertTrue(result["old_aot_keyerror_reproduced"])
        self.assertTrue(result["new_aot_replays_passed"])
        self.assertEqual(result["skipped"], 0, result)
        print("ENGRAM_LOOKUP_CPU_RESULT=" + records[0])


def _native_cpu_child():
    if os.environ.get(CHILD_ENV) != "1" or any(os.environ.get(k) != "-1" for k in GPU_MASKS):
        raise RuntimeError("isolated CPU child and all GPU selectors=-1 required")
    import torch
    from torch import nn
    from torch._dynamo.aot_compile import aot_compile_fullgraph
    from torch._dynamo.aot_compile_types import SerializableCallable
    from torch._dynamo.hooks import Hooks

    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    source = _source()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    evidence = {"old_aot_keyerror_reproduced": False, "new_aot_replays_passed": False,
                "aot_backend": "SerializableCallable wrapping FX graph.forward; NOT Inductor",
                "aot_runs": []}

    def lookup_rows_cpu_stub(table, indices, out, vocab_start, vocab_end):
        # Float gather avoids requiring FP8 index_select support on CPU. Cast back
        # before the actual production forward applies FP8 row scales.
        valid = (indices >= vocab_start) & (indices < vocab_end)
        local = (indices - vocab_start).clamp(0, table.shape[0] - 1).reshape(-1)
        values = torch.index_select(table.float(), 0, local)
        values = values.reshape(*indices.shape, *table.shape[1:])
        mask = valid.reshape(*indices.shape, *([1] * (table.ndim - 1)))
        out.copy_(torch.where(mask, values, 0).to(out.dtype))

    class RawByteKernelCPUStub:
        def __getitem__(self, grid):
            def launch(weight, indices, output, row_bytes, start, end, **kwargs):
                lookup_rows_cpu_stub(weight.reshape(-1, row_bytes), indices,
                                     output.reshape(*indices.shape, row_bytes), start, end)
            return launch

    def default_weight_loader_stub(parameter, loaded):
        with torch.no_grad():
            parameter.copy_(loaded)

    def set_weight_attrs_stub(parameter, attrs):
        for key, value in attrs.items():
            setattr(parameter, key, value)

    def extract(legacy=False):
        chosen = (Path(os.environ[LEGACY_ENV])
                  if legacy and os.environ.get(LEGACY_ENV) else source)
        cls = deepcopy(_embedding_ast(chosen))
        if legacy and not os.environ.get(LEGACY_ENV):
            cls.body = [ast.parse(LEGACY_LOOKUP).body[0]
                        if isinstance(n, ast.FunctionDef) and n.name == "_lookup" else n
                        for n in cls.body]
        uva_calls = []

        def fake_uva_tensor(parameter):
            # Detached clone is deliberately NOT a real UVA mapping. It exposes
            # stale-cache bugs on reload and cannot register as an nn.Parameter.
            uva_calls.append(parameter)
            return parameter.detach().clone()

        namespace = {"__name__": __name__, "torch": torch, "nn": nn,
                     "get_tensor_model_parallel_world_size": lambda: 1,
                     "get_tensor_model_parallel_rank": lambda: 0,
                     "head_range": lambda sizes, tp, rank: (0, len(sizes)),
                     "is_uva_available": lambda: True,
                     "get_accelerator_view_from_cpu_tensor": fake_uva_tensor,
                     "lookup_rows": lookup_rows_cpu_stub,
                     "_lookup_ple_embedding_from_pinned_kernel": RawByteKernelCPUStub(),
                     "triton": NS(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
                     "default_weight_loader": default_weight_loader_stub,
                     "set_weight_attrs": set_weight_attrs_stub,
                     "logger": NS(info=lambda *a, **kw: None)}
        # AOT guard serialization pickles the class by module/name. This is
        # only an AST fixture holder, not a replacement production/Core module.
        module_name = "_hcu_engram_lookup_ast_" + ("legacy" if legacy else "active")
        holder = ModuleType(module_name)
        namespace["__name__"] = module_name
        holder.__dict__.update(namespace)
        namespace = holder.__dict__
        sys.modules[module_name] = holder
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                     str(chosen), "exec"), namespace)
        return namespace["FlashEmbedding"], namespace, uva_calls

    def table(dtype, shift=0):
        return ((torch.arange(48, device="cpu").reshape(12, 4) % 8) / 2 + shift).to(dtype)

    def make(dtype, *, offload=False, legacy=False):
        cls, namespace, calls = extract(legacy)
        # Exercise the actual resident constructor. For offload, switch the flag
        # AFTER construction: no pin_memory allocator or real UVA is exercised.
        with torch.device("cpu"):
            module = cls([5, 7], 4, dtype, False)
        module.cpu_offload = offload
        module.load_weight("weight", table(dtype))
        if dtype == torch.float8_e4m3fn:
            module.load_weight("weight_scale_inv", torch.arange(1, 13).to(torch.bfloat16) / 4)
        return module, namespace, calls

    def indices(rows):
        return ((torch.arange(rows * 2, device="cpu").reshape(rows, 2) * 3) % 14) - 1

    def expected(module, ids):
        valid = (ids >= 0) & (ids < 12)
        safe = ids.clamp(0, 11)
        values = module.embedding.weight.detach().float()[safe]
        if module.fp8:
            values = values * module.embedding.weight_scale_inv.detach().float()[safe].unsqueeze(-1)
        return torch.where(valid.unsqueeze(-1), values, 0)

    def registry(module):
        # include recursive raw _parameters keys: named_parameters de-duplicates
        # aliases and alone would miss exactly the original regression.
        return {"parameters": tuple((name, tuple(child._parameters))
                                     for name, child in module.named_modules()),
                "state_dict": tuple(module.state_dict()),
                "named_parameters": tuple(n for n, _ in module.named_parameters()),
                "dict_cache": tuple(k for k in ("_uva_weight", "_uva_scale")
                                    if k in module.__dict__)}

    class FXSerializableCPU(SerializableCallable):
        def __init__(self, graph):
            self.graph = graph

        def __call__(self, *args, **kwargs):
            return self.graph(*args, **kwargs)

        @classmethod
        def serialize_compile_artifacts(cls, fn):
            raise NotImplementedError("CPU replay witness only; serialization not qualified")

        @classmethod
        def deserialize_compile_artifacts(cls, data):
            raise NotImplementedError("CPU replay witness only; serialization not qualified")

    def compile_aot(module, ids):
        graphs = []

        def backend(graph, example_inputs):
            graphs.append(graph)
            return FXSerializableCPU(graph)

        compiled = aot_compile_fullgraph(module.forward, ((ids,), {}), hooks=Hooks(),
                                        backend=backend, dynamic=True)
        # AOTCompiledFunction carries the unbound forward signature, as does
        # Torch AOTCompiledModel: replay receives the Module explicitly.
        def replay(hashes):
            return compiled(module, hashes)

        replay.aot_compiled_function = compiled
        return replay, graphs

    class RealTorchLookupContracts(unittest.TestCase):
        def tearDown(self):
            torch._dynamo.reset()
            self.assertFalse(torch.cuda.is_initialized())

        def test_resident_bf16_and_fp8_scale_do_not_change_registry(self):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                with self.subTest(dtype=str(dtype)):
                    module, namespace, calls = make(dtype)
                    before = registry(module)
                    self.assertEqual(before["state_dict"], ("embedding.weight",) +
                                     (("embedding.weight_scale_inv",) if module.fp8 else ()))
                    seen = []

                    def spy(param, *args):
                        seen.append(param)
                        return lookup_rows_cpu_stub(param, *args)

                    namespace["lookup_rows"] = spy
                    for rows in (3, 1, 7, 0, 3):
                        ids = indices(rows).roll(1, 0)
                        torch.testing.assert_close(module(ids), expected(module, ids), rtol=0, atol=0)
                        self.assertEqual(registry(module), before)
                    self.assertTrue(all(v is module.embedding.weight or
                                        v is module.embedding.weight_scale_inv for v in seen))
                    self.assertEqual(calls, [])
                    self.assertIsNone(module._uva_weight)
                    self.assertIsNone(module._uva_scale)

        def test_resident_reload_and_parameter_replacement_never_read_old_alias(self):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                with self.subTest(dtype=str(dtype)):
                    module, _, calls = make(dtype)
                    before = registry(module)
                    ids = indices(3)
                    first = module(ids).clone()
                    module.load_weight("weight", table(dtype, 2))
                    if module.fp8:
                        module.load_weight("weight_scale_inv", torch.full((12,), 2, dtype=torch.bfloat16))
                    second = module(ids)
                    torch.testing.assert_close(second, expected(module, ids), rtol=0, atol=0)
                    self.assertFalse(torch.equal(first, second))
                    # A new Parameter object is a stronger witness than in-place
                    # reload: an accidentally retained resident alias is stale.
                    module.embedding.weight = nn.Parameter(table(dtype, 3), requires_grad=False)
                    if module.fp8:
                        module.embedding.weight_scale_inv = nn.Parameter(
                            torch.full((12,), 3, dtype=torch.bfloat16), requires_grad=False)
                    torch.testing.assert_close(module(ids), expected(module, ids), rtol=0, atol=0)
                    self.assertEqual(registry(module), before)
                    self.assertEqual(calls, [])

        def test_offload_fake_uva_cache_reuse_invalidation_and_scale(self):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                with self.subTest(dtype=str(dtype)):
                    module, _, calls = make(dtype, offload=True)
                    before = registry(module)
                    ids = indices(3)
                    torch.testing.assert_close(module(ids), expected(module, ids), rtol=0, atol=0)
                    count = 2 if module.fp8 else 1
                    self.assertEqual(len(calls), count)
                    cached_weight, cached_scale = module._uva_weight, module._uva_scale
                    self.assertIs(type(cached_weight), torch.Tensor)
                    if module.fp8:
                        self.assertIs(type(cached_scale), torch.Tensor)
                    for rows in (1, 5, 3):
                        ids = indices(rows)
                        torch.testing.assert_close(module(ids), expected(module, ids), rtol=0, atol=0)
                        self.assertIs(module._uva_weight, cached_weight)
                        self.assertIs(module._uva_scale, cached_scale)
                    self.assertEqual(len(calls), count)
                    module.load_weight("weight", table(dtype, 2))
                    self.assertIsNone(module._uva_weight)
                    self.assertIsNone(module._uva_scale)
                    if module.fp8:
                        module.load_weight("weight_scale_inv", torch.full((12,), 2, dtype=torch.bfloat16))
                    torch.testing.assert_close(module(ids), expected(module, ids), rtol=0, atol=0)
                    self.assertEqual(len(calls), 2 * count)
                    self.assertIsNot(module._uva_weight, cached_weight)
                    if module.fp8:
                        self.assertIsNot(module._uva_scale, cached_scale)
                    self.assertEqual(registry(module), before)

        def test_legacy_eager_reproduces_parameter_alias_registration(self):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                with self.subTest(dtype=str(dtype)):
                    module, _, _ = make(dtype, legacy=True)
                    module(indices(3))
                    self.assertIs(module._parameters["_uva_weight"], module.embedding.weight)
                    self.assertNotIn("_uva_weight", module.__dict__)
                    self.assertIn("_uva_weight", module.state_dict())
                    if module.fp8:
                        self.assertIs(module._parameters["_uva_scale"], module.embedding.weight_scale_inv)
                        self.assertNotIn("_uva_scale", module.__dict__)

        def test_actual_dynamo_aot_legacy_replay_raises_uva_weight_keyerror(self):
            module, _, _ = make(torch.bfloat16, legacy=True)
            compiled, graphs = compile_aot(module, indices(3))
            self.assertEqual(len(graphs), 1)
            # Guarded Torch reports the alias migration first. Core033's normal
            # evaluate_guards=False policy executes the same AOT bytecode without
            # this check, exposing the exact service KeyError. This local fixture
            # exercises both paths; it does not change Core or service settings.
            with self.assertRaisesRegex(RuntimeError, r"KeyError on .*_uva_weight"):
                for _ in range(3):
                    compiled(indices(3))
            compiled.aot_compiled_function.disable_guard_check()
            with self.assertRaises(KeyError) as caught:
                for _ in range(3):
                    compiled(indices(3))
            self.assertEqual(caught.exception.args, ("_uva_weight",))
            evidence["old_aot_keyerror_reproduced"] = True

        def test_actual_dynamo_aot_new_repeated_shapes_registry_and_reload(self):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                with self.subTest(dtype=str(dtype)):
                    torch._dynamo.reset()
                    module, _, calls = make(dtype)
                    before = registry(module)
                    compiled, graphs = compile_aot(module, indices(3))
                    self.assertEqual(registry(module), before)
                    for rows in (3, 5, 2, 3):
                        ids = indices(rows).roll(1, 0)
                        torch.testing.assert_close(compiled(ids), expected(module, ids), rtol=0, atol=0)
                        self.assertEqual(registry(module), before)
                    module.load_weight("weight", table(dtype, 2))
                    if module.fp8:
                        module.load_weight("weight_scale_inv", torch.full((12,), 2, dtype=torch.bfloat16))
                    torch.testing.assert_close(compiled(indices(3)), expected(module, indices(3)), rtol=0, atol=0)
                    self.assertEqual(registry(module), before)
                    self.assertEqual(calls, [])
                    self.assertEqual(len(graphs), 1)
                    evidence["aot_runs"].append({"dtype": str(dtype), "graphs": len(graphs),
                                                  "replays": 5, "registry_unchanged": True})
            evidence["new_aot_replays_passed"] = len(evidence["aot_runs"]) == 2

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RealTorchLookupContracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence.update(passed=result.wasSuccessful(), tests_run=result.testsRun,
                    failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                    torch=torch.__version__, source=str(source), source_sha256=source_hash,
                    legacy_source=os.environ.get(LEGACY_ENV, "frozen _lookup fixture, rest active class"),
                    cuda_initialized=torch.cuda.is_initialized(),
                    scope="real Torch nn.Module/Parameter/Dynamo AOT; AST class with explicit CPU lookup/TP/loader/fake-UVA stubs; NOT HCU/Inductor/Graph")
    print("ENGRAM_LOOKUP_CPU_RESULT=" + json.dumps(evidence, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--native-cpu-child"]:
        raise SystemExit(_native_cpu_child())
    unittest.main(verbosity=2)
