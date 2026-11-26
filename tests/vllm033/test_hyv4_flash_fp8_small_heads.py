"""CPU contracts for the actual Flash leaf; no attention math is simulated.

The native provider and tensor storage are explicit boundary doubles. The class
under test is compiled from the repository source, with no copied Core code.
Run with pytest or ``python -m unittest discover -s tests/vllm033 -p '*small_heads.py'``.
"""

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch


class Device:
    def __init__(self, kind="cuda", index=2):
        self.type, self.index = kind, index

    def __eq__(self, other):
        return isinstance(other, Device) and vars(self) == vars(other)


class Tensor:
    def __init__(self, shape, dtype="bfloat16", device=None, parent=None,
                 contiguous=True, last_stride=1):
        self.shape = tuple(shape)
        self.dtype, self.device = dtype, device or Device()
        self.parent, self.contiguous_layout = parent, contiguous
        self.last_stride = last_stride
        self.identity = object()

    @property
    def ndim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self.contiguous_layout

    def contiguous(self):
        if self.is_contiguous():
            return self
        return Tensor(self.shape, self.dtype, self.device, self)

    def stride(self, axis):
        assert axis == -1
        return self.last_stride

    def clone(self):
        return Tensor(self.shape, self.dtype, self.device, self)

    def view(self, dtype):
        assert dtype == "uint8"
        return Tensor(self.shape, dtype, self.device, self)

    def unsqueeze(self, axis):
        dimensions = list(self.shape)
        if axis < 0:
            axis += self.ndim + 1
        dimensions.insert(axis, 1)
        return Tensor(dimensions, self.dtype, self.device, self)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if Ellipsis in indices:
            i = indices.index(Ellipsis)
            missing = self.ndim - len(indices) + 1
            indices = (*indices[:i], *([slice(None)] * missing), *indices[i + 1:])
        indices = (*indices, *([slice(None)] * (self.ndim - len(indices))))
        dimensions = [len(range(*selected.indices(length)))
                      for length, selected in zip(self.shape, indices)
                      if isinstance(selected, slice)]
        return Tensor(dimensions, self.dtype, self.device, self)

    def __bool__(self):
        raise AssertionError("No device value read is allowed")

    def item(self):
        raise AssertionError("No device synchronization is allowed")


class WindowProducer:
    window = 512

    def __init__(self):
        self.topk_indices_buffer = Tensor((4096, 2048), "int32")


class HiSparseGroup:
    def cache(self, index):
        return self


class Provider:
    def __init__(self):
        self.calls, self.fresh = [], []
        self.error = None

    def get_metadata(self):
        scheduler = NS(initialized=False)
        self.fresh.append(scheduler)
        return scheduler, None

    def with_kvcache(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        q = kwargs["q"]
        return (Tensor((*q.shape[:3], 512), device=q.device),
                Tensor((q.shape[0], q.shape[2], q.shape[1]), device=q.device))


class SmallHeadContracts(unittest.TestCase):
    def setUp(self):
        self.source = (Path(__file__).resolve().parents[2] /
                       "adapter_extension/src/hcu_vllm_das/models/"
                       "hy_v4_flash/sparse_backend.py")
        tree = ast.parse(self.source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == "FlashSparseImpl")
        self.provider = Provider()
        self.master = True
        self.master_reads = 0
        self.cuda = NS(available=True, current=2, arch="gfx938:sramecc+",
                       property_reads=[])
        self.cuda.is_available = lambda: self.cuda.available
        self.cuda.current_device = lambda: self.cuda.current

        def properties(index):
            self.cuda.property_reads.append(index)
            return NS(gcnArchName=self.cuda.arch, multi_processor_count=64)

        self.cuda.get_device_properties = properties

        def master_policy():
            self.master_reads += 1
            return self.master

        class Core:
            def __init__(self, **kwargs):
                self._indexer = kwargs.get("indexer")
                self.index_group = kwargs.get("index_group", object())
                self.index_group_index = 0
                self.fp8_decode_padded_heads = 64
                self.softmax_scale = .0625
                self.sinks = kwargs.get("sinks")

            def _sinks_for_query(self, q, axis, padded):
                if self.sinks is None:
                    return None
                assert self.sinks.shape == (q.shape[axis],)
                assert self.sinks.dtype == "float32"
                assert self.sinks.device == q.device
                if padded == q.shape[axis]:
                    return self.sinks
                return Tensor((padded,), "float32", q.device, self.sinks)

            def _fp8_flash_mla_kernel(self, *args):
                return "core-out", "core-lse"

        def pad(t, values):
            return Tensor((*t.shape[:-2], t.shape[-2] + values[3],
                           t.shape[-1]), t.dtype, t.device, t)

        package = "_hy4_small_head_contract"
        parent = ModuleType(package)
        parent.__path__ = []
        swa = ModuleType(package + ".swa")
        swa.FlashWindowIndexer = WindowProducer
        # Load the actual pure tuple predicate for the added gfx936 branch.
        # This fixture remains TP4/64CU, so gfx936 stays an unqualified fallback.
        gate_name = package + ".fullgraph_runner"
        spec = importlib.util.spec_from_file_location(
            gate_name, self.source.with_name("fullgraph_runner.py"))
        gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
        # Constructor selection executes the real pure format policy. Only its
        # host discovery dependency is doubled; no codec decision is mocked.
        runtime = ModuleType(package + ".runtime")
        runtime.is_hcu_platform = lambda: True
        policy_name = package + ".flashmla_ops"
        policy_spec = importlib.util.spec_from_file_location(
            policy_name, self.source.with_name("flashmla_ops.py"))
        policy = importlib.util.module_from_spec(policy_spec)
        group = ModuleType("vllm.v1.attention.backends.mla.index_group")
        group.HiSparseMLAIndexGroup = HiSparseGroup
        self.modules = patch.dict(sys.modules, {
            package: parent, package + ".swa": swa, gate_name: gate,
            package + ".runtime": runtime, policy_name: policy,
            "vllm.v1.attention.backends.mla.index_group": group,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        policy_spec.loader.exec_module(policy)
        self.namespace = {
            "__package__": package, "__name__": package + ".sparse_backend",
            "CoreSinkImpl": Core,
            "torch": NS(cuda=self.cuda, device=Device, bfloat16="bfloat16",
                        uint8="uint8", int32="int32"),
            "F": NS(pad=pad), "fp8_fusions_enabled": master_policy,
            "get_current_vllm_config": lambda: NS(parallel_config=NS(
                tensor_parallel_size=4,
                prefill_context_parallel_size=1, decode_context_parallel_size=1)),
            "get_flashmla_ops": lambda: self.provider,
            "is_hcu_platform": lambda: True,
            "install_hisparse_prewrite_hook": lambda *args: None,
        }
        module = ast.Module(body=[ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0),
            node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(self.source), "exec"),
             self.namespace)
        self.cls = self.namespace["FlashSparseImpl"]

    def make(self, heads=8, width=2048, window=False, hisparse=False):
        sinks = Tensor((heads,), "float32", contiguous=False)
        obj = self.cls(indexer=WindowProducer() if window else None,
                       index_group=HiSparseGroup() if hisparse else object(),
                       sinks=sinks)
        q = Tensor((1, 33, heads, 576), contiguous=False)
        cache = Tensor((64, 64, 656), "uint8")
        indices = Tensor((1, 33, width), "int32")
        metadata = NS(scheduler_metadata=NS(initialized=True, h_q=64, topk=2048),
                      dummy_block_table=object(), cache_lens=object())
        return [obj, q, cache, indices, metadata]

    def invoke(self, values):
        obj, q, cache, indices, metadata = values
        return obj._fp8_flash_mla_kernel(q, cache, indices, metadata)

    def test_real_heads_and_independent_scheduler(self):
        for heads, width in ((8, 2048), (16, 512)):
            with self.subTest(heads=heads, width=width):
                values = self.make(heads, width)
                obj, q, cache, ids, metadata = values
                output, lse = self.invoke(values)
                call = self.provider.calls[-1]
                self.assertEqual(call["q"].shape, (1, 33, heads, 576))
                self.assertTrue(call["q"].is_contiguous())
                self.assertIs(call["q"].parent, q)
                self.assertEqual(call["attn_sink"].shape, (heads,))
                self.assertTrue(call["attn_sink"].is_contiguous())
                self.assertEqual(call["attn_sink"].dtype, "float32")
                self.assertIs(call["attn_sink"].parent, obj.sinks)
                self.assertIsNot(call["tile_scheduler_metadata"],
                                 metadata.scheduler_metadata)
                self.assertIsNone(call["block_table"])
                self.assertIsNone(call["cache_seqlens"])
                self.assertIs(call["indices"].parent, ids)
                self.assertEqual(ids.shape, (1, 33, width))
                self.assertEqual(output.shape, (1, 33, heads, 512))
                self.assertEqual(lse.shape, (1, heads, 33))
                self.assertEqual(obj.fp8_decode_padded_heads, 64)
                self.assertEqual(metadata.scheduler_metadata.h_q, 64)

    def test_window_prefix_and_compressed_have_fresh_schedulers(self):
        window = self.make(16, window=True)
        compressed = self.make(8)
        compressed[-1] = window[-1]
        for values in (compressed, window, compressed):
            self.invoke(values)
        calls = self.provider.calls
        self.assertEqual([c["indices"].shape[-1] for c in calls], [2048, 512, 2048])
        self.assertEqual(len({id(c["tile_scheduler_metadata"]) for c in calls}), 3)
        self.assertEqual([c["q"].shape[2] for c in calls], [8, 16, 8])

    def test_metadata_none_and_no_sink(self):
        values = self.make()
        values[0].sinks = None
        values[-1] = None
        self.invoke(values)
        self.assertEqual(self.provider.calls[-1]["q"].shape[2], 8)
        self.assertIsNone(self.provider.calls[-1]["attn_sink"])
        self.assertEqual(len(self.provider.fresh), 1)

    def test_master_frozen_before_capture_and_off_keeps_h64(self):
        values = self.make()
        reads = self.master_reads
        properties = list(self.cuda.property_reads)
        self.master = False
        self.cuda.arch = "gfx936"
        self.invoke(values)
        self.assertEqual(self.provider.calls[-1]["q"].shape[2], 8)
        self.assertEqual(self.master_reads, reads)
        self.assertEqual(self.cuda.property_reads, properties)
        self.cuda.arch = "gfx938"
        off = self.make()
        self.invoke(off)
        self.assertEqual(self.provider.calls[-1]["q"].shape[2], 64)
        self.assertIs(self.provider.calls[-1]["tile_scheduler_metadata"],
                      off[-1].scheduler_metadata)

    def test_unqualified_shapes_preserve_h64(self):
        mutations = {
            "head32": lambda v: setattr(v[1], "shape", (1, 33, 32, 576)),
            "fp16": lambda v: setattr(v[1], "dtype", "float16"),
            "q512": lambda v: setattr(v[1], "shape", (1, 33, 8, 512)),
            "batch2": lambda v: setattr(v[1], "shape", (2, 33, 8, 576)),
            "cache584": lambda v: setattr(v[2], "shape", (64, 64, 584)),
            "cachebf16": lambda v: setattr(v[2], "dtype", "bfloat16"),
            "cache_strided": lambda v: setattr(v[2], "contiguous_layout", False),
            "cache_device": lambda v: setattr(v[2], "device", Device(index=1)),
            "q_device": lambda v: setattr(v[1], "device", Device(index=1)),
            "ids_int64": lambda v: setattr(v[3], "dtype", "int64"),
            "ids1024": lambda v: setattr(v[3], "shape", (1, 33, 1024)),
            "ids_rows": lambda v: setattr(v[3], "shape", (1, 32, 2048)),
            "ids_device": lambda v: setattr(v[3], "device", Device(index=1)),
            "ids_stride": lambda v: setattr(v[3], "last_stride", 2),
            "padding128": lambda v: setattr(v[0], "fp8_decode_padded_heads", 128),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                values = self.make()
                mutate(values)
                # No math: malformed sinks are irrelevant to the guard itself.
                values[0].sinks = None
                self.invoke(values)
                self.assertEqual(self.provider.calls[-1]["q"].shape[2],
                                 values[0].fp8_decode_padded_heads)
                self.assertIs(self.provider.calls[-1]["tile_scheduler_metadata"],
                              values[-1].scheduler_metadata)

    def test_other_arch_hidden_gpu_and_hisparse_keep_h64(self):
        for name in ("arch", "hidden", "hisparse"):
            with self.subTest(name=name):
                self.cuda.arch = "gfx936" if name == "arch" else "gfx938"
                self.cuda.available = name != "hidden"
                values = self.make(hisparse=name == "hisparse")
                self.invoke(values)
                self.assertEqual(self.provider.calls[-1]["q"].shape[2], 64)

    def test_native_errors_propagate_without_retry(self):
        values = self.make()
        self.provider.error = RuntimeError("native failure")
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            self.invoke(values)
        self.assertEqual(len(self.provider.calls), 1)

    def test_non_hcu_keeps_core_leaf(self):
        values = self.make()
        values[0]._hcu_ops = None
        self.assertEqual(self.invoke(values), ("core-out", "core-lse"))
        self.assertFalse(self.provider.calls)


if __name__ == "__main__":
    unittest.main()
