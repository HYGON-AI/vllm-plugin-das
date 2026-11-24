# SPDX-License-Identifier: Apache-2.0
"""Attention compile-boundary contracts; NOT GPU/serving acceptance.

Installed portable CI runs the AST contracts and the real native-CPU boundary
case by default with Torch/Core033 and the production plugin installed. A
stdlib-only local check can select PortableAttentionBoundaryContracts explicitly.
The native lane spawns a fresh process, hides every GPU, imports the actual
production module, and adds only this op's CPU dispatch implementation there.
Context/owner/configuration doubles are explicitly labeled; no attention kernel,
model construction, GPU capture, or installed-file modification is exercised.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest
import weakref


MODEL = (Path(__file__).resolve().parents[2]
         / "adapter_extension/src/hcu_vllm_das/models/hy_v4_flash")
OP_NAME = "hcu_hyv4_flash_indexer_and_attn"
OP = "vllm::" + OP_NAME
ARGS = ["hidden_states", "q_c", "positions", "q", "kv_c_normed", "k_pe",
        "out", "layer_name"]
GPU_MASKS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
             "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
             "NVIDIA_VISIBLE_DEVICES")
CHILD_ENV = "_HCU_HYV4_ATTENTION_CPU_CHILD"


def _tree(filename="attention.py"):
    return ast.parse((MODEL / filename).read_text(encoding="utf-8"))


def _named(body, name):
    return next(n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name == name)


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and ast.unparse(n.func) == name]


def _binder():
    """Execute exact production AST, not a reimplemented binding algorithm.

    CompilationMode/config/MLA owner below are doubles, not real Core objects.
    """
    cls = _named(_tree().body, "HYV4FlashAttention")
    method = deepcopy(_named(cls.body, "_bind_attention_compile_boundary"))
    namespace = {"ref": weakref.ref, "_FLASH_ATTENTION_OP": OP,
                 "CompilationMode": NS(VLLM_COMPILE=3)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],
                type_ignores=[])), str(MODEL / "attention.py"), "exec"), namespace)
    return namespace[method.name]


class _CompilationDouble:
    def __init__(self, mode, splitting_ops):
        self.mode = mode
        self.splitting_ops = splitting_ops
        self.default_calls = []

    def set_splitting_ops_for_v1(self, **kwargs):
        self.default_calls.append(kwargs)
        self.splitting_ops = ["vllm::unified_attention", "vllm::other_default"]


class PortableAttentionBoundaryContracts(unittest.TestCase):
    def test_impl_and_fake_have_exact_mutating_void_signature(self):
        tree = _tree()
        for name in ("_flash_indexer_and_attn_impl", "_flash_indexer_and_attn_fake"):
            with self.subTest(function=name):
                fn = _named(tree.body, name)
                self.assertEqual([n.arg for n in fn.args.args], ARGS)
                self.assertFalse(fn.args.posonlyargs or fn.args.kwonlyargs
                                 or fn.args.defaults or fn.args.vararg
                                 or fn.args.kwarg)
                self.assertEqual([ast.unparse(n.annotation) for n in fn.args.args],
                                 ["torch.Tensor"] * 7 + ["str"])
                self.assertEqual(ast.unparse(fn.returns), "None")

    def test_registration_is_production_op_with_only_out_mutable(self):
        tree = _tree()
        registrations = _calls(tree, "direct_register_custom_op")
        matches = [n for n in registrations if any(
            k.arg == "op_name" and ast.literal_eval(k.value) == OP_NAME
            for k in n.keywords)]
        self.assertEqual(len(matches), 1)
        call = matches[0]
        self.assertIn(call, [n.value for n in tree.body if isinstance(n, ast.Expr)])
        kw = {k.arg: k.value for k in call.keywords}
        self.assertEqual(ast.unparse(kw["op_func"]), "_flash_indexer_and_attn_impl")
        self.assertEqual(ast.unparse(kw["fake_impl"]), "_flash_indexer_and_attn_fake")
        self.assertEqual(ast.literal_eval(kw["mutates_args"]), ["out"])
        constant = next(n for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "_FLASH_ATTENTION_OP"
                                for t in n.targets))
        self.assertEqual(ast.literal_eval(constant.value), OP)

    def test_impl_resolves_current_context_layer_and_owner_inside_leaf(self):
        fn = _named(_tree().body, "_flash_indexer_and_attn_impl")
        self.assertEqual(len(_calls(fn, "get_forward_context")), 1)
        lookups = [n for n in ast.walk(fn) if isinstance(n, ast.Subscript)
                   and ast.unparse(n) ==
                   "get_forward_context().no_compile_layers[layer_name]"]
        self.assertEqual(len(lookups), 1)
        self.assertTrue(any(isinstance(n, ast.Constant)
                            and n.value == "_hcu_flash_attention_owner"
                            for n in ast.walk(fn)))
        delegate, = _calls(fn, "owner._indexer_and_attn")
        self.assertEqual([ast.unparse(a) for a in delegate.args], ARGS[:-1])
        self.assertFalse(delegate.keywords)
        self.assertEqual(len(_calls(fn, "owner_ref")), 1)

    def test_fake_has_no_context_events_or_tensor_effects(self):
        fn = _named(_tree().body, "_flash_indexer_and_attn_fake")
        body = [n for n in fn.body if not (isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
        self.assertTrue(body)
        for statement in body:
            self.assertTrue(isinstance(statement, ast.Pass) or
                            (isinstance(statement, ast.Return) and
                             (statement.value is None or
                              (isinstance(statement.value, ast.Constant)
                               and statement.value.value is None))))

    def test_both_initializers_bind_after_mla_child_construction(self):
        for filename, name in (("attention.py", "HYV4FlashAttention"),
                               ("dfly.py", "HYV4DFlyAttention")):
            with self.subTest(attention=name):
                cls = _named(_tree(filename).body, name)
                init = _named(cls.body, "__init__")
                call, = _calls(init, "self._bind_attention_compile_boundary")
                self.assertEqual([ast.unparse(a) for a in call.args], ["vllm_config"])
                assignments = [n for n in ast.walk(init) if isinstance(n, ast.Assign)
                               and any(ast.unparse(t) == "self.mla_attn" for t in n.targets)]
                self.assertEqual(len(assignments), 1)
                self.assertLess(assignments[0].lineno, call.lineno)
                if filename == "dfly.py":
                    self.assertIn("HYV4FlashAttention", [ast.unparse(b) for b in cls.bases])
                    self.assertFalse(any(isinstance(n, ast.FunctionDef) and n.name in
                                         ("forward", "_bind_attention_compile_boundary")
                                         for n in cls.body))

    def test_forward_calls_opaque_leaf_without_inline_metadata_or_events(self):
        cls = _named(_tree().body, "HYV4FlashAttention")
        forward = _named(cls.body, "forward")
        call, = _calls(forward, "torch.ops.vllm." + OP_NAME)
        self.assertEqual([ast.unparse(a) for a in call.args],
                         ["hidden_states", "q_c", "positions", "q", "kv_c",
                          "k_pe", "out", "self.mla_attn.layer_name"])
        self.assertFalse(call.keywords)
        forbidden = {"_indexer_and_attn", "get_forward_context", "record_logical_topk_ready",
                     "Event", "record", "wait_event", "wait_stream", "current_stream",
                     "attn_metadata", "no_compile_layers", "item", "mla_attn", "indexer"}
        for node in ast.walk(forward):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else (
                    node.func.id if isinstance(node.func, ast.Name) else "")
                self.assertNotIn(name, forbidden)
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, {"attn_metadata", "no_compile_layers"})
        allocation, = _calls(forward, "torch.empty")
        projection, = _calls(forward, "self.o_proj")
        self.assertLess(allocation.lineno, call.lineno)
        self.assertLess(call.lineno, projection.lineno)

    def test_indexer_snapshot_event_and_consumer_stay_in_order_inside_leaf(self):
        cls = _named(_tree().body, "HYV4FlashAttention")
        leaf = _named(cls.body, "_indexer_and_attn")
        indexer, = _calls(leaf, "self.indexer")
        snapshot, = _calls(leaf, "self.topk_snapshot[:num_rows].copy_")
        event, = _calls(leaf, "self.mla_attn.impl.record_logical_topk_ready")
        consumer, = _calls(leaf, "self.mla_attn")
        self.assertLess(indexer.lineno, snapshot.lineno)
        self.assertLess(snapshot.lineno, event.lineno)
        self.assertLess(event.lineno, consumer.lineno)
        copy_out, = _calls(leaf, "out.copy_")
        self.assertIs(copy_out.args[0], consumer)
        forward = _named(cls.body, "forward")
        for node in ast.walk(forward):
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, {"indexer", "topk_snapshot",
                                            "topk_indices_buffer", "skip_topk"})

    def test_binder_preserves_explicit_splits_and_is_idempotent(self):
        bind = _binder()
        for initial in ([], ["vllm::user_op", "vllm::unified_attention"],
                        ["vllm::user_op", OP]):
            with self.subTest(initial=initial):
                original = list(initial)
                compilation = _CompilationDouble(3, original)
                owner = type("OwnerDouble", (), {})()
                owner.mla_attn = NS()
                config = NS(compilation_config=compilation)
                bind(owner, config)
                bind(owner, config)
                self.assertEqual(compilation.splitting_ops,
                                 initial if OP in initial else [*initial, OP])
                self.assertEqual(compilation.splitting_ops.count(OP), 1)
                self.assertEqual(original, initial)  # No in-place default/user-list mutation.
                self.assertEqual(compilation.default_calls, [])
                self.assertIs(owner.mla_attn._hcu_flash_attention_owner(), owner)

    def test_binder_materializes_core_defaults_before_appending(self):
        compilation = _CompilationDouble(3, None)
        config = NS(compilation_config=compilation,
                    parallel_config=NS(all2all_backend="double-backend", data_parallel_size=2))
        owner = type("OwnerDouble", (), {})()
        owner.mla_attn = NS()
        bind = _binder()
        bind(owner, config)
        bind(owner, config)
        self.assertEqual(compilation.default_calls,
                         [{"all2all_backend": "double-backend", "data_parallel_size": 2}])
        self.assertEqual(compilation.splitting_ops,
                         ["vllm::unified_attention", "vllm::other_default", OP])

    def test_binder_does_not_change_non_compile_modes_and_owner_is_weak(self):
        bind = _binder()
        for mode in (0, 1, 2):
            for splits in (None, [], ["vllm::user_op"]):
                with self.subTest(mode=mode, splits=splits):
                    compilation = _CompilationDouble(mode, splits)
                    layer = NS()
                    owner = type("OwnerDouble", (), {})()
                    owner.mla_attn = layer
                    bind(owner, NS(compilation_config=compilation))
                    self.assertIs(compilation.splitting_ops, splits)
                    self.assertEqual(compilation.default_calls, [])
                    self.assertIsInstance(layer._hcu_flash_attention_owner, weakref.ReferenceType)
                    self.assertIs(layer._hcu_flash_attention_owner(), owner)
                    del owner
                    gc.collect()
                    self.assertIsNone(layer._hcu_flash_attention_owner())


class NativeAttentionBoundaryContracts(unittest.TestCase):
    def test_real_production_op_fake_fullgraph_and_live_metadata_in_isolated_cpu_child(self):
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                           VLLM_PLUGINS="__disabled__", HCU_ADAPTER_HOST="none",
                           PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONOPTIMIZE="0")
        environment.update({key: "-1" for key in GPU_MASKS})
        environment[CHILD_ENV] = "1"
        run = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                              "--native-cpu-child"], env=environment,
                             capture_output=True, text=True, timeout=150)
        self.assertEqual(run.returncode, 0, run.stdout + "\n" + run.stderr)
        lines = [line for line in run.stdout.splitlines() if line.startswith("ATTENTION_CPU_RESULT=")]
        self.assertEqual(len(lines), 1, run.stdout)
        result = json.loads(lines[0].split("=", 1)[1])
        self.assertEqual(result["graphs"], 1)
        self.assertEqual(result["runtime_calls"], 4)
        self.assertFalse(result["cuda_initialized"])
        print(lines[0])


def _native_cpu_child():
    """Real production import/registration/op body; explicit context/owner doubles."""
    if os.environ.get(CHILD_ENV) != "1" or any(os.environ.get(k) != "-1" for k in GPU_MASKS):
        raise RuntimeError("CPU dispatch registration requires isolated child and all GPU masks=-1")
    import importlib
    from unittest.mock import patch

    import torch
    import vllm
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    from vllm.utils.torch_utils import direct_register_custom_op
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    assert vllm.__version__.split("+")[0] == "0.33.0", vllm.__version__
    assert not torch.cuda.is_initialized()
    attention = importlib.import_module("hcu_vllm_das.models.hy_v4_flash.attention")
    assert attention.direct_register_custom_op is direct_register_custom_op
    # Reject a stale installed module. This is source/target parity, not wheel provenance.
    assert Path(attention.__file__).read_text(encoding="utf-8") == (
        MODEL / "attention.py").read_text(encoding="utf-8"), attention.__file__
    op = getattr(torch.ops.vllm, OP_NAME).default
    schema = op._schema
    assert [a.name for a in schema.arguments] == ARGS, schema
    assert not schema.returns, schema
    assert [a.name for a in schema.arguments if a.alias_info and a.alias_info.is_write] == ["out"]
    assert not torch._C._dispatch_has_kernel_for_dispatch_key(OP, "CPU")
    # Never register a replacement namespace or monkeypatch Core/platform globals.
    # This real production op alone gets a CPU kernel, only in this short-lived child.
    cpu_library = torch.library.Library("vllm", "IMPL", "CPU")
    cpu_library.impl(OP_NAME, attention._flash_indexer_and_attn_impl)
    assert torch._C._dispatch_has_kernel_for_dispatch_key(OP, "Meta")

    layer_name = "cpu_contract.attn"
    slot = [None]
    reads = []
    calls = []

    def context_double():
        reads.append(True)
        if slot[0] is None:
            raise AssertionError("fake tracing must not read live context")
        return slot[0]

    class OwnerDouble:
        """Arithmetic witness only: no indexer, MLA, event or GPU implementation."""
        def __init__(self, bias):
            self.mla_attn = NS()
            self.bias = bias
            self.mla_attn._hcu_flash_attention_owner = weakref.ref(self)

        def _indexer_and_attn(self, hidden_states, q_c, positions, q,
                              kv_c_normed, k_pe, out):
            metadata = context_double().attn_metadata
            calls.append((self.bias, metadata))
            # Profile branch deliberately returns zeros; replay must NOT freeze it.
            if metadata is None:
                out.zero_()
            else:
                out.copy_(hidden_states * metadata["scale"] + self.bias)

    def invoke(hidden_states, q_c, positions, q, kv_c_normed, k_pe, out):
        op(hidden_states, q_c, positions, q, kv_c_normed, k_pe, out, layer_name)
        return out

    inputs = [torch.arange(12, dtype=torch.float32).reshape(3, 4),
              torch.ones(3, 2), torch.arange(3), torch.ones(3, 1, 4),
              torch.ones(3, 2), torch.ones(3, 1, 2), torch.full((3, 4), -71.0)]
    snapshots = [value.clone() for value in inputs]
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    with patch.object(attention, "get_forward_context", context_double):
        # Call the actual registered fake kernel with real FakeTensors.
        with FakeTensorMode() as mode:
            fake_inputs = [mode.from_tensor(value) for value in inputs]
            assert op(*fake_inputs, layer_name) is None
        fake_graph = make_fx(invoke, tracing_mode="fake")(*inputs)
        fake_leaves = [n for n in fake_graph.graph.nodes if n.op == "call_function"]
        assert len(fake_leaves) == 1 and fake_leaves[0].target == op, fake_graph.code
        assert not reads and not calls
        for value, before in zip(inputs, snapshots):
            torch.testing.assert_close(value, before)

        # Real Core effective default splits, with only container/owner doubles.
        compilation = CompilationConfig(mode=CompilationMode.VLLM_COMPILE,
                                        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE)
        baseline = CompilationConfig(mode=CompilationMode.VLLM_COMPILE,
                                     cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE)
        baseline.set_splitting_ops_for_v1(all2all_backend="naive", data_parallel_size=1)
        binding_owner = OwnerDouble(0)
        binding_config = NS(compilation_config=compilation,
                            parallel_config=NS(all2all_backend="naive", data_parallel_size=1))
        attention.HYV4FlashAttention._bind_attention_compile_boundary(binding_owner, binding_config)
        attention.HYV4FlashAttention._bind_attention_compile_boundary(binding_owner, binding_config)
        assert compilation.splitting_ops == [*baseline.splitting_ops, OP]
        assert compilation.splitting_ops.count(OP) == 1
        assert compilation.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
        assert binding_owner.mla_attn._hcu_flash_attention_owner() is binding_owner

        owner = OwnerDouble(5)
        context = NS(no_compile_layers={layer_name: owner.mla_attn}, attn_metadata=None)
        slot[0] = context
        compiled = torch.compile(invoke, backend=backend, fullgraph=True)
        result = compiled(*inputs)
        assert result is inputs[-1]
        torch.testing.assert_close(result, torch.zeros_like(inputs[0]))
        context.attn_metadata = {"scale": torch.full((3, 1), 2.0)}
        torch.testing.assert_close(compiled(*inputs), inputs[0] * 2 + 5)
        # Replace both context and live owner without changing tensor shapes/graph.
        replacement = OwnerDouble(17)
        slot[0] = NS(no_compile_layers={layer_name: replacement.mla_attn},
                     attn_metadata={"scale": torch.full((3, 1), 3.0)})
        torch.testing.assert_close(compiled(*inputs), inputs[0] * 3 + 17)
        slot[0].attn_metadata = {"scale": torch.full((3, 1), -1.0)}
        torch.testing.assert_close(compiled(*inputs), -inputs[0] + 17)
        assert len(graphs) == 1, len(graphs)
        leaves = [n for n in graphs[0].graph.nodes if n.op == "call_function"]
        assert len(leaves) == 1 and leaves[0].target == op, graphs[0].code
        assert len(calls) == 4 and [x[0] for x in calls] == [5, 5, 17, 17], calls
        for value, before in zip(inputs[:-1], snapshots[:-1]):
            torch.testing.assert_close(value, before)

        # Actual production lifetime/matching guards, not owner-double behavior.
        for bad_layer in (NS(), NS(_hcu_flash_attention_owner=weakref.ref(owner))):
            slot[0].no_compile_layers[layer_name] = bad_layer
            try:
                op(*inputs, layer_name)
            except RuntimeError as error:
                assert "no live matching owner" in str(error), str(error)
            else:
                raise AssertionError("unbound/mismatched owner was accepted")
        dead_owner = OwnerDouble(99)
        dead_layer = dead_owner.mla_attn
        del dead_owner
        gc.collect()
        assert dead_layer._hcu_flash_attention_owner() is None
        slot[0].no_compile_layers[layer_name] = dead_layer
        try:
            op(*inputs, layer_name)
        except RuntimeError as error:
            assert "no live matching owner" in str(error), str(error)
        else:
            raise AssertionError("expired weak owner was accepted")
    assert not torch.cuda.is_initialized()
    print("ATTENTION_CPU_RESULT=" + json.dumps({
        "torch": torch.__version__, "core": vllm.__version__,
        "production_module": attention.__file__, "schema": str(schema),
        "graphs": len(graphs), "runtime_calls": len(calls),
        "effective_splitting_ops": compilation.splitting_ops,
        "cudagraph_mode": str(compilation.cudagraph_mode),
        "fake_context_reads": 0, "cuda_initialized": torch.cuda.is_initialized(),
        "scope": "real Torch/Core op; context and arithmetic owner are doubles; NOT GPU acceptance",
    }))


if __name__ == "__main__":
    if sys.argv[1:] == ["--native-cpu-child"]:
        _native_cpu_child()
    else:
        unittest.main(verbosity=2)
