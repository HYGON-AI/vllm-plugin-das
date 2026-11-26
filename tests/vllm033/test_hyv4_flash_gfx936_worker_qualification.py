"""CPU contracts for worker admission; these are not GPU qualification."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = (Path(__file__).resolve().parents[2] / 'adapter_extension/src'
          / 'hcu_vllm_das/models/hy_v4_flash')


def load_runner():
    spec = importlib.util.spec_from_file_location(
        'flash_worker_qualification_fixture', SOURCE / 'fullgraph_runner.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def postload_prefix():
    tree = ast.parse((SOURCE / 'aiter_w4a8_moe.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'HYV4FlashAiterW4A8Method')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'process_weights_after_loading')
    # Execute the actual admission prefix up to Core's first postload call.
    boundary = next(i for i, n in enumerate(method.body)
                    if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Attribute)
                    and n.value.func.attr == 'process_weights_after_loading')
    method.body = method.body[:boundary] + [ast.parse('raise ReachedPrepare').body[0]]
    ast.fix_missing_locations(method)
    class ReachedPrepare(Exception):
        pass
    namespace = {'__package__': 'hy4_worker_fixture',
                 '_device_supported': lambda _: True,
                 'ReachedPrepare': ReachedPrepare}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual-postload>', 'exec'),
         namespace)
    return namespace[method.name], ReachedPrepare


class WorkerQualificationTests(unittest.TestCase):
    def test_qualified_worker_pairs_and_cross_products(self):
        predicate = load_runner()._qualified_worker_tuple
        for triple, expected in [
            ((4, 'gfx938', 64), True), ((8, 'gfx936', 80), True),
            ((8, 'gfx938', 64), False), ((4, 'gfx936', 80), False),
            ((8, 'gfx936', 64), False), ((4, 'gfx938', 80), False),
            ((8, 'gfx942', 80), False), (('8', 'gfx936', 80), False),
        ]:
            with self.subTest(triple=triple):
                self.assertEqual(predicate(*triple), expected)

    def test_actual_postload_rejects_before_core_prepare(self):
        method, reached = postload_prefix()
        runner = load_runner()
        package = ModuleType('hy4_worker_fixture')
        for tp, arch, cu, accepted in [
            (4, 'gfx938', 64, True), (8, 'gfx936', 80, True),
            (8, 'gfx938', 64, False), (4, 'gfx936', 80, False),
            (8, 'gfx936', 64, False), (8, 'gfx942', 80, False),
        ]:
            with self.subTest(tp=tp, arch=arch, cu=cu):
                props = SimpleNamespace(gcnArchName=arch + ':sramecc+',
                                        multi_processor_count=cu)
                torch = SimpleNamespace(cuda=SimpleNamespace(
                    get_device_properties=lambda _: props))
                self_obj = SimpleNamespace(_weights_processed=False, _prepared=None,
                                           _graph_admitted=True,
                                           moe=SimpleNamespace(moe_backend='aiter', tp_size=tp))
                layer = SimpleNamespace(w13_weight_packed=SimpleNamespace(device='cuda:0'))
                with patch.dict(sys.modules, {'torch': torch,
                        'hy4_worker_fixture': package,
                        'hy4_worker_fixture.fullgraph_runner': runner}):
                    with self.assertRaises(reached if accepted else NotImplementedError):
                        method(self_obj, layer)

    def test_eager_postload_does_not_query_graph_worker_properties(self):
        method, reached = postload_prefix()
        self_obj = SimpleNamespace(_weights_processed=False, _prepared=None,
                                   _graph_admitted=False,
                                   moe=SimpleNamespace(moe_backend='aiter', tp_size=8))
        layer = SimpleNamespace(w13_weight_packed=SimpleNamespace(device='cuda:0'))
        with patch.dict(sys.modules, {'torch': None}):
            with self.assertRaises(reached):
                method(self_obj, layer)

    def test_real_h4_h8_admission_and_hisparse_opt_out(self):
        tree = ast.parse((SOURCE / 'sparse_backend.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'FlashSparseImpl')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == '_use_fp8_small_heads')
        namespace = {'torch': SimpleNamespace(bfloat16='bf16', uint8='u8', int32='i32')}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual-head-guard>', 'exec'),
             namespace)
        predicate = namespace[method.name]
        owner = SimpleNamespace(_hcu_small_head_device='cuda:0',
                                _hcu_small_head_counts=(4, 8), fp8_decode_padded_heads=64)
        cache = SimpleNamespace(device='cuda:0', dtype='u8', ndim=3,
                                shape=(2, 64, 656), is_contiguous=lambda: True)
        indices = SimpleNamespace(device='cuda:0', dtype='i32', ndim=3,
                                  shape=(1, 7, 512), stride=lambda _: 1)
        for h, expected in [(4, True), (8, True), (16, False), (3, False)]:
            q = SimpleNamespace(device='cuda:0', ndim=4, shape=(1, 7, h, 576), dtype='bf16')
            self.assertEqual(predicate(owner, q, cache, indices), expected)
        owner._hcu_small_head_device = None
        self.assertFalse(predicate(owner, q, cache, indices))

    def test_existing_938_attention_ctor_scope_is_preserved(self):
        tree = ast.parse((SOURCE / 'sparse_backend.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'FlashSparseImpl')
        ctor = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == '__init__')
        # Core construction is outside this CPU test. Execute the actual
        # capability-publication body, preserving its original branch logic.
        ctor.body = [n for n in ctor.body if not (
            isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == '__init__')]
        runner = load_runner()
        for tp, arch, cu, expected in [
            (2, 'gfx938', 64, (8, 16)), (8, 'gfx938', 64, (8, 16)),
            (2, 'gfx938', 80, (8, 16)), (4, 'gfx938', 64, (8, 16)),
            (8, 'gfx936', 80, (4, 8)), (4, 'gfx936', 80, None),
            (8, 'gfx936', 64, None), (8, 'gfx942', 80, None),
        ]:
            with self.subTest(tp=tp, arch=arch, cu=cu):
                parallel = SimpleNamespace(
                    prefill_context_parallel_size=1, decode_context_parallel_size=1)
                # Only the added gfx936 path may read TP or import its gate.
                if arch == 'gfx936':
                    parallel.tensor_parallel_size = tp
                props = SimpleNamespace(gcnArchName=arch, multi_processor_count=cu)
                torch = SimpleNamespace(cuda=SimpleNamespace(
                    is_available=lambda: True, current_device=lambda: 0,
                    get_device_properties=lambda _: props),
                    device=lambda kind, index: f'{kind}:{index}')
                namespace = {'__package__': 'hy4_worker_fixture', 'torch': torch,
                    'get_current_vllm_config': lambda: SimpleNamespace(parallel_config=parallel),
                    'is_hcu_platform': lambda: True, 'get_flashmla_ops': lambda: object(),
                    'fp8_fusions_enabled': lambda: True}
                exec(compile(ast.Module(body=[ctor], type_ignores=[]),
                             '<actual-attention-ctor>', 'exec'), namespace)
                owner = SimpleNamespace(
                    kv_cache_dtype='fp8_ds_mla',
                    num_heads=4 if arch == 'gfx936' else 8,
                    _install_hisparse_hook=lambda: None)
                with patch.dict(sys.modules, {
                    'hy4_worker_fixture': ModuleType('hy4_worker_fixture'),
                    'hy4_worker_fixture.fullgraph_runner': (
                        runner if arch == 'gfx936' else None)}):
                    namespace['__init__'](owner)
                if expected is None:
                    self.assertIsNone(owner._hcu_small_head_device)
                else:
                    self.assertEqual(owner._hcu_small_head_device, 'cuda:0')
                    self.assertEqual(owner._hcu_small_head_counts, expected)


if __name__ == '__main__':
    unittest.main()
