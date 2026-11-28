"""CPU contracts for the native-writer composition, not GPU numerical tests."""
import ast
from contextlib import contextmanager
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as N
import unittest
from unittest.mock import patch

SOURCE = (Path(__file__).resolve().parents[2] / 'adapter_extension/src'
          / 'hcu_vllm_das/models/hy_v4_flash/fp8_ds_mla_cache.py')


class Tensor(N):
    def __getitem__(self, key):
        assert isinstance(key, slice) and key.start is None and key.step is None
        return tensor((min(key.stop, self.shape[0]), *self.shape[1:]), self.dtype,
                      self._strides, self.device)


def tensor(shape, dtype, strides=None, device='cuda:0'):
    if strides is None:
        strides = []
        step = 1
        for s in reversed(shape):
            strides.insert(0, step)
            step *= s
    return Tensor(shape=shape, ndim=len(shape), dtype=dtype,
                  is_cuda=device.startswith('cuda'), device=device,
                  _strides=strides, stride=lambda i: strides[i],
                  is_contiguous=lambda: len(shape) == 1 and strides == [1])


def args(n=7):
    return [tensor((n, 512), 'bf16'), tensor((n, 1, 64), 'bf16'),
            tensor((8, 64, 656), 'u8'), tensor((n,), 'i64')]


def load():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('_validate_inputs', 'write_fp8_ds_mla_cache')]
    namespace = {'torch': N(bfloat16='bf16', uint8='u8', int64='i64')}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class WriterContracts(unittest.TestCase):
    def test_geometry_zero_and_real_pitch(self):
        f = load()['_validate_inputs']
        self.assertEqual(f(*args()), 7)
        self.assertEqual(f(*args(0)), 0)
        a = args()
        a[2] = tensor((8, 64, 656), 'u8', (64 * 672 + 128, 672, 1))
        self.assertEqual(f(*a), 7)

    def test_prefix_rows_accepted_and_short_rows_rejected(self):
        f = load()['_validate_inputs']
        a = args(2)
        a[0] = tensor((5, 512), 'bf16')
        a[1] = tensor((4, 1, 64), 'bf16')
        self.assertEqual(f(*a), 2)
        a[3] = tensor((0,), 'i64')
        self.assertEqual(f(*a), 0)
        a[3] = tensor((5,), 'i64')
        with self.assertRaises(ValueError):
            f(*a)

    def test_dtype_shape_and_device_rejections(self):
        for idx, bad in [
            (0, tensor((7, 512), 'f16')), (1, tensor((7, 1, 64), 'f32')),
            (2, tensor((8, 64, 584), 'u8')), (3, tensor((7,), 'i32')),
            (0, tensor((6, 512), 'bf16')),
            (2, tensor((8, 64, 656), 'u8', (64 * 656, 656, 2))),
            (2, tensor((8, 64, 656), 'u8', device='cuda:1')),
        ]:
            with self.subTest(idx=idx, shape=bad.shape):
                a = args()
                a[idx] = bad
                with self.assertRaises((ValueError, TypeError)):
                    load()['_validate_inputs'](*a)

    def test_actual_wrapper_slices_prefix_and_preserves_native_errors(self):
        namespace = load()
        calls = []
        namespace['torch'].empty = lambda shape, dtype, device: tensor(shape, dtype, device=device)
        namespace['torch'].arange = lambda n, dtype, device: tensor((n,), dtype, device=device)
        class Scatter:
            def __getitem__(self, grid):
                return lambda *a, **kw: calls.append(('scatter', grid, a, kw))
        namespace['_scatter_656'] = Scatter()
        provider = ModuleType('lightop.kvcache')
        class Missing:
            pass
        base = ModuleType('hcu_adapter._base')
        base.Missing = Missing
        a = args(2)
        a[0], a[1] = tensor((5, 512), 'bf16'), tensor((4, 1, 64), 'bf16')
        provider.fused_quantize_and_store_mla_kv_cache = lambda *a: calls.append(('native', a))
        modules = {'lightop': ModuleType('lightop'), 'lightop.kvcache': provider,
                   'hcu_adapter': ModuleType('hcu_adapter'), 'hcu_adapter._base': base}
        with patch.dict(sys.modules, modules):
            namespace['write_fp8_ds_mla_cache'](*a)
            self.assertEqual(calls[0][0], 'native')
            self.assertEqual(calls[0][1][0].shape, (2, 512))
            self.assertEqual(calls[0][1][1].shape, (2, 1, 64))
            self.assertEqual(calls[0][1][2].shape, (2, 1, 656))
            self.assertEqual(calls[1][0], 'scatter')
            error = ValueError('real native failure')
            def native_failure(*_):
                raise error
            provider.fused_quantize_and_store_mla_kv_cache = native_failure
            with self.assertRaises(ValueError) as raised:
                namespace['write_fp8_ds_mla_cache'](*a)
            self.assertIs(raised.exception, error)

    def test_zero_slots_never_import_or_allocate_native(self):
        namespace = load()
        with patch.dict(sys.modules, {'lightop': None}):
            namespace['write_fp8_ds_mla_cache'](*args(0))

    def test_hisparse_write_uses_the_same_fp8_codec_inside_the_binding(self):
        backend = SOURCE.with_name('sparse_backend.py')
        cls = next(n for n in ast.parse(backend.read_text(encoding='utf-8')).body
                   if isinstance(n, ast.ClassDef) and n.name == 'FlashSparseImpl')
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'do_kv_cache_update']
        calls = []

        class Core:
            def do_kv_cache_update(self, *args):
                calls.append(('core', args))

        class Binding:
            def validate_batch(self, metadata, **kwargs):
                calls.append(('validate', metadata, kwargs))

            def write(self, writer, **kwargs):
                calls.append(('write', kwargs))
                return writer()

        namespace = {'CoreSinkImpl': Core, '__package__': '_hisparse_codec_test'}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(backend), 'exec'), namespace)
        impl = namespace['FlashSparseImpl']()
        impl._hcu_fp8_ds_mla_writer = True
        impl.topk_indices_buffer = N(shape=(6, 2176))
        metadata = object()
        impl._checked_hisparse_call = lambda: (Binding(), metadata)
        codec = ModuleType('_hisparse_codec_test.fp8_ds_mla_cache')
        codec.write_fp8_ds_mla_cache = lambda *args: calls.append(('fp8', args)) or 'encoded'
        a = args(6)
        with patch.dict(sys.modules, {codec.__name__: codec}):
            self.assertEqual(impl.do_kv_cache_update(*a, 'fp8_ds_mla', object()), 'encoded')
            self.assertEqual(calls, [
                ('validate', metadata, {'query_rows': 6, 'topk_width': 2176}),
                ('write', {'kv_cache': a[2], 'slot_mapping': a[3], 'rows': 6}),
                ('fp8', tuple(a)),
            ])
            calls.clear()
            impl.do_kv_cache_update(*a, 'auto', object())
            self.assertEqual([call[0] for call in calls], ['validate', 'write', 'core'])

    def test_resident_hisparse_scope_only_wraps_batches_with_prefill_rows(self):
        backend = SOURCE.with_name('sparse_backend.py')
        cls = next(n for n in ast.parse(backend.read_text(encoding='utf-8')).body
                   if isinstance(n, ast.ClassDef) and n.name == 'FlashSparseImpl')
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_forward_fp8_kv_separate_prefill_decode']
        calls = []

        class Core:
            def _forward_fp8_kv_separate_prefill_decode(self, *args):
                calls.append('core')

        @contextmanager
        def scope(group):
            calls.append(('enter', group))
            yield
            calls.append('exit')

        namespace = {'CoreSinkImpl': Core, '__package__': '_hisparse_codec_test',
                     'hisparse_fp8_prefill_scope': scope}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(backend), 'exec'), namespace)
        impl = namespace['FlashSparseImpl']()
        impl._hcu_use_resident_hisparse = True
        impl.index_group = object()
        q = N(shape=(6,))
        metadata = N(fp8_extra_metadata=N(num_decode_tokens=6))
        impl._forward_fp8_kv_separate_prefill_decode(q, object(), object(), metadata)
        self.assertEqual(calls, ['core'])
        calls.clear()
        metadata.fp8_extra_metadata.num_decode_tokens = 4
        impl._forward_fp8_kv_separate_prefill_decode(q, object(), object(), metadata)
        self.assertEqual(calls, [('enter', impl.index_group), 'core', 'exit'])


if __name__ == '__main__':
    unittest.main()
