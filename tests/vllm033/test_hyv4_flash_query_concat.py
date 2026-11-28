"""Portable actual-source control contracts; native math is qualified separately."""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import unittest

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / 'adapter_extension/src/hcu_vllm_das/models/hy_v4_flash'


class Tensor:
    def __init__(self, shape, dtype='bf16', device='cuda:0', contiguous=True, owner=None):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device
        self.contiguous = contiguous
        self.owner = owner or self

    def stride(self, axis):
        return 1 if axis == -1 else 576

    def is_contiguous(self):
        return self.contiguous

    def __getitem__(self, item):
        return Tensor((item.stop, *self.shape[1:]), self.dtype, self.device,
                      self.contiguous, self.owner)

    def reshape(self, *shape):
        return Tensor(shape, self.dtype, self.device, True, self.owner)

    view = reshape


class HiSparse:
    def cache(self, index):
        return 'owned-cache'


class Missing:
    def __call__(self, *args):
        raise AssertionError('Missing must not be invoked')


class Core:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.num_heads = kwargs.get('num_heads', 4)
        self.kv_lora_rank = 512
        self.qk_rope_head_dim = 64
        self.kv_cache_dtype = kwargs.get('kv_cache_dtype', 'fp8_ds_mla')
        self.index_group = kwargs.get('index_group')
        self.index_group_index = 0
        self.workspace_specs = [((16, self.num_heads, 576), 'bf16')]
        self.seen = None

    def _checked_hisparse_call(self, layer=None):
        return None

    def forward_mqa(self, q, cache, metadata, layer):
        self.seen = (q, cache, metadata, layer)
        return q


@contextmanager
def harness(tp=8, arch='gfx936', cu=80, master=True, provider='native', available=True):
    calls = []
    workspace = [Tensor((16, 4, 576))]
    state = {'probes': 0}

    def properties(device):
        state['probes'] += 1
        return SimpleNamespace(gcnArchName=arch, multi_processor_count=cu)

    def native(*args):
        calls.append(('native', args))
        if provider == 'error':
            raise RuntimeError('real-native-error')

    def cat(parts, *, dim, out):
        calls.append(('aten', (parts, dim, out)))
        return out

    torch = SimpleNamespace(bfloat16='bf16', int64='i64',
        device=lambda kind, device: f'{kind}:{device}',
        arange=lambda n, **kw: Tensor((n,), kw['dtype'], kw['device']), cat=cat,
        cuda=SimpleNamespace(is_available=lambda: available,
            current_device=lambda: 0, get_device_properties=properties))
    runner_ns = {}
    node = next(n for n in ast.parse((FAMILY / 'fullgraph_runner.py').read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_qualified_worker_tuple')
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'actual-runner', 'exec'), runner_ns)
    modules = {}
    for name in ('lightop.kvcache', 'hcu_adapter._base',
                 'vllm.v1.attention.backends.mla.index_group', '_concat_contract.fullgraph_runner'):
        modules[name] = ModuleType(name)
    if provider != 'absent':
        modules['lightop.kvcache'].fused_concat_and_store_mla_kv_cache = (
            Missing() if provider == 'missing' else native)
    modules['hcu_adapter._base'].Missing = Missing
    modules['vllm.v1.attention.backends.mla.index_group'].HiSparseMLAIndexGroup = HiSparse
    modules['_concat_contract.fullgraph_runner']._qualified_worker_tuple = runner_ns['_qualified_worker_tuple']
    source = ast.parse((FAMILY / 'sparse_backend.py').read_text())
    original = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'FlashSparseImpl')
    names = {'__init__', '_install_hisparse_hook', '_concat_query_for_mqa', 'forward_mqa'}
    cls = ast.ClassDef(name='FlashSparseImpl', bases=[ast.Name(id='Core', ctx=ast.Load())],
        keywords=[], body=[n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names], decorator_list=[])
    ns = dict(__name__='_concat_contract.sparse_backend', __package__='_concat_contract',
        Core=Core, torch=torch, is_hcu_platform=lambda: True, get_flashmla_ops=lambda: object(),
        fp8_fusions_enabled=lambda: master,
        get_current_vllm_config=lambda: SimpleNamespace(parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, decode_context_parallel_size=1, tensor_parallel_size=tp)),
        install_hisparse_prewrite_hook=lambda *args: calls.append(('hisparse_hook', args)),
        current_workspace_manager=lambda: SimpleNamespace(get_simultaneous=lambda *specs: tuple(workspace)))
    with patch.dict('sys.modules', modules):
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(FAMILY / 'sparse_backend.py'), 'exec'), ns)
        yield ns['FlashSparseImpl'], workspace, calls, state


def query(t=2, h=4):
    return Tensor((t, h, 512)), Tensor((t, h, 64), contiguous=False)


class QueryConcatTests(unittest.TestCase):
    def test_qualified_tuple_uses_same_workspace_and_core_reader(self):
        with harness() as (Impl, buffers, calls, state):
            obj = Impl(num_heads=4)
            q = query()
            result = obj.forward_mqa(q, 'cache', 'metadata', 'layer')
            self.assertIs(result.owner, buffers[0])
            self.assertEqual(result.shape, (2, 4, 576))
            self.assertEqual(obj.seen[1:], ('cache', 'metadata', 'layer'))
            op, args = calls[0]
            self.assertEqual(op, 'native')
            self.assertEqual([x.shape for x in args[:4]], [(8, 512), (8, 64), (8, 1, 576), (8,)])
            self.assertIs(args[2].owner, buffers[0])
            self.assertEqual(state['probes'], 1)

    def test_head8_and_zero_rows(self):
        for t in (0, 1, 33):
            with self.subTest(t=t), harness() as (Impl, buffers, calls, state):
                obj = Impl(num_heads=8)
                buffers[0] = Tensor((64, 8, 576))
                result = obj.forward_mqa(query(t, 8), None, None, None)
                self.assertEqual(result.shape, (t, 8, 576))
                self.assertIs(result.owner, buffers[0])
                self.assertEqual(len(calls), int(t > 0))

    def test_gfx936_tp4_uses_existing_workspace_for_both_head_layouts(self):
        for heads in (8, 16):
            with self.subTest(heads=heads), harness(tp=4) as (Impl, buffers, calls, _):
                obj = Impl(num_heads=heads)
                buffers[0] = Tensor((64, heads, 576))
                result = obj.forward_mqa(query(33, heads), 'cache', 'metadata', 'layer')
                self.assertIs(result.owner, buffers[0])
                self.assertEqual(result.shape, (33, heads, 576))
                self.assertEqual(obj.seen[1:], ('cache', 'metadata', 'layer'))
                self.assertEqual([op for op, _ in calls], ['native'])

    def test_unsupported_devices_keep_original_tuple(self):
        for tp, arch, cu in ((4, 'gfx938', 64), (8, 'gfx938', 80),
                             (4, 'gfx936', 80), (8, 'gfx936', 64), (8, 'other', 80)):
            with self.subTest(tp=tp, arch=arch, cu=cu), harness(tp, arch, cu) as (Impl, buffers, calls, state):
                obj = Impl(num_heads=4)
                q = query()
                self.assertIs(obj.forward_mqa(q, None, None, None), q)
                self.assertEqual(calls, [])

    def test_hisparse_keeps_original_route(self):
        with harness() as (Impl, buffers, calls, state):
            obj = Impl(num_heads=4, index_group=HiSparse())
            q = query()
            self.assertIs(obj.forward_mqa(q, None, None, None), q)
            self.assertEqual([x[0] for x in calls], ['hisparse_hook'])

    def test_master_off_keeps_compatibility_copy(self):
        with harness(master=False) as (Impl, buffers, calls, state):
            obj = Impl(num_heads=4)
            self.assertNotIsInstance(obj.forward_mqa(query(), None, None, None), tuple)
            self.assertEqual(calls[0][0], 'native')

    def test_unsupported_metadata_preserves_original(self):
        variants = [(Tensor((2, 4, 511)), Tensor((2, 4, 64))),
                    (Tensor((2, 4, 512), dtype='fp32'), Tensor((2, 4, 64))),
                    (Tensor((2, 4, 512)), Tensor((2, 4, 64), device='cuda:1')),
                    (Tensor((2, 16, 512)), Tensor((2, 16, 64)))]
        for q in variants:
            with harness() as (Impl, buffers, calls, state):
                obj = Impl(num_heads=4)
                self.assertIs(obj.forward_mqa(q, None, None, None), q)
                self.assertEqual(calls, [])

    def test_padded64_or_bad_workspace_not_admitted(self):
        for output in (Tensor((16, 64, 576)), Tensor((16, 4, 576), contiguous=False),
                       Tensor((16, 4, 576), dtype='fp32'), Tensor((1, 4, 576))):
            with harness() as (Impl, buffers, calls, state):
                obj = Impl(num_heads=4)
                buffers[0] = output
                q = query()
                self.assertIs(obj.forward_mqa(q, None, None, None), q)
                self.assertEqual(calls, [])

    def test_missing_public_api_uses_aten_same_output(self):
        for provider in ('absent', 'missing'):
            with self.subTest(provider=provider), harness(provider=provider) as (Impl, buffers, calls, state):
                obj = Impl(num_heads=4)
                result = obj.forward_mqa(query(), None, None, None)
                self.assertIs(result.owner, buffers[0])
                self.assertEqual(calls[0][0], 'aten')
                self.assertIs(calls[0][1][2].owner, buffers[0])

    def test_native_errors_propagate(self):
        with harness(provider='error') as (Impl, buffers, calls, state):
            obj = Impl(num_heads=4)
            with self.assertRaisesRegex(RuntimeError, 'real-native-error'):
                obj.forward_mqa(query(), None, None, None)
            self.assertIsNone(obj.seen)


if __name__ == '__main__':
    unittest.main(verbosity=2)
