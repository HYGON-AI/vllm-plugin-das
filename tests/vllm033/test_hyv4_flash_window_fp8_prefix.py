"""Actual leaf/constructor with explicit control doubles, never GPU numerics."""
import ast
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest


class Device:
    def __init__(self, kind='cuda', index=2):
        self.type, self.index = kind, index

    def __eq__(self, other):
        return isinstance(other, Device) and (self.type, self.index) == (other.type, other.index)


class Tensor:
    """Shape/ownership recording only; no attention math is simulated."""
    def __init__(self, shape, dtype='bfloat16', device=None, parent=None):
        self.shape, self.dtype = tuple(shape), dtype
        self.device = device or Device()
        self.parent = parent
        self.identity = object()

    @property
    def ndim(self):
        return len(self.shape)

    def stride(self, axis):
        assert axis == -1
        return 1

    def clone(self):
        return Tensor(self.shape, self.dtype, self.device, self)

    def view(self, dtype):
        assert dtype == 'uint8'
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
            position = indices.index(Ellipsis)
            indices = (*indices[:position], *([slice(None)] * (self.ndim-len(indices)+1)), *indices[position+1:])
        dimensions = []
        for length, selected in zip(self.shape, (*indices, *([slice(None)] * (self.ndim-len(indices))))):
            if isinstance(selected, slice):
                dimensions.append(len(range(*selected.indices(length))))
        return Tensor(dimensions, self.dtype, self.device, self)

    def __bool__(self):
        raise AssertionError('Device values cannot drive the dispatch predicate')

    def __eq__(self, other):
        raise AssertionError('No GPU value comparison is allowed')

    def item(self):
        raise AssertionError('No GPU value read or synchronization is allowed')


class WindowProducer:
    def __init__(self, window=512, device=None):
        self.window = window
        self.topk_indices_buffer = Tensor((8192, 2048), 'int32', device)


class Scheduler:
    def __init__(self, width=None):
        self.config = NS(topk=width)
        self.have_initialized = width is not None


class Provider:
    def __init__(self):
        self.fresh, self.calls = [], []

    def get_metadata(self):
        value = Scheduler()
        self.fresh.append(value)
        return value, None

    def with_kvcache(self, **kwargs):
        self.calls.append(kwargs)
        q, indices = kwargs['q'], kwargs['indices']
        scheduler = kwargs['tile_scheduler_metadata']
        scheduler.config.topk = indices.shape[-1]
        scheduler.have_initialized = True
        return Tensor((*q.shape[:3], 512), device=q.device), Tensor((q.shape[0], q.shape[2], q.shape[1]), device=q.device)


@pytest.fixture
def setup(monkeypatch):
    source = Path(__file__).resolve().parents[2] / 'adapter_extension/src/hcu_vllm_das/models/hy_v4_flash/sparse_backend.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FlashSparseImpl')
    provider = Provider()
    cuda = NS(available=True, current=2, architectures={0:'gfx936', 2:'gfx938:sramecc+:xnack-'}, property_reads=[])
    cuda.is_available = lambda: cuda.available
    cuda.current_device = lambda: cuda.current

    def properties(index):
        cuda.property_reads.append(index)
        return NS(gcnArchName=cuda.architectures[index],
                  multi_processor_count={0: 80, 2: 64}[index])
    cuda.get_device_properties = properties

    class Core:
        def __init__(self, *args, **kwargs):
            self._indexer = kwargs.get('indexer')
            self._topk_indices_buffer = kwargs.get('topk_indices_buffer')
            self.kv_cache_dtype = 'fp8_ds_mla'
            self.num_heads = kwargs.get('num_heads', 16)
            self.fp8_decode_padded_heads = 64 if kwargs.get('num_heads',16) <= 64 else 128
            self.softmax_scale = .25

        @property
        def topk_indices_buffer(self):
            return self._indexer.topk_indices_buffer if self._indexer is not None else self._topk_indices_buffer

        @topk_indices_buffer.setter
        def topk_indices_buffer(self, value):
            self._indexer, self._topk_indices_buffer = None, value

        def _fp8_flash_mla_kernel(self, *args):
            return 'core-output', 'core-lse'

        def _sinks_for_query(self, q, axis, padded):
            return Tensor((padded,), 'float32', q.device)

    package = '_hy4_window_prefix_contract'
    parent = ModuleType(package); parent.__path__ = []
    swa = ModuleType(package+'.swa'); swa.FlashWindowIndexer = WindowProducer
    monkeypatch.setitem(sys.modules, package, parent)
    monkeypatch.setitem(sys.modules, package+'.swa', swa)
    runner_name = package + '.fullgraph_runner'
    spec = spec_from_file_location(runner_name, source.with_name('fullgraph_runner.py'))
    runner = module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setitem(sys.modules, runner_name, runner)

    def pad(t, values):
        assert values[:3] == (0,0,0)
        return Tensor((*t.shape[:-2], t.shape[-2]+values[3], t.shape[-1]), t.dtype, t.device, t)
    # Keep these original H64/window-width contracts on the rollback path.
    parallel = NS(tensor_parallel_size=4, prefill_context_parallel_size=1,
                  decode_context_parallel_size=1)
    namespace = {'fp8_fusions_enabled':lambda:False, '__package__':package, 'CoreSinkImpl':Core, 'torch':NS(cuda=cuda,device=Device,bfloat16='bfloat16',uint8='uint8',int32='int32'), 'F':NS(pad=pad), 'get_current_vllm_config':lambda:NS(parallel_config=parallel), 'get_flashmla_ops':lambda:provider, 'is_hcu_platform':lambda:True}
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(source)+'::<actual-class-control-test>','exec'),namespace)
    cls = namespace['FlashSparseImpl']
    # HiSparse ownership has independent tests; this double does not simulate it.
    monkeypatch.setattr(cls, '_install_hisparse_hook', lambda self:None)

    def make(owner=None, heads=16):
        if owner is None:
            owner = WindowProducer()
        obj = cls(indexer=owner, num_heads=heads)
        q = Tensor((1,384,heads,576))
        cache = Tensor((64,64,656),'uint8')
        indices = Tensor((1,384,2048),'int32')
        metadata = NS(scheduler_metadata=Scheduler(2048),dummy_block_table=object(),cache_lens=object())
        return obj,q,cache,indices,metadata
    return NS(make=make,provider=provider,cuda=cuda,cls=cls,namespace=namespace)


def invoke(values):
    obj,q,cache,indices,metadata = values
    return obj._fp8_flash_mla_kernel(q,cache,indices,metadata)


def test_only_window_metadata_forces_fp8_mixed_on_hisparse():
    source = (Path(__file__).resolve().parents[2] /
              'adapter_extension/src/hcu_vllm_das/models/hy_v4_flash/swa.py')
    names = {'FlashSparseMetadataBuilder', 'FlashSWAMetadataBuilder'}
    tree = ast.parse(source.read_text())
    builders = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]

    class CoreBuilder:
        def __init__(self, _spec, _layers, config, _device):
            self.use_hisparse = config.hisparse
            self.fp8_use_mixed_batch = False

    ns = {'CompactPrefillMetadataBuilder':CoreBuilder, 'is_hcu_platform':lambda:False,
          'AttentionCGSupport':NS(UNIFORM_BATCH=object())}
    exec(compile(ast.Module(body=builders,type_ignores=[]),str(source),'exec'),ns)
    sparse, window = (ns[name] for name in ('FlashSparseMetadataBuilder','FlashSWAMetadataBuilder'))
    assert not sparse(None,[],NS(hisparse=True),None).fp8_use_mixed_batch
    assert sparse(None,[],NS(hisparse=False),None).fp8_use_mixed_batch
    assert window(None,[],NS(hisparse=True),None).fp8_use_mixed_batch


def test_window_uses_private512_clone_and_fresh_scheduler(setup):
    values = setup.make(); obj,q,cache,indices,metadata = values
    original = (indices.identity,obj.topk_indices_buffer.identity)
    output,lse = invoke(values)
    call = setup.provider.calls[-1]
    assert call['indices'].shape == (1,384,512)
    assert call['indices'].parent.shape == (1,384,512)
    assert call['indices'] is not indices
    assert call['tile_scheduler_metadata'] is not metadata.scheduler_metadata
    assert metadata.scheduler_metadata.config.topk == 2048
    assert call['block_table'] is metadata.dummy_block_table
    assert call['cache_seqlens'] is metadata.cache_lens
    assert call['attn_sink'].shape == (64,)
    assert output.shape == (1,384,16,512) and lse.shape == (1,16,384)
    assert (indices.identity,obj.topk_indices_buffer.identity) == original
    first = call['tile_scheduler_metadata']; reads = list(setup.cuda.property_reads)
    invoke(values)
    assert setup.provider.calls[-1]['tile_scheduler_metadata'] is not first
    assert setup.cuda.property_reads == reads, 'No per-leaf device query'


def test_constructor_qualifies_current_device_not_zero(setup):
    values = setup.make()
    assert values[0]._hcu_window_device == Device('cuda',2)
    assert setup.cuda.property_reads == [2]
    setup.cuda.current = 0
    invoke(setup.make())
    assert setup.provider.calls[-1]['indices'].shape[-1] == 2048


def test_hidden_gpu_does_not_add_an_architecture_query(setup):
    setup.cuda.available = False
    invoke(setup.make())
    assert not setup.cuda.property_reads
    assert setup.provider.calls[-1]['indices'].shape[-1] == 2048


def test_shared_metadata_interleaves_window_and_compressed(setup):
    window = setup.make(); compressed = setup.make(NS(topk_indices_buffer=Tensor((8192,2048),'int32')))
    metadata = window[-1]; compressed = (*compressed[:-1],metadata)
    invoke(compressed); shared = setup.provider.calls[-1]['tile_scheduler_metadata']
    invoke(window)
    assert setup.provider.calls[-1]['tile_scheduler_metadata'] is not shared
    invoke(compressed)
    assert setup.provider.calls[-1]['tile_scheduler_metadata'] is shared
    assert shared.config.topk == 2048
    assert [x['indices'].shape[-1] for x in setup.provider.calls] == [2048,512,2048]


def test_metadata_none_creates_one_fresh_scheduler_per_call(setup):
    values = setup.make(); values=(*values[:-1],None)
    invoke(values)
    assert len(setup.provider.fresh)==1
    assert setup.provider.calls[-1]['indices'].shape[-1]==512
    assert setup.provider.calls[-1]['block_table'] is None


@pytest.mark.parametrize('mutation', ['owner-none','foreign-subclass','wrong-window','tensor-window','rebound-width','explicit-buffer','owner-device','q-device','cache-device','index-device','fp16','cache-dtype','cache-record','index-dtype','physical-width','q-batch','q-rank','q-heads','q-dim','index-rank','index-rows'])
def test_unqualified_owners_and_layouts_keep_original_call(setup,mutation):
    values=list(setup.make()); obj,q,cache,indices,metadata=values
    if mutation=='owner-none':obj._indexer=None
    elif mutation=='foreign-subclass':
        class Foreign(WindowProducer):pass
        obj._indexer=Foreign()
    elif mutation=='wrong-window':obj._indexer.window=1024
    elif mutation=='tensor-window':obj._indexer.window=Tensor((),'int32')
    elif mutation=='rebound-width':obj._indexer.topk_indices_buffer=Tensor((8192,4096),'int32')
    elif mutation=='explicit-buffer':obj.topk_indices_buffer=Tensor((8192,2048),'int32')
    elif mutation=='owner-device':obj._indexer.topk_indices_buffer.device=Device('cuda',1)
    elif mutation=='q-device':q.device=Device('cuda',1)
    elif mutation=='cache-device':cache.device=Device('cuda',1)
    elif mutation=='index-device':indices.device=Device('cuda',1)
    elif mutation=='fp16':q.dtype='float16'
    elif mutation=='cache-dtype':cache.dtype='float16'
    elif mutation=='cache-record':cache.shape=(64,64,352)
    elif mutation=='index-dtype':indices.dtype='int64'
    elif mutation=='physical-width':indices.shape=(1,384,4096)
    elif mutation=='q-batch':q.shape=(2,384,16,576)
    elif mutation=='q-rank':q.shape=(384,16,576)
    elif mutation=='q-heads':q.shape=(1,384,32,576)
    elif mutation=='q-dim':q.shape=(1,384,16,512)
    elif mutation=='index-rank':indices.shape=(384,2048)
    elif mutation=='index-rows':indices.shape=(1,383,2048)
    # Bad q rank is outside even the existing API, so inspect eligibility only.
    if mutation in ['q-rank','index-rank']:
        assert not obj._use_window_fp8_prefix(q,cache,indices)
        return
    invoke(values)
    call=setup.provider.calls[-1]
    assert call['indices'].shape[-1] == indices.shape[-1]
    assert call['tile_scheduler_metadata'] is metadata.scheduler_metadata
    assert not setup.provider.fresh


def test_owner_buffer_is_resolved_after_draft_rebinding(setup):
    values=setup.make(); obj=values[0]
    invoke(values)
    before=obj._indexer.topk_indices_buffer
    obj._indexer.topk_indices_buffer=Tensor((8192,2048),'int32')
    invoke(values)
    assert obj.topk_indices_buffer is not before
    assert setup.provider.calls[-1]['indices'].shape[-1]==512
    obj.topk_indices_buffer=Tensor((8192,2048),'int32')
    invoke(values)
    assert setup.provider.calls[-1]['indices'].shape[-1]==2048


def test_non_hcu_keeps_inherited_leaf(setup):
    values=setup.make(); values[0]._hcu_ops=None
    assert invoke(values)==('core-output','core-lse')
    assert not setup.provider.calls
