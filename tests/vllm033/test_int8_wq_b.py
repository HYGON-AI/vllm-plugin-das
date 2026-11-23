# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the owned INT8 WQ-B hooks (no vLLM/HIP imports)."""
import ast
import hashlib
import inspect
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import pytest

_SRC = Path(__file__).resolve().parents[2] / 'adapter_extension/src/hcu_vllm_das'


class Tensor:
    def __init__(self, shape, dtype, contiguous=True):
        self.shape, self.dtype = shape, dtype
        self._contiguous = contiguous
        self.device = NS(index=0)
        self.is_cuda = True
        self.ndim = len(shape)

    def is_contiguous(self):
        return self._contiguous

    def contiguous(self):
        return self if self._contiguous else Tensor(self.shape, self.dtype)

    def view(self, *shape):
        self.view_shape = shape
        return self

    def stride(self):
        return (1, 1280)

    def data_ptr(self):
        return id(self)


class Linear:
    def forward(self):
        raise AssertionError('fixture must call the owned projection seam')


class Column(Linear):
    pass


class Replicated(Linear):
    pass


class Method:
    def apply(self):
        pass


class Scheme:
    def apply_weights(self):
        pass


class Kernel:
    def apply_weights(self):
        pass


class DecodeMeta:
    decode, prefill = object(), None
    num_decodes = num_decode_tokens = 6
    num_prefills = num_prefill_tokens = 0


def layer(cls, n):
    kernel = Kernel()
    kernel.config = NS(is_channelwise=True, is_static_input_scheme=False, input_symmetric=True)
    kernel.layer_param_names = ('weight', 'weight_scale', 'input_scale', 'input_zero_point', 'azp_adj')
    scheme = Scheme()
    vars(scheme).update(kernel=kernel, strategy='channel', is_static_input_scheme=False,
                        input_symmetric=True)
    linear = cls()
    weight, scale = Tensor((1280, n), 'int8', False), Tensor((n, 1), 'float32')
    vars(linear).update(weight=weight, weight_scale=scale, quant_method=Method(), scheme=scheme,
                       bias=None, has_bias=False, skip_bias_add=False, return_bias=cls is Replicated,
                       gather_output=False, tp_size=1, tp_rank=0, input_size=1280, output_size=n,
                       input_size_per_partition=1280, output_size_per_partition=n,
                       output_partition_sizes=[n], input_scale=None, input_zero_point=None,
                       azp_adj=None, _hcu_int8_empty_state='ready', _hcu_int8_empty_witness=NS(
                           scheme=scheme, kernel=kernel, weight=weight, shape=weight.shape,
                           stride=weight.stride(), data_ptr=weight.data_ptr(),
                           device=weight.device, dtype=weight.dtype))
    return linear


@pytest.fixture
def route(monkeypatch):
    """Use the installed source hooks with tensor metadata and GPU leaves stubbed."""
    mod = ModuleType('int8_test.int8_wq_b')
    cfg = NS(use_v2_model_runner=True, model_config=NS(enforce_eager=False),
             parallel_config=NS(tensor_parallel_size=1, prefill_context_parallel_size=1,
                                decode_context_parallel_size=1, use_ubatching=False),
             compilation_config=NS(mode='NONE', cudagraph_mode='FULL_AND_PIECEWISE'))
    context = NS(cudagraph_runtime_mode='NONE', attn_metadata={'index': DecodeMeta()},
                 ubatch_slices=None)
    q = Tensor((6, 1280), 'bfloat16', False)
    payload, scale = Tensor(q.shape, 'int8'), Tensor((6, 1), 'float32')
    quant, mm = Mock(return_value=(payload, scale, None)), Mock()
    label = Mock(side_effect=lambda _: nullcontext())
    torch = NS(Tensor=Tensor, int8='int8', bfloat16='bfloat16', float32='float32',
               version=NS(hip=True), compiler=NS(is_compiling=Mock(return_value=False)),
               cuda=NS(current_device=Mock(return_value=0),
                       get_device_properties=Mock(return_value=NS(gcnArchName='gfx938:sramecc+')),
                       is_current_stream_capturing=Mock(return_value=True)),
               profiler=NS(record_function=label),
               ops=NS(vllm=NS(int8_scaled_mm_func=NS(default=mm))))
    tree = ast.parse((_SRC / 'int8_wq_b.py').read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    vars(mod).update(torch=torch, ops=NS(scaled_int8_quant=quant), ColumnParallelLinear=Column,
                     ReplicatedLinear=Replicated, CompressedTensorsW8A8Int8=Scheme,
                     adapter_module={'dsv41_int8_linear': NS(HcuDSV41Int8LinearMethod=Method)}.__getitem__,
                      triton=NS(TritonInt8ScaledMMLinearKernel=Kernel),
                     CompilationMode=NS(NONE='NONE'), CUDAGraphMode=NS(
                         NONE='NONE', FULL_AND_PIECEWISE='FULL_AND_PIECEWISE',
                         FULL_DECODE_ONLY='FULL_DECODE_ONLY'),
                     get_forward_context=lambda: context,
                     is_forward_context_available=Mock(return_value=True),
                     current_platform=NS(is_rocm=lambda: True, device_name='hcu'),
                     init_logger=lambda _: NS(info_once=Mock(), warning_once=Mock()),
                     version=lambda _: mod._CORE_WHEEL,
                     DeepseekV32IndexerMetadata=DecodeMeta, hashlib=hashlib, inspect=inspect)
    exec(compile(tree, str(_SRC / 'int8_wq_b.py'), 'exec'), vars(mod))
    monkeypatch.setattr(mod, '_reviewed_source', Mock(return_value=True))
    monkeypatch.setattr(mod, '_registered_ops', Mock(return_value=True))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    env = ModuleType('int8_test.environment')
    exec(compile((_SRC / 'environment.py').read_text(), str(_SRC / 'environment.py'), 'exec'),
         vars(env))
    monkeypatch.setitem(sys.modules, env.__name__, env)
    env_key = 'VLLM_HCU_DSV41_INT8_WQ_B'
    monkeypatch.delenv(env_key, raising=False)
    monkeypatch.setenv('HCU_DSV41_WQ_B_INT8_DIRECT', '0')

    seen = []
    native_qr_scale = NS(value=None)
    kv = object()

    class NativeAttention:
        def __init__(self, config):
            self.indexer = Fp4Indexer()
            self.indexer.vllm_config = config
            self.wq_b = layer(Column, 32768)
            self.n_local_heads, self.head_dim, self.q_lora_rank = 64, 512, 1280
            self.use_pcp = False
            self.kv_mxfp8 = False
            self.accepts_unnormed_unroped_query = False
            self.padded_heads = self.n_local_heads
            self.rotary_emb = NS(cos_sin_cache=q)
            self.wo_a = NS(weight=q)
            self._prepare_and_attn = Mock()
            self._prepare_and_attn_fn = self._prepare_and_attn

        def _split_qkv_and_norm(self, inp):
            seen.append(('native split', inp))
            return inp, native_qr_scale.value, kv

        def _wq_b_proj(self, qr, qr_scale):
            seen.append(('native main', qr, qr_scale))
            return qr

    class NativeIndexer:
        def __init__(self):
            self.wq_b = layer(Replicated, 4096)
            self.n_head, self.head_dim, self.q_lora_rank = 32, 128, 1280
            self.k_cache, self.softmax_scale = NS(prefix='index'), 0.125

        def _wq_b_proj(self, qr, qr_scale):
            seen.append(('native index', qr, qr_scale))
            return qr

    index_q = Mock(return_value=(('pack', 'scales'), 'weights'))
    source = _SRC / 'fp4_indexer.py'
    fp4 = ast.parse(source.read_text())
    definitions = {}
    for node in fp4.body:
        if isinstance(node, ast.ClassDef) and node.name in ('Fp4Attention', 'Fp4Indexer'):
            keep = ('__init__', '_split_qkv_and_norm', '_wq_b_proj') if node.name == 'Fp4Attention' else (
                '_wq_b_proj', 'forward_q')
            node.body = [method for method in node.body if method.name in keep]
            definitions[node.name] = node
    ns = dict(__name__='int8_test.fp4_indexer', torch=torch,
              common=NS(DeepseekV4Indexer=NativeIndexer, fused_indexer_q_rope_quant=index_q),
              rocm=NS(DeepseekV41ROCMAiterMLAAttention=NativeAttention),
              _Fp4CommonAttention=object, Fp4Query=lambda *args: args,
              rotate_bf16=lambda *_: 'bf16', qualify_qkv_split=lambda _: False,
              wo_a_layout=NS(configured=lambda *_: False))
    exec(compile(ast.Module(body=list(definitions.values()), type_ignores=[]), str(source), 'exec'), ns)
    Fp4Attention, Fp4Indexer = ns['Fp4Attention'], ns['Fp4Indexer']
    mm.side_effect = lambda qr, weight, *args: Tensor((qr.shape[0], weight.shape[1]), 'bfloat16')
    attn = Fp4Attention(cfg)
    # The optional 0006/0010 stack keeps the same projection hook.
    attn.indexer._skip_bf16_q = lambda *_: False
    return NS(attn=attn, attn_cls=Fp4Attention, mod=mod, cfg=cfg, context=context, torch=torch,
              q=q, kv=kv, quant=quant, mm=mm, label=label, payload=payload, scale=scale,
              seen=seen, index_q=index_q, env_key=env_key, native_qr_scale=native_qr_scale)


def test_one_native_quant_shared_by_both_exact_gemms(route):
    r = route
    q, scale, kv = r.attn._split_qkv_and_norm(r.q)
    assert (q, scale, kv) == (r.payload, r.scale, r.kv)
    args, kwargs = r.quant.call_args
    r.quant.assert_called_once()
    assert args[0] is not r.q and args[0].is_contiguous() and args[1:] == (None, None)
    assert kwargs == {'symmetric': True}
    assert r.seen == [('native split', r.q)]

    assert r.attn._wq_b_proj(q, scale).dtype == 'bfloat16'
    pack, out_scale, weights = r.attn.indexer.forward_q(q, scale, None, None, NS(cos_sin_cache=None))
    assert (pack, out_scale, weights) == (('pack', 'scales', 'bf16'), None, 'weights')
    assert r.index_q.call_args.args[1].dtype == 'bfloat16'
    assert r.index_q.call_args.args[1].view_shape == (-1, 32, 128)
    assert [call.args for call in r.mm.call_args_list] == [
        (q, owner.wq_b.weight, scale, owner.wq_b.weight_scale, 'bfloat16', None)
        for owner in (r.attn, r.attn.indexer)
    ]
    assert [call.args[0] for call in r.label.call_args_list] == [
        'hcu::dsv41_int8_wq_b_quant', 'hcu::dsv41_int8_wq_b_main', 'hcu::dsv41_int8_wq_b_index',
    ]


@pytest.mark.parametrize('mismatch', [
    'off', 'eager', 'compile', 'piecewise', 'profile', 'prefill', 'empty', 'wide',
    'tp', 'pcp', 'dcp', 'ubatch', 'wrong arch', 'schema', 'source',
    'fp8 pair', 'skip main return', 'gather', 'bias', 'static', 'asymmetric',
    'other kernel', 'no witness', 'replaced weight', 'scale shape', 'missing scale', 'other device',
])
def test_ineligible_pair_stays_native(route, monkeypatch, mismatch):
    r = route
    main, index = r.attn.wq_b, r.attn.indexer.wq_b
    if mismatch == 'off':
        monkeypatch.setenv(r.env_key, '0')
    elif mismatch == 'eager':
        r.torch.cuda.is_current_stream_capturing.return_value = False
    elif mismatch == 'compile':
        r.cfg.compilation_config.mode = 'VLLM_COMPILE'
    elif mismatch == 'piecewise':
        r.context.cudagraph_runtime_mode = 'PIECEWISE'
    elif mismatch == 'profile':
        r.context.attn_metadata = None
    elif mismatch == 'prefill':
        r.context.attn_metadata['index'].num_prefills = 1
    elif mismatch in ('empty', 'wide'):
        r.q.shape = (0 if mismatch == 'empty' else 257, 1280)
    elif mismatch in ('tp', 'pcp', 'dcp'):
        field = dict(tp='tensor', pcp='prefill_context', dcp='decode_context')[mismatch]
        setattr(r.cfg.parallel_config, f'{field}_parallel_size', 2)
    elif mismatch == 'ubatch':
        r.context.ubatch_slices = object()
    elif mismatch == 'wrong arch':
        r.torch.cuda.get_device_properties.return_value = NS(gcnArchName='gfx936')
    elif mismatch == 'schema':
        r.mod._registered_ops.return_value = False
    elif mismatch == 'source':
        r.mod._reviewed_source.return_value = False
    elif mismatch == 'fp8 pair':
        r.q.dtype, r.native_qr_scale.value = 'fp8', object()
    elif mismatch == 'skip main return':
        main.return_bias = True
    elif mismatch == 'gather':
        main.gather_output = True
    elif mismatch == 'bias':
        index.bias = object()
    elif mismatch == 'static':
        index.scheme.is_static_input_scheme = True
    elif mismatch == 'asymmetric':
        index.scheme.kernel.config.input_symmetric = False
    elif mismatch == 'other kernel':
        index.scheme.kernel = object()
    elif mismatch == 'no witness':
        index._hcu_int8_empty_witness = None
    elif mismatch == 'replaced weight':
        index.weight = Tensor((1280, 4096), 'int8', False)
    elif mismatch == 'scale shape':
        main.weight_scale.shape = (32768,)
    elif mismatch == 'missing scale':
        main.weight_scale = None
    elif mismatch == 'other device':
        r.q.device = NS(index=1)
    if mismatch in ('off', 'tp', 'pcp', 'dcp', 'wrong arch', 'schema', 'source'):
        r.attn = r.attn_cls(r.cfg)
        assert r.attn._int8_wq_b_device is None
    pair = r.q, r.native_qr_scale.value
    assert r.attn._split_qkv_and_norm(r.q) == (*pair, r.kv)
    assert r.attn._wq_b_proj(*pair) is r.q
    assert r.attn.indexer._wq_b_proj(*pair) is r.q
    assert r.seen == [('native split', r.q), ('native main', *pair), ('native index', *pair)]
    r.quant.assert_not_called()
    r.mm.assert_not_called()


def test_surprising_quantization_results_are_rejected(route):
    r = route
    r.quant.return_value = (r.payload, r.scale, object())
    with pytest.raises(RuntimeError, match='invalid operand'):
        r.attn._split_qkv_and_norm(r.q)
    with pytest.raises(ValueError, match='contiguous INT8'):
        r.attn.indexer._wq_b_proj(r.payload, None)
    r.scale.dtype = 'bfloat16'
    with pytest.raises(ValueError, match='contiguous INT8'):
        r.attn._wq_b_proj(r.payload, r.scale)
    r.mm.assert_not_called()
