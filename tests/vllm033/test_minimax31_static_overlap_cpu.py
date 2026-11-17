"""Check the changed shared-stream/static-frame boundary without GPU mocks posing as numerics."""
from types import SimpleNamespace
import pytest
import torch
from hcu_vllm_das.parallel import minimax31_deepep_runner as runner
from hcu_vllm_das.parallel import minimax31_static_runtime as runtime

def test_static_completion_is_inside_lock_after_shared_join(monkeypatch):
    events=[]
    def forward(*args):
        assert runner._buffer_inflight.locked()
        events.append('shared-joined')
        return 'result'
    def complete(layer):
        assert runner._buffer_inflight.locked()
        events.append(('static-completed',layer))
    monkeypatch.setattr(runtime,'complete_layer',complete)
    obj=SimpleNamespace(_forward_locked=forward,routed_experts=SimpleNamespace(_static_placement=object(),_static_layer_id=3))
    assert runner.MiniMax31DeepEpRunner._forward_impl(obj,None,None,None)=='result'
    assert events==['shared-joined',('static-completed',3)]
    assert not runner._buffer_inflight.locked()

@pytest.mark.parametrize('backend', ['bolt', 'deepgemm_hipc_v1'])
@pytest.mark.parametrize('rows,overlap',[(8,False),(2047,False),(2048,True),(4096,True),(6144,True),(8192,True)])
def test_prefill_sizes_keep_136_physical_slots_and_join(monkeypatch,rows,overlap,backend):
    events=[]
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    def aligned(layer, hidden):
        assert layer.global_num_experts == 136
        def execute(hidden, ids, weights, *, expert_weights):
            events.append('aligned136')
            return hidden
        return execute, object()
    monkeypatch.setattr(runner,'_get_aligned_moe',aligned)
    monkeypatch.setattr(runner,'get_ep_group',lambda:None)
    monkeypatch.setattr(runner,'_get_deepep_buffer',lambda *a:None)
    def dispatch(hidden,*args,**kwargs):
        assert kwargs['physical_experts']==136
        events.append('dispatch136');return hidden
    monkeypatch.setattr(runner,'dispatch_routed',dispatch)
    class Shared:
        output='shared'
        def maybe_forward_async(self,x):events.append('stock');return False
        def wait(self):events.append('join')
    expert=SimpleNamespace(global_num_experts=136,quant_method=SimpleNamespace(is_monolithic=False,quant_config=SimpleNamespace(routed_backend=backend)),_ensure_moe_quant_config_init=lambda:None)
    obj=SimpleNamespace(_pcp_ep_group=lambda:SimpleNamespace(rank_in_group=0),gate=None,routed_experts=expert,_shared_experts=Shared(),_hcu_shared_pending=False,_maybe_apply_shared_experts=lambda *a:None,_launch_long_shared=lambda x:events.append('aux') or True)
    obj.router=SimpleNamespace(select_experts=lambda **kw:(torch.ones(rows,1),torch.zeros(rows,1,dtype=torch.int64)))
    obj._quant_method=SimpleNamespace(topk_indices_dtype=torch.int64)
    x=torch.zeros(rows,1,dtype=torch.bfloat16)
    assert runner.MiniMax31DeepEpRunner._forward_locked(obj,x,None,x)==('shared',x)
    route='aligned136' if backend=='deepgemm_hipc_v1' else 'dispatch136'
    assert events==(['aux',route,'join'] if overlap else ['stock',route])

@pytest.mark.parametrize('rows', [2047, 2048])
@pytest.mark.parametrize('backend,kernel,stream,disabled,dbo', [
    ('bolt', 'MiniMax31HipBlasLtInt8Kernel', True, False, False),
    ('deepgemm_hipc_v1', 'TritonInt8ScaledMMLinearKernel', True, False, False),
    ('deepgemm_hipc_v1', 'MiniMax31HipBlasLtInt8Kernel', False, False, False),
    ('deepgemm_hipc_v1', 'MiniMax31HipBlasLtInt8Kernel', True, True, False),
    ('deepgemm_hipc_v1', 'MiniMax31HipBlasLtInt8Kernel', True, False, True),
    ('deepgemm_hipc_v1', 'MiniMax31HipBlasLtInt8Kernel', True, False, False),
])
def test_long_prefill_real_overlap_guard_preserves_legal_fallback(
        monkeypatch, rows, backend, kernel, stream, disabled, dbo):
    # Real launch/admission code; only CUDA events and the numerical body are
    # isolated. This is a scheduling contract, not a GPU numerical test.
    from contextlib import nullcontext
    from types import MethodType
    from vllm.models.minimax_m3.amd.model import MiniMaxM3MLP
    events = []
    class Event:
        def record(self, stream): events.append('record')
        def wait(self, stream): events.append('event-wait')
    class SharedValue:
        shape = (rows, 1)
        ndim = 2
        dtype = torch.bfloat16
        device = torch.device('cpu')
        def record_stream(self, stream): events.append('record-stream')
    value = SharedValue()
    mlp = MiniMaxM3MLP.__new__(MiniMaxM3MLP)
    torch.nn.Module.__init__(mlp)
    mlp.gate_up_proj = SimpleNamespace(tp_size=1,
        scheme=SimpleNamespace(kernel=type(kernel, (), {})()))
    mlp.down_proj = SimpleNamespace(tp_size=1, reduce_results=False,
        scheme=SimpleNamespace(kernel=type(kernel, (), {})()))
    monkeypatch.setattr(MiniMaxM3MLP, 'forward', lambda self, x: events.append('native-shared') or value)
    class Shared:
        enable_dbo = dbo
        _disable_shared_experts_overlap = disabled
        _stream = object() if stream else None
        _layer = mlp
        def __init__(self):
            self._output = [None]
            self._input_ready_event = [Event()]
            self._output_ready_event = [Event()]
        @property
        def output(self): return self._output[0] if self._output[0] is not None else value
        def maybe_forward_async(self, x): events.append('stock'); return False
        def wait(self): events.append('shared-join')
    method = SimpleNamespace(is_monolithic=False, topk_indices_dtype=torch.int64,
        quant_config=SimpleNamespace(routed_backend=backend))
    layer = SimpleNamespace(global_num_experts=128, quant_method=method,
                            _ensure_moe_quant_config_init=lambda: None)
    obj = SimpleNamespace(_pcp_ep_group=lambda: SimpleNamespace(rank_in_group=0),
        gate=None, routed_experts=layer, _shared_experts=Shared(), _quant_method=method,
        _hcu_shared_pending=False,
        _maybe_apply_shared_experts=lambda *a: events.append('no-overlap'),
        router=SimpleNamespace(select_experts=lambda **kw: (torch.ones(rows, 1),
            torch.zeros(rows, 1, dtype=torch.int64))))
    obj._launch_long_shared = MethodType(runner.MiniMax31DeepEpRunner._launch_long_shared, obj)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *a: object())
    monkeypatch.setattr(torch.cuda, 'stream', lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(runner, 'get_ep_group', lambda: None)
    monkeypatch.setattr(runner, '_get_deepep_buffer', lambda *a: None)
    monkeypatch.setattr(runner, 'dispatch_routed', lambda hidden, *a, **kw: hidden)
    monkeypatch.setattr(runner, '_get_aligned_moe', lambda *a:
        (lambda hidden, *a, **kw: hidden, object()))
    hidden = torch.zeros(rows, 1, dtype=torch.bfloat16)
    result = runner.MiniMax31DeepEpRunner._forward_locked(obj, hidden, None, value)
    assert result[0] is value and result[1] is hidden
    eligible = (rows >= 2048 and backend == 'deepgemm_hipc_v1'
                and kernel == 'MiniMax31HipBlasLtInt8Kernel'
                and stream and not disabled and not dbo)
    assert ('native-shared' in events) == eligible
    assert ('stock' in events) == (not eligible)
    assert ('no-overlap' in events) == (not eligible)
    assert ('shared-join' in events) == eligible
