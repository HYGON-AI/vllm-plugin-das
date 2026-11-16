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

@pytest.mark.parametrize('rows,overlap',[(8,False),(2047,False),(2048,True),(4096,True),(6144,True),(8192,True)])
def test_prefill_sizes_keep_136_physical_slots_and_join(monkeypatch,rows,overlap):
    events=[]
    monkeypatch.setenv('HCU_MINIMAX_DEEPEP_FIXEDCOUNTS','1');monkeypatch.setenv('HCU_MINIMAX_SHARED_OVERLAP','1')
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
    expert=SimpleNamespace(global_num_experts=136,quant_method=SimpleNamespace(is_monolithic=False),_ensure_moe_quant_config_init=lambda:None)
    obj=SimpleNamespace(_pcp_ep_group=lambda:SimpleNamespace(rank_in_group=0),gate=None,routed_experts=expert,_shared_experts=Shared(),_hcu_shared_pending=False,_maybe_apply_shared_experts=lambda *a:None,_launch_long_shared=lambda x:events.append('aux') or True)
    x=torch.zeros(rows,1,dtype=torch.bfloat16)
    assert runner.MiniMax31DeepEpRunner._forward_locked(obj,x,None,x)==('shared',x)
    assert events==(['aux','dispatch136','join'] if overlap else ['stock','dispatch136'])
