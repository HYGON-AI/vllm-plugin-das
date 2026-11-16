"""CPU reproduction of producer registered counter vs consumer meta capacity."""
import json
import torch
from hcu_vllm_das.quantization.minimax31 import restore_minimax31_scratch
from hcu_vllm_das.parallel.minimax31_pcp import MiniMax31PCPSparseAttention
from vllm.model_executor.model_loader.weight_cache.ipc_loader import IpcModelLoader

class Impl(torch.nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.topk_indices_buffer = shared
        self.register_buffer('topk_completion_counter', torch.zeros(shared.shape[:2], dtype=torch.int32), persistent=False)
class Indexer(torch.nn.Module):
    def __init__(self, shared):
        super().__init__(); self.impl = Impl(shared)
class Layer(torch.nn.Module):
    def __init__(self, shared):
        super().__init__()
        a = MiniMax31PCPSparseAttention.__new__(MiniMax31PCPSparseAttention)
        torch.nn.Module.__init__(a)
        a.topk_indices_buffer = shared; a.sparse_bt_buffer = a.sparse_ctx_buffer = None
        a.indexer = Indexer(shared); self.self_attn = a
class MiniMaxM3Model(torch.nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self.topk_indices_buffer = torch.empty((4,tokens,16), dtype=torch.int32)
        self.sparse_bt_buffer = self.sparse_ctx_buffer = None
        self.layers = torch.nn.ModuleList([Layer(self.topk_indices_buffer), Layer(self.topk_indices_buffer)])
MiniMaxM3Model.__module__ = 'vllm.models.minimax_m3.amd.model'

class Entry:
    kind = 'buffer'
    def __init__(self, tensor): self.tensor = tensor
    def rebuild(self, _): return self.tensor
class State:
    aliases = {}
    def __init__(self, tensors):
        self.entries = {f'layers.{i}.self_attn.indexer.impl.topk_completion_counter':Entry(t) for i,t in enumerate(tensors)}

def make(tokens, shape=(4,2048), dtype=torch.int32):
    with torch.device('meta'): model = MiniMaxM3Model(tokens)
    producer = [torch.full(shape, 7, dtype=dtype), torch.full(shape, 9, dtype=dtype)]
    loader = object.__new__(IpcModelLoader); loader.mode = 'zero_copy'
    loader._apply_entries(model, State(producer), 0)
    return model, producer

def test_ipc_counter_capacity_and_ownership():
    model, producer = make(65536)
    assert all('topk_completion_counter' not in x.self_attn.indexer.impl._non_persistent_buffers_set for x in model.layers)
    old = model.topk_indices_buffer
    assert restore_minimax31_scratch(model, torch.device('cpu')) == 2
    replacement = model.topk_indices_buffer
    counters=[]
    for layer in model.layers:
        a=layer.self_attn; impl=a.indexer.impl; c=impl.topk_completion_counter
        assert a.topk_indices_buffer is replacement and impl.topk_indices_buffer is replacement
        assert c.shape == (4,65536) and c.dtype == torch.int32 and torch.count_nonzero(c) == 0
        assert 'topk_completion_counter' in impl._non_persistent_buffers_set
        counters.append(c)
    assert replacement is not old and counters[0].data_ptr()!=counters[1].data_ptr()
    assert all(c.data_ptr()!=p.data_ptr() for c in counters for p in producer)
    assert torch.all(producer[0]==7) and torch.all(producer[1]==9)
    assert restore_minimax31_scratch(model,torch.device('cpu')) == 0

    for shape,dtype in [((3,2048),torch.int32),((4,0),torch.int32),((4,2048),torch.float32),((4,1,1),torch.int32)]:
        bad, source=make(65536,shape,dtype)
        before=bad.topk_indices_buffer
        try: restore_minimax31_scratch(bad,torch.device('cpu'))
        except RuntimeError: pass
        else: raise AssertionError((shape,dtype))
        assert bad.topk_indices_buffer is before
        assert all(layer.self_attn.indexer.impl.topk_completion_counter is source[i] for i,layer in enumerate(bad.layers))
    # Consumer capacity can be smaller than the producer's; no source mutation.
    small, src=make(32)
    assert restore_minimax31_scratch(small,torch.device('cpu')) == 2
    assert all(layer.self_attn.indexer.impl.topk_completion_counter.shape == (4,32) for layer in small.layers)
    assert torch.all(src[0] == 7)
    print(json.dumps({'test':'scratch_capacity','status':'PASS','consumer':65536,'producer':2048,
                      'checks':['real_ipc_apply_entries','private_zero','aliases','nonpersistent','source_unchanged','negative_shapes','smaller_consumer']}))
