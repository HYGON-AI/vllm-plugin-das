# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU-only byte-copy regressions for the FlashAttention Mooncake wire views.

Load the actual small helpers and registration method without initializing the
device runtime. Real CPU tensors exercise storage aliasing, offsets and padding.
"""
from __future__ import annotations
import ast
import ctypes
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[2] / 'vllm_hcu/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py'

class AttentionSpec:
    def __init__(self, size, padded=None):
        self.page_size_bytes = size if padded is None else padded
        self.unpadded_page_size_bytes = size
class MLAAttentionSpec(AttentionSpec): pass
class SlidingWindowMLASpec(AttentionSpec): pass
class KpoolTailSpec: pass

@pytest.fixture(scope='module')
def api():
    tree = ast.parse(SOURCE.read_text())
    names = {'_attention_transfer_page_views', '_compute_sender_transfer_plan'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MooncakeConnectorWorker')
    nodes += [next(n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == 'register_kv_caches')]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    spec = importlib.util.spec_from_file_location("step5_aux_views_test", SOURCE.with_name("step5_aux.py"))
    auxiliary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auxiliary)
    ns = dict(create_step5_aux_manager=auxiliary.create_step5_aux_manager,
              torch=torch, AttentionSpec=AttentionSpec, MLAAttentionSpec=MLAAttentionSpec,
              SlidingWindowMLASpec=SlidingWindowMLASpec, KpoolTailSpec=KpoolTailSpec,
              logger=logging.getLogger(__name__), extract_layer_index=lambda name:int(name.split('.')[2]))
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), ns)
    return SimpleNamespace(**{name:ns[name] for name in names|{'register_kv_caches'}})

def cache(blocks=4,heads=4,tokens=3,dim=2,dtype=torch.uint8,padding=19):
    e = torch.empty((),dtype=dtype).element_size()
    stride = 2*heads*tokens*dim+padding
    backing = torch.full((blocks*stride,),37,dtype=dtype)
    physical = torch.as_strided(backing,(blocks,2,heads,tokens,dim),(stride,heads*tokens*dim,tokens*dim,dim,1))
    physical.copy_(torch.arange(physical.numel()).reshape(physical.shape).to(dtype))
    return physical.permute(0,1,3,2,4), backing, 2*heads*tokens*dim*e

def worker(layer_spec, group=1):
    engine = SimpleNamespace(calls=[])
    engine.batch_register_memory=lambda ptrs,sizes: (engine.calls.append((list(ptrs),list(sizes))) or 0)
    return SimpleNamespace(use_mla=False, _layer_specs={'model.layers.0.attn':layer_spec},
        _layer_group_indices={'model.layers.0.attn':group}, _pp_layer_range=None,
        dsa_layer_split_enabled=False, _physical_blocks_per_logical_kv_block=1,
        _log_debug_cache_registration=lambda *args:None, engine=engine,is_kv_consumer=True,
        _step5_pd=True)

@pytest.mark.parametrize('dtype',[torch.uint8,torch.bfloat16])
def test_explicit_kv_registration_preserves_storage_and_group(api,dtype):
    tensor,backing,payload=cache(dtype=dtype)
    w=worker(AttentionSpec(payload))
    api.register_kv_caches(w,{'model.layers.0.attn':tensor})
    views=api._attention_transfer_page_views(tensor,allow_explicit_kv=True)
    assert len(views)==2
    assert w.kv_caches_base_addr == [v.data_ptr() for v in views]
    assert w.block_len_per_layer == [tensor.stride(0)*tensor.element_size()]*2
    assert w.kv_block_len_per_layer == [payload//2]*2
    assert w.registered_layer_names == ['model.layers.0.attn']*2
    assert w.registered_group_indices == [1,1]
    assert w.num_blocks_per_layer == [4,4]
    assert w.engine.calls == [([backing.data_ptr()],[backing.untyped_storage().nbytes()])]
    assert all(v.untyped_storage().data_ptr()==backing.data_ptr() for v in views)
    assert w.device_kv_caches['model.layers.0.attn'] is tensor

@pytest.mark.parametrize('dtype',[torch.uint8,torch.bfloat16])
def test_tp1_to_tp4_byte_copy_with_padded_nonidentity_blocks(api,dtype):
    source,_,payload=cache(dtype=dtype)
    p=worker(AttentionSpec(payload));api.register_kv_caches(p,{'model.layers.0.attn':source})
    topo=SimpleNamespace(tp_rank=0,tp_ratio=lambda _: -4)
    for rank in range(4):
        target,backing,dpayload=cache(blocks=3,heads=1,dtype=dtype,padding=23)
        backing.fill_(37)
        d=worker(AttentionSpec(dpayload));api.register_kv_caches(d,{'model.layers.0.attn':target})
        for region in range(2):
            send,src_offset,dst_offset,length=api._compute_sender_transfer_plan(topo,rank,4,p.kv_block_len_per_layer[region],d.kv_block_len_per_layer[region],False)
            assert send and length==dpayload//2
            for src_block,dst_block in [(3,0),(1,2)]:
                ctypes.memmove(d.kv_caches_base_addr[region]+dst_block*d.block_len_per_layer[region]+dst_offset,
                               p.kv_caches_base_addr[region]+src_block*p.block_len_per_layer[region]+src_offset,length)
        for src_block,dst_block in [(3,0),(1,2)]:
            assert torch.equal(target[dst_block,:,:,0,:].view(torch.uint8),source[src_block,:,:,rank,:].view(torch.uint8))
        assert (target[1].to(torch.float32)==37).all()
        stride=target.stride(0)
        page_elements=2*3*2
        for block in range(3):
            assert (backing[block*stride+page_elements:(block+1)*stride].to(torch.float32)==37).all()

def test_existing_packed_four_dimensional_view_is_unchanged(api):
    tensor=torch.zeros((3,4,5,6),dtype=torch.uint8)
    w=worker(AttentionSpec(4*5*6),group=0)
    api.register_kv_caches(w,{'model.layers.0.attn':tensor})
    assert w.kv_caches_base_addr == [tensor.data_ptr()]
    assert w.kv_block_len_per_layer == [4*5*6]
    assert api._attention_transfer_page_views(tensor)[0] is tensor

def test_rejects_noncontiguous_head_pages_and_wrong_kv_axis(api):
    with pytest.raises(ValueError,match='head-contiguous'):
        api._attention_transfer_page_views(torch.empty(2,2,3,4,5),allow_explicit_kv=True)
    with pytest.raises(ValueError,match='requires'):
        api._attention_transfer_page_views(torch.empty(2,3,3,4,5))


def test_non_step5_keeps_existing_four_dimensional_cache_contract(api):
    tensor, _, payload = cache()
    w = worker(AttentionSpec(payload))
    w._step5_pd = False
    with pytest.raises(ValueError, match='requires'):
        api.register_kv_caches(w, {'model.layers.0.attn': tensor})
    assert not w.engine.calls

def test_rejects_mismatched_spec_payload(api):
    tensor,_,payload=cache()
    with pytest.raises(ValueError,match='shape does not match'):
        api.register_kv_caches(worker(AttentionSpec(payload-2)),{'model.layers.0.attn':tensor})
