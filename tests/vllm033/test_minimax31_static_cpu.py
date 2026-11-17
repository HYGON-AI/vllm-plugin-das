"""Static capacity, raw-loading and native-plus-publication negative contracts."""
import copy,io,json,unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import torch
from hcu_vllm_das.parallel import minimax31_static_contract as c
from hcu_vllm_das.parallel.minimax31_static_experts import StaticRoutedExperts
from hcu_vllm_das.parallel import minimax31_static_ipc as ipc
from vllm.model_executor.model_loader.weight_cache.protocol import WeightCacheKey,recv_msg

class Contracts(unittest.TestCase):
    def contract(self):
        rows=[list(range(128))+list(range(8)) for _ in range(57)]
        return dict(schema='minimax31-static-runtime-contract-v1',logical_experts=128,physical_experts=136,ep_size=8,slots_per_rank=17,model_layer_ids=list(c.LAYERS),physical_to_logical=rows,map_sha256=c.sha({'model_layer_ids':list(c.LAYERS),'physical_to_logical':rows}),selector=c.SELECTOR.copy(),loader_abi=c.ABI,capture_sha256='1'*64,dataset_manifest_sha256='2'*64,precision_decision_sha256='3'*64)
    def test_all_layers_and_digest_fail_closed(self):
        value=self.contract();p=c.validate(value);self.assertEqual(len(p.local(59,7)),17)
        for mutate in [lambda v:v['physical_to_logical'][56].__setitem__(135,9),lambda v:v.__setitem__('physical_experts',True),lambda v:v['selector'].__setitem__('seed_u64',0),lambda v:v.__setitem__('model_layer_ids',list(range(57)))]:
            wrong=copy.deepcopy(value);mutate(wrong)
            with self.assertRaises(ValueError):c.validate(wrong)
    def test_nested_text_model_scope_is_bounded(self):
        outer=SimpleNamespace(hf_config=SimpleNamespace(quantization_config={"hcu_minimax_static_placement":self.contract()}))
        inner=SimpleNamespace(hf_config=SimpleNamespace(architectures=["MiniMaxM3SparseForCausalLM"]))
        with patch("vllm.config.get_current_vllm_config",return_value=SimpleNamespace(model_config=inner)):
            self.assertIsNone(c.from_current())
            with c.model_construction_scope(outer):
                self.assertEqual(c.from_current().map_sha256,self.contract()["map_sha256"])
                inner.hf_config.architectures=["OtherModel"]
                with self.assertRaises(RuntimeError):c.from_current()
                inner.hf_config.architectures=["MiniMaxM3SparseForCausalLM"]
            self.assertIsNone(c.from_current())
    def test_model_frame_rejects_reentry_and_out_of_order_layers(self):
        from hcu_vllm_das.parallel import minimax31_static_runtime as rt
        old=rt._LAST_LAYER;rt._LAST_LAYER=4;key="00"*32;rt._REQUESTS["frame-test"]=(key,c.request_base(key))
        batch=SimpleNamespace(num_tokens=2,num_tokens_after_padding=2,num_scheduled_tokens=torch.tensor([2]),req_ids=["frame-test"],num_computed_tokens_np=torch.tensor([131072]),positions=torch.tensor([131072,131073]))
        try:
            rt.set_forward_layout(batch)
            self.assertEqual(rt.layout(2)["next_layer"],3)
            with self.assertRaises(RuntimeError):rt.set_forward_layout(batch)
            rt.complete_layer(3);rt.complete_layer(4)
            with self.assertRaises(RuntimeError):rt.layout(2)
            rt.set_forward_layout(batch);rt.complete_layer(3);rt.complete_layer(4)
        finally:
            if rt._FRAME_LOCK.locked():rt._FRAME_LOCK.release()
            rt._LAYOUT=None;rt._LAST_LAYER=old;rt._REQUESTS.pop("frame-test",None)
    def test_selector_frozen_golden(self):
        h=c.request_base('00'*32)
        for x in (131072,3,5):h=c.mix(h^x)
        self.assertEqual([2,17,135][h%3],17)
    def test_slot16_and_duplicate_logical_load(self):
        layer=StaticRoutedExperts.__new__(StaticRoutedExperts);torch.nn.Module.__init__(layer)
        layer._static_local_owners=tuple(range(16))+(5,);layer._static_load_receipts={};layer.expert_map_manager=SimpleNamespace(ep_rank=2);layer.layer_name='language_model.model.layers.3.block_sparse_moe.experts';layer.moe_config=SimpleNamespace(intermediate_size_per_partition=8)
        for n,shape,dtype in [('w13_weight',(17,16,16),torch.int8),('w2_weight',(17,32,4),torch.int8),('w13_weight_scale',(17,16,1),torch.float32),('w2_weight_scale',(17,32,1),torch.float32)]:layer.register_parameter(n,torch.nn.Parameter(torch.zeros(shape,dtype=dtype),requires_grad=False))
        for n,shard in [('w13_weight','w1'),('w13_weight','w3'),('w2_weight','w2'),('w13_weight_scale','w1'),('w13_weight_scale','w3'),('w2_weight_scale','w2')]:
            p=getattr(layer,n);shape=tuple(p.shape[1:]);shape=(8,)+shape[1:] if n.startswith('w13') else shape;src=torch.arange(torch.tensor(shape).prod().item()).reshape(shape).to(p.dtype)
            self.assertTrue(layer.weight_loader(p,src,n,shard,5,True))
            for slot in [5,16]:
                target=p[slot];target=target[:8] if shard=='w1' else target[8:] if shard=='w3' else target;self.assertTrue(torch.equal(target,src))
            with self.assertRaises(RuntimeError):layer.weight_loader(p,src,n,shard,5)
        self.assertEqual(len(layer._static_load_receipts),12)
        with self.assertRaises(ValueError):layer.weight_loader(layer.w13_weight,torch.ones(8,16,dtype=torch.int8),'w13_weight','w1',128)
    def native_key(self):
        return WeightCacheKey(checkpoint='checkpoint',model_arch='MiniMaxM3',tp_size=1,tp_rank=0,dtype='torch.bfloat16',quantization='slimquant_w4a8',quant_config_hash='q',rope_parameters_hash='r',kv_cache_dtype='fp8',kv_cache_dtype_skip_layers=(),revision=None,vllm_version='unit',enable_expert_parallel=True)
    def daemon(self):
        model=torch.nn.Linear(2,3,bias=False);tensors,aliases,refs=ipc.model_inventory(model);manifest={'generation':'a'*32,'rank':0,'gpu_uuid':'test0','map_sha256':'b'*64,'tensors':tensors,'aliases':aliases,'layer_loading_proofs':{}}
        obj=SimpleNamespace(model=model,cache_key=self.native_key(),role='target',global_rank=0,_static_manifest=manifest,_static_manifest_sha256=c.sha(manifest),_static_storage_snapshot={n:(id(t),t.data_ptr(),ipc.version(t)) for n,t in refs.items()})
        return obj
    def test_native_key_and_generation_reject_before_handles(self):
        daemon=self.daemon();good={'cache_key':daemon.cache_key,'is_reload':False,'publication':ipc.publication_identity(daemon)}
        wrong=copy.deepcopy(good);wrong['publication']['generation']='c'*32;wire=ipc.CaptureWire()
        with patch('vllm.model_executor.model_loader.weight_cache.daemon.export_entries',side_effect=AssertionError('export before generation validation')):ipc.serve_static(daemon,wire,wrong)
        self.assertEqual(recv_msg(wire)['status'],'mismatch')
        wrong=copy.deepcopy(good);wrong['cache_key']=replace(wrong['cache_key'],checkpoint='wrong');wire=ipc.CaptureWire()
        with patch('vllm.model_executor.model_loader.weight_cache.daemon.export_entries',side_effect=AssertionError('export before native full key validation')):ipc.serve_static(daemon,wire,wrong)
        self.assertEqual(recv_msg(wire)['status'],'mismatch')
    def test_storage_changed_rejects(self):
        daemon=self.daemon();daemon.model.weight.data.add_(1)
        # .data can evade version counters: actual weight samples are checked
        # for StaticRoutedExperts. Metadata/Parameter replacement must reject.
        daemon.model.weight=torch.nn.Parameter(daemon.model.weight.detach().clone())
        with self.assertRaises(RuntimeError):ipc.assert_unchanged(daemon)
if __name__=='__main__':unittest.main()
