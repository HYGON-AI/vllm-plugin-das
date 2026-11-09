"""Synthetic CPU contract tests; no GPU equivalence is claimed."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('vl_inspection_probe',HERE/'vl_inspection_probe.py')
inspection=None
if Path(spec.origin).exists():
    inspection=importlib.util.module_from_spec(spec);spec.loader.exec_module(inspection)

class InspectorTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(inspection,'idle VL static inspector implementation required')
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.row=list(range(256))+list(range(16))
        self.rows=[self.row.copy() for _ in range(77)]
        path=Path(self.folder.name)/'map.json'
        raw=json.dumps({'version':2,'model_maps':{'HYV4VLForConditionalGeneration':{
            'physical_to_logical_map':self.rows}}}).encode();path.write_bytes(raw)
        self.plan=SimpleNamespace(model_key='HYV4VLForConditionalGeneration',source_path=str(path),
            source_sha256=hashlib.sha256(raw).hexdigest(),num_logical_experts=256,
            num_physical_experts=272,num_redundant_experts=16,layer_map=lambda i:self.rows[i])
        self.model=type('HYV4VLForConditionalGeneration',(),{})()
        self.model._vllm_hcu_static_eplb_plan=self.plan
        self.model.num_logical_experts=256;self.model.num_physical_experts=272;self.model.num_redundant_experts=16
        self.model.moe_layers=[]
        for _ in range(77):
            count=torch.tensor([2]*16+[1]*240,dtype=torch.int64)
            inverse=torch.full((256,2),-1,dtype=torch.int64)
            for logical in range(256):
                slots=[p for p,e in enumerate(self.row) if e==logical]
                inverse[logical,:len(slots)]=torch.tensor(slots)
            owner=SimpleNamespace(local_num_experts=17,_vllm_hcu_static_eplb_row=self.row.copy(),
                moe_config=SimpleNamespace(num_logical_experts=256,num_experts=272),
                w13_weight=torch.arange(1700,dtype=torch.int32).reshape(17,100),
                w2_weight=torch.arange(1700,dtype=torch.int32).reshape(17,100),
                w13_weight_scale=torch.arange(1700,dtype=torch.float32).reshape(17,100),
                w2_weight_scale=torch.arange(1700,dtype=torch.float32).reshape(17,100))
            router=SimpleNamespace(capture_fn=None,eplb_state=SimpleNamespace(
                logical_replica_count=count,logical_to_physical_map=inverse))
            self.model.moe_layers.append(SimpleNamespace(router=router,routed_experts=owner))
        parallel=SimpleNamespace(tensor_parallel_size=1,prefill_context_parallel_size=16,
            data_parallel_size=1,decode_context_parallel_size=1,pipeline_parallel_size=1,
            enable_eplb=True,enable_expert_parallel=True,all2all_backend='deepep_high_throughput')
        self.runner=SimpleNamespace(model=self.model,model_state=SimpleNamespace(),
            vllm_config=SimpleNamespace(parallel_config=parallel,speculative_config=None,
                model_config=SimpleNamespace(architectures=['HYV4VLForConditionalGeneration'],enforce_eager=True),
                kernel_config=SimpleNamespace(moe_backend='deep_gemm')),
            num_speculative_steps=0)

    def inspect(self,**kwargs):
        return inspection.inspect_runner(self.runner,rank=0,world_size=16,**kwargs)

    def test_valid_77_layer_plan_and_local_fingerprint_coverage(self):
        owner=self.model.moe_layers[0].routed_experts
        before=owner.w13_weight.clone()
        result=self.inspect(samples=True)
        self.assertEqual(result['models'][0]['physical'],272)
        self.assertEqual(len(result['layers']),77)
        self.assertEqual(len(result['fingerprints']),1309)
        first=result['fingerprints'][0]
        self.assertEqual((first['layer'],first['physical'],first['logical']),(0,0,0))
        self.assertEqual(first['components']['w13_weight_scale']['mode'],'full')
        self.assertEqual(first['components']['w13_weight_scale']['selected_elements'],100)
        self.assertEqual(first['components']['w13_weight']['selected_elements'],32)
        self.assertTrue(torch.equal(owner.w13_weight,before))
        self.assertIsNone(self.model.moe_layers[0].router.capture_fn)

    def test_bad_rows_owner_dimensions_and_replica_counts_fail(self):
        self.rows[0][-1]=256
        with self.assertRaises(ValueError):self.inspect()
        self.rows[0][-1]=15
        self.model.moe_layers[0].routed_experts._vllm_hcu_static_eplb_row[0]=1
        with self.assertRaises(ValueError):self.inspect()
        self.model.moe_layers[0].routed_experts._vllm_hcu_static_eplb_row[0]=0
        state=self.model.moe_layers[0].router.eplb_state
        state.logical_replica_count[0]=1
        with self.assertRaises(ValueError):self.inspect()
        state.logical_replica_count[0]=2
        state.logical_to_physical_map[0,1]=257
        with self.assertRaises(ValueError):self.inspect()
        state.logical_to_physical_map[0,1]=256
        self.model.moe_layers[0].routed_experts.w2_weight_scale=torch.zeros(16,100)
        with self.assertRaises(ValueError):self.inspect()

    def test_full_scales_catch_change_outside_weight_sample(self):
        owner=self.model.moe_layers[0].routed_experts
        first=inspection.fingerprint_expert(owner,0)
        self.assertNotIn(1,first['components']['w13_weight']['flat_indices'])
        owner.w13_weight_scale[0,1]+=1
        second=inspection.fingerprint_expert(owner,0)
        self.assertNotEqual(first['sha256'],second['sha256'])
        self.assertEqual(first['components']['w13_weight']['sha256'],second['components']['w13_weight']['sha256'])
        self.assertNotEqual(first['components']['w13_weight_scale']['sha256'],second['components']['w13_weight_scale']['sha256'])

    def test_fingerprint_independent_of_physical_slot_and_covers_dtype_shape(self):
        owner=self.model.moe_layers[0].routed_experts
        for name in inspection.TENSORS:
            value=getattr(owner,name);value[1].copy_(value[0])
        first=inspection.fingerprint_expert(owner,0)
        self.assertEqual(first['sha256'],inspection.fingerprint_expert(owner,1)['sha256'])
        owner.w2_weight_scale=owner.w2_weight_scale.to(torch.float64)
        self.assertNotEqual(first['sha256'],inspection.fingerprint_expert(owner,0)['sha256'])

    def test_mtp_active_hooks_wrong_topology_and_no_plan_fail(self):
        with self.assertRaises(ValueError):self.inspect(calibration_active=True)
        with self.assertRaises(ValueError):self.inspect(diagnostic_active=True)
        self.runner.speculator=object()
        with self.assertRaises(ValueError):self.inspect()
        self.runner.speculator=None
        self.runner.vllm_config.speculative_config={}
        with self.assertRaises(ValueError):self.inspect()
        self.runner.vllm_config.speculative_config=None
        self.runner.vllm_config.parallel_config.data_parallel_size=16
        with self.assertRaises(ValueError):self.inspect()
        self.runner.vllm_config.parallel_config.data_parallel_size=1
        self.model.moe_layers[0].router.capture_fn=lambda ids:None
        with self.assertRaises(ValueError):self.inspect()
        self.model.moe_layers[0].router.capture_fn=None
        self.model._vllm_hcu_static_eplb_plan=None
        with self.assertRaises(ValueError):self.inspect()

    def test_r0_uses16_slots_and_rejects_boolean_map_ids(self):
        self.rows[:]=[list(range(256)) for _ in range(77)]
        self.plan.num_physical_experts=256;self.plan.num_redundant_experts=0
        self.model.num_physical_experts=256;self.model.num_redundant_experts=0
        for layer in self.model.moe_layers:
            owner=layer.routed_experts;owner.local_num_experts=16
            owner._vllm_hcu_static_eplb_row=list(range(256));owner.moe_config.num_experts=256
            for name in inspection.TENSORS:setattr(owner,name,getattr(owner,name)[:16])
            layer.router.eplb_state.logical_replica_count=torch.ones(256,dtype=torch.int32)
            layer.router.eplb_state.logical_to_physical_map=torch.arange(256).reshape(256,1)
        raw=json.dumps({'rows':self.rows}).encode();Path(self.plan.source_path).write_bytes(raw)
        self.plan.source_sha256=hashlib.sha256(raw).hexdigest()
        self.assertEqual(self.inspect()['models'][0]['local_experts'],16)
        self.rows[0][0]=False
        with self.assertRaises(ValueError):self.inspect()

    def test_noncontiguous_weights_hash_the_same_values_as_contiguous_view(self):
        owner=self.model.moe_layers[0].routed_experts
        owner.w13_weight=owner.w13_weight.reshape(17,10,10).transpose(1,2)
        first=inspection.fingerprint_expert(owner,0)
        owner.w13_weight=owner.w13_weight.contiguous()
        self.assertEqual(first['sha256'],inspection.fingerprint_expert(owner,0)['sha256'])

    def test_map_source_hash_mismatch_fails(self):
        Path(self.plan.source_path).write_text('{}')
        with self.assertRaisesRegex(ValueError,'SHA256'):self.inspect()

if __name__=='__main__':unittest.main()
