"""CPU contract tests; synthetic tensors do not establish HCU correctness."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import asyncio
import torch
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('vl_diagnostic_probe',HERE/'vl_diagnostic_probe.py')
diag=None
if Path(spec.origin).exists():
    diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)

class Rope:
    def __init__(self):
        self.positions=torch.zeros(3,17,dtype=torch.int64)
        self.prefill_positions=SimpleNamespace(gpu=torch.stack([torch.arange(10)+d for d in (10,20,30)]))
        self.prefill_delta=SimpleNamespace(np=np.array([2]),gpu=torch.tensor([2]))
        self.num_dims=3;self.has_delta=True;self.get_calls=0
    def get_positions(self,n):
        self.get_calls+=1
        return self.positions[:,:n]
    def read_prefill_positions(self,index,length):
        return self.prefill_positions.gpu[3*index:3*(index+1),:length]

class State:
    def __init__(self,manager):
        self.pcp_manager=manager;self._pcp_size=16;self._dummy_run=False
        self.rope_state=Rope();self.calls=0
        self._local_positions=torch.zeros(3,17,dtype=torch.int64)
        self._local_inputs_embeds=torch.zeros(16,4)
        self.encoder_runner=SimpleNamespace(inputs_embeds=torch.zeros(16,4))
        self._hcu_vl_persistent_memory_bytes=1024
    def prepare_inputs(self,batch,req_states):
        self.calls+=1
        self.rope_state.positions[:,:4].copy_(self.rope_state.prefill_positions.gpu[:,3:7])
        self._local_positions[:,:4].copy_(self.rope_state.positions[:,[2,0,1,0]])
        self._local_positions[:,3]=0
        return {'positions':self._local_positions[:,:4]}

class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(diag,'bounded native-position diagnostic implementation required')
        self.global_batch=SimpleNamespace(positions=torch.tensor([3,4,5,6]),num_tokens=4,num_tokens_after_padding=4,
            num_reqs=1,req_ids=['request-A'],idx_mapping=torch.tensor([0]),query_start_loc=torch.tensor([0,4]),
            prefill_len_np=np.array([10]),num_computed_tokens_np=np.array([3]),num_scheduled_tokens=np.array([4]))
        self.local=SimpleNamespace(positions=torch.tensor([5,3,4,0]),num_tokens=3,num_tokens_after_padding=4,
            num_reqs=1,req_ids=['request-A'],is_padding=torch.tensor([False,False,False,True]))
        self.manager=SimpleNamespace(pcp_rank=0,pcp_size=16,
            get_global_batch=lambda batch:self.global_batch,
            get_local_token_mapping=lambda batch:(torch.tensor([2,0,1,0]),~batch.is_padding))
        self.state=State(self.manager)
        model=type('HYV4VLForConditionalGeneration',(),{})()
        model.num_logical_experts=256;model.moe_layers=[object() for _ in range(77)]
        budget=SimpleNamespace(max_model_len=10,max_num_reqs=1,mm_limits={'image':1},
            encoder_compute_budget=16,encoder_cache_size=32,mm_max_toks_per_item={'image':8},
            mm_max_items_per_prompt={'image':1},mm_max_items_per_batch={'image':2})
        self.worker=diag.WorkerExtension();self.worker.device=torch.device('cpu')
        self.worker.model_runner=SimpleNamespace(model=model,model_state=self.state,device=self.worker.device,
            model_memory_usage=4096,mm_budget=budget,
            encoder_cache=SimpleNamespace(encoder_outputs={}))
        self.req_states=SimpleNamespace(prefill_len=SimpleNamespace(gpu=torch.tensor([10])),
            num_computed_tokens=SimpleNamespace(gpu=torch.tensor([3])))

    def test_all_core_rpc_reaches_every_independent_dp_core(self):
        class Core:
            core_engines=list(range(16))
            async def _call_utility_async(self,method,rpc,timeout,args,kwargs,*,engine):
                if (method,rpc,timeout,args,kwargs)!=('collective_rpc','hy4_vl_diag_info',120,(),None):
                    raise AssertionError('native utility argument contract changed')
                return [{'rank':engine}]
        engine=SimpleNamespace(engine_core=Core())
        function=getattr(diag,'all_core_rpc',None)
        self.assertIsNotNone(function,'diagnostic RPC must fan out to all DP cores')
        workers=asyncio.run(function(engine,'hy4_vl_diag_info',()))
        self.assertEqual([row['rank'] for row in workers],list(range(16)))

    def test_exact_pcp_and_fullprompt_window_and_canonical_unchanged(self):
        self.worker.hy4_vl_diag_start(limit=1)
        result=self.state.prepare_inputs(self.local,self.req_states)
        info=self.worker.hy4_vl_diag_info(include_records=True)
        record=info['records'][0]
        self.assertEqual(self.state.calls,1)
        self.assertEqual(self.state.rope_state.get_calls,1)
        self.assertEqual(result['positions'].tolist(),[[15,13,14,0],[25,23,24,0],[35,33,34,0]])
        self.assertTrue(record['pcp']['selected_positions_equal_native'])
        self.assertEqual(record['pcp']['global_indices'],[2,0,1,0])
        self.assertEqual(record['pcp']['valid_tokens'],[True,True,True,False])
        self.assertEqual(record['requests'][0]['computed_tokens'],3)
        self.assertTrue(record['requests'][0]['prefill_chunk_equal_staged_fullprompt'])
        self.assertEqual(record['model_positions']['values'],result['positions'].tolist())
        self.assertTrue(record['canonical_local_unchanged'])
        self.assertTrue(record['canonical_global_unchanged'])
        self.assertFalse(info['hook_installed'])
        self.assertNotIn('prepare_inputs',self.state.__dict__)
        self.assertEqual(record['memory']['model_memory_usage'],4096)
        self.assertEqual(record['persistent_buffers']['_local_positions']['shape'],[3,17])

    def test_limit_eight_auto_restores_and_does_not_capture_ninth(self):
        self.worker.hy4_vl_diag_start(limit=8)
        for _ in range(9):self.state.prepare_inputs(self.local,self.req_states)
        info=self.worker.hy4_vl_diag_info(include_records=True)
        self.assertEqual(self.state.calls,9)
        self.assertEqual(len(info['records']),8)
        self.assertEqual(self.state.rope_state.get_calls,8)
        self.assertNotIn('prepare_inputs',self.state.__dict__)
        with self.assertRaises(ValueError):self.worker.hy4_vl_diag_start(limit=9)

    def test_error_restores_original_instance_override_and_retains_error(self):
        def fail(*args,**kwargs):raise RuntimeError('native failure')
        self.state.prepare_inputs=fail
        self.worker.hy4_vl_diag_start(limit=1)
        with self.assertRaisesRegex(RuntimeError,'native failure'):
            self.state.prepare_inputs(self.local,self.req_states)
        self.assertIs(self.state.prepare_inputs,fail)
        self.assertIn('native failure',self.worker.hy4_vl_diag_info()['error'])

    def test_canonical_mutation_is_evidence_and_does_not_change_return(self):
        original=self.state.prepare_inputs
        def mutate(batch,req_states):
            output=original(batch,req_states);batch.positions[0]=999
            return output
        self.state.prepare_inputs=mutate
        self.worker.hy4_vl_diag_start(limit=1)
        result=self.state.prepare_inputs(self.local,self.req_states)
        self.assertEqual(result['positions'][0,0].item(),15)
        record=self.worker.hy4_vl_diag_info(include_records=True)['records'][0]
        self.assertFalse(record['canonical_local_unchanged'])
        self.assertIs(self.state.prepare_inputs,mutate)

    def test_canonical_storage_replacement_is_detected(self):
        original=self.state.prepare_inputs
        def replace(batch,req_states):
            output=original(batch,req_states);batch.positions=batch.positions.clone()
            return output
        self.state.prepare_inputs=replace
        self.worker.hy4_vl_diag_start(limit=1)
        self.state.prepare_inputs(self.local,self.req_states)
        record=self.worker.hy4_vl_diag_info(include_records=True)['records'][0]
        self.assertFalse(record['canonical_local_unchanged'])

    def test_bad_pcp_selection_fails_verification_without_mutating_return(self):
        original=self.state.prepare_inputs
        def wrong(batch,req_states):
            output=original(batch,req_states);output['positions'][0,0]=99
            return output
        self.state.prepare_inputs=wrong
        self.worker.hy4_vl_diag_start(limit=1)
        output=self.state.prepare_inputs(self.local,self.req_states)
        self.assertEqual(output['positions'][0,0].item(),99)
        self.assertFalse(self.worker.hy4_vl_diag_info(include_records=True)['records'][0]['pcp']['selected_positions_equal_native'])

    def test_calibration_conflict_limits_and_stop_error_cleanup(self):
        for limit in (0,9,True,1.5):
            with self.assertRaises(ValueError):self.worker.hy4_vl_diag_start(limit=limit)
        self.worker._hy4_vl_capture=[object()]
        with self.assertRaises(ValueError):self.worker.hy4_vl_diag_start()
        self.worker._hy4_vl_capture=None
        self.worker.hy4_vl_diag_start(limit=2)
        with self.assertRaises(ValueError):self.worker.hy4_vl_eplb_start()
        with self.assertRaises(Exception):self.worker.hy4_vl_diag_stop('/tmp/forbidden')
        self.assertNotIn('prepare_inputs',self.state.__dict__)

if __name__=='__main__':unittest.main()
