"""CPU-only regression tests for the owned VL calibration harness."""
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent

def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    if not spec or not Path(spec.origin).exists():
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

maps = load('build_vl_maps')
probe = load('vl_calibration_probe')

class BalanceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(maps, 'map validation and balance implementation is required')

    def test_uniform_and_single_rank(self):
        uniform = maps.balance_metrics([[3]*16])
        self.assertEqual(uniform['weighted_layer_max_over_mean'], 1)
        self.assertEqual(uniform['mean_layer_cv'], 0)
        self.assertEqual(maps.balance_metrics([[16]+[0]*15])['weighted_layer_max_over_mean'],16)

    def test_weighted_ratio_and_cumulative_are_distinct(self):
        loads = [[1]*16, [160]+[0]*15]
        result = maps.balance_metrics(loads)
        self.assertAlmostEqual(result['weighted_layer_max_over_mean'],161/11)
        self.assertNotEqual(result['weighted_layer_max_over_mean'],8.5)
        alternating = maps.balance_metrics([[16]+[0]*15,[0,16]+[0]*14])
        self.assertEqual(alternating['weighted_layer_max_over_mean'],16)
        self.assertEqual(alternating['cumulative_rank_max_over_mean'],8)
        permuted = maps.balance_metrics([list(reversed(row)) for row in loads])
        self.assertEqual(result,permuted)

    def test_zero_layers_are_explicit(self):
        result = maps.balance_metrics([[0]*16])
        self.assertIsNone(result['weighted_layer_max_over_mean'])
        self.assertIsNone(result['mean_layer_cv'])
        self.assertEqual(result['zero_layer_indexes'],[0])

    def test_272_physical_slots_are_17_per_rank(self):
        hist = [[0]*272]
        hist[0][16] = 5
        hist[0][17] = 3
        self.assertEqual(maps.destination_loads(hist),[[5,3]+[0]*14])

    def test_uneven_replica_scatter_and_per_selection_identity(self):
        row = list(range(256))+list(range(16))
        physical = [0]*272
        physical[0],physical[256] = 3,5
        self.assertEqual(maps.scatter_logical(physical,row),[8]+[0]*255)
        for broken in [[-1]+physical[1:],[True]+physical[1:],[0.5]+physical[1:]]:
            with self.assertRaises(ValueError): maps.scatter_logical(broken,row)
        with self.assertRaises(ValueError): maps.validate_map([row[:-1]+[256]]*77,272)
        with self.assertRaises(ValueError): maps.validate_map([row[:-1]+[False]]*77,272)

    def test_capture_rejects_bad_counts_and_mtp(self):
        data = capture(0)
        maps.validate_capture(data)
        for mutation in [lambda d:d['records'][0].update(model_key='HYV4MTP'),
                         lambda d:d['records'][0]['histogram'].__setitem__(0,-1),
                         lambda d:d['records'][0]['histogram'].__setitem__(0,True),
                         lambda d:d['records'][0].update(rows=2),
                         lambda d:d['records'].__setitem__(1,copy.deepcopy(d['records'][0]))]:
            bad=copy.deepcopy(data);mutation(bad)
            with self.assertRaises(ValueError):maps.validate_capture(bad)

    def test_complete_rank_manifest_sums_real_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            workers=[]
            for rank in range(16):
                path=Path(folder)/('rank'+str(rank)+'.json')
                raw=json.dumps(capture(rank)).encode();path.write_bytes(raw)
                workers.append({'rank':rank,'file':str(path),'sha256':hashlib.sha256(raw).hexdigest()})
            result=maps.load_capture_manifest({'workers':workers})
            self.assertEqual(result['logical'][0],[128]+[0]*255)
            self.assertEqual(result['actual_count_balance']['weighted_layer_max_over_mean'],16)
            self.assertEqual(len(result['sources']),16)

    def test_analysis_cli_prints_actual_balance_without_policy_or_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            workers=[]
            for rank in range(16):
                path=Path(folder)/('rank'+str(rank)+'.json')
                raw=json.dumps(capture(rank)).encode();path.write_bytes(raw)
                workers.append({'rank':rank,'file':str(path),'sha256':hashlib.sha256(raw).hexdigest()})
            manifest=Path(folder)/'manifest.json';manifest.write_text(json.dumps({'workers':workers}))
            result=subprocess.run([sys.executable,str(HERE/'build_vl_maps.py'),'--analyze-manifest',str(manifest)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            output=json.loads(result.stdout)
            self.assertEqual(output['actual_count_balance']['weighted_layer_max_over_mean'],16)
            self.assertEqual(output['total_selected_ids'],9856)
            self.assertIn('PCP padding',output['scope'])

    def test_ideal_prediction_splits_replicas_and_has_explicit_label(self):
        row=list(range(256))+list(range(16))
        result=maps.ideal_equal_split([[8]+[0]*255],[row])
        self.assertEqual(result['per_layer_rank_loads'],[[4]+[0]*14+[4]])
        self.assertEqual(result['metrics']['weighted_layer_max_over_mean'],8)
        self.assertIn('ideal equal-split',result['label'])

    def test_provenance_rejects_checkpoint_mismatch_and_split_overlap(self):
        data={'model_path':maps.MODEL_PATH,'model_class':maps.MODEL_KEY,'speculative_config':None,
              'config_sha256':maps.CONFIG_SHA256,'index_sha256':maps.INDEX_SHA256,
              'corpus_sha256':'a'*64,'core_sha256':'b'*64,'plugin_sha256':'c'*64,
              'item_image_hashes':{'fit':{'item_sha256':'d'*64,'image_sha256':'e'*64},
                                   'heldout':{'item_sha256':'f'*64,'image_sha256':'1'*64}},
              'fit_item_ids':['fit'],'heldout_item_ids':['heldout'],
              'preprocessing_limits':{},'input_output_lengths':{},'concurrency':1,
              'prefix_cache_state':'disabled','prompt_templates':'frozen',
              'source_files':[{'file':str(Path(__file__).resolve()),
                               'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}]}
        maps.validate_provenance(data)
        for key,value in [('config_sha256','0'*64),('heldout_item_ids',['fit']),('source_files',[{'file':__file__,'sha256':'0'*64}])]:
            bad=copy.deepcopy(data);bad[key]=value
            with self.assertRaises(ValueError):maps.validate_provenance(bad)

    def test_output_rejects_escape_and_policy_rejects_wrong_sha(self):
        for value in ['/tmp/capture','/workspace/hy4_mm_20261108/../elsewhere','relative']:
            with self.assertRaises(ValueError):maps.output_path(value)
        with self.assertRaisesRegex(ValueError,'policy SHA256'):
            maps.load_policy(__file__,'0'*64)

    def test_policy_generator_passes_exact_cpu_topology(self):
        class Policy:
            @staticmethod
            def rebalance_experts(weight,**kwargs):
                # Strict boundary double: the native module is not installed on
                # this Windows CPU test host. Test real caller contract.
                if weight.device.type!='cpu' or weight.dtype!=torch.int64:
                    raise AssertionError('generation must remain CPU int64')
                if {k:v for k,v in kwargs.items() if k!='old_global_expert_indices'}!={
                    'num_replicas':272,'num_groups':1,'num_nodes':1,'num_ranks':16}:
                    raise AssertionError('wrong native policy topology')
                return kwargs['old_global_expert_indices']
        result=maps.generate([[8]+[0]*255]*77,16,Policy)
        self.assertEqual(list(result['model_maps']),['HYV4VLForConditionalGeneration'])
        entry=result['model_maps']['HYV4VLForConditionalGeneration']
        self.assertEqual(entry['num_redundant_experts'],16)
        self.assertEqual(len(entry['physical_to_logical_map'][0]),272)

    def test_missing_ranks_and_hash_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'rank0.json';raw=json.dumps(capture(0)).encode();path.write_bytes(raw)
            entry={'rank':0,'file':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
            with self.assertRaises(ValueError):maps.load_capture_manifest({'workers':[entry]})
            workers=[dict(entry,rank=i) for i in range(16)]
            workers[0]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'SHA256'):maps.load_capture_manifest({'workers':workers})


def capture(rank):
    row={'model_key':'HYV4VLForConditionalGeneration','layer_index':0,
         'num_logical_experts':256,'top_k':8,'calls':1,'mapping_calls':1,'rows':1,
         'ignored_invalid_slots':0,'physical_ignored_invalid_slots':0,
         'histogram':[8]+[0]*255,'physical_histogram':[8]+[0]*255,
         'physical_to_logical':list(range(256)), 'physical_mapping_verified':True}
    rows=[]
    for i in range(77):
        r=copy.deepcopy(row);r['layer_index']=i;rows.append(r)
    return {'rank':rank,'records':rows,'scope':'executed real top-k including PCP padding; -1 slots excluded'}

class Router:
    def __init__(self,row):self.capture_fn=None;self.row=row;self.top_k=8
    def set_capture_fn(self,callback):self.capture_fn=callback
    def _apply_eplb_mapping(self,ids):
        # Deliberately in-place, with uneven replica selection.
        ids[ids==0]=256
        return ids
    def route(self,ids):
        if self.capture_fn:self.capture_fn(ids)
        return self._apply_eplb_mapping(ids)

class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(probe,'external routing capture implementation is required')
        row=list(range(256))+list(range(16))
        self.routers=[Router(row) for _ in range(77)]
        model=type('HYV4VLForConditionalGeneration',(),{})()
        model.moe_layers=[SimpleNamespace(router=r) for r in self.routers]
        model.num_logical_experts=256
        model._vllm_hcu_static_eplb_plan=SimpleNamespace(layer_map=lambda i:row)
        self.worker=probe.WorkerExtension()
        self.worker.model_runner=SimpleNamespace(model=model)
        self.worker.device=torch.device('cpu')

    def test_capture_preserves_logical_before_inplace_mapping_and_restores(self):
        router=self.routers[0]
        original=router._apply_eplb_mapping
        self.worker.hy4_vl_eplb_start()
        for r in self.routers:r.route(torch.tensor([[0,1,2,3,4,5,6,-1]]))
        records=self.worker._hy4_vl_capture
        self.assertEqual(records[0]['histogram'][0].item(),1)
        self.assertEqual(records[0]['physical_histogram'][256].item(),1)
        self.assertEqual(records[0]['ignored_invalid_slots'],1)
        self.worker._restore_vl_capture()
        self.assertIsNone(router.capture_fn)
        self.assertNotIn('_apply_eplb_mapping',router.__dict__)
        self.assertEqual(router._apply_eplb_mapping,original)
        self.assertIsNone(self.worker._hy4_vl_capture)

    def test_mapping_error_restores_all_routers(self):
        self.worker.hy4_vl_eplb_start()
        with self.assertRaises(ValueError):self.routers[0].route(torch.tensor([[257]*8]))
        for r in self.routers:
            self.assertIsNone(r.capture_fn)
            self.assertNotIn('_apply_eplb_mapping',r.__dict__)

    def test_stop_with_bad_output_restores_before_validation(self):
        self.worker.hy4_vl_eplb_start()
        with self.assertRaises(Exception):self.worker.hy4_vl_eplb_stop('/tmp/forbidden')
        for router in self.routers:
            self.assertIsNone(router.capture_fn)
            self.assertNotIn('_apply_eplb_mapping',router.__dict__)

    def test_wrong_physical_logical_pair_restores_hooks(self):
        self.routers[0].row=list(range(256))+list(range(16))
        def wrong(ids):
            return torch.full_like(ids,1)
        self.routers[0]._apply_eplb_mapping=wrong
        self.worker.hy4_vl_eplb_start()
        with self.assertRaisesRegex(ValueError,'selected logical'):
            self.routers[0].route(torch.tensor([[0]*8]))
        self.assertIs(self.routers[0]._apply_eplb_mapping,wrong)
        self.assertIsNone(self.routers[0].capture_fn)

    def test_duplicate_start_and_simulated_router_fail(self):
        self.worker.hy4_vl_eplb_start()
        with self.assertRaises(ValueError):self.worker.hy4_vl_eplb_start()
        self.worker._restore_vl_capture()
        self.routers[0].__class__=type('SimulatorRouter',(Router,),{})
        with self.assertRaises(ValueError):self.worker.hy4_vl_eplb_start()
        self.assertIsNone(self.routers[1].capture_fn)

    def test_start_rejects_conflicts_and_draft_before_hooking(self):
        sentinel=lambda x:None
        self.routers[-1].capture_fn=sentinel
        with self.assertRaises(ValueError):self.worker.hy4_vl_eplb_start()
        self.assertIsNone(self.routers[0].capture_fn)
        self.assertIs(self.routers[-1].capture_fn,sentinel)
        self.routers[-1].capture_fn=None
        self.worker.model_runner.drafter=SimpleNamespace(model=object())
        with self.assertRaises(ValueError):self.worker.hy4_vl_eplb_start()

if __name__=='__main__':unittest.main()
