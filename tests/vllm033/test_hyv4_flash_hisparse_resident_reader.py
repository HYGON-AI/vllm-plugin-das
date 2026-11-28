# SPDX-License-Identifier: Apache-2.0
"""Exercise the instance-local resident lease without constructing a model."""

from pathlib import Path
import runpy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


SOURCE = (Path(__file__).resolve().parents[2] / 'adapter_extension/src/hcu_vllm_das'
          / 'models/hy_v4_flash/hisparse_runtime.py')


def tensor(pointer, shape=(128, 656)):
    return NS(data_ptr=lambda: pointer, shape=shape, stride=lambda: (shape[-1], 1),
              dtype='uint8', device='cpu', storage_offset=lambda: 0)


def pair(union=2176):
    target_cache = tensor(3)
    resident = tensor(5)
    owner_group = NS(followers=[], lru_init=object(), device_global_indices=object(),
                     lru_slots=object(), swap_stats=object())
    owner_rt = NS(index_group=owner_group, kv_dtype='uint8', row_width=656, region_stride=4096,
                  max_num_reqs=4, device='cuda', _swap_staged=False, hot=NS(cache=resident),
                  hot_backing=object(), hot_block_table=object(), _host_cache=target_cache,
                  registered_host_pool=object(), request_state_indices=object(),
                  resident_source_index=2, shared_host_region=None)
    reader_group = NS(followers=[], shared_topk=object(), copy_stream=object(), request_ids=object(),
                      lru_init=object(), device_global_indices=object(), lru_slots=object(), swap_stats=object())
    reader_rt = NS(index_group=reader_group, kv_dtype='uint8', row_width=656, region_stride=2304,
                   max_num_reqs=4, device='cuda', is_group_leader=True,
                   _swap_staged=False, max_union_rows=union)
    reader_group.leader = reader_rt
    owner = NS(runtime=owner_rt, draft_layer=False, view=NS(cache=resident), block_table=object(),
               residency=object(), slot_mapping=object(), mirror_slot_mapping=object(),
               source_block_table=object())
    reader = NS(runtime=reader_rt, view=None, submit_layer_mirror=None)
    return owner, reader, target_cache


class TestResidentReader(unittest.TestCase):
    def setUp(self):
        self.ns = runpy.run_path(str(SOURCE))

    def test_aliases_resident_slots_and_replacement_but_retains_reader_workspaces(self):
        owner, reader, source = pair()
        r, t = reader.runtime, owner.runtime
        private = (r.index_group.shared_topk, r.index_group.copy_stream, r.index_group.request_ids)
        bind = self.ns['bind_hisparse_resident_reader']
        bind(owner, reader, source)
        self.assertIs(reader.slot_mapping, owner.slot_mapping)
        self.assertIs(reader.mirror_slot_mapping, owner.mirror_slot_mapping)
        self.assertIs(reader.view, owner.view)
        self.assertIs(r._host_cache, t._host_cache)
        self.assertIs(r.hot, t.hot)
        self.assertIs(r.request_state_indices, t.request_state_indices)
        self.assertIs(r.index_group.lru_slots, t.index_group.lru_slots)
        self.assertEqual(r.region_stride, t.region_stride)
        self.assertEqual(private, (r.index_group.shared_topk, r.index_group.copy_stream,
                                   r.index_group.request_ids))
        self.assertTrue(reader.draft_layer)
        self.assertIsNone(reader.submit_layer_mirror)
        bind(owner, reader, source)

    def test_rejects_different_source_and_union_over_capacity_before_changing_reader(self):
        for union, address in ((4097, 3), (2176, 4)):
            with self.subTest(union=union, address=address):
                owner, reader, _ = pair(union)
                bind = self.ns['bind_hisparse_resident_reader']
                with self.assertRaises(self.ns['HiSparseBindingError']):
                    bind(owner, reader, tensor(address))
                self.assertIsNone(reader.view)
                self.assertFalse(hasattr(reader.runtime, '_host_cache'))

    def test_profile_cleanup_releases_views_after_core_teardown_and_can_rebind(self):
        owner, reader, source = pair()
        original_lru = reader.runtime.index_group.lru_slots
        self.ns['bind_hisparse_resident_reader'](owner, reader, source, profiling=True)
        for attr in ('_host_cache', 'registered_host_pool', 'hot_backing'):
            delattr(reader.runtime, attr)
        self.ns['release_hisparse_resident_reader'](reader)
        self.assertIsNone(reader.view)
        self.assertIsNone(reader.slot_mapping)
        self.assertIsNone(reader.runtime.request_state_indices)
        self.assertIs(reader.runtime.index_group.lru_slots, original_lru)
        self.assertFalse(reader.draft_layer)
        self.ns['bind_hisparse_resident_reader'](owner, reader, source)
        self.assertIs(reader.slot_mapping, owner.slot_mapping)

    def test_synthetic_capture_deduplicates_and_restores_target_callbacks(self):
        calls = []
        mirror = lambda: calls.append('mirrored')
        handle = NS(submit_layer_mirror=mirror)
        layer = NS(hisparse_cache=handle)
        with self.assertRaisesRegex(RuntimeError, 'capture failed'):
            with self.ns['suspend_hisparse_mirrors']([layer, layer, NS()]):
                self.assertIsNone(handle.submit_layer_mirror)
                raise RuntimeError('capture failed')
        self.assertIs(handle.submit_layer_mirror, mirror)
        handle.submit_layer_mirror()
        self.assertEqual(calls, ['mirrored'])

    def test_runner_accepts_only_explicit_fp8_hisparse_connector(self):
        runner = runpy.run_path(str(SOURCE.with_name('fullgraph_runner.py')))
        cfg = NS(
            use_v2_model_runner=True,
            model_config=NS(enforce_eager=False, enable_prompt_embeds=False,
                            hf_text_config=NS(model_type='hy_v4_flash'), multimodal_config=None),
            parallel_config=NS(tensor_parallel_size=4, data_parallel_size=1,
                               pipeline_parallel_size=1, prefill_context_parallel_size=1,
                               decode_context_parallel_size=1, enable_dbo=False,
                               enable_eplb=False, disable_custom_all_reduce=False),
            compilation_config=NS(mode=NS(name='VLLM_COMPILE'),
                                  cudagraph_mode=NS(name='FULL_AND_PIECEWISE')),
            speculative_config=NS(method='dfly', num_speculative_tokens=5),
            cache_config=NS(cache_dtype='fp8_ds_mla', enable_prefix_caching=False),
            lora_config=None, attention_config=NS(hisparse_config=object()),
            kv_transfer_config=NS(kv_connector='HCUHiSparseConnector',
                                  kv_connector_module_path='hcu_vllm_das.models.hy_v4_flash.hisparse_planner_compat'),
        )
        self.assertTrue(runner['is_qualified_config'](cfg, breakable=False))
        with patch.dict('os.environ', {'HCU_VLLM_HY4_HISPARSE': '0'}):
            self.assertFalse(runner['_admits_hisparse_graph'](cfg))
        with patch.dict('os.environ', {'HCU_VLLM_HY4_HISPARSE': '1'}):
            self.assertTrue(runner['_admits_hisparse_graph'](cfg))
            cfg.attention_config.hisparse_config = None
            self.assertFalse(runner['_admits_hisparse_graph'](cfg))
            cfg.attention_config.hisparse_config = object()
        cfg.parallel_config.tensor_parallel_size = 8
        self.assertFalse(runner['is_qualified_config'](cfg, breakable=False))
        cfg.parallel_config.tensor_parallel_size = 4
        cfg.kv_transfer_config.kv_connector_module_path = 'other'
        self.assertFalse(runner['is_qualified_config'](cfg, breakable=False))
