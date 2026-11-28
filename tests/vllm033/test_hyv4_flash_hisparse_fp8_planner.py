# SPDX-License-Identifier: Apache-2.0
"""CPU-only geometry of the 64K HY4 Flash FP8-DS HiSparse plan.

Run directly with Python in a Core033/Plugin environment. Set
HYV4_HISPARSE_MODEL_CONFIG to also read the checkpoint's config.json. The
fallback fixture captures the same cache geometry, with no weights required.
"""

from importlib.util import find_spec
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def model_geometry():
    config_path = os.environ.get("HYV4_HISPARSE_MODEL_CONFIG")
    if config_path:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        return SimpleNamespace(**raw["text_config"])
    sparse_layers = {0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41}
    return SimpleNamespace(
        layer_types=["deepseek_sparse_attention" if i in sparse_layers else
                     "sliding_attention" for i in range(42)],
        kv_lora_rank=512, qk_rope_head_dim=64, swa_qk_rope_head_dim=64,
        sliding_window=512, index_topk=2048, compressed_indexer_compress_ratio=4,
    )


@unittest.skipUnless(find_spec("vllm") and find_spec("hcu_vllm_das"),
                     "requires installed Core033 and Plugin")
class TestHYV4FlashHiSparseFP8Planner(unittest.TestCase):
    def test_64k_dfly_fp8_plan_preserves_cache_geometry(self):
        import torch
        from vllm.config import AttentionConfig, CacheConfig, KVTransferConfig, VllmConfig
        from vllm.config import set_current_vllm_config
        from vllm.config.attention import HiSparseConfig
        from vllm.distributed.kv_transfer.kv_connector.v1.hisparse import connector
        from vllm.model_executor.layers.attention import MLAAttention
        from vllm.v1.core import kv_cache_config_builder, kv_cache_planning
        from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
        from vllm.v1.hisparse import layout
        from vllm.v1.kv_cache_interface import (
            CircularBufferSpec, HiSparseHotSpec, HiSparseResidentSpec,
            KVCacheGroupRole, SlidingWindowMLASpec, UniformTypeKVCacheSpecs,
        )
        from vllm.v1.worker.utils import customize_attention_spec
        from hcu_vllm_das.models.hy_v4_flash import hisparse_planner_compat as planner
        from hcu_vllm_das.models.hy_v4_flash.indexpool_cache import IndexPoolCache
        from hcu_vllm_das.models.hy_v4_flash.swa import FlashSparseBackend, FlashSWABackend

        hf_config = model_geometry()
        class PlannerConfig(SimpleNamespace):
            max_concurrent_batches = VllmConfig.max_concurrent_batches
            max_in_flight_tokens = VllmConfig.max_in_flight_tokens

        config = PlannerConfig(
            attention_config=AttentionConfig(hisparse_config=HiSparseConfig()),
            cache_config=CacheConfig(block_size=64, cache_dtype="fp8_ds_mla",
                                     enable_prefix_caching=False),
            compilation_config=SimpleNamespace(static_forward_context={}),
            model_config=SimpleNamespace(
                hf_text_config=hf_config, dtype=torch.bfloat16, max_model_len=65792,
                original_max_model_len=65792, supports_kv_cache_grouping=False,
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=4, pipeline_parallel_size=1,
                prefill_context_parallel_size=1, decode_context_parallel_size=1,
                world_size=4, distributed_executor_backend="mp", nnodes_within_dp=1,
            ),
            scheduler_config=SimpleNamespace(
                disable_hybrid_kv_cache_manager=False, async_scheduling=True,
                max_num_batched_tokens=8256, max_num_seqs=4,
            ),
            speculative_config=SimpleNamespace(
                method="dfly", num_speculative_tokens=5, parallel_drafting=False,
                use_eagle_block_drop=lambda: False,
            ),
            num_speculative_tokens=5, num_prefill_lookahead_tokens=0,
            use_v2_model_runner=True,
            kv_transfer_config=KVTransferConfig(
                kv_connector="HCUHiSparseConnector",
                kv_connector_module_path=planner.__name__, kv_role="kv_both",
                kv_buffer_device="cuda", kv_connector_extra_config={"host_pool_gib": 2},
            ),
        )
        connector_cls, builder_cls = planner._make_types(kv_cache_planning, layout, connector)
        config.cache_config.kv_cache_layout = connector_cls.get_required_kvcache_layout(config)
        self.assertEqual(config.cache_config.kv_cache_layout, "BLHNC")

        layers = {}
        source_names, window_names = set(), set()
        with set_current_vllm_config(config):
            for layer_id, layer_type in enumerate(hf_config.layer_types):
                prefix = f"model.layers.{layer_id}.self_attn"
                name = f"{prefix}.attn"
                sliding = layer_type == "sliding_attention"
                # Real cache-spec producer with just its constructor's geometry fields.
                attn = MLAAttention.__new__(MLAAttention)
                torch.nn.Module.__init__(attn)
                attn.kv_cache_dtype = config.cache_config.cache_dtype
                attn.head_size = hf_config.kv_lora_rank + (hf_config.swa_qk_rope_head_dim
                    if sliding else hf_config.qk_rope_head_dim)
                attn.attn_backend = FlashSWABackend if sliding else FlashSparseBackend
                attn.sliding_window = hf_config.sliding_window if sliding else None
                attn.topk_buffer_width = hf_config.index_topk
                attn.indexer = None if sliding else object()
                attn.non_causal_multi_token_decode = False
                layers[name] = attn
                if sliding:
                    window_names.add(name)
                else:
                    source_names.add(name)
                    for cache_name, tail in (("k_cache", False), ("tail_cache", True)):
                        index_name = f"{prefix}.indexer.{cache_name}"
                        layers[index_name] = IndexPoolCache(
                            index_name, config.cache_config,
                            hf_config.compressed_indexer_compress_ratio, tail=tail,
                        )

        self.assertEqual((len(source_names), len(window_names)), (12, 30))
        specs = {
            name: customize_attention_spec(layer.get_attn_backend(), layer.get_kv_cache_spec(config))
            for name, layer in layers.items()
        }
        with patch.dict(os.environ, {"HCU_VLLM_HY4_HISPARSE": "0"}):
            worker_specs = planner._resolve_indexpool_specs(config, specs, layers)
        # Exercise the external connector's builder and Core's cross-worker planner.
        # Config admission and process-wide installation are outside this CPU test.
        builder = builder_cls()
        worker_budgets = [12 << 30] * 4
        worker_maps = [dict(worker_specs) for _ in worker_budgets]
        with (patch.dict(os.environ, {"HCU_VLLM_HY4_HISPARSE": "0"}),
              patch.object(planner, "_validate_owned", return_value=None),
              patch.object(kv_cache_config_builder, "get_kv_cache_config_builder",
                           return_value=builder)):
            groups = kv_cache_planning.get_global_kv_cache_groups(config, worker_maps)
            for max_num_seqs in (4, 8, 16, 32):
                config.scheduler_config.max_num_seqs = max_num_seqs
                plans = kv_cache_planning.get_kv_cache_configs(config, worker_maps, worker_budgets, groups)
                self.assertEqual([p.num_blocks for p in plans], [9300] * 4, f"{max_num_seqs=}")
            config.scheduler_config.max_num_batched_tokens = 16384
            with self.assertRaisesRegex(ValueError, "larger than the available KV cache"):
                kv_cache_planning.get_kv_cache_configs(config, worker_maps, worker_budgets, groups)
            config.scheduler_config.async_scheduling = False
            synchronous = kv_cache_planning.get_kv_cache_configs(config, worker_maps, worker_budgets, groups)
            self.assertEqual([p.num_blocks for p in synchronous], [9300] * 4)
            config.scheduler_config.async_scheduling = True
            config.scheduler_config.max_num_batched_tokens = 8256

        plan = plans[0]
        required = kv_cache_planning._max_memory_usage_bytes_from_groups(builder, config, groups)
        self.assertEqual(required, 9136 * 1385472)
        self.assertEqual(resolve_kv_cache_block_sizes(plan, config), (256, 64))
        (host,) = [g for g in plan.kv_cache_groups if g.host_resident]
        self.assertEqual(host.role, KVCacheGroupRole.HISPARSE_SOURCE)
        self.assertTrue(host.enable_kv_transfer)
        self.assertEqual(set(host.layer_names), source_names)
        self.assertEqual(host.kv_cache_spec.block_size, 64)
        self.assertEqual(plan.hisparse_host_block_stride, 503808)
        self.assertEqual(plan.hisparse_host_num_blocks, 4262)

        for name in source_names:
            source = host.kv_cache_spec.kv_cache_specs[name]
            self.assertEqual((source.state_content_size_bytes, source.page_size_bytes), (656, 41984))
            for suffix, cls in ((layout.HISPARSE_RESIDENT_SUFFIX, HiSparseResidentSpec),
                                (layout.HISPARSE_HOT_SUFFIX, HiSparseHotSpec)):
                layer_name = name + suffix
                (group,) = [g for g in plan.kv_cache_groups if layer_name in g.layer_names]
                self.assertFalse(group.host_resident)
                self.assertFalse(group.enable_kv_transfer)
                spec = group.kv_cache_spec
                self.assertIsInstance(spec, cls)
                self.assertEqual((spec.block_size, spec.page_size_bytes), (64, 41984))
                if isinstance(spec, HiSparseHotSpec):
                    self.assertEqual(spec.blocks_per_request, 224)
                (tensor,) = [t for t in plan.kv_cache_tensors if layer_name in t.layers]
                self.assertFalse(tensor.host_resident)
                self.assertEqual((tensor.layer_stride, tensor.block_stride), (41984, 1385472))

        host_tensors = [t for t in plan.kv_cache_tensors if t.host_resident]
        self.assertEqual({name for t in host_tensors for name in t.layers}, source_names)
        self.assertTrue(all(t.block_stride == 41984 for t in host_tensors))

        indexer_count, tail_count = 0, 0
        for group in plan.kv_cache_groups:
            spec = group.kv_cache_spec
            if group.role is KVCacheGroupRole.HISPARSE_INDEXER:
                indexer_count += 1
                (name,) = group.layer_names
                self.assertEqual((spec.block_size, spec.tokens_per_state, spec.page_size_bytes),
                                 (256, 4, 8448))
                self.assertEqual(spec.get_block_stride_alignment(), 8448)
                self.assertEqual(spec.page_size_bytes, specs[name].page_size_bytes)
                self.assertEqual(spec.dtype, torch.uint8)
            for name in group.layer_names:
                if name.endswith(".tail_cache"):
                    tail_count += 1
                    tail = spec.kv_cache_specs[name] if isinstance(spec, UniformTypeKVCacheSpecs) else spec
                    self.assertIsInstance(tail, CircularBufferSpec)
                    self.assertEqual(tail, specs[name])
                    self.assertEqual((tail.block_size, tail.page_size_bytes), (16, 8384))
        self.assertEqual((indexer_count, tail_count), (12, 12))

        (window,) = [g for g in plan.kv_cache_groups if window_names.intersection(g.layer_names)]
        self.assertEqual(set(window.layer_names), window_names)
        self.assertFalse(window.host_resident)
        self.assertEqual(window.role, KVCacheGroupRole.DEFAULT)
        self.assertEqual(window.kv_cache_spec.max_memory_usage_bytes(config)
                         // window.kv_cache_spec.page_size_bytes, 267)
        for name in window.layer_names:
            spec = (window.kv_cache_spec.kv_cache_specs[name]
                    if isinstance(window.kv_cache_spec, UniformTypeKVCacheSpecs) else window.kv_cache_spec)
            self.assertIsInstance(spec, SlidingWindowMLASpec)
            self.assertEqual(spec, specs[name])
            self.assertIs(layers[name].get_attn_backend(), FlashSWABackend)
            self.assertEqual((spec.block_size, spec.sliding_window, spec.page_size_bytes),
                             (64, 512, 41984))
        (window_tensor,) = [t for t in plan.kv_cache_tensors if window_names.intersection(t.layers)]
        self.assertEqual(set(window_tensor.layers), window_names)
        self.assertEqual((window_tensor.layer_stride, window_tensor.block_stride), (41984, 1385472))
        self._assert_four_context_plan(config, worker_specs, plan)

    def _assert_four_context_plan(self, config, worker_specs, legacy):
        from copy import deepcopy
        from dataclasses import replace
        import torch
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.kv_cache_utils import (
            generate_scheduler_kv_cache_config, get_kv_cache_capacity, resolve_kv_cache_block_sizes,
        )
        from vllm.v1.hisparse.coordinator import get_hisparse_coordinator
        from vllm.v1.hisparse.layout import get_hisparse_kv_cache_config
        from vllm.v1.hisparse.runtime import PagedCacheView
        from vllm.v1.kv_cache_interface import HiSparseHotSpec, HiSparseResidentSpec, KVCacheGroupRole
        from hcu_vllm_das.models.hy_v4_flash import hisparse_planner_compat as planner
        from hcu_vllm_das.models.hy_v4_flash.dfly_compat import _add_dfly_sharing

        with patch.dict(os.environ, {"HCU_VLLM_HY4_HISPARSE": "1"}):
            groups = planner._indexpool_groups(config, worker_specs)
            (indexer,) = [g for g in groups if g.role is KVCacheGroupRole.HISPARSE_INDEXER]
            self.assertEqual(len(indexer.layer_names), 12)
            self.assertEqual(indexer.kv_cache_spec.block_size, 256)
            self.assertEqual(indexer.kv_cache_spec.kv_cache_specs,
                             {name: worker_specs[name] for name in indexer.layer_names})
            name = indexer.layer_names[0]
            for block_size, ratio in ((64, 1), (512, 8)):
                mixed = dict(worker_specs)
                mixed[name] = replace(worker_specs[name], block_size=block_size, tokens_per_state=ratio)
                mixed_groups = planner._indexpool_groups(config, mixed)
                self.assertEqual([g.kv_cache_spec.block_size for g in mixed_groups
                                  if g.role is KVCacheGroupRole.HISPARSE_INDEXER],
                                 [block_size] + [256] * 11)
            mixed = dict(worker_specs)
            source_name = name.removesuffix(".indexer.k_cache") + ".attn"
            mixed[source_name] = replace(worker_specs[source_name], state_content_bytes=672)
            mixed_groups = planner._indexpool_groups(config, mixed)
            self.assertEqual([g.kv_cache_spec.page_size_bytes for g in mixed_groups
                              if isinstance(g.kv_cache_spec, HiSparseHotSpec)], [43008] + [41984] * 11)

        resident = [g for g in groups if isinstance(g.kv_cache_spec, HiSparseResidentSpec)]
        hot = [g for g in groups if isinstance(g.kv_cache_spec, HiSparseHotSpec)]
        self.assertEqual([len(g.layer_names) for g in resident], [4, 4, 4])
        self.assertEqual([len(g.layer_names) for g in hot], [4, 4, 4])
        self.assertEqual([g.kv_cache_spec.blocks_per_request for g in hot], [224] * 3)
        self.assertEqual([[n.removesuffix(".hisparse_resident") for n in g.layer_names] for g in resident],
                         [[n.removesuffix(".hisparse_hot") for n in g.layer_names] for g in hot])

        plan = get_hisparse_kv_cache_config(config, groups, 12 << 30, 2 << 30)
        self.assertEqual((plan.num_blocks, plan.hisparse_host_num_blocks),
                         (legacy.num_blocks, legacy.hisparse_host_num_blocks))
        self.assertEqual(resolve_kv_cache_block_sizes(plan, config), (256, 64))
        self.assertEqual(get_kv_cache_capacity(config, plan)[0], 272768)
        schedule = generate_scheduler_kv_cache_config([plan])
        manager = KVCacheManager(schedule, 65792, 256, 64, max_in_flight_tokens=16512,
                                 enable_caching=False)
        coordinator = get_hisparse_coordinator(manager)
        self.assertEqual((len(coordinator.resident_managers), len(coordinator.hot_managers)), (3, 3))
        self.assertEqual(coordinator.transition_watermark, 930)

        # Aliases are added after planning, only to their target source group.
        bound_groups = deepcopy(plan.kv_cache_groups)
        (bound_source,) = [g for g in bound_groups if g.host_resident]
        aliases = {f"dfly.layers.{i}.attn": name for i, name in enumerate(bound_source.layer_names[:4])}
        _add_dfly_sharing(aliases, bound_groups)
        for alias, target in aliases.items():
            self.assertEqual([g for g in bound_groups if alias in g.layer_names], [bound_source])
            self.assertEqual(bound_source.kv_cache_spec.kv_cache_specs[alias],
                             bound_source.kv_cache_spec.kv_cache_specs[target])

        # Two CPU blocks suffice to check Core's physical view identity.
        raw = torch.empty(2 * 1385472, dtype=torch.uint8)
        views = {
            name: PagedCacheView.bind(raw, dtype=torch.uint8, row_width=656,
                byte_offset=tensor.offset + i * tensor.layer_stride,
                block_stride=tensor.block_stride, num_blocks=2, block_size=64).cache
            for tensor in plan.kv_cache_tensors for i, name in enumerate(tensor.layers)
            if name.endswith((".hisparse_resident", ".hisparse_hot"))
        }
        for group in resident:
            pointers = []
            for name in group.layer_names:
                resident_view = views[name]
                hot_view = views[name.removesuffix(".hisparse_resident") + ".hisparse_hot"]
                self.assertEqual(resident_view.data_ptr(), hot_view.data_ptr())
                self.assertEqual(resident_view.stride(), hot_view.stride())
                self.assertEqual(hot_view.untyped_storage().data_ptr(), raw.data_ptr())
                pointers.append(resident_view.data_ptr())
            self.assertEqual(pointers, [raw.data_ptr() + i * 41984 for i in range(4)])


if __name__ == "__main__":
    unittest.main()
