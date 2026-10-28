# SPDX-License-Identifier: Apache-2.0
"""CPU byte-copy replay of the existing Mooncake plan, without format negotiation.

Uses the real allocator and connector planner; CPU memmove stands in for the
TransferEngine. This does not execute the GPU writer, reader or native transfer.
"""
import asyncio
import ctypes
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.mark.parametrize("ratio", [1, 2])
def test_fp4_pages_survive_existing_mooncake_copy_plan(ratio):
    from vllm.models.deepseek_v41.attention import DeepseekV4IndexerCache
    from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
    from vllm.v1.kv_cache_interface import (
        KVCacheGroupSpec, KVCacheLayout, MLAAttentionSpec, UniformTypeKVCacheSpecs,
    )
    from vllm.v1.worker.utils import allocate_kv_cache
    from vllm_hcu.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector as mc
    from vllm_hcu.model_executor.layers.fp4_indexer import fp4_flat_page_view

    main_name = "model.layers.0.self_attn"
    index_name = main_name + ".indexer.k_cache"
    layout = KVCacheLayout.BLNHC
    config = NS(
        cache_config=NS(block_size=64, cache_dtype="fp8_ds_mla",
                        num_gpu_blocks_override=3, prefix_cache_retention_interval=None,
                        get_resolved_kv_cache_layout=lambda: layout),
        attention_config=NS(indexer_sparse_logits=False, hisparse_config=None),
    )
    owner = object.__new__(DeepseekV4IndexerCache)
    owner.cache_config = config.cache_config
    owner.dtype, owner.head_dim = torch.uint8, 68
    owner.compress_ratio, owner.sparse_logits = ratio, False
    index_spec = owner.get_kv_cache_spec(config)
    main_spec = MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=512, dtype=torch.uint8,
        tokens_per_state=ratio, cache_dtype_str="fp8_ds_mla", alignment=576,
        model_version="deepseek_v4", state_content_bytes=584,
    )
    specs = {main_name: main_spec, index_name: index_spec}
    group = KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs(
        block_size=64, kv_cache_specs=specs,
    ))
    cfg = get_kv_cache_config_from_groups(config, [group], 10_000_000)

    class Engine:
        def batch_register_memory(self, pointers, sizes):
            return 0

    def worker(tp_size):
        caches = allocate_kv_cache(cfg, torch.device("cpu"), layout,
                                   kernel_block_sizes=[64])
        value = object.__new__(mc.MooncakeConnectorWorker)
        value.shutdown = lambda: None
        value.use_mla, value.is_kv_consumer = True, True
        value.transfer_topo = NS(is_mamba=False, local_replicates_kv_cache=True)
        value.engine = Engine()
        value._layer_specs, value._layer_group_indices = specs, dict.fromkeys(specs, 0)
        value.kv_cache_config = cfg
        value._physical_blocks_per_logical_kv_block = 1
        value.tp_rank, value.tp_size = 0, tp_size
        value.register_kv_caches(caches)
        return value, caches

    producer, source = worker(8)
    consumer, destination = worker(1)
    rows = 64 // ratio
    assert index_spec.page_size_bytes == ((rows * 68 + 575) // 576) * 576
    assert main_spec.page_size_bytes == ((rows * 584 + 575) // 576) * 576
    assert all(size % 576 == 0 for size in producer.kv_block_len_per_layer)
    for name in specs:
        source[name].fill_(0x51)
        destination[name].zero_()
    source_index = fp4_flat_page_view(source[index_name].squeeze(1))
    source_index[0, :rows * 64] = 0x12
    source_index[0, rows * 64:] = 0x7F
    assert source[index_name].stride(-2) == 68
    metadata = mc.MooncakeXferMetadata(
        remote_hostname="cpu", remote_port=1, remote_tp_size=1, remote_tp_rank=0,
        req_blocks={"d": ("x", [[2]])},
        kv_caches_base_addr=consumer.kv_caches_base_addr,
        block_lens=consumer.block_len_per_layer,
        kv_block_lens=consumer.kv_block_len_per_layer,
        registered_layer_names=consumer.registered_layer_names,
        registered_layer_indices=consumer.registered_layer_indices,
        registered_group_indices=consumer.registered_group_indices,
    )
    def regions(value):
        return value._get_transfer_regions(
            value.kv_caches_base_addr, value.block_len_per_layer,
            value.kv_block_len_per_layer, value.registered_layer_names,
            value.registered_layer_indices, value.registered_group_indices,
        )
    send = mc.SendBlockMeta("p", "x", [[0]], asyncio.Event())
    src, dst, lengths, errors, message = asyncio.run(producer._build_transfer_params(
        [("d", send)], metadata, regions(producer), regions(consumer),
    ))
    assert not errors and message is None
    assert lengths == producer.kv_block_len_per_layer
    # The planner emits one descriptor per registered region. Copying those
    # spans models Mooncake's raw byte transfer without checking a format tag.
    for start, end, size in zip(src, dst, lengths):
        ctypes.memmove(end, start, size)
    for name, source_cache in source.items():
        dest_cache = destination[name]
        assert source_cache[0].reshape(-1).tolist() == dest_cache[2].reshape(-1).tolist()
        assert not dest_cache[0].any() and not dest_cache[1].any()
    copied = fp4_flat_page_view(destination[index_name].squeeze(1))
    assert torch.equal(copied[2, :rows * 64], source_index[0, :rows * 64])
    assert torch.equal(copied[2, rows * 64:], source_index[0, rows * 64:])
