# SPDX-License-Identifier: Apache-2.0
"""Bind speculative staging bounds to Core's actual per-step PCP ownership."""

from hcu_vllm_das.adapter_api import adapter_module as _adapter_module
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from hcu_vllm_das.fp4_indexer_lightop import decode_rows
from vllm.v1.worker.gpu.pcp_manager import PCPManager


@pytest.mark.parametrize('requests', [1, 7, 8, 9, 128])
def test_capacity_covers_native_decode_owners(requests):
    manager = PCPManager.__new__(PCPManager)
    manager.pcp_world_size = 8
    manager.dcp_world_size = 1
    manager.shard_decode_requests = True
    scheduled = np.full(requests, 6, dtype=np.int64)
    is_prefilling = np.zeros(requests, dtype=np.bool_)
    actual = [sum(size for _, _, size in manager._iter_rank_chunks(rank, scheduled, is_prefilling))
              for rank in range(8)]
    cfg = NS(scheduler_config=NS(max_num_seqs=requests),
             parallel_config=NS(pcp_shard_decode_requests=True, prefill_context_parallel_size=8),
             speculative_config=NS(num_speculative_tokens=5))
    assert max(actual) == decode_rows(cfg)


def test_unsharded_and_nonspeculative_capacities_preserved():
    cfg = NS(scheduler_config=NS(max_num_seqs=128),
             parallel_config=NS(pcp_shard_decode_requests=False, prefill_context_parallel_size=1),
             speculative_config=NS(num_speculative_tokens=5))
    assert decode_rows(cfg) == 768
    cfg.parallel_config.pcp_shard_decode_requests = True
    cfg.parallel_config.prefill_context_parallel_size = 8
    cfg.speculative_config = None
    assert decode_rows(cfg) == 128


@pytest.mark.parametrize('sharded,speculative,expected',
                         [(True, 5, 96), (False, 5, 768), (True, None, 128)])
def test_real_indexer_constructor_freezes_the_same_workspace_and_staging_bound(
        monkeypatch, sharded, speculative, expected):
    dsv41_channel_fp8 = _adapter_module('dsv41_channel_fp8')
    from hcu_vllm_das import v41_indexer

    cfg = NS(scheduler_config=NS(max_num_seqs=128, max_num_batched_tokens=32768),
             parallel_config=NS(pcp_shard_decode_requests=sharded,
                                prefill_context_parallel_size=8),
             speculative_config=NS(num_speculative_tokens=speculative) if speculative else None,
             cache_config=NS(block_size=64), model_config=NS(hf_config=NS()))

    def base_init(self):
        torch.nn.Module.__init__(self)
        self._hcu_fp4 = True
        self.max_model_len = 65536
        self.compress_ratio = 2

    monkeypatch.setattr(dsv41_channel_fp8, '_is_v41', lambda cfg: False)
    monkeypatch.setattr(v41_indexer, 'current_config', lambda: cfg)
    monkeypatch.setattr(v41_indexer._api['base'], '__init__', base_init)
    layer = v41_indexer.HcuV41SparseAttnIndexer()
    assert layer.fp4_decode_rows == expected
    assert layer.fp4_staging_plan['requests'] == expected
