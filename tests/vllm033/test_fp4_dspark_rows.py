# SPDX-License-Identifier: Apache-2.0
"""Bind speculative staging bounds to Core's actual per-step PCP ownership."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

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
