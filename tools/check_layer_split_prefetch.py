# SPDX-License-Identifier: Apache-2.0
"""2-GPU standalone check: dedicated broadcast comm + async prefetch.

torchrun --nproc-per-node=2 tools/check_layer_split_prefetch.py

Asserts:
- _init_layer_broadcast_comm creates the dedicated PyNccl comm.
- prefetch_dsa_cache_from_owner takes the async path (pending flag set).
- After finalize via broadcast_dsa_cache_from_owner, non-owner holds the
  owner's bytes exactly.
"""
import os

import torch
import torch.distributed as dist

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=world, rank=rank, local_rank=rank,
        distributed_init_method="env://",
    )
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            prefill_context_model_parallel_size=world,
        )

    import vllm_hcu.v1.dsa_cache_layer_split as ls

    state = ls.LayerSplitState(shard_rank=rank, shard_size=world, total_layers=4)
    ls._RUNTIME_STATE = state
    ls._RUNTIME_STATE_INSTALLED = True
    ls._init_layer_broadcast_comm(state)
    assert ls._BCAST_COMM is not None, "dedicated broadcast comm not created"
    assert ls._BCAST_STREAM is not None

    # Layers 0-1 owned by rank0, 2-3 by rank1. Simulate the packed scratch:
    # one tensor per layer; owner fills a signature, non-owner zeros.
    caches = {}
    for layer in range(4):
        t = torch.zeros(64, 8, dtype=torch.float16, device="cuda")
        if state.is_owned(layer):
            t.fill_(float(layer + 1))
        caches[layer] = t

    # Prefetch layer 2 (owned by rank1) — async path.
    name = "model.layers.2.self_attn.attn"
    ls.prefetch_dsa_cache_from_owner(caches[2], name)
    fam = ls._family_state(caches[2], name)
    assert fam.pending_remote_broadcast, "prefetch did not take the async path"
    assert fam.pending_remote_layer_id == 2

    # Reader path finalizes the pending broadcast.
    ls.broadcast_dsa_cache_from_owner(caches[2], name)
    assert fam.remote_layer_id == 2
    assert not fam.pending_remote_broadcast
    torch.cuda.synchronize()
    expected = torch.full_like(caches[2], 3.0)
    assert torch.equal(caches[2], expected), (
        f"rank{rank}: layer2 bytes wrong after async prefetch+finalize"
    )

    # Dedup: second broadcast for same layer must not re-enter NCCL.
    ls.broadcast_dsa_cache_from_owner(caches[2], name)

    # Sync broadcast of layer 0 (owned by rank0) still works alongside.
    name0 = "model.layers.0.self_attn.attn"
    ls.broadcast_dsa_cache_from_owner(caches[0], name0)
    torch.cuda.synchronize()
    assert torch.equal(caches[0], torch.full_like(caches[0], 1.0))

    dist.barrier()
    if rank == 0:
        print("LAYER_SPLIT_ASYNC_PREFETCH_OK")


if __name__ == "__main__":
    main()
