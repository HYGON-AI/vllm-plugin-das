"""CPU-only EP16 regression for HY4 MegaMoE collective scheduling."""
from datetime import timedelta
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous):
    impl = runpy.run_path(str(Path(__file__).resolve().parents[2]
                              / "vllm_hcu/models/hy_v4/megamoe.py"))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=16, timeout=timedelta(seconds=60))
    try:
        calls = []

        def run(provider, hidden, ids, weights, packed, buffer, clamp):
            assert len(hidden) == len(ids) == len(weights) <= buffer.num_max_tokens_per_rank
            calls.append(len(hidden))
            return hidden.clone()

        group = SimpleNamespace(cpu_group=dist.group.WORLD)
        kernels = [impl["Hy4MegaKernel"](None, None, SimpleNamespace(num_max_tokens_per_rank=2),
                                         group, 10) for _ in range(77)]
        with patch.dict(kernels[0].apply.__globals__, run_hy4_mega_experts=run), \
                patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reduce:
            for step, (rows, rounds) in enumerate((
                (0, 1), (2, 1), (5 if rank == 15 else 0, 3), (rank % 4, 2), (1, 1)
            ), start=1):
                hidden = torch.arange(rows * 8).reshape(rows, 8)
                ids = torch.zeros_like(hidden)
                calls.clear()
                with impl["mega_forward_batch"](rows):
                    for kernel in kernels:
                        out = kernel.apply(hidden, None, None, ids.float(), ids, None,
                                           272, None, False)
                        assert torch.equal(out, hidden)
                assert reduce.call_count == step
                assert calls == [min(2, max(rows - 2*i, 0)) for i in range(rounds)] * 77
    finally:
        dist.destroy_process_group()


def test_ep16_rounds_once_per_backbone_forward(tmp_path):
    mp.start_processes(_worker, args=((tmp_path / "gloo").as_uri(),),
                       nprocs=16, join=True, start_method="fork")
