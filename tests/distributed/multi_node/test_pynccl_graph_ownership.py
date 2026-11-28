# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""TP16 eager/graph RCCL regression, run with torchrun on two eight-GPU nodes.

The default exercises the actual HCU communicator helper. ``--unpatched``
replays the same workload without the helper for a controlled reproduction.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytestmark = [pytest.mark.multi_node, pytest.mark.node_count(2)]


def _worker_main(*, unpatched: bool = False, capture_gather: bool = False) -> None:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm_hcu.distributed.device_communicators.pynccl_graph import (
        close_graph_ownership,
        install_graph_ownership,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 16, "This regression requires two nodes with eight ranks each"
    torch.cuda.set_device(local_rank)
    timeout = datetime.timedelta(seconds=120)
    dist.init_process_group("gloo", timeout=timeout)
    gpu_group = dist.new_group(backend="nccl", timeout=timeout)
    comm = PyNcclCommunicator(dist.group.WORLD, device=local_rank)
    assert comm.available and not comm.disabled
    owner = SimpleNamespace(pynccl_comm=comm)

    main_stream = torch.cuda.current_stream()
    copy_stream = torch.cuda.Stream()
    sizes = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
    inputs = {n: torch.ones(n * 5120, device="cuda", dtype=torch.bfloat16) for n in sizes}
    outputs = {n: torch.zeros_like(value) for n, value in inputs.items()}
    logits = {
        n: torch.full((n * 9472,), rank + 1, device="cuda", dtype=torch.bfloat16)
        for n in (1, 4)
    }
    gathered = {
        n: torch.empty(world_size * n * 9472, device="cuda", dtype=torch.bfloat16)
        for n in (1, 4)
    }
    draft_gathered = torch.empty_like(gathered[1])
    host = torch.empty(world_size * 4 * 9472, dtype=torch.bfloat16, pin_memory=True)
    graphs = {}

    def all_reduce(n):
        for _ in range(185):
            comm.all_reduce(inputs[n], outputs[n])

    def all_gather(n, *, draft=False):
        output = draft_gathered if draft else gathered[n]
        output.copy_(owner.all_gather(logits[n], dim=0))

    def gather_flat(input_, dim=-1):
        output = torch.empty(world_size * input_.numel(), device="cuda", dtype=input_.dtype)
        dist.all_gather_into_tensor(output, input_.contiguous(), group=gpu_group)
        return output

    owner.world_size = world_size
    owner.all_gather = gather_flat
    if not unpatched:
        install_graph_ownership(owner)

    def synchronize():
        torch.cuda.synchronize()
        dist.barrier()

    try:
        for n in sizes:
            warmup = torch.cuda.Stream()
            warmup.wait_stream(main_stream)
            with torch.cuda.stream(warmup):
                dist.all_reduce(outputs[n], group=gpu_group)
                if n in logits:
                    all_gather(n)
            main_stream.wait_stream(warmup)
            synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                all_reduce(n)
            graphs[n] = graph
            synchronize()

        draft_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(draft_graph):
            all_reduce(1)
            if capture_gather:
                all_gather(1, draft=True)
        synchronize()

        for iteration in range(200):
            n = 1 if iteration % 4 == 0 else 4
            for batch_size in (1, 4):
                inputs[batch_size].fill_(rank + 1 + iteration % 7)
                logits[batch_size].fill_(rank + 1 + iteration % 7)
            if iteration % 17 == 0:
                all_reduce(n)
            else:
                graphs[n].replay()
            all_gather(n)
            copied = torch.cuda.Event(blocking=True)
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_stream(main_stream)
                host[:gathered[n].numel()].copy_(gathered[n], non_blocking=True)
                copied.record(copy_stream)
            for _ in range(3):
                draft_graph.replay()
                if not capture_gather:
                    all_gather(1, draft=True)
            copied.synchronize()
            expected_logits = (
                torch.arange(1, world_size + 1, dtype=torch.bfloat16)
                + iteration % 7
            ).repeat_interleave(n * 9472)
            torch.testing.assert_close(
                host[:gathered[n].numel()], expected_logits, rtol=0, atol=0
            )
            if iteration % 10 == 0:
                print(json.dumps(dict(rank=rank, iteration=iteration)), flush=True)
        synchronize()
        torch.testing.assert_close(
            draft_gathered.cpu(),
            (torch.arange(1, world_size + 1, dtype=torch.bfloat16) + 199 % 7)
            .repeat_interleave(9472),
            rtol=0, atol=0,
        )
        expected = world_size * (world_size + 1) // 2 + world_size * (199 % 7)
        for n in (1, 4):
            torch.testing.assert_close(outputs[n], torch.full_like(outputs[n], expected))
        print(json.dumps(dict(rank=rank, result="PASS", iterations=200)), flush=True)
    finally:
        graphs.clear()
        if "draft_graph" in locals():
            del draft_graph
        close_graph_ownership(owner)
        comm.destroy()
        dist.destroy_process_group()


@pytest.mark.skipif("RANK" not in os.environ, reason="Requires a two-node torchrun job")
def test_pynccl_graph_ownership():
    _worker_main()


if __name__ == "__main__":
    _worker_main(
        unpatched="--unpatched" in sys.argv,
        capture_gather="--capture-gather" in sys.argv,
    )
