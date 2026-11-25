# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun on four or eight idle gfx938 cards."""
from datetime import timedelta
import json
import os

import torch
import torch.distributed as dist

from hcu_vllm_das.moe.ct_w4a8_prepare import (
    FixedHTExpertTokens,
    HcuCTHTPrepareAndFinalize,
    fixed_ht_batch_capacity,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)


def combine(prepare, capacity, x, ids, weights, num_experts, expert_map):
    config = FusedMoEQuantConfig.make(torch.int8, per_act_token_quant=True, weight_dtype="int4")
    with fixed_ht_batch_capacity(capacity):
        q, scale, counts, received_ids, received_weights = prepare.prepare_async(
            x, weights, ids, num_experts, expert_map, False, config,
        )()
    live = received_ids >= 0
    if capacity is None:
        live = expert_map[received_ids] >= 0
    else:
        assert isinstance(counts, FixedHTExpertTokens)
        assert q.size(0) == capacity and counts.expert_num_tokens_cpu is None
        assert not q[~live.any(dim=1)].any()
    # Make each expert return the quantized input, weighted per live route.
    # Use the native combine handle to return those rows to their senders.
    reduced = (q.float() * scale * received_weights.masked_fill(~live, 0).sum(
        dim=1, keepdim=True
    )).bfloat16()
    output = torch.empty_like(x)
    prepare.finalize(output, reduced, weights, ids, False, TopKWeightAndReduceNoOP())
    return output, counts.expert_num_tokens


def check_case(prepare, capacity, x, ids, weights, num_experts, expert_map):
    expected, expected_counts = combine(prepare, None, x, ids, weights,
                                        num_experts, expert_map)
    actual, actual_counts = combine(prepare, capacity, x, ids, weights,
                                    num_experts, expert_map)
    torch.testing.assert_close(actual_counts, expected_counts, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def check_clamp_quant(experts):
    from hcu_vllm_das.moe.ct_w4a8_runtime import (
        _accepts_device_count,
        _aligned_expert_row_count,
        contiguous_aligned_rows,
    )
    from lightop import fuse_silu_mul_clamp_quant

    assert _accepts_device_count(fuse_silu_mul_clamp_quant)
    capacity = contiguous_aligned_rows(32768, 6, experts)
    x = torch.empty((capacity, 4608), device="cuda", dtype=torch.bfloat16)
    x.normal_(std=7)
    ref_q, ref_s = fuse_silu_mul_clamp_quant(x, limit=10)
    q, s = torch.empty_like(ref_q), torch.empty_like(ref_s)

    def launch(counts):
        return fuse_silu_mul_clamp_quant(
            x, limit=10, output=q, scales=s,
            num_local_tokens_tensor=_aligned_expert_row_count(counts, 256),
        )

    for counts, rows in (
        ([0] * experts, 0),
        ([0, 1, 256, 257] + [0] * (experts - 4), 1024),
        ([381] * experts, 512 * experts),
        ([196608] + [0] * (experts - 1), 196608),
    ):
        counts = torch.tensor(counts, device="cuda", dtype=torch.int32)
        q[rows:rows + 1].fill_(-128)
        s[rows:rows + 1].fill_(-1)
        launch(counts)
        assert torch.equal(q[:rows], ref_q[:rows])
        assert torch.equal(s[:rows], ref_s[:rows])
        assert (q[rows:rows + 1] == -128).all()
        assert (s[rows:rows + 1] == -1).all()

    def time_ms(call):
        for _ in range(5):
            call()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(20):
            call()
        end.record()
        end.synchronize()
        return begin.elapsed_time(end) / 20

    counts = torch.full((experts,), 381, device="cuda", dtype=torch.int32)
    dense_ms = time_ms(lambda: fuse_silu_mul_clamp_quant(x, limit=10, output=q, scales=s))
    counted_ms = time_ms(lambda: launch(counts))
    print(json.dumps(dict(clamp_quant_exact=True, capacity_rows=capacity, active_rows=512 * experts,
                          dense_ms=dense_ms, device_count_ms=counted_ms)), flush=True)
    assert counted_ms < dense_ms


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    assert torch.cuda.get_device_properties("cuda").gcnArchName.split(":")[0] == "gfx938"
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    import deep_ep

    rank, ranks = dist.get_rank(), dist.get_world_size()
    assert ranks in (4, 8)
    hidden, local_experts, rows = 5120, 416 // ranks, 17
    size = max(make_config(ranks).get_nvl_buffer_size_hint(hidden * 2, ranks)
               for make_config in (deep_ep.Buffer.get_dispatch_config,
                                   deep_ep.Buffer.get_combine_config))
    buffer = deep_ep.Buffer(dist.group.WORLD, num_nvl_bytes=size)
    assert buffer.runtime.get_num_rdma_ranks() == 1
    worst = sum(rows + sender for sender in range(ranks))
    prepare = HcuCTHTPrepareAndFinalize(buffer, ranks, 1, rank * local_experts,
                                        fixed_receive_tokens=worst)
    num_experts = local_experts * ranks
    expert_map = torch.full((num_experts,), -1, device="cuda", dtype=torch.int32)
    expert_map[rank * local_experts:(rank + 1) * local_experts] = torch.arange(
        local_experts, device="cuda", dtype=torch.int32
    )
    x = torch.arange((rows + rank) * hidden, device="cuda").reshape(rows + rank, hidden)
    x = ((x % 29 - 14).float() / 16).bfloat16()
    weights = torch.full((x.size(0), 2), 0.5, device="cuda")
    rows_index = torch.arange(x.size(0), device="cuda")
    try:
        for skewed in (False, True):
            dest = torch.full_like(rows_index, ranks - 1) if skewed else (rows_index + rank) % ranks
            ids = dest[:, None] * local_experts + torch.arange(2, device="cuda")
            check_case(prepare, worst, x, ids, weights, num_experts, expert_map)
        for capacity, local_rows in ((2, 1), (16, 2)):
            if rank == 0:
                print(f"checking {capacity}-token warmup geometry", flush=True)
            physical = (ranks - 3) * local_experts + torch.arange(6, device="cuda")
            routes = physical.expand(local_rows, -1).clone()
            if capacity == 2 and rank >= 2:
                routes.fill_(-1)
            route_weights = torch.full(routes.shape, 0.125, device="cuda")
            route_weights.masked_fill_(routes < 0, 0)
            check_case(prepare, capacity, x[:local_rows], routes, route_weights,
                       num_experts, expert_map)
        print(f"rank {rank}: fixed and dynamic dispatch/combined outputs match", flush=True)
        if rank == 0:
            check_clamp_quant(local_experts)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
