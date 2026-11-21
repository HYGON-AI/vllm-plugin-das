"""Run with torchrun --standalone --nproc-per-node=16 against a HY4-capable DeepGEMM."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def physical_routing(row, tokens, rank):
    # Exercise every physical slot, including replicas, across the ranks.
    start = torch.arange(tokens, dtype=torch.int64)[:, None] * 8 + rank * 17
    ids = (start + torch.arange(8)) % len(row)
    return ids, torch.tensor(row, dtype=torch.int64)[ids]


def expert_weights(physical_row, device):
    # All output channels read one input channel. Each logical expert has a
    # distinct scale, so the reference detects misplaced or swapped replicas.
    count = len(physical_row)
    w13 = torch.zeros(count, 4096, 6144, dtype=torch.float8_e4m3fn, device=device)
    w2 = torch.zeros(count, 6144, 2048, dtype=w13.dtype, device=device)
    w13[:, :, 0] = 1
    w2[:, :, 0] = 1
    gains = (torch.tensor(physical_row, device=device).float() + 1) / 16
    s13 = torch.cat((gains[:, None].expand(-1, 2048),
                     torch.ones(count, 2048, device=device)), dim=1)
    s2 = torch.ones(count, 6144, device=device)
    return (w13, s13), (w2, s2)


def run(row, tokens, decode=False):
    import deepgemm.mega as mega
    from vllm_hcu.models.hy_v4.megamoe import pack_hy4_mega_weights, run_hy4_mega_experts

    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    count = len(row) // dist.get_world_size()
    local = row[count * rank:count * (rank + 1)]
    (w13, s13), (w2, s2) = expert_weights(local, device)
    packed = pack_hy4_mega_weights(mega, w13, w2, s13, s2)
    del w13, s13, w2, s2
    capacity = tokens if decode else (tokens + 127) // 128 * 128
    buf = mega.get_symm_buffer_for_mega_moe(
        dist.group.WORLD, len(row), capacity, 8, 6144, 2048, use_fp8_dispatch=True)
    try:
        ids, logical = physical_routing(row, tokens, rank)
        # Nonuniform routing weights also exercise the combine operation.
        weights = (torch.arange(1, 9, device=device).float() / 36).expand(tokens, -1).contiguous()
        x = torch.ones(tokens, 6144, device=device, dtype=torch.bfloat16)
        x[1::2].neg_()
        ids = ids.to(device)
        def forward():
            return run_hy4_mega_experts(mega, x, ids, weights, packed, buf, 10.0)

        def verify(out):
            sign = x[:, :1].float()
            gate = (sign * ((logical.to(device).float() + 1) / 16)).clamp(max=10)
            ref = (torch.nn.functional.silu(gate) * sign * weights).sum(-1, keepdim=True)
            torch.testing.assert_close(out.float(), ref.expand_as(out), rtol=0.025, atol=0.015)

        verify(forward())
        if decode:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = forward()
            x.neg_()
            graph.replay()
            verify(out)
            # Uneven local batch: rank zero still serves its local experts.
            n = 0 if rank == 0 else rank
            x, ids, weights, logical = x[:n], ids[:n], weights[:n], logical[:n]
            verify(forward())
    finally:
        buf.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()
    if args.tokens <= 0 or int(os.environ["WORLD_SIZE"]) != 16:
        parser.error("Requires EP16 and a positive token count")
    if args.decode:
        row = list(range(256))
    else:
        if args.map is None:
            parser.error("Prefill requires --map")
        maps = json.loads(args.map.read_text())["model_maps"]
        rows = maps["HYV4VLForConditionalGeneration"]["physical_to_logical_map"]
        if not 0 <= args.layer < len(rows):
            parser.error("Layer index outside VL map")
        row = rows[args.layer]
        if len(row) != 272 or set(row) != set(range(256)):
            parser.error("Expected HY4 R16 physical-to-logical map")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        run(row, args.tokens, args.decode)
        dist.barrier()
        if dist.get_rank() == 0:
            print("HY4 channel FP8 MegaMoE EP16 numerical check passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
