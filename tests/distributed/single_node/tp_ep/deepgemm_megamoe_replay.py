"""8-rank eager/Graph checks against a separate PyTorch mathematical reference.

Run with torchrun --standalone --nproc-per-node=8, or directly with Python.
This tests the compiled EP8/H4096/I1536 decode instance without calling
DeepEP dispatch/combine or the previous custom G2 library.
"""
import json
import os
from pathlib import Path
import statistics
from datetime import timedelta

import torch
import torch.distributed as dist
from deepgemm import mega

ROOT = Path(os.environ.get('DG_MEGA_TEST_OUTPUT_DIR', 'deepgemm-megamoe-results')).resolve()
H, I, E, K, CAP = 4096, 1536, 352, 8, 64


def gather_cpu(value, group=None):
    result = [None] * dist.get_world_size(group)
    dist.all_gather_object(result, value, group=group)
    return result


def quantize(x):
    scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.
    return (x.float() / scale).to(torch.float8_e4m3fn), scale.squeeze(-1)


def error_report(actual, expected):
    if actual.numel() == 0:
        return dict(finite=True, max_abs=0., relative_rms=0., tolerance_failures=0)
    actual, expected = actual.float(), expected.float()
    error = (actual - expected).abs()
    amax = expected.abs().max()
    # Same elementwise criterion as the branch's published randn-init test.
    tolerance = .00355 + .03 * amax + .02 * expected.abs()
    result = dict(finite=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
                  max_abs=error.max().item(),
                  relative_rms=(error.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-12)).item(),
                  tolerance_failures=int((error > tolerance).sum()))
    assert result['finite'] and result['tolerance_failures'] == 0, result
    return result


def main():
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(rank)
    torch.set_grad_enabled(False)
    dist.init_process_group('gloo', timeout=timedelta(minutes=15))
    torch.manual_seed(7723 + rank)
    device = torch.device('cuda', rank)
    w1, s1 = quantize(torch.randn(44, 2 * I, H, device=device, dtype=torch.bfloat16) * .03)
    w2, s2 = quantize(torch.randn(44, H, I, device=device, dtype=torch.bfloat16) * .03)
    packed = mega.transform_weights_for_mega_moe((w1, s1), (w2, s2))
    buffer = mega.get_symm_buffer_for_mega_moe(dist.group.WORLD, E, CAP, K, H, I,
        use_fp8_dispatch=True, activation='swiglu', peer_memory_mode='ipc')
    assert buffer.decode and all(p[0].ndim == 7 for p in packed)
    ROOT.mkdir(parents=True, exist_ok=True)
    records, retained = [], []

    def reference(x, ids, weights):
        # Use exact input-quantizer output bytes. GEMMs, activation, intermediate
        # quantization, routing and reductions below are independent torch ops.
        mega.mega_moe_preprocess(x, ids, weights, buffer, len(x))
        decoded = buffer.x[:len(x)].float() * buffer.x_sf[:len(x)].reshape(-1, 1)
        objects = gather_cpu((decoded.cpu(), ids.cpu(), weights.cpu()))
        sizes = [len(v[0]) for v in objects]
        inputs = torch.cat([v[0] for v in objects]).to(device)
        routes = torch.cat([v[1] for v in objects]).to(device)
        scores = torch.cat([v[2] for v in objects]).to(device)
        total = torch.zeros_like(inputs)
        for expert in range(44):
            tokens, slots = torch.where(routes == rank * 44 + expert)
            if tokens.numel() == 0:
                continue
            gate_up = (inputs[tokens] @ (w1[expert].float() * s1[expert, :, None]).t()).bfloat16()
            gate, up = gate_up.float().chunk(2, -1)
            # The user's branch defaults to swiglu_nv, including clamp ordering.
            act = (torch.nn.functional.silu(gate.clamp(max=7.)) * up.clamp(-7., 7.)).bfloat16()
            act_q, act_sf = quantize(act)
            result = ((act_q.float() * act_sf[:, None]) @
                      (w2[expert].float() * s2[expert, :, None]).t()).bfloat16()
            total.index_add_(0, tokens, result.float() * scores[tokens, slots, None])
        cpu_total = total.cpu()
        dist.all_reduce(cpu_total)
        start = sum(sizes[:rank])
        return cpu_total[start:start + sizes[rank]].to(device).bfloat16()

    for count, route in [(0, 'spread'), (4, 'spread'), (16, 'spread'),
                         (32, 'spread'), (64, 'spread'), (16, 'concentrated'),
                         (64, 'concentrated'), (16, 'empty-rank'), (4, 'clamp')]:
        actual_count = 0 if route == 'empty-rank' and rank == 0 else count
        x = torch.randn(actual_count, H, device=device, dtype=torch.bfloat16) * (3.6 if route == 'clamp' else .6)
        starts = torch.arange(actual_count, device=device)[:, None] * K + rank * 17
        ids = ((starts + torch.arange(K, device=device)) % E).long()
        if route == 'concentrated':
            ids[:] = torch.arange(K, device=device)
        weights = torch.rand(actual_count, K, device=device)
        weights = weights / weights.sum(-1, keepdim=True)
        expected = reference(x, ids, weights)
        output = torch.empty_like(x)
        def run():
            mega.mega_moe_preprocess(x, ids, weights, buffer, actual_count)
            mega.fp8_mega_moe(output, *packed, buffer, activation_clamp=7., fast_math=True)
        run()
        torch.cuda.synchronize()
        eager = error_report(output, expected)
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        retained.append(graph)
        checks = []
        for _ in range(10):
            graph.replay()
            torch.cuda.synchronize()
            checks.append(error_report(output, expected))
        samples = []
        for _ in range(30):
            dist.barrier()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            graph.replay()
            b.record()
            b.synchronize()
            samples.append(a.elapsed_time(b) * 1000.)
        ranks = gather_cpu(samples)
        worst_rank_median = statistics.median(max(row) for row in zip(*ranks))
        records.append(dict(tokens=actual_count, route=route, eager=eager, graph=checks,
                            slowest_rank_median_us=worst_rank_median))
        (ROOT / f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank, records=records), indent=2))

    # Exercise the actual framework adapter and graph-capturable TP4 gather.
    import vllm_hcu.hcu_ops
    from types import SimpleNamespace
    from vllm_hcu.distributed.device_communicators.ipc_all_gather import IpcAllGather
    from vllm_hcu.model_executor.layers.fused_moe.deepgemm_megamoe import DeepGemmMegaMoEMethod
    groups = [dist.new_group(list(range(first, first + 4)), backend='gloo') for first in (0, 4)]
    tp_group = groups[rank // 4]
    gather = IpcAllGather(tp_group, device, CAP * H * 2)
    method = DeepGemmMegaMoEMethod.__new__(DeepGemmMegaMoEMethod)
    method.moe = SimpleNamespace(experts_per_token=K, activation='swiglustep', swiglu_limit=None)
    method._buffer, method._weights, method._gather = buffer, packed, gather
    method._capacity, method._tp_size, method._tp_rank = CAP, 4, rank % 4
    for rows in (0, 5, 64):
        generator = torch.Generator(device=device).manual_seed(9000 + rank // 4)
        x = torch.randn(rows, H, generator=generator, device=device, dtype=torch.bfloat16) * (3.6 if rows == 64 else .6)
        ids = ((torch.arange(rows, device=device)[:, None] * K + torch.arange(K, device=device) +
                (rank // 4) * 19) % E).long()
        weights = torch.full((rows, K), 1. / K, device=device)
        chunk = (rows + 3) // 4
        start, stop = min((rank % 4) * chunk, rows), min((rank % 4 + 1) * chunk, rows)
        expected_local = reference(x[start:stop], ids[start:stop], weights[start:stop])
        expected = torch.cat(gather_cpu(expected_local.cpu(), tp_group)).to(device)
        def run_adapter():
            return method.apply(SimpleNamespace(swiglu_limit=None), x, weights, ids)
        output = run_adapter()
        torch.cuda.synchronize()
        eager = error_report(output, expected)
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = run_adapter()
        retained.append(graph)
        for _ in range(10):
            graph.replay()
            torch.cuda.synchronize()
            error_report(graph_output, expected)
        records.append(dict(tp4_rows=rows, eager=eager, graph_replays=10))
    (ROOT / f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank, passed=True, records=records), indent=2))
    torch.cuda.synchronize()
    dist.barrier()
    gather.close()
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


def spawn_rank(rank):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='8',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT='19549')
    main()


if __name__ == '__main__':
    if 'RANK' in os.environ:
        main()
    else:
        torch.multiprocessing.spawn(spawn_rank, nprocs=8, join=True)
