import torch
from flash_mla import flash_mla_with_kvcache, get_mla_metadata
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_decode
from vllm.models.deepseek_v4.common.ops import quantize_and_insert_k_cache

dev = 'cuda:0'
heads = 4
TOL = 0.01


def cache(blocks, size):
    c = torch.zeros(blocks, size, 584, device=dev, dtype=torch.uint8)
    k = torch.randn(blocks * size, 512, device=dev, dtype=torch.bfloat16)
    slots = torch.arange(blocks * size, device=dev, dtype=torch.int64)
    quantize_and_insert_k_cache(k, c.view(blocks, -1), slots, block_size=size, use_fnuz=False)
    return c


def make_inputs(batch, length, ratio):
    torch.manual_seed(13 + batch + length + ratio)
    q = torch.randn(batch, heads, 512, device=dev, dtype=torch.bfloat16)
    swa = cache(batch, 128)
    idx = torch.arange(128, device=dev, dtype=torch.int32).view(1, 1, 128).expand(batch, 1, -1).clone()
    idx += torch.arange(batch, device=dev, dtype=torch.int32).view(batch, 1, 1) * 128
    lengths = torch.full((batch,), min(length, 128), device=dev, dtype=torch.int32)
    extra = extra_idx = extra_lens = None
    if ratio != 1:
        extra = cache(batch, 64)
        extra_idx = torch.arange(64, device=dev, dtype=torch.int32).view(1, 1, 64).expand(batch, 1, -1).clone()
        extra_idx += torch.arange(batch, device=dev, dtype=torch.int32).view(batch, 1, 1) * 64
        extra_lens = torch.full((batch,), min(length // ratio, 64), device=dev, dtype=torch.int32)
    sink = torch.randn(heads, device=dev, dtype=torch.float32)
    return q, swa, idx, lengths, extra, extra_idx, extra_lens, sink


def reference(q, swa, idx, lengths, extra, extra_idx, extra_lens, sink, ratio):
    a = torch.empty_like(q)
    rocm_sparse_attn_decode(q, extra, swa, ratio == 1, extra_idx, extra_lens, idx,
                            lengths, None, None, None, None, sink, 512 ** -0.5, 512, 448, 64, a)
    return a


def flashmla(q, swa, idx, lengths, extra, extra_idx, extra_lens, sink, sched=None):
    f, _ = flash_mla_with_kvcache(
        q.unsqueeze(1), swa.unsqueeze(2), None, None, 512,
        sched if sched is not None else get_mla_metadata()[0],
        softmax_scale=512 ** -0.5, is_fp8_kvcache=True,
        indices=idx, topk_length=lengths, attn_sink=sink,
        extra_k_cache=extra.unsqueeze(2) if extra is not None else None,
        extra_indices_in_kvcache=extra_idx, extra_topk_length=extra_lens,
    )
    return f.squeeze(1)


# Eager comparison. length=6000 makes the C128A compressed side nonempty
# (6000 // 128 = 46 of 64 slots), which lengths 17/93 cannot reach.
for batch in (1, 8):
    for length in (17, 93, 6000):
        for ratio in (1, 4, 128):
            args = make_inputs(batch, length, ratio)
            a = reference(*args, ratio)
            f = flashmla(*args)
            err = (f.float() - a.float()).abs()
            extra_lens = args[6]
            n_extra = int(extra_lens[0]) if extra_lens is not None else 0
            print(batch, length, ratio, f'extra_len={n_extra}',
                  'max', err.max().item(), 'mean', err.mean().item(),
                  'refmax', a.abs().max().item(), flush=True)
            assert err.max().item() <= TOL, (batch, length, ratio, err.max().item())

# CUDA graph: capture once, then mutate lengths and indices in place and
# replay. FULL_DECODE_ONLY replays the captured planner kernel with new
# metadata values, so the plan must follow the device buffers, not the
# values seen at capture.
batch = 8
for ratio in (1, 4, 128):
    q, swa, idx, lengths, extra, extra_idx, extra_lens, sink = make_inputs(batch, 6000, ratio)
    # Warm up lazy HIP/kernel state with a THROWAWAY sched. The captured sched
    # must be fresh (have_initialized=False) so the planner kernels are recorded
    # inside the graph and re-run on every replay, re-reading the topk_length
    # device buffers. This mirrors vLLM, where the metadata builder allocates a
    # new FlashMLASchedMeta each step, so the capture-step planner lands in the
    # graph. A sched pre-initialized outside the graph freezes the plan and
    # VM-faults on replay once the lengths change.
    flashmla(q, swa, idx, lengths, extra, extra_idx, extra_lens, sink, get_mla_metadata()[0])
    torch.cuda.synchronize()
    sched = get_mla_metadata()[0]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = flashmla(q, swa, idx, lengths, extra, extra_idx, extra_lens, sink, sched)

    # Replay 1: same values as capture (sanity).
    graph.replay()
    torch.cuda.synchronize()
    a = reference(q, swa, idx, lengths, extra, extra_idx, extra_lens, sink, ratio)
    err = (captured.float() - a.float()).abs().max().item()
    print('graph same-values', ratio, err, flush=True)
    assert err <= TOL, (ratio, 'same-values', err)

    # Replay 2: new lengths and new query, written into the SAME buffers.
    torch.manual_seed(999 + ratio)
    q.copy_(torch.randn_like(q))
    lengths.fill_(31)
    if extra_lens is not None:
        extra_lens.fill_(9)
    graph.replay()
    torch.cuda.synchronize()
    a = reference(q, swa, idx, lengths, extra, extra_idx, extra_lens, sink, ratio)
    err = (captured.float() - a.float()).abs().max().item()
    print('graph changed-lengths', ratio, err, flush=True)
    assert err <= TOL, (
        ratio, 'changed-lengths', err,
        'the captured tile-scheduler plan does not track topk_length; '
        'FULL_DECODE_ONLY replay would compute wrong attention',
    )

    # Replay 3: grow lengths past the captured values.
    lengths.fill_(128)
    if extra_lens is not None:
        extra_lens.fill_(46)
    graph.replay()
    torch.cuda.synchronize()
    a = reference(q, swa, idx, lengths, extra, extra_idx, extra_lens, sink, ratio)
    err = (captured.float() - a.float()).abs().max().item()
    print('graph grown-lengths', ratio, err, flush=True)
    assert err <= TOL, (ratio, 'grown-lengths', err)

print('ALL CHECKS PASSED', flush=True)
