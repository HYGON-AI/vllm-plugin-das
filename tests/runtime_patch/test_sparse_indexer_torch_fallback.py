# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Sparse indexer PyTorch fallback numerical and GPU Graph regression tests."""

import ast
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
import pytest

def test_sparse_indexer_torch_reference(monkeypatch):
    import ast
    import sys
    from types import SimpleNamespace, ModuleType
    from pathlib import Path
    import torch
    import torch.nn.functional as F
    source = (Path(__file__).resolve().parents[2] / 'vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py').read_text()
    tree = ast.parse(source)
    names = {'fp8_paged_mqa_logits_torch', 'fp8_mqa_logits_torch'}
    selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    namespace = dict(torch=torch, F=F, current_platform=SimpleNamespace(fp8_dtype=lambda : torch.float8_e4m3fn))
    stub = ModuleType('vllm.utils.math_utils')
    stub.cdiv = lambda a, b: (a + b - 1) // b
    monkeypatch.setitem(sys.modules, 'vllm.utils.math_utils', stub)
    exec(compile(selected, '<reference>', 'exec'), namespace)
    torch.manual_seed(7)
    (b, d, h) = (4, 8, 3)
    values = torch.randint(-4, 5, (3, b, d)).to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.5, 1, 2, 4], [2, 1, 0.5, 4], [1, 2, 4, 0.5]], dtype=torch.float32)
    packed = torch.cat([values.view(torch.uint8).reshape(3, -1), scales.view(torch.uint8).reshape(3, -1)], dim=1).view(3, b, 1, d + 4)
    q = torch.randn(1, 1, h, d)
    w = torch.rand(1, h)
    table = torch.tensor([[2, 0]], dtype=torch.int32)
    for length in [1, 4, 7]:
        actual = namespace['fp8_paged_mqa_logits_torch'](q, packed, w, torch.tensor([length]), table, 8)
        k = values.float()[table[0].long()].reshape(-1, d)[:length]
        scale = scales[table[0].long()].reshape(-1)[:length]
        expected = ((q[0, 0] @ k.T).relu() * w[0, :, None]).sum(0) * scale
        torch.testing.assert_close(actual[0, :length], expected)
        assert torch.isneginf(actual[0, length:]).all()
    for (m, n) in [(9, 2), (2, 9), (3, 3)]:
        q = torch.randint(-3, 4, (m, 3, 8)).to(torch.float8_e4m3fn)
        k = torch.randint(-3, 4, (n, 8)).to(torch.float8_e4m3fn)
        scales = torch.arange(1, n + 1, dtype=torch.float32).reshape(n, 1) / 2
        weights = torch.rand(m, 3)
        lo = torch.zeros(m, dtype=torch.int32)
        hi = torch.full((m,), n, dtype=torch.int32)
        hi[0] = 1
        actual = namespace['fp8_mqa_logits_torch'](q, (k, scales), weights, lo, hi)
        expected = torch.empty(m, n)
        for row in range(m):
            for col in range(n):
                expected[row, col] = sum((max(float(torch.dot(q[row, head].float(), k[col].float())), 0) * float(weights[row, head]) * float(scales[col, 0]) for head in range(3))) if col < int(hi[row]) else -torch.inf
        torch.testing.assert_close(actual, expected)

def test_decode_graph_replay():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip('GPU graph test requires a visible GPU')
    source = (Path(__file__).resolve().parents[2] / 'vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py').read_text()
    node = next((n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'fp8_paged_mqa_logits_torch'))
    node.body = [n for n in node.body if not isinstance(n, ast.ImportFrom)]
    ns = dict(torch=torch, F=F, current_platform=SimpleNamespace(fp8_dtype=lambda : torch.float8_e4m3fn))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<decode>', 'exec'), ns)
    fn = ns['fp8_paged_mqa_logits_torch']
    torch.manual_seed(17)
    for batch in (1, 8):
        (block, dim, heads, limit) = (256, 128, 8, 513)
        values = torch.randint(-3, 4, (5, block, dim), device='cuda').to(torch.float8_e4m3fn)
        scales = torch.rand(5, block, device='cuda') + 0.25
        cache = torch.cat((values.view(torch.uint8).reshape(5, -1), scales.view(torch.uint8).reshape(5, -1)), 1).view(5, block, 1, dim + 4)
        q = torch.randn(batch, 1, heads, dim, device='cuda').to(torch.float8_e4m3fn)
        weights = torch.rand(batch, heads, device='cuda')
        lengths = torch.full((batch,), 1, device='cuda', dtype=torch.int32)
        tables = torch.tensor([[2, 0, 4]] * batch, device='cuda', dtype=torch.int32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn(q, cache, weights, lengths, tables, limit)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fn(q, cache, weights, lengths, tables, limit)
        for step in range(3):
            lens = [[0, 255, 513][(i + step) % 3] for i in range(batch)]
            lengths.copy_(torch.tensor(lens, device='cuda', dtype=torch.int32))
            tables.copy_(torch.tensor([[4, 2, 1] if step % 2 else [1, 0, 3]] * batch, device='cuda', dtype=torch.int32))
            q.copy_(torch.randn_like(q.float()).to(q.dtype))
            weights.mul_(0.9)
            graph.replay()
            expected = torch.full((batch, limit), -torch.inf, device='cuda')
            for (i, length) in enumerate(lens):
                ids = tables[i].long()
                k = values.float()[ids].reshape(-1, dim)[:length]
                scale = scales[ids].reshape(-1)[:length]
                expected[i, :length] = ((q[i, 0].float() @ k.T).relu() * weights[i, :, None]).sum(0) * scale
            torch.testing.assert_close(output, expected, atol=0.002, rtol=0.0002)

def test_prefill_gather_graph_replay():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip('GPU graph test requires a visible GPU')
    source = (Path(__file__).resolve().parents[2] / 'vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py').read_text()
    branch = next((n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and (n.test.id == 'v4_fp8_fallback') and any((isinstance(x, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'page_size' for t in x.targets)) for x in n.body))))
    code = compile(ast.Module(body=branch.body, type_ignores=[]), '<gather>', 'exec')
    (block, dim, count) = (4, 8, 7)
    values = torch.arange(3 * block * dim, device='cuda').reshape(3, block, dim).remainder(7).to(torch.float8_e4m3fn)
    scales = torch.arange(12, device='cuda', dtype=torch.float32).reshape(3, 4) + 1
    cache = torch.cat((values.view(torch.uint8).reshape(3, -1), scales.view(torch.uint8).reshape(3, -1)), 1).view(3, block, dim + 4)
    chunk = SimpleNamespace(total_seq_lens=count, cu_seq_lens=torch.tensor([0, 3, 7], device='cuda', dtype=torch.int32), block_table=torch.tensor([[2, 1], [0, 2]], device='cuda', dtype=torch.int32))
    out = torch.empty(count, dim, device='cuda', dtype=torch.float8_e4m3fn)
    out_scale = torch.empty(count, 4, device='cuda', dtype=torch.uint8)
    ns = dict(torch=torch, chunk=chunk, kv_cache=cache, head_dim=dim, fp8_dtype=torch.float8_e4m3fn, k_fp8=out, k_scale=out_scale)

    def gather():
        exec(code, ns)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            gather()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gather()
    for boundary in (0, 2, 5):
        chunk.cu_seq_lens.copy_(torch.tensor([0, boundary, count], device='cuda', dtype=torch.int32))
        graph.replay()
        expected = []
        expected_scales = []
        for (seq, (start, end)) in enumerate(((0, boundary), (boundary, count))):
            for j in range(end - start):
                page = int(chunk.block_table[seq, j // block].item())
                expected.append(values[page, j % block].float())
                expected_scales.append(scales[page, j % block])
        torch.testing.assert_close(out.float(), torch.stack(expected))
        torch.testing.assert_close(out_scale.view(torch.float32).flatten(), torch.stack(expected_scales))

def _load_decode_fn():
    path = Path(__file__).resolve().parents[2] / 'vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py'
    node = next((n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'fp8_paged_mqa_logits_torch'))
    ns = dict(torch=torch, current_platform=SimpleNamespace(fp8_dtype=lambda : torch.float8_e4m3fn))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    return ns[node.name]

@pytest.mark.parametrize('next_n', [1, 2, 6])
@pytest.mark.parametrize('per_query', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_multitoken(next_n, per_query, device):
    if device == 'cuda' and (not torch.cuda.is_available()):
        pytest.skip('GPU required')
    fn = _load_decode_fn()
    torch.manual_seed(9)
    (batch, block, dim, heads, limit) = (2, 4, 8, 3, 11)
    values = torch.randint(-3, 4, (5, block, dim), device=device).to(torch.float8_e4m3fn)
    scales = torch.rand(5, block, device=device) + 0.25
    cache = torch.cat([values.view(torch.uint8).reshape(5, -1), scales.view(torch.uint8).reshape(5, -1)], 1).view(5, block, 1, dim + 4)
    q = torch.randn(batch, next_n, heads, dim, device=device).to(torch.float8_e4m3fn)
    weights = torch.rand(batch * next_n, heads, device=device)
    table = torch.tensor([[4, 1, 0], [2, 3, 1]], device=device, dtype=torch.int32)
    lengths = torch.full((batch, next_n) if per_query else (batch,), 7, device=device, dtype=torch.int32)
    graph = None
    if device == 'cuda':
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn(q, cache, weights, lengths, table, limit)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn(q, cache, weights, lengths, table, limit)
    for step in (0, 1, 2):
        if per_query:
            data = [[min(11, max(0, step * 3 + j - 1)) for j in range(next_n)], [0] * next_n]
        else:
            data = [11 - step * 4, step * 4]
        lengths.copy_(torch.tensor(data, device=device, dtype=torch.int32))
        table.copy_(torch.tensor([[step, 4, 1], [4, step, 0]], device=device, dtype=torch.int32))
        q.copy_(torch.randn_like(q.float()).to(q.dtype))
        if graph:
            graph.replay()
        else:
            out = fn(q, cache, weights, lengths, table, limit)
        expected = torch.full((batch, next_n, limit), -torch.inf, device=device)
        for b in range(batch):
            keys = values.float()[table[b].long()].reshape(-1, dim)
            scale = scales[table[b].long()].reshape(-1)
            for j in range(next_n):
                end = data[b][j] if per_query else max(0, data[b] - next_n + j + 1)
                expected[b, j, :end] = ((q[b, j].float() @ keys[:end].T).relu() * weights[b * next_n + j, :, None]).sum(0) * scale[:end]
        torch.testing.assert_close(out, expected.reshape(batch * next_n, limit), atol=0.0002, rtol=0.0002)


def test_long_context_bounds_decode_workspace(monkeypatch):
    """A large configured limit must not expand all requests' KV at once."""
    fn = _load_decode_fn()
    batch, block, dim, heads, limit = 9, 256, 128, 8, 32769
    values = torch.ones(1, block, dim).to(torch.float8_e4m3fn)
    scales = torch.ones(1, block)
    cache = torch.cat((values.view(torch.uint8).reshape(1, -1),
                       scales.view(torch.uint8).reshape(1, -1)), 1)
    cache = cache.view(1, block, 1, dim + 4)
    table = torch.full((batch, (limit + block - 1) // block), -999, dtype=torch.int32)
    table[:, 0] = 0
    q = torch.ones(batch, 1, heads, dim)
    weights = torch.ones(batch, heads)
    lengths = torch.arange(batch, dtype=torch.int32)
    original_bmm = torch.bmm
    workspaces = []

    def bounded_bmm(a, b):
        workspaces.append((b.numel() + a.shape[0] * a.shape[1] * b.shape[2]) * 4)
        return original_bmm(a, b)

    monkeypatch.setattr(torch, 'bmm', bounded_bmm)
    out = fn(q, cache, weights, lengths, table, limit)
    assert len(workspaces) > 2
    assert max(workspaces) <= 8 * 1024 * 1024
    for row in range(batch):
        torch.testing.assert_close(out[row, :row], torch.full((row,), float(heads * dim)))
        assert torch.isneginf(out[row, row:]).all()


def test_fallback_packs_weights_with_variable_decode_lengths():
    """Exercise the dispatch branch with a short decode beside a full decode."""
    path = Path(__file__).resolve().parents[2] / 'vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py'
    branch = next(
        n for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
        and n.test.id == 'v4_fp8_fallback'
        and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                and c.func.id == 'fp8_paged_mqa_logits_torch' for c in ast.walk(n))
    )
    block, dim, heads = 4, 8, 3
    values = torch.ones(1, block, dim).to(torch.float8_e4m3fn)
    scales = torch.ones(1, block)
    cache = torch.cat((values.view(torch.uint8).reshape(1, -1),
                       scales.view(torch.uint8).reshape(1, -1)), 1).view(1, block, 1, dim + 4)
    weights = torch.arange(1, 5).float()[:, None].expand(-1, heads)
    calls = []

    def pack(tensor, lengths):
        calls.append(tensor.clone())
        out = torch.zeros(2, 3, heads)
        out[0, 0] = tensor[0]
        out[1] = tensor[1:4]
        return out

    namespace = dict(
        fp8_paged_mqa_logits_torch=_load_decode_fn(), pack_seq_triton=pack,
        padded_q_fp8_decode_tokens=torch.ones(2, 3, heads, dim),
        kv_cache=cache, weights=weights, num_decode_tokens=4, num_padded_tokens=6,
        seq_lens=torch.tensor([[1, 0, 0], [1, 2, 3]]),
        decode_lens=torch.tensor([1, 3]), max_model_len=4,
        decode_metadata=SimpleNamespace(requires_padding=True,
                                        block_table=torch.zeros(2, 1, dtype=torch.int32)),
    )
    exec(compile(ast.Module(body=branch.body, type_ignores=[]), str(path), 'exec'), namespace)
    assert len(calls) == 1
    out = namespace['logits']
    for row, length, weight in [(0, 1, 1), (3, 1, 2), (4, 2, 3), (5, 3, 4)]:
        torch.testing.assert_close(out[row, :length], torch.full((length,), float(heads * dim * weight)))
        assert torch.isneginf(out[row, length:]).all()


def _load_prefill_fn():
    path = Path(__file__).resolve().parents[2] / 'vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'fp8_mqa_logits_torch')
    namespace = dict(torch=torch)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[node.name]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_prefill_bounds_workspace_and_preserves_masks(monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('GPU required')
    fn = _load_prefill_fn()
    # Cross both query and key chunk boundaries with the V4 head count.
    m, n, h, d = 65, 1025, 64, 8
    torch.manual_seed(81)
    q = torch.randint(-3, 4, (m, h, d), device=device).to(torch.float8_e4m3fn)
    k = torch.randint(-3, 4, (n, d), device=device).to(torch.float8_e4m3fn)
    scales = torch.rand(n, 1, device=device) + 0.25
    weights = torch.randn(m, h, device=device)
    lo = torch.arange(m, device=device, dtype=torch.int32) * 7
    hi = torch.clamp(lo + 300, max=n)
    lo[0], hi[0] = 0, n
    hi[1] = lo[1]  # Empty interval must stay entirely masked.
    calls = []
    original_einsum = torch.einsum

    def measured_einsum(equation, a, b):
        result = original_einsum(equation, a, b)
        calls.append((a.shape[0], b.shape[0], result.numel() * 6))
        return result

    monkeypatch.setattr(torch, 'einsum', measured_einsum)
    actual = fn(q, (k, scales), weights, lo, hi)
    assert len(calls) > 1
    assert max(size for _, _, size in calls) <= 8 * 1024 * 1024
    # Independent head-by-head reference keeps test memory small.
    expected = torch.zeros(m, n, device=device)
    for head in range(h):
        score = (q[:, head].float() @ k.float().T) * scales.flatten()[None, :]
        expected += score.relu() * weights[:, head, None]
    offsets = torch.arange(n, device=device)
    expected.masked_fill_((offsets[None, :] < lo[:, None]) |
                          (offsets[None, :] >= hi[:, None]), -torch.inf)
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.0002)
    torch.testing.assert_close(fn(q, (k, scales.flatten()), weights, lo, hi), actual)


def test_prefill_logits_graph_replay():
    if not torch.cuda.is_available():
        pytest.skip('GPU graph test requires a visible GPU')
    fn = _load_prefill_fn()
    m, n, h, d = 65, 1025, 64, 8
    q = torch.ones(m, h, d, device='cuda').to(torch.float8_e4m3fn)
    k = torch.ones(n, d, device='cuda').to(torch.float8_e4m3fn)
    scales = torch.ones(n, 1, device='cuda')
    weights = torch.ones(m, h, device='cuda')
    lo = torch.zeros(m, dtype=torch.int32, device='cuda')
    hi = torch.full_like(lo, n)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(q, (k, scales), weights, lo, hi)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(q, (k, scales), weights, lo, hi)
    for step in range(3):
        q.copy_(torch.full_like(q.float(), step + 1).to(q.dtype))
        k.copy_(torch.full_like(k.float(), 3 - step).to(k.dtype))
        scales.fill_(0.5 * (step + 1))
        weights.fill_(0.25 * (step + 1))
        lo.fill_(step * 300)
        hi.fill_(n - step * 100)
        hi[1] = lo[1]
        graph.replay()
        value = h * d * (step + 1) * (3 - step) * 0.5 * (step + 1) * 0.25 * (step + 1)
        expected = torch.full((m, n), value, device='cuda')
        offsets = torch.arange(n, device='cuda')
        expected.masked_fill_((offsets[None, :] < lo[:, None]) |
                              (offsets[None, :] >= hi[:, None]), -torch.inf)
        torch.testing.assert_close(out, expected)
