# SPDX-License-Identifier: Apache-2.0
"""Real HIP tests of the active IndexPool package, not legacy/AST aliases.

Run with HIP_VISIBLE_DEVICES=2 and an active source snapshot on PYTHONPATH.
Missing HIP, runtime activation, provider ABI, or kernels is a failure, not skip.
This suite validates arithmetic/provider contracts, not an installed model/wheel.
"""

import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="session")
def active():
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "2"
    assert torch.version.hip is not None and torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    device = torch.cuda.get_device_properties(0)
    assert (device.pci_domain_id, device.pci_bus_id, device.pci_device_id) == (
        0,
        0x5E,
        0,
    )
    from hcu_adapter.bridges.vllm import hcu_runtime_plugin

    hcu_runtime_plugin()  # Real guarded activation; errors must propagate.
    dependency_sources = {}
    for name in (
        "torch",
        "triton",
        "vllm",
        "vllm.triton_utils",
        "hcu_adapter",
        "hcu_adapter.bridges.vllm",
    ):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        dependency_sources[name] = dict(
            file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    print(
        "DEPENDENCY_IMPORTS=" + json.dumps(dependency_sources, sort_keys=True),
        flush=True,
    )
    root = "hcu_vllm_das.models.hy_v4_flash"
    modules = {
        name: importlib.import_module(root + "." + name)
        for name in (
            "runtime",
            "indexpool_ops",
            "indexpool_prepare",
            "indexpool_triton",
        )
    }
    assert modules["runtime"].is_hcu_platform()
    modules["runtime"].validate_flash_platform()
    expected = Path(
        os.environ.get(
            "INDEXPOOL_ACTIVE_PACKAGE_ROOT",
            Path(__file__).resolve().parents[2] / "adapter_extension/src",
        )
    ).resolve()
    provenance = {}
    for name, module in modules.items():
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(expected), (path, expected)
        data = path.read_bytes()
        assert b"\0" not in data
        assert b"vllm_hcu" not in data
        provenance[name] = dict(
            module=module.__name__,
            file=str(path),
            sha256=hashlib.sha256(data).hexdigest(),
        )
    print("ACTIVE_IMPORTS=" + json.dumps(provenance, sort_keys=True), flush=True)
    torch.cuda.reset_peak_memory_stats()
    yield SimpleNamespace(**modules)
    torch.cuda.synchronize()
    print(
        "HIP_PEAK_ALLOCATED_BYTES=" + str(torch.cuda.max_memory_allocated()), flush=True
    )


def _fwht_reference(x):
    x = x.float()
    width = 1
    while width < 128:
        pairs = x.reshape(-1, 2, width)
        x = torch.stack(
            (pairs[:, 0] + pairs[:, 1], pairs[:, 0] - pairs[:, 1]), 1
        ).reshape(-1, 128)
        width *= 2
    x *= 128**-0.5
    scale = torch.exp2(torch.ceil(torch.log2(x.abs().amax(-1).clamp_min(1e-4) / 448)))
    return (x / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn), scale


@pytest.mark.parametrize("rows", [0, 1, 32, 513])
@pytest.mark.parametrize("strided", [False, True])
def test_fwht_fusion_bit_exact(active, rows, strided):
    torch.manual_seed(rows)
    backing = torch.randn(rows + 2, 260 if strided else 128, device="cuda").bfloat16()
    x = backing[1 : rows + 1, 2:258:2] if strided else backing[1 : rows + 1]
    before = backing.clone()
    # The arithmetic oracle runs on CPU, not through the fused implementation.
    q, s = _fwht_reference(x.cpu())
    direct = active.indexpool_triton.fwht_quantize_triton(x)
    dispatched = active.indexpool_ops.fwht_quantize(x)
    for actual_q, actual_s in (direct, dispatched):
        torch.testing.assert_close(actual_s.cpu(), s, rtol=0, atol=0)
        assert torch.equal(actual_q.view(torch.uint8).cpu(), q.view(torch.uint8))
    assert torch.equal(backing, before)


def test_fwht_extremes_and_nan(active):
    torch.manual_seed(33)
    x = torch.randn(12, 128, device="cuda").bfloat16()
    x[:4] *= torch.tensor([0, 2**-20, 2**12, 2**-4], device="cuda")[:, None]
    x[4, 0], x[5, 0], x[6, 0] = float("nan"), float("inf"), -float("inf")
    q, s = _fwht_reference(x.cpu())
    a, b = active.indexpool_triton.fwht_quantize_triton(x)
    torch.testing.assert_close(b.cpu(), s, rtol=0, atol=0, equal_nan=True)
    # NaN payloads need not agree; finite quantization must be byte exact.
    finite = ~q.float().isnan()
    assert torch.equal(a.view(torch.uint8).cpu()[finite], q.view(torch.uint8)[finite])
    torch.testing.assert_close(
        a.float().cpu(), q.float(), rtol=0, atol=0, equal_nan=True
    )


def _expand_reference(groups, positions, ratio, active_mask):
    groups, positions = groups.cpu().tolist(), positions.cpu().tolist()
    live = (
        [True] * len(positions) if active_mask is None else active_mask.cpu().tolist()
    )
    rows = []
    for history, pos, enabled in zip(groups, positions, live):
        candidates = []
        for group in history:
            for offset in range(ratio):
                candidates.append(
                    group * ratio + offset
                    if enabled and pos >= 0 and 0 <= group < pos // ratio
                    else -1
                )
        candidates.extend(
            (
                pos // ratio * ratio + offset
                if enabled and pos >= 0 and pos // ratio * ratio + offset <= pos
                else -1
            )
            for offset in range(ratio)
        )
        # Preserve candidate int32 cast *before* validity/compaction.
        candidates = [((v + 2**31) % 2**32) - 2**31 for v in candidates]
        rows.append(
            [v for v in candidates if v >= 0] + [v for v in candidates if v < 0]
        )
    return torch.tensor(rows, dtype=torch.int32)


@pytest.mark.parametrize("ratio,budget", [(1, 0), (4, 7), (4, 511), (3, 17)])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("use_active", [False, True])
def test_expand_groups_causal_compaction(active, ratio, budget, strided, use_active):
    positions = torch.tensor([-1, 0, 31, 2047, 2048, 4095, 4096, 8191], device="cuda")
    groups = torch.arange(budget, device="cuda").repeat(8, 1).int()
    if budget:
        groups[:, ::5], groups[:, ::7] = -1, 50000
    mask = (
        torch.tensor([True, True, False, True, True, False, True, True], device="cuda")
        if use_active
        else None
    )
    if strided:
        storage = torch.full(
            (10, budget * 2 + 4), 123, device="cuda", dtype=torch.int32
        )
        storage[1:9, 2 : 2 + budget * 2 : 2] = groups
        groups = storage[1:9, 2 : 2 + budget * 2 : 2]
        p = torch.zeros(18, device="cuda", dtype=torch.long)
        p[1:17:2] = positions
        positions = p[1:17:2]
    expected = _expand_reference(groups, positions, ratio, mask)
    a = active.indexpool_triton.expand_groups_triton(groups, positions, ratio, mask)
    b = active.indexpool_ops.expand_groups(groups, positions, ratio, mask)
    torch.testing.assert_close(a.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(b.cpu(), expected, rtol=0, atol=0)


def test_expand_cast_broadcast_and_empty(active):
    groups = torch.tensor([[0, 2**29, 2**30, -1]], device="cuda")
    positions = torch.tensor([2**33], device="cuda")
    actual = active.indexpool_ops.expand_groups(groups, positions, 4)
    torch.testing.assert_close(
        actual.cpu(), _expand_reference(groups, positions, 4, None), rtol=0, atol=0
    )
    groups = torch.arange(7, device="cuda", dtype=torch.int32)[None]
    positions = torch.tensor([31, 4095], device="cuda")
    actual = active.indexpool_ops.expand_groups(groups, positions, 4)
    expected = _expand_reference(groups.expand(2, -1), positions, 4, None)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert active.indexpool_ops.expand_groups(groups[:0], positions[:0], 4).shape == (
        0,
        32,
    )


def _written_page_tail(active, width, lengths):
    torch.manual_seed(width)
    pages = width // 64
    backing = torch.full(
        (2 * pages, 64 * 132 + 256), 67, device="cuda", dtype=torch.uint8
    )
    cache = backing[:, 128 : 128 + 64 * 132].view(2 * pages, 64, 132)
    table = torch.randperm(2 * pages, device="cuda").int().reshape(2, pages)
    owners = torch.tensor([0] * lengths[0] + [1] * lengths[1] + [0], device="cuda")
    groups = torch.cat(
        (torch.arange(lengths[0]), torch.arange(lengths[1]), torch.tensor([0]))
    ).to("cuda")
    mask = torch.ones(groups.numel(), device="cuda", dtype=torch.bool)
    mask[-1] = False  # Duplicate group0 must not overwrite committed data.
    quant, scales = active.indexpool_ops.fwht_quantize(
        torch.randn(groups.numel(), 128, device="cuda").bfloat16()
    )
    quant_storage = torch.empty(
        groups.numel() + 1, 260, device="cuda", dtype=quant.dtype
    )
    quant_storage[1:, 3:131] = quant
    quant = quant_storage[1:, 3:131]  # Row stride and nonzero pointer offset.
    scale_storage = torch.empty(groups.numel() * 2 + 1, device="cuda")
    scale_storage[1::2] = scales
    scales = scale_storage[1::2]
    expected = backing.cpu()
    qbytes, sbytes = quant.view(torch.uint8).cpu(), scales.contiguous().view(
        torch.uint8
    ).cpu().reshape(-1, 4)
    for owner, group, enabled, key, scale in zip(
        owners.cpu(), groups.cpu(), mask.cpu(), qbytes, sbytes
    ):
        if enabled:
            page, row = int(table[int(owner), int(group) // 64]), int(group) % 64
            expected[page, 128 + row * 128 : 128 + (row + 1) * 128] = key
            expected[
                page, 128 + 64 * 128 + row * 4 : 128 + 64 * 128 + (row + 1) * 4
            ] = scale
    active.indexpool_ops.write_pool(cache, table, owners, groups, mask, quant, scales)
    assert torch.equal(backing.cpu(), expected)  # Includes outer-offset canaries.
    # Empty writes are supported and must not alter the cache.
    active.indexpool_ops.write_pool(
        cache, table, owners[:0], groups[:0], mask[:0], quant[:0], scales[:0]
    )
    assert torch.equal(backing.cpu(), expected)
    return cache, table, quant, scales


@pytest.mark.parametrize("width,lengths", [(128, (65, 3)), (2048, (2047, 1024))])
def test_page_tail_writer_stride_offset_exact(active, width, lengths):
    _written_page_tail(active, width, lengths)


@pytest.mark.parametrize("width,lengths", [(128, (65, 3)), (2048, (2047, 1024))])
def test_real_public_provider_reads_active_page_tail(active, width, lengths):
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device("cuda:0"))
    cache, table, quant, scales = _written_page_tail(active, width, lengths)
    # Call the production seam first: a missing installed public ABI must fail
    # with its original chained ImportError, never a legacy/fake replacement.
    scheduler, scorer = active.runtime.get_paged_mqa_provider()
    from hcu_adapter.bridges.vllm_paged_mqa import get_paged_mqa_provider

    assert (scheduler, scorer) == get_paged_mqa_provider()
    heads = 32
    query, query_scales = active.indexpool_ops.fwht_quantize(
        torch.randn(2 * heads, 128, device="cuda").bfloat16()
    )
    query = query.reshape(2, 1, heads, 128)
    weights = torch.randn(2, heads, device="cuda")
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)[:, None]
    print(
        "PAGED_MQA_PROVIDER="
        + json.dumps(
            [
                dict(
                    module=f.__module__, name=f.__name__, file=inspect.getsourcefile(f)
                )
                for f in (scheduler, scorer)
            ]
        ),
        flush=True,
    )
    actual = scorer(
        (query, None),
        cache.unsqueeze(2),
        weights * query_scales.reshape(2, heads),
        lens,
        table,
        scheduler(lens, 64, 2),
        max_model_len=width,
    )
    reference = torch.zeros_like(actual)
    keys = quant.float() * scales[:, None]
    queries = query[:, 0].float() * query_scales.reshape(2, heads, 1)
    start = 0
    for request, length in enumerate(lengths):
        scores = queries[request] @ keys[start : start + length].T
        reference[request, :length] = (scores.relu() * weights[request, :, None]).sum(0)
        assert torch.count_nonzero(actual[request, length:]) == 0
        start += length
    torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-4)
    print(
        "PAGE_TAIL_MAX_ABS_ERROR=" + str((actual - reference).abs().max().item()),
        flush=True,
    )


@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("three", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_prepare_groups_then_write_tails_exact(active, ratio, three, strided):
    contexts, counts, mapping = [3, 8, 6, 0, -2], [7, 0, 3, 3, 2], [4, 2, 0, 3, 1]
    starts_cpu = torch.tensor(
        [0] + list(torch.tensor(counts).cumsum(0).tolist()), dtype=torch.int32
    )
    lengths_cpu = torch.tensor(
        [c + n for c, n in zip(contexts, counts)], dtype=torch.int32
    )
    tokens, ring = sum(counts), 8
    torch.manual_seed(ratio + three)
    keys_storage = torch.randn(
        tokens + 2, 260 if strided else 128, device="cuda"
    ).bfloat16()
    keys = (
        keys_storage[1 : tokens + 1, 2:258:2]
        if strided
        else keys_storage[1 : tokens + 1]
    )
    xyz = torch.empty(3, tokens, dtype=torch.long)
    for request, (context, count) in enumerate(zip(contexts, counts)):
        start = int(starts_cpu[request])
        for axis in range(3):
            xyz[axis, start : start + count] = (
                torch.arange(context, context + count) + request * 100 + axis * 10000
            )
    positions = xyz.to("cuda") if three else xyz[0].to("cuda")
    if strided:
        storage = torch.zeros(
            (3, tokens * 2 + 4) if three else (tokens * 2 + 4,),
            device="cuda",
            dtype=torch.long,
        )
        storage[..., 2 : 2 + tokens * 2 : 2] = positions
        positions = storage[..., 2 : 2 + tokens * 2 : 2]
    tail_storage = torch.full((5, ring + 2, 139), -777.0, device="cuda")
    tail = tail_storage[:, 1 : ring + 1, 3:134]
    expected_storage = tail_storage.cpu()
    expected_tail = expected_storage[:, 1 : ring + 1, 3:134]
    for request, context in enumerate(contexts):
        for logical in range(max(0, context - ring), max(0, context)):
            old = (
                ((torch.arange(128) + request * 100 + logical * 3) / 16)
                .bfloat16()
                .float()
            )
            expected_tail[mapping[request], logical % ring, :128] = old
            expected_tail[mapping[request], logical % ring, 128:] = torch.tensor(
                [logical + request * 100 + axis * 10000 for axis in range(3)]
                if three
                else [logical + request * 100] * 3
            )
    tail_storage.copy_(expected_storage)
    table_storage = torch.full((5, 4), -1, device="cuda", dtype=torch.int32)
    table_storage[:, 1] = torch.tensor(mapping, device="cuda")
    table = table_storage[:, 1:2]
    starts, lengths = starts_cpu.to("cuda"), lengths_cpu.to("cuda")
    prepared = active.indexpool_prepare.prepare_groups(
        keys, positions, tail, table, starts, lengths, tokens, ratio
    )
    (
        raw,
        coordinates,
        requests,
        groups,
        group_live,
        owner,
        logical,
        upper,
        token_live,
    ) = [t.cpu() for t in prepared]
    key_ref = keys.cpu()
    expected_groups = []
    for request, (context, count) in enumerate(zip(contexts, counts)):
        if count > 0 and context >= 0:
            for group in range(context // ratio, (context + count) // ratio):
                expected_groups.append((request, group))
    assert group_live.tolist() == [True] * len(expected_groups) + [False] * (
        group_live.numel() - len(expected_groups)
    )
    for slot, (request, group) in enumerate(expected_groups):
        assert (int(requests[slot]), int(groups[slot])) == (request, group)
        first = group * ratio
        for member in range(ratio):
            pos = first + member
            expected = (
                expected_tail[mapping[request], pos % ring, :128]
                if pos < contexts[request]
                else key_ref[int(starts_cpu[request]) + pos - contexts[request]]
            )
            torch.testing.assert_close(
                raw[slot, member], expected.to(raw.dtype), rtol=0, atol=0
            )
        expected_xyz = torch.tensor(
            [first + request * 100 + axis * 10000 for axis in range(3)]
            if three
            else [first + request * 100]
        )
        torch.testing.assert_close(
            coordinates[:, slot] if three else coordinates[slot : slot + 1],
            expected_xyz,
            rtol=0,
            atol=0,
        )
    assert torch.count_nonzero(raw[~group_live]) == 0
    assert (
        torch.count_nonzero(
            coordinates[:, ~group_live] if three else coordinates[~group_live]
        )
        == 0
    )
    for request, (context, count) in enumerate(zip(contexts, counts)):
        start = int(starts_cpu[request])
        torch.testing.assert_close(
            owner[start : start + count], torch.full((count,), request), rtol=0, atol=0
        )
        pos = torch.arange(max(context, 0), max(context, 0) + count)
        torch.testing.assert_close(logical[start : start + count], pos, rtol=0, atol=0)
        torch.testing.assert_close(
            upper[start : start + count].long(),
            pos // ratio if context >= 0 else torch.zeros_like(pos),
            rtol=0,
            atol=0,
        )
        assert token_live[start : start + count].tolist() == [context >= 0] * count
        if context >= 0:
            for pos in range(max(context, context + count - ring), context + count):
                source = start + pos - context
                expected_tail[mapping[request], pos % ring, :128] = key_ref[
                    source
                ].float()
                expected_tail[mapping[request], pos % ring, 128:] = (
                    xyz[:, source] if three else xyz[0, source].repeat(3)
                )
    active.indexpool_prepare.write_tails(keys, positions, tail, table, starts, lengths)
    torch.testing.assert_close(tail_storage.cpu(), expected_storage, rtol=0, atol=0)
    # Gathered outputs own their storage even after new keys replace tail slots.
    assert torch.equal(prepared[0].cpu(), raw)


def test_prepare_empty_batch(active):
    keys = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.empty(0, device="cuda", dtype=torch.long)
    tail = torch.empty(0, 8, 131, device="cuda")
    table = torch.empty(0, 1, device="cuda", dtype=torch.int32)
    starts = torch.zeros(1, device="cuda", dtype=torch.int32)
    lengths = starts[:0]
    result = active.indexpool_prepare.prepare_groups(
        keys, positions, tail, table, starts, lengths, 0, 4
    )
    assert result[0].shape == (0, 4, 128)
    assert all(t.numel() == 0 for t in result)
    active.indexpool_prepare.write_tails(keys, positions, tail, table, starts, lengths)


def test_fwht_expand_changing_input_hip_graph(active):
    x = torch.randn(32, 128, device="cuda").bfloat16()
    groups = torch.arange(511, device="cuda", dtype=torch.int32)[None].repeat(2, 1)
    positions = torch.tensor([4095, 8191], device="cuda")
    mask = torch.ones(2, device="cuda", dtype=torch.bool)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            active.indexpool_ops.fwht_quantize(x)
            active.indexpool_ops.expand_groups(groups, positions, 4, mask)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        quant, scales = active.indexpool_ops.fwht_quantize(x)
        indices = active.indexpool_ops.expand_groups(groups, positions, 4, mask)
    for step in range(3):
        x.copy_(torch.randn_like(x) * 2**step)
        groups[:, ::3] = -step - 1
        positions.add_(1)
        mask[1] = step != 1
        graph.replay()
        q, s = _fwht_reference(x.cpu())
        torch.testing.assert_close(scales.cpu(), s, rtol=0, atol=0)
        assert torch.equal(quant.view(torch.uint8).cpu(), q.view(torch.uint8))
        torch.testing.assert_close(
            indices.cpu(), _expand_reference(groups, positions, 4, mask), rtol=0, atol=0
        )
