# SPDX-License-Identifier: Apache-2.0
"""Native cache-reuse regression; run with one assigned gfx938 device.

The break caught is compiling another gather/query binary for each token count
while batch, dtype, ratio and tensor strides are unchanged. The subprocess owns
a fresh on-disk and in-memory cache. No platform or numerical operator is mocked.
"""

import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.hcu


def _production():
    import torch
    from hcu_adapter.bridges.vllm import hcu_runtime_plugin

    hcu_runtime_plugin()
    from vllm.platforms import current_platform
    from hcu_vllm_das.models.hy_v4_flash import indexpool_prepare

    assert torch.version.hip and torch.cuda.is_available()
    assert torch.cuda.device_count() == 1, "Assign exactly one logical HCU device"
    torch.cuda.set_device(0)
    assert type(current_platform).__module__ == "hcu_adapter.bridges.vllm"
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx938"
    return torch, indexpool_prepare


def _case(torch, tokens, ratio, three, strided, batch=16):
    # Fixed backing allocations keep every required stride and input alignment
    # independent of TOKENS. Unused capacity never participates in the oracle.
    capacity, ring = 8224, 32
    counts = [0] * batch
    participants = [r for r in range(batch) if r % 4 != 3]
    for token in range(tokens):
        counts[participants[token % len(participants)]] += 1
    contexts = [r * 1048576 + ratio - 1 for r in range(batch)]
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    lengths = [context + count for context, count in zip(contexts, counts)]
    mapping = list(reversed(range(batch)))
    width = 260 if strided else 128
    backing = torch.zeros(capacity, width, dtype=torch.bfloat16)
    keys = backing[1:tokens + 1, 2:258:2] if strided else backing[1:tokens + 1]
    for r, count in enumerate(counts):
        for local in range(count):
            logical = contexts[r] + local
            keys[starts[r] + local] = (
                (torch.arange(128) + r * 13 + logical % 4096) / 64
            ).bfloat16()
    position_backing = torch.zeros((3, capacity * 2), dtype=torch.int64)
    positions = position_backing[:, 2:2 + tokens * 2:2]
    for r, count in enumerate(counts):
        for axis in range(3):
            positions[axis, starts[r]:starts[r + 1]] = (
                torch.arange(contexts[r], lengths[r]) + axis * 100000
            )
    if not three:
        positions = positions[0]
    tail_storage = torch.full((batch, ring + 2, 139), -777.0)
    tail = tail_storage[:, 1:ring + 1, 3:134]
    for r, context in enumerate(contexts):
        for logical in range(max(0, context - ring), context):
            row = tail[mapping[r], logical % ring]
            row[:128] = (
                (torch.arange(128) + r * 13 + logical % 4096) / 64
            ).bfloat16().float()
            row[128:] = torch.tensor(
                [logical + a * 100000 for a in range(3)] if three else [logical] * 3
            )
    table_storage = torch.full((batch, 4), -1, dtype=torch.int32)
    table_storage[:, 1] = torch.tensor(mapping, dtype=torch.int32)
    table = table_storage[:, 1:2]
    # Move backing storage, then recreate views; .to on the views would erase
    # the very strides whose reuse this test is intended to hold fixed.
    kb, pb, tb, mb = (x.to("cuda") for x in (
        backing, position_backing, tail_storage, table_storage
    ))
    gpu_keys = kb[1:tokens + 1, 2:258:2] if strided else kb[1:tokens + 1]
    gpu_positions = pb[:, 2:2 + tokens * 2:2]
    if not three:
        gpu_positions = gpu_positions[0]
    inputs = (
        gpu_keys, gpu_positions, tb[:, 1:ring + 1, 3:134], mb[:, 1:2],
        torch.tensor(starts, dtype=torch.int32, device="cuda"),
        torch.tensor(lengths, dtype=torch.int32, device="cuda"), tokens, ratio,
    )
    reference = dict(
        keys=keys, positions=positions, tail=tail, starts=starts,
        contexts=contexts, counts=counts, mapping=mapping, ring=ring,
        tail_storage=tail_storage,
    )
    return inputs, reference


def _check(torch, outputs, ref, ratio):
    raw, xyz, requests, groups, live, owner, logical, upper, token_live = (
        value.cpu() for value in outputs
    )
    expected_groups = []
    for r, (context, count) in enumerate(zip(ref["contexts"], ref["counts"])):
        # Independent request/group enumeration, not the GPU prefix/binary search.
        for end in range(context + 1, context + count + 1):
            if end % ratio == 0:
                expected_groups.append((r, end // ratio - 1))
    assert live.tolist() == [True] * len(expected_groups) + [False] * (
        live.numel() - len(expected_groups)
    )
    for slot, (r, group) in enumerate(expected_groups):
        assert (int(requests[slot]), int(groups[slot])) == (r, group)
        first = group * ratio
        for member in range(ratio):
            position = first + member
            expected = (
                ref["tail"][ref["mapping"][r], position % ref["ring"], :128]
                if position < ref["contexts"][r] else
                ref["keys"][ref["starts"][r] + position - ref["contexts"][r]]
            )
            torch.testing.assert_close(raw[slot, member], expected.to(raw.dtype),
                                       rtol=0, atol=0)
        if first < ref["contexts"][r]:
            expected_xyz = ref["tail"][ref["mapping"][r], first % ref["ring"], 128:].long()
        else:
            offset = ref["starts"][r] + first - ref["contexts"][r]
            expected_xyz = ref["positions"][..., offset].reshape(-1)
        if xyz.ndim == 1:
            expected_xyz = expected_xyz[:1]
        actual_xyz = xyz[:, slot] if xyz.ndim == 2 else xyz[slot:slot + 1]
        torch.testing.assert_close(actual_xyz, expected_xyz, rtol=0, atol=0)
    assert not torch.count_nonzero(raw[~live])
    assert not torch.count_nonzero(xyz[:, ~live] if xyz.ndim == 2 else xyz[~live])
    for r, count in enumerate(ref["counts"]):
        section = slice(ref["starts"][r], ref["starts"][r + 1])
        positions = torch.arange(ref["contexts"][r], ref["contexts"][r] + count)
        assert owner[section].tolist() == [r] * count
        assert logical[section].tolist() == positions.tolist()
        assert upper[section].tolist() == (positions // ratio).tolist()
        assert token_live[section].tolist() == [True] * count


def _check_tail_write(torch, module, inputs, ref, outputs):
    before_outputs = [value.clone() for value in outputs]
    expected = ref["tail_storage"].clone()
    expected_tail = expected[:, 1:ref["ring"] + 1, 3:134]
    for r, count in enumerate(ref["counts"]):
        context = ref["contexts"][r]
        for logical in range(max(context, context + count - ref["ring"]), context + count):
            offset = ref["starts"][r] + logical - context
            target = expected_tail[ref["mapping"][r], logical % ref["ring"]]
            target[:128] = ref["keys"][offset].float()
            position = ref["positions"][..., offset]
            target[128:] = position if position.ndim else position.repeat(3)
    module.write_tails(*inputs[:6])
    torch.cuda.synchronize()
    # Check padding as well as valid rows; it belongs to neighboring state.
    actual_storage = inputs[2]._base
    assert actual_storage is not None and actual_storage.shape == expected.shape
    torch.testing.assert_close(actual_storage.cpu(), expected, rtol=0, atol=0)
    for actual, previous in zip(outputs, before_outputs):
        assert torch.equal(actual, previous), "Tail writes aliased prepared outputs"


@pytest.mark.parametrize("ratio", [2, 4, 8, 16])
@pytest.mark.parametrize("three,strided", [(False, False), (True, True)])
def test_partial_carry_and_query_bounds_are_exact(ratio, three, strided):
    torch, module = _production()
    for tokens in (0, 1, 6, 63, 64, 65, 192, 224, 230, 384, 433, 463, 768, 8192):
        inputs, ref = _case(torch, tokens, ratio, three, strided)
        outputs = module.prepare_groups(*inputs)
        torch.cuda.synchronize()
        _check(torch, outputs, ref, ratio)
        _check_tail_write(torch, module, inputs, ref, outputs)


def test_empty_batch_has_no_prepared_or_retained_rows():
    torch, module = _production()
    inputs = (
        torch.empty(0, 128, dtype=torch.bfloat16, device="cuda"),
        torch.empty(0, dtype=torch.int64, device="cuda"),
        torch.empty(0, 32, 131, device="cuda"),
        torch.empty(0, 1, dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        torch.empty(0, dtype=torch.int32, device="cuda"), 0, 8,
    )
    outputs = module.prepare_groups(*inputs)
    assert outputs[0].shape == (0, 8, 128)
    assert len(outputs) == 9 and all(value.numel() == 0 for value in outputs)
    module.write_tails(*inputs[:6])
    torch.cuda.synchronize()


@pytest.mark.parametrize("three", [False, True])
def test_captured_counts_and_private_pools_retain_outputs(three):
    torch, module = _production()
    captured = []
    for tokens in (224, 463):
        inputs, ref = _case(torch, tokens, 8, three, True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                module.prepare_groups(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph, pool = torch.cuda.CUDAGraph(), torch.cuda.graph_pool_handle()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            outputs = module.prepare_groups(*inputs)
        graph.replay()
        torch.cuda.synchronize()
        _check(torch, outputs, ref, 8)
        captured.append((graph, pool, inputs, ref, outputs))
    first, second = captured
    retained = [value.clone() for value in first[4]]
    for step in range(3):
        graph, _, inputs, ref, outputs = second
        # Change actual key/position bytes, retaining every captured argument and
        # required stride. Reusing a binary must not freeze either input values
        # or another graph's scalar bounds/output allocation.
        inputs[0].add_(1)
        ref["keys"].add_(1)
        inputs[1].add_(16)
        ref["positions"].add_(16)
        graph.replay()
        torch.cuda.synchronize()
        _check(torch, outputs, ref, 8)
        for actual, previous in zip(first[4], retained):
            assert torch.equal(actual, previous), "Independent pool output overwritten"
    first[0].replay()
    torch.cuda.synchronize()
    _check(torch, first[4], first[3], 8)


def _cache_child(report_path, three):
    torch, module = _production()
    cache = Path(os.environ["TRITON_CACHE_DIR"])
    assert cache.is_dir() and not list(cache.iterdir()), "Cache must start empty"
    rows = []
    # R4 matches the actual mixed-refill gather TTIR (4x128 copied groups).
    for tokens in (224, 230, 433, 463, 192, 384, 768, 8192, 64, 65):
        inputs, ref = _case(torch, tokens, 4, three, True)
        output = module.prepare_groups(*inputs)
        torch.cuda.synchronize()
        _check(torch, output, ref, 4)
        rows.append(dict(tokens=tokens, slots=output[0].shape[0],
                         key_stride=list(inputs[0].stride()),
                         position_stride=list(inputs[1].stride())))
    variants = {}
    for name in ("_gather_groups", "_query_metadata"):
        entries = []
        for path in sorted(cache.rglob(name + ".json")):
            metadata = json.loads(path.read_text())
            assert metadata["name"] == name
            ir = path.with_suffix(".ttir")
            assert ir.is_file()
            entries.append(dict(metadata=str(path), hash=metadata["hash"],
                                ttir_sha256=hashlib.sha256(ir.read_bytes()).hexdigest()))
        variants[name] = entries
    source = Path(inspect.getsourcefile(module))
    report = dict(cases=rows, source=str(source),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  variants=variants, numerical_cases_passed=len(rows),
                  claim="one binary per fixed batch/ratio/layout/stride signature")
    Path(report_path).write_text(json.dumps(report, indent=2))
    # This is the intended pre-fix RED, after all real numerical controls passed.
    assert all(len(entries) == 1 for entries in variants.values()), {
        name: len(entries) for name, entries in variants.items()
    }


@pytest.mark.parametrize("three", [False, True], ids=["1d", "3d-fixed-plane"])
def test_refill_token_counts_reuse_preparation_binaries(tmp_path, three):
    cache = tmp_path / "triton-cache"
    cache.mkdir()
    report = tmp_path / "cache-report.json"
    env = dict(os.environ, TRITON_CACHE_DIR=str(cache))
    run = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--cache-child",
         str(report), str(int(three))],
        env=env, capture_output=True, text=True, timeout=240,
    )
    (tmp_path / "child.stdout").write_text(run.stdout)
    (tmp_path / "child.stderr").write_text(run.stderr)
    assert run.returncode == 0, run.stderr[-6000:]
    result = json.loads(report.read_text())
    assert result["numerical_cases_passed"] == 10
    assert all(len(entries) == 1 for entries in result["variants"].values())


if __name__ == "__main__":
    assert len(sys.argv) == 4 and sys.argv[1] == "--cache-child"
    _cache_child(sys.argv[2], bool(int(sys.argv[3])))
