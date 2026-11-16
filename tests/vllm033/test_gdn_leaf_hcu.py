# SPDX-License-Identifier: Apache-2.0
"""Opt-in complete FLA dataflow parity and changed-input Graph checks.

Run each provider selection in a fresh process. This component test does not
qualify a model, another architecture, or serving performance.
"""

import os
from importlib import import_module
from types import FunctionType, SimpleNamespace

import pytest
import torch

pytestmark = [
    pytest.mark.hcu,
    pytest.mark.skipif(
        os.environ.get("HCU_V033_TEST_GDN") != "1",
        reason="explicit HCU GDN native component run required",
    ),
]


@pytest.fixture(scope="module")
def runtime():
    assert torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info()
    budget_mib = int(os.environ.get("HCU_V033_TEST_GDN_ALLOCATOR_MIB", "512"))
    assert budget_mib >= 64
    assert free >= (budget_mib + 16) * 1024**2, (
        "insufficient free memory for the bounded native probe"
    )
    # Keep the full matrix while bounding this process's Torch allocator on
    # shared nodes. This does not reserve or release another process's memory.
    # Cold Triton autotuning alone needs a 256 MiB benchmark cache. A 64 MiB
    # budget can run decode only; it cannot qualify the complete prefill path.
    torch.cuda.set_per_process_memory_fraction(budget_mib * 1024**2 / total)
    from hcu_vllm_das import gdn
    from hcu_vllm_das import gdn_activation as activation

    chunk = import_module(activation.CHUNK)
    original = chunk.chunk_gated_delta_rule_fwd
    from hcu_vllm_das.gdn_contracts import OWNERS

    reference_globals = dict(original.__globals__)
    for name in gdn.PREFILL:
        reference_globals[name] = getattr(import_module(OWNERS[name]), name)
    # Test-only independent reference: identical official function code with
    # its ORIGINAL leaf aliases frozen before production activation. No copy
    # of this algorithm is shipped in the Plugin package.
    official = FunctionType(
        original.__code__,
        reference_globals,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    if os.environ.get("HCU_V033_TEST_GDN_PUBLIC_STARTUP") == "1":
        from hcu_adapter.bridges import vllm as adapter

        adapter.require_extension_healthy(activation.CAPABILITY)
        assert activation._published is not None
        assert any(
            report.entry_point_name == "das_hcu_gdn_fla"
            and report.status.value == "active"
            for report in adapter.extension_reports()
        )
    else:
        activation.activate(
            SimpleNamespace(
                vllm_version=import_module("vllm").__version__,
                requested_capabilities={activation.CAPABILITY},
            )
        )
    assert activation._published.policy[0] in {"hcu-boltops", "hcu-aiter"}
    assert activation._published.providers
    activation.check_health()
    return chunk.chunk_gated_delta_rule_fwd, official, gdn, activation


def _decode_reference(
    q, k, v, a, b, A_log, dt_bias, initial, indices, cu, accepted=None
):
    """CPU FP32 scalar-sequence recurrence, independent of either GPU kernel."""
    q, k, v, a, b, A_log, dt_bias, initial = (
        x.detach().cpu().float() for x in (q, k, v, a, b, A_log, dt_bias, initial)
    )
    indices = indices.detach().cpu()
    cu = cu.detach().cpu().tolist()
    accepted = accepted.detach().cpu().tolist() if accepted is not None else None
    final = initial.clone()
    output = torch.zeros(v.shape, dtype=torch.float32)
    ratio = v.shape[-2] // k.shape[-2]
    for sequence, (start, end) in enumerate(zip(cu, cu[1:])):
        slots = indices[sequence].reshape(-1)
        slot = int(slots[(accepted[sequence] - 1) if accepted is not None else 0])
        if slot <= 0:
            continue
        h = initial[slot].clone()
        for token in range(start, end):
            for head in range(v.shape[-2]):
                qi = q[0, token, head // ratio]
                ki = k[0, token, head // ratio]
                qi = qi / torch.sqrt((qi * qi).sum() + 1e-6)
                ki = ki / torch.sqrt((ki * ki).sum() + 1e-6)
                decay = torch.exp(
                    -torch.exp(A_log[head])
                    * torch.nn.functional.softplus(a[token, head] + dt_bias[head])
                )
                h[head].mul_(decay)
                delta = (v[0, token, head] - h[head] @ ki) * torch.sigmoid(
                    b[token, head]
                )
                h[head].add_(delta[:, None] * ki[None, :])
                output[0, token, head] = h[head] @ (qi * q.shape[-1] ** -0.5)
            write_slot = int(slots[token - start] if slots.numel() > 1 else slots[0])
            if write_slot > 0:
                final[write_slot].copy_(h)
    return output, final


def _assert_decode_close(actual, reference, *, state=False):
    actual, reference = actual.detach().float().cpu(), reference.detach().float().cpu()
    difference = actual - reference
    print(
        "GDN_DECODE_ERROR",
        "state" if state else "output",
        "max_abs=",
        difference.abs().max().item(),
        "rms=",
        difference.square().mean().sqrt().item(),
        flush=True,
    )
    torch.testing.assert_close(
        actual, reference, atol=5e-4 if state else 1e-4, rtol=1e-2
    )


@pytest.mark.parametrize("heads", [8, 12])
def test_packed_decode_null_slots_state_and_changed_input_graph(
    runtime, monkeypatch, heads
):
    _, _, gdn, activation = runtime
    from hcu_vllm_das.gdn_contracts import OWNERS

    name = "fused_recurrent_gated_delta_rule_packed_decode"
    official = getattr(import_module(OWNERS[name]), name)
    candidate = getattr(import_module(activation.QWEN), name)
    batch, key_heads, dim = 4, 4, 128
    mixed = torch.empty(
        (batch, (key_heads * 2 + heads) * dim), device="cuda", dtype=torch.bfloat16
    )
    a = torch.empty((batch, heads), device="cuda", dtype=mixed.dtype)
    b = torch.empty_like(a)
    A_log = torch.empty(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.empty(heads, device="cuda", dtype=mixed.dtype)
    initial = torch.empty((5, heads, dim, dim), device="cuda", dtype=torch.float32)
    state, ref_state = torch.empty_like(initial), torch.empty_like(initial)
    out = torch.empty((batch, 1, heads, dim), device="cuda", dtype=mixed.dtype)
    ref_out = torch.empty_like(out)
    indices = torch.tensor([0, 1, 3, -1], device="cuda", dtype=torch.int32)
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    seen = []
    call = gdn._call

    def record(op, provider, kernel, arguments):
        seen.append((op, provider))
        return call(op, provider, kernel, arguments)

    monkeypatch.setattr(gdn, "_call", record)

    def change(step):
        mixed.normal_(std=0.2)
        a.normal_(std=0.2)
        b.normal_(std=0.2)
        A_log.uniform_(-1.2, -0.5)
        dt_bias.normal_(std=0.1)
        initial.normal_(std=0.02)
        indices.copy_(
            torch.tensor(
                ([0, 1, 3, -1], [1, 0, 3, -1], [2, 1, 0, -1])[step % 3],
                device="cuda",
                dtype=indices.dtype,
            )
        )
        state.copy_(initial)
        ref_state.copy_(initial)

    def run(function, target_state, target_out):
        return function(
            mixed,
            a,
            b,
            A_log,
            dt_bias,
            dim**-0.5,
            target_state,
            target_out,
            indices,
            True,
        )

    def compare(result):
        official_result = run(official, ref_state, ref_out)
        q, k, v = torch.split(
            mixed, [key_heads * dim, key_heads * dim, heads * dim], dim=-1
        )
        independent_out, independent_state = _decode_reference(
            q.view(1, batch, key_heads, dim),
            k.view(1, batch, key_heads, dim),
            v.view(1, batch, heads, dim),
            a,
            b,
            A_log,
            dt_bias,
            initial,
            indices,
            cu,
        )
        _assert_decode_close(result[0], official_result[0])
        _assert_decode_close(
            out.reshape(1, batch, heads, dim).float().cpu(),
            independent_out,
        )
        _assert_decode_close(state, ref_state, state=True)
        _assert_decode_close(state, independent_state, state=True)
        assert torch.count_nonzero(out[indices <= 0]) == 0
        assert result[0].data_ptr() == out.data_ptr()
        assert result[1].data_ptr() == state.data_ptr()
        torch.testing.assert_close(state[0], initial[0], atol=0, rtol=0)

    change(0)
    result = run(candidate, state, out)
    compare(result)
    assert (name, activation._published.policy[1]) in seen
    for _ in range(2):
        state.copy_(initial)
        run(candidate, state, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(candidate, state, out)
    addresses = tuple(
        x.data_ptr() for x in (mixed, a, b, A_log, dt_bias, state, out, indices)
    )
    for step in range(3):
        change(step)
        graph.replay()
        compare(captured)
        assert addresses == tuple(
            x.data_ptr() for x in (mixed, a, b, A_log, dt_bias, state, out, indices)
        )


@pytest.mark.parametrize("heads", [8, 12])
def test_sigmoid_spec_varlen_accepted_tokens_and_changed_input_graph(
    runtime, monkeypatch, heads
):
    _, _, gdn, activation = runtime
    from hcu_vllm_das.gdn_contracts import OWNERS

    name = "fused_sigmoid_gating_delta_rule_update"
    official = getattr(import_module(OWNERS[name]), name)
    candidate = getattr(import_module(activation.QWEN), name)
    key_heads, dim, tokens = 4, 128, 6
    q = torch.empty((1, tokens, key_heads, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty((1, tokens, heads, dim), device="cuda", dtype=q.dtype)
    a = torch.empty((tokens, heads), device="cuda", dtype=q.dtype)
    b = torch.empty_like(a)
    A_log = torch.empty(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.empty(heads, device="cuda", dtype=q.dtype)
    initial = torch.empty((7, heads, dim, dim), device="cuda", dtype=torch.float32)
    state, ref_state = torch.empty_like(initial), torch.empty_like(initial)
    indices = torch.tensor(
        [[1, 2, 3], [4, 5, 0], [0, 0, 0]], device="cuda", dtype=torch.int32
    )
    accepted = torch.tensor([2, 1, 1], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, 3, 5, 6], device="cuda", dtype=torch.int32)
    seen = []
    call = gdn._call

    def record(op, provider, kernel, arguments):
        seen.append((op, provider))
        return call(op, provider, kernel, arguments)

    monkeypatch.setattr(gdn, "_call", record)

    def change(step):
        q.normal_(std=0.2)
        k.normal_(std=0.2)
        v.normal_(std=0.2)
        a.normal_(std=0.2)
        b.normal_(std=0.2)
        A_log.uniform_(-1.2, -0.5)
        dt_bias.normal_(std=0.1)
        initial.normal_(std=0.02)
        accepted.copy_(
            torch.tensor(
                [1 + step % 3, 1 + step % 2, 1], device="cuda", dtype=torch.int32
            )
        )
        state.copy_(initial)
        ref_state.copy_(initial)

    def run(function, target):
        return function(
            A_log,
            a,
            b,
            dt_bias,
            q,
            k,
            v,
            initial_state=target,
            cu_seqlens=cu,
            ssm_state_indices=indices,
            num_accepted_tokens=accepted,
            use_qk_l2norm_in_kernel=True,
        )

    def compare(result):
        reference = run(official, ref_state)
        independent_out, independent_state = _decode_reference(
            q, k, v, a, b, A_log, dt_bias, initial, indices, cu, accepted
        )
        # Official sigmoid returns without writing padded rows; their output
        # bytes are unspecified, unlike packed decode's explicit zeros.
        _assert_decode_close(result[0][:, :5], reference[0][:, :5])
        _assert_decode_close(result[0][:, :5], independent_out[:, :5])
        _assert_decode_close(state, ref_state, state=True)
        _assert_decode_close(state, independent_state, state=True)
        assert result[1].data_ptr() == state.data_ptr()
        torch.testing.assert_close(state[0], initial[0], atol=0, rtol=0)
        torch.testing.assert_close(state[6], initial[6], atol=0, rtol=0)

    change(0)
    result = run(candidate, state)
    compare(result)
    assert (name, activation._published.policy[1]) in seen
    for _ in range(2):
        state.copy_(initial)
        run(candidate, state)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(candidate, state)
    addresses = tuple(
        x.data_ptr() for x in (q, k, v, a, b, A_log, dt_bias, state, accepted, indices)
    )
    for step in range(3):
        change(step)
        graph.replay()
        compare(captured)
        assert addresses == tuple(
            x.data_ptr()
            for x in (q, k, v, a, b, A_log, dt_bias, state, accepted, indices)
        )


@pytest.mark.parametrize("heads", [8, 12])
@pytest.mark.parametrize("varlen", [False, True])
def test_complete_gated_fla_state_output_and_changed_input_graph(
    runtime, monkeypatch, heads, varlen
):
    candidate, official, gdn, activation = runtime
    from vllm.third_party.flash_linear_attention.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )

    tokens = 97
    device = "cuda"
    q = torch.empty((1, tokens, 4, 128), device=device, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty((1, tokens, heads, 128), device=device, dtype=q.dtype)
    g = torch.empty((1, tokens, heads), device=device, dtype=torch.float32)
    beta = torch.empty_like(g)
    cu = (
        torch.tensor([0, 17, 65, 97], dtype=torch.int32, device=device)
        if varlen
        else None
    )
    indices = prepare_chunk_indices(cu, 64) if varlen else None
    offsets = prepare_chunk_offsets(cu, 64) if varlen else None
    state = torch.empty(
        (3 if varlen else 1, heads, 128, 128), device=device, dtype=torch.float32
    )
    out = torch.full((v.numel() + 23,), -99.0, dtype=v.dtype, device=device)
    ref_out = torch.empty_like(out)
    inputs = (q, k, v, g, beta, 128**-0.5, state, True, cu, indices, offsets)
    seen = []
    call = gdn._call

    def record(name, provider, kernel, arguments):
        seen.append((name, provider))
        return call(name, provider, kernel, arguments)

    monkeypatch.setattr(gdn, "_call", record)

    def change():
        q.copy_(torch.nn.functional.normalize(torch.randn_like(q), dim=-1))
        k.copy_(torch.nn.functional.normalize(torch.randn_like(k), dim=-1))
        v.normal_(std=0.2)
        g.uniform_(-0.15, -0.01)
        beta.uniform_(0.05, 0.4)
        state.normal_(std=0.01)

    def compare(result):
        reference = official(*inputs, core_attn_out=ref_out)
        torch.testing.assert_close(result[1], reference[1], atol=1e-4, rtol=0.02)
        torch.testing.assert_close(result[3], reference[3], atol=0.002, rtol=0.02)
        assert result[1].data_ptr() == out.data_ptr()
        assert torch.all(out[v.numel() :] == -99.0)

    change()
    for _ in range(3):
        result = candidate(*inputs, core_attn_out=out)
    compare(result)
    assert ("recompute_w_u_fwd", "hcu-boltops") in seen
    assert any(
        name == "chunk_fwd_o" and provider == "hcu-boltops" for name, provider in seen
    )
    assert not any(
        name == "chunk_fwd_o" and provider == "hcu-aiter" for name, provider in seen
    )
    if activation._published.policy[0] == "hcu-aiter":
        assert ("chunk_gated_delta_rule_fwd_h", "hcu-aiter") in seen
    elif heads == 8:
        assert ("chunk_gated_delta_rule_fwd_h", "hcu-boltops") in seen
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = candidate(*inputs, core_attn_out=out)
    addresses = tuple(x.data_ptr() for x in (q, k, v, g, beta, state, out))
    for _ in range(3):
        change()
        graph.replay()
        compare(captured)
        assert addresses == tuple(x.data_ptr() for x in (q, k, v, g, beta, state, out))
    activation.check_health()
