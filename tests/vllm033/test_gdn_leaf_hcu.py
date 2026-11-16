# SPDX-License-Identifier: Apache-2.0
"""Opt-in complete FLA dataflow parity and changed-input Graph checks.

Run each provider selection in a fresh process. This component test does not
qualify a model, decode, another architecture, or serving performance.
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
    free, _ = torch.cuda.mem_get_info()
    assert free >= 1024**3, "insufficient free memory for an isolated native probe"
    from hcu_vllm_das import gdn
    from hcu_vllm_das import gdn_activation as activation

    chunk = import_module(activation.CHUNK)
    original = chunk.chunk_gated_delta_rule_fwd
    # Test-only independent reference: identical official function code with
    # its ORIGINAL leaf aliases frozen before production activation. No copy
    # of this algorithm is shipped in the Plugin package.
    official = FunctionType(
        original.__code__,
        dict(original.__globals__),
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
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
        torch.testing.assert_close(result[1], reference[1], atol=0.02, rtol=0.02)
        torch.testing.assert_close(result[3], reference[3], atol=0.002, rtol=0.02)
        assert result[1].data_ptr() == out.data_ptr()
        assert torch.all(out[v.numel() :] == -99.0)

    change()
    for _ in range(3):
        result = candidate(*inputs, core_attn_out=out)
    compare(result)
    assert any(name == "recompute_w_u_fwd" for name, _ in seen)
    assert any(
        name == "chunk_fwd_o" and provider == "hcu-boltops" for name, provider in seen
    )
    assert not any(
        name == "chunk_fwd_o" and provider == "hcu-aiter" for name, provider in seen
    )
    if heads == 8:
        assert any(name == "chunk_gated_delta_rule_fwd_h" for name, _ in seen)
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
