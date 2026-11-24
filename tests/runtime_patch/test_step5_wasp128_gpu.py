# SPDX-License-Identifier: Apache-2.0
"""Real nightly GEMMs for EP16 expert counts; run in a gfx938 environment."""
import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="gfx938 GPU required")


def _provider():
    path = Path(__file__).parents[2] / "vllm_hcu/model_executor/layers/fused_moe/step5_wasp128.py"
    spec = importlib.util.spec_from_file_location("wasp128_gpu_provider", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("experts", [22, 23])
@pytest.mark.parametrize("k,n", [(4096, 3072), (1536, 4096)])
def test_nightly_wasp128_ep16_matches_fp32_and_graph(experts, k, n):
    if "gfx938" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("gfx938 required")
    mod = _provider()
    from deepgemm.group_gemm_weight_pack import pack_fp8_weight_pair32

    torch.manual_seed(42)
    device = "cuda"
    # Include empty experts and both 128- and 256-row boundaries.
    counts = [0 if i % 5 == 0 else 128 * (1 + i % 2) for i in range(experts)]
    ids = torch.repeat_interleave(torch.arange(experts, dtype=torch.int32, device=device),
                                  torch.tensor(counts, device=device))
    rows = sum(counts)
    x = (torch.randn(rows, k, device=device) * .25).to(torch.float8_e4m3fn)
    w = (torch.randn(experts, n, k, device=device) * .25).to(torch.float8_e4m3fn)
    scale_a = torch.full((rows, 1), .5, device=device)
    scale_b = torch.full((experts, n), .25, device=device)
    packed = pack_fp8_weight_pair32(w)
    out = torch.empty(rows, n, device=device, dtype=torch.bfloat16)
    ref = torch.empty_like(out)
    start = 0
    for expert, count in enumerate(counts):
        if count:
            ref[start:start+count] = ((x[start:start+count].float() @ w[expert].float().T) * .125).to(ref.dtype)
        start += count

    mod.grouped_fp8((x, scale_a), (packed, scale_b), out, ids)
    torch.cuda.synchronize()
    error = torch.linalg.vector_norm((out-ref).float()) / torch.linalg.vector_norm(ref.float())
    assert error.item() < .004, error.item()
    eager = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mod.grouped_fp8((x, scale_a), (packed, scale_b), out, ids)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


@pytest.mark.parametrize("k,n", [(4096, 3072), (1536, 4096)])
def test_same_nightly_masked_api_for_decode(k, n):
    from deepgemm import marlin_fp8_masked_weight, m_grouped_fp8_gemm_nt_masked

    torch.manual_seed(43)
    experts, capacity = 22, 64
    counts = torch.tensor([0 if e % 5 == 0 else 1 + e % 31 for e in range(experts)],
                          dtype=torch.int32, device="cuda")
    x = (torch.randn(experts, capacity, k, device="cuda") * .25).to(torch.float8_e4m3fn)
    w = (torch.randn(experts, n, k, device="cuda") * .25).to(torch.float8_e4m3fn)
    packed = marlin_fp8_masked_weight(w)
    a_scale = torch.full((experts, capacity), .5, device="cuda")
    b_scale = torch.full((experts, n), .25, device="cuda")
    out = torch.empty(experts, capacity, n, device="cuda", dtype=torch.bfloat16)
    m_grouped_fp8_gemm_nt_masked((x, a_scale), (packed, b_scale), out, counts, 16)
    torch.cuda.synchronize()
    # Masked padding is unspecified; compare only rows owned by real tokens.
    for expert, count in enumerate(counts.tolist()):
        if count:
            ref = ((x[expert, :count].float() @ w[expert].float().T) * .125).to(out.dtype)
            error = torch.linalg.vector_norm((out[expert, :count]-ref).float()) / torch.linalg.vector_norm(ref.float())
            assert error.item() < .004, (expert, error.item())
