# SPDX-License-Identifier: Apache-2.0
"""Native correctness and graph tests for fused AITER CK GDN prefill."""

import pytest
import torch
import torch.nn.functional as F

from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

pytestmark = pytest.mark.hcu


def _inputs(lengths, hq=2, hv=8, seed=7, nonzero_state=True):
    torch.manual_seed(seed)
    total = sum(lengths)
    q = F.normalize(torch.randn(1, total, hq, 128, device="cuda").float(), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn_like(q).float(), dim=-1).to(torch.bfloat16)
    v = torch.randn(1, total, hv, 128, device="cuda", dtype=torch.bfloat16)
    g = -0.01 - 0.02 * torch.rand(1, total, hv, device="cuda")
    beta = torch.sigmoid(torch.randn(1, total, hv, device="cuda"))
    state = torch.zeros(len(lengths), hv, 128, 128, device="cuda")
    if nonzero_state:
        state.normal_(0, 0.01)
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), device="cuda", dtype=torch.int64)
    return q, k, v, g, beta, state, cu


@pytest.fixture(scope="module")
def methods():
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as qwen
    from vllm_hcu.patch.worker.op_opt import patch_gdn_ck_prefill

    reset_workspace_manager()
    init_workspace_manager(torch.device("cuda"))
    # These are explicit native characterization tests. Production routing is
    # fail-closed until AITER fixes its legal strong-decay NaN behavior.
    patch_gdn_ck_prefill.aiter_ck_runtime_safety_reason = lambda: ""
    patch_gdn_ck_prefill.apply_to_module(qwen)
    wrapped = qwen.ChunkGatedDeltaRule.forward_native
    original = qwen.ChunkGatedDeltaRule._vllm_hcu_original_forward_native
    return wrapped, original


def _run(method, data, output_final_state=True):
    q, k, v, g, beta, state, cu = data
    return method(
        object(), q, k, v, g, beta, state, output_final_state,
        cu_seqlens=cu, use_qk_l2norm_in_kernel=False,
    )


def _assert_close(got, ref):
    for actual, expected in zip(got, ref):
        if actual is None or expected is None:
            assert actual is expected
            continue
        assert torch.isfinite(actual).all()
        delta = (actual.float() - expected.float()).abs()
        nrmse = delta.square().mean().sqrt() / expected.float().square().mean().sqrt().clamp_min(1e-8)
        assert delta.max().item() <= 0.01
        assert nrmse.item() <= 0.01


@pytest.mark.parametrize("length", [1, 16, 63, 64, 65, 127, 128, 256, 512, 1024, 2048, 4096])
def test_ck_prefill_matches_official_across_lengths(methods, length):
    wrapped, original = methods
    data = _inputs([length])
    ref = _run(original, data)
    got = _run(wrapped, data)
    torch.cuda.synchronize()
    _assert_close(got, ref)


@pytest.mark.parametrize(("hq", "hv"), [(4, 4), (4, 8), (2, 8)])
def test_ck_prefill_matches_official_supported_head_ratios(methods, hq, hv):
    wrapped, original = methods
    data = _inputs([65], hq=hq, hv=hv)
    _assert_close(_run(wrapped, data), _run(original, data))


@pytest.mark.parametrize("lengths", [[1, 63], [17, 48], [64, 65], [97, 511, 450]])
def test_ck_prefill_matches_official_packed_varlen_eager(methods, lengths):
    wrapped, original = methods
    data = _inputs(lengths)
    _assert_close(_run(wrapped, data), _run(original, data))


def test_ck_prefill_fixed_graph_replay_matches_eager(methods):
    wrapped, _ = methods
    data = _inputs([65], seed=7)
    for _ in range(3):
        _run(wrapped, data)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = _run(wrapped, data)
    graph.replay()
    torch.cuda.synchronize()
    old = graph_out[0].clone()
    changed = _inputs([65], seed=17)
    for dst, src in zip(data[:6], changed[:6]):
        dst.copy_(src)
    graph.replay()
    torch.cuda.synchronize()
    replay = tuple(x.clone() if x is not None else None for x in graph_out)
    eager = _run(wrapped, changed)
    assert not torch.equal(old, replay[0])
    for actual, expected in zip(replay, eager):
        assert torch.equal(actual, expected)


def test_ck_prefill_ratio_three_uses_original(methods):
    wrapped, original = methods
    data = _inputs([65], hq=2, hv=6)
    _assert_close(_run(wrapped, data), _run(original, data))


def test_ck_prefill_fixed_int32_metadata_matches_model_contract(methods):
    wrapped, original = methods
    data = _inputs([65])
    data = (*data[:-1], data[-1].to(torch.int32))
    _assert_close(_run(wrapped, data), _run(original, data))


@pytest.mark.parametrize("nonzero_state", [False, True])
@pytest.mark.parametrize("output_final_state", [False, True])
def test_ck_prefill_state_and_final_state_semantics(
    methods, nonzero_state, output_final_state
):
    wrapped, original = methods
    data = _inputs([65], nonzero_state=nonzero_state)
    got = _run(wrapped, data, output_final_state=output_final_state)
    ref = _run(original, data, output_final_state=output_final_state)
    assert (got[1] is None) is (not output_final_state)
    _assert_close(got, ref)


def test_ck_prefill_packed_varlen_graph_uses_original(methods, caplog):
    wrapped, original = methods
    data = _inputs([17, 48])
    from vllm.third_party.flash_linear_attention.ops.index import (
        prepare_chunk_indices,
    )

    cu = data[-1]
    chunk_indices = prepare_chunk_indices(cu, 64)
    chunk_offsets = torch.tensor([0, 1], device="cuda", dtype=torch.int64)

    def run(method):
        q, k, v, g, beta, state, _ = data
        return method(
            object(), q, k, v, g, beta, state, True,
            cu_seqlens=cu, chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets, use_qk_l2norm_in_kernel=False,
        )

    for _ in range(3):
        run(wrapped)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run(wrapped)
    graph.replay()
    torch.cuda.synchronize()
    eager_original = run(original)
    _assert_close(graph_out, eager_original)
    assert "packed variable-length metadata is not graph-safe" in caplog.text


@pytest.mark.parametrize(
    ("decay", "expected_finite"),
    [(-1.0, True), (-2.0, False), (-5.0, False), (-10.0, False)],
)
def test_native_ck_prefill_documents_strong_decay_boundary(
    decay, expected_finite
):
    """Preserve the AITER 0.1.7 failure that keeps production fail-closed."""
    import aiter

    q, k, v, _, beta, state, _ = _inputs([128])
    g = torch.full_like(beta[0], decay)
    flags = {
        "use_g": True, "use_gk": False, "use_exp2": False,
        "transpose_state": True, "qk_l2norm": False,
        "gate_in_kernel": False, "has_dt_bias": False,
        "beta_sigmoid_in_kernel": False, "output_final_state": True,
        "cp_context": False, "cp_world_size": 2,
    }
    workspace = aiter.ck_gdn_prefill_alloc_workspace(
        q[0], v[0], None, **flags
    )
    output, final_state = aiter.ck_gdn_prefill(
        q[0], k[0], v[0], g, beta[0], workspace,
        initial_state=state, scale=128**-0.5, **flags, cp_rank=1,
    )
    torch.cuda.synchronize()

    assert bool(torch.isfinite(output).all()) is expected_finite
    assert bool(torch.isfinite(final_state).all()) is expected_finite
