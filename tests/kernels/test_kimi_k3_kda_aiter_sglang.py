# SPDX-License-Identifier: Apache-2.0
"""GPU-only KDA comparison with SGLang's AITER path; no inference service.

Run with SGLang source on PYTHONPATH and SGLANG_KDA_USE_HCU_OP=1,
SGLANG_KDA_DISABLE_TRITON_CHUNK_H=1, VLLM_PLUGINS=__disabled__.
"""
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers import kimi_k3_kda as plugin

pytestmark = pytest.mark.hcu


def _metadata(length, checkpoint):
    chunks = (length + 63) // 64
    return (torch.tensor([0, length], dtype=torch.int32, device="cuda"),
            torch.tensor([(0, i) for i in range(chunks)], dtype=torch.int32, device="cuda"),
            SimpleNamespace(nums_dict={plugin.CHECKPOINT_KEY: plugin.Checkpoints(
                torch.tensor([1], device="cuda"),
                torch.tensor([checkpoint // 64], device="cuda"), (), (checkpoint,))}))


@pytest.mark.parametrize("length,checkpoint", [(4096, 3072), (8175, 7680)])
@torch.inference_mode()
def test_aiter_checkpoint_and_resume_matches_sglang(monkeypatch, length, checkpoint):
    if not torch.cuda.is_available():
        pytest.skip("requires HCU")
    from sglang.kernels.ops.attention.fla import kda as reference
    from sglang.kernels.ops.attention.fla import chunk_delta_h as reference_state
    assert reference_state._USE_KDA_HCU_FROM_AITER, "SGLang AITER path must be enabled"
    monkeypatch.setenv("VLLM_KIMI_KDA_STATE_BACKEND", "aiter")
    torch.manual_seed(20261027)
    heads, dim = 2, 128
    shape = (1, length, heads, dim)
    tensors = {name: torch.randn(shape, device="cuda", dtype=torch.bfloat16)
               for name in ("q", "k", "v", "raw_g")}
    tensors["raw_beta"] = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)
    params = dict(A_log=torch.randn(heads, device="cuda"),
                  g_bias=torch.randn(heads * dim, device="cuda"), lower_bound=-5.0)
    initial = torch.randn((1, heads, dim, dim), device="cuda") * 0.1
    cu, ci, metadata = _metadata(length, checkpoint)
    cache = torch.zeros((2, heads, dim, dim), device="cuda", dtype=torch.float32)
    token = plugin.ACTIVE_LAYER.set((SimpleNamespace(kv_cache=(None, cache)), metadata))
    try:
        out, final = plugin.chunk_kda(**tensors, **params, initial_state=initial,
            cu_seqlens=cu, chunk_indices=ci, use_qk_l2norm_in_kernel=True)
    finally:
        plugin.ACTIVE_LAYER.reset(token)
    ref_state = initial.clone()
    ref_out, ref_history = reference.chunk_kda(
        q=tensors["q"], k=tensors["k"], v=tensors["v"].clone(),
        g=tensors["raw_g"], beta=tensors["raw_beta"].float().sigmoid().to(torch.bfloat16),
        initial_state=ref_state, initial_state_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
        cu_seqlens=cu, A_log=params["A_log"], dt_bias=params["g_bias"],
        lower_bound=params["lower_bound"], use_qk_l2norm_in_kernel=True,
        output_intermediate_states=True)
    print("reference history dtype", ref_history.dtype)
    for label, actual, expected in (("output", out, ref_out), ("final", final, ref_state)):
        error = (actual.float() - expected.float()).abs()
        print(label, "max_abs", error.max().item(), "rms", error.square().mean().sqrt().item())
    assert ref_history.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref_out.float(), atol=0.03, rtol=0.03)
    torch.testing.assert_close(final, ref_state, atol=0.03, rtol=0.03)
    plugin.finalize_checkpoints(metadata)
    assert torch.equal(cache[1], cache[1].to(torch.bfloat16).float())
    delta = (cache[1].float() - ref_history[0, checkpoint // 64].float()).abs()
    print("checkpoint", "max_abs", delta.max().item(), "rms", delta.square().mean().sqrt().item())
    torch.testing.assert_close(cache[1].float(), ref_history[0, checkpoint // 64].float(),
                               atol=0.03, rtol=0.03)
    tail = {name: value[:, checkpoint:].contiguous() for name, value in tensors.items()}
    tail_cu, tail_ci, _ = _metadata(length - checkpoint, 0)
    saved = cache[1:2].float()
    saved_before = saved.clone()
    resumed, resumed_final = plugin.chunk_kda(**tail, **params, initial_state=saved,
        cu_seqlens=tail_cu, chunk_indices=tail_ci, use_qk_l2norm_in_kernel=True)
    torch.testing.assert_close(saved, saved_before, atol=0, rtol=0)
    ref_resume_state = ref_history[0, checkpoint // 64:checkpoint // 64 + 1].float().clone()
    ref_resume = reference.chunk_kda(
        q=tail["q"], k=tail["k"], v=tail["v"].clone(), g=tail["raw_g"],
        beta=tail["raw_beta"].float().sigmoid().to(torch.bfloat16),
        initial_state=ref_resume_state, initial_state_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
        cu_seqlens=tail_cu, A_log=params["A_log"], dt_bias=params["g_bias"],
        lower_bound=params["lower_bound"], use_qk_l2norm_in_kernel=True)
    for label, actual, expected in (("resumed_output", resumed, ref_resume),
                                    ("resumed_final", resumed_final, ref_resume_state)):
        error = (actual.float() - expected.float()).abs()
        print(label, "max_abs", error.max().item(), "rms", error.square().mean().sqrt().item())
    torch.testing.assert_close(resumed.float(), ref_resume.float(), atol=0.03, rtol=0.03)
    torch.testing.assert_close(resumed_final, ref_resume_state, atol=0.03, rtol=0.03)


@torch.inference_mode()
def test_aiter_state_varlen_batch_matches_sglang_and_preserves_input():
    if not torch.cuda.is_available():
        pytest.skip("requires HCU")
    from sglang.kernels.ops.attention.fla import chunk_delta_h as reference
    assert reference._USE_KDA_HCU_FROM_AITER
    torch.manual_seed(407)
    lengths = [256, 511]
    shape = (1, sum(lengths), 2, 128)
    k, w = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.01
            for _ in range(2))
    u = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    gk = torch.full(shape, -0.01, device="cuda", dtype=torch.float32)
    cu = torch.tensor([0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32)
    ci = torch.tensor([(row, i) for row, length in enumerate(lengths)
                       for i in range((length + 63) // 64)], device="cuda", dtype=torch.int32)
    initial = torch.randn((2, 2, 128, 128), device="cuda") * 0.1
    before = initial.clone()
    h, vn, final = plugin._chunk_gated_delta_rule_fwd_h_aiter(
        k=k, w=w, u=u, gk=gk, initial_state=initial, cu_seqlens=cu, chunk_indices=ci)
    ref_state = before.clone()
    ref_h, ref_vn = reference.chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, gk=gk, initial_state=ref_state,
        initial_state_indices=torch.arange(2, device="cuda", dtype=torch.int32),
        cu_seqlens=cu, chunk_indices=ci, use_exp2=True)
    assert h.dtype == torch.bfloat16
    torch.testing.assert_close(initial, before, atol=0, rtol=0)
    torch.testing.assert_close(h, ref_h, atol=0, rtol=0)
    torch.testing.assert_close(vn, ref_vn, atol=0, rtol=0)
    torch.testing.assert_close(final, ref_state, atol=0, rtol=0)
