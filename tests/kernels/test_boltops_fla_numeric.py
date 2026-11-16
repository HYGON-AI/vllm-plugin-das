# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU numerical parity and routing for optional FLA providers."""

from __future__ import annotations

from types import ModuleType

import pytest
import torch


@pytest.mark.parametrize(
    ("provider", "package"),
    [("hcu-boltops", "boltops.fla.gdn"), ("hcu-aiter", "aiter.ops.fla")],
)
@pytest.mark.parametrize("lengths", [(64,), (32, 64)])
def test_hcu_chunk_pipeline_matches_official(
    monkeypatch: pytest.MonkeyPatch,
    lengths: tuple[int, ...],
    provider: str,
    package: str,
):
    if not torch.cuda.is_available():
        pytest.skip("HCU is required")
    pytest.importorskip(package)

    from vllm.third_party.flash_linear_attention.ops import chunk
    from vllm_hcu.patch.worker.op_opt import (
        patch_fla_chunk_delta_h,
        patch_fla_chunk_o,
        patch_fla_recompute_w_u,
    )
    from vllm_hcu.patch.worker.op_opt import _gdn_backend

    patch_fla_chunk_delta_h.apply_to_module(chunk)
    patch_fla_chunk_o.apply_to_module(chunk)
    patch_fla_recompute_w_u.apply_to_module(chunk)

    total_tokens = sum(lengths)
    torch.manual_seed(7)
    # K=V=128: the aiter HIP kernels only support this head size.
    q = torch.randn(
        1, total_tokens, 2, 128, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn_like(q) * 0.1
    g = torch.nn.functional.logsigmoid(
        torch.randn(1, total_tokens, 2, device="cuda", dtype=torch.float32)
    )
    beta = torch.sigmoid(
        torch.randn(1, total_tokens, 2, device="cuda", dtype=torch.float32)
    )
    if len(lengths) == 1:
        cu_seqlens = None
        initial_state = None
    else:
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.long)
        initial_state = torch.randn(
            len(lengths), 2, 128, 128, device="cuda", dtype=torch.float32
        ) * 0.01

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")

    def run(backend: str):
        monkeypatch.setattr(_gdn_backend, "_prefill_requested", backend)
        output_buffer = torch.empty(
            v.numel() + 16, device="cuda", dtype=v.dtype
        )
        output, final_state = chunk.chunk_gated_delta_rule(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            initial_state=(
                None if initial_state is None else initial_state.clone()
            ),
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            core_attn_out=output_buffer,
        )
        torch.cuda.synchronize()
        assert output.data_ptr() == output_buffer.data_ptr()
        return output.clone(), final_state.clone()

    expected_output, expected_state = run("triton")
    actual_output, actual_state = run(provider)
    torch.testing.assert_close(
        actual_output, expected_output, rtol=5e-2, atol=5e-3
    )
    torch.testing.assert_close(
        actual_state, expected_state, rtol=5e-2, atol=5e-3
    )


def test_chunk_h_uses_official_for_three_to_one_value_head_ratio(monkeypatch):
    from vllm_hcu.patch.worker.op_opt import _gdn_backend
    from vllm_hcu.patch.worker.op_opt import patch_fla_chunk_delta_h as adapter

    calls = []

    def original(
        k,
        w,
        u,
        g=None,
        gk=None,
        initial_state=None,
        output_final_state=False,
        chunk_size=64,
        save_new_value=True,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_offsets=None,
        use_exp2=False,
    ):
        calls.append("official")
        return k, u, initial_state

    def boltops_kernel(*args, **kwargs):
        calls.append("boltops")
        return args[0], args[2], kwargs["initial_state"]

    module = ModuleType(adapter.TARGET_MODULE)
    module.FLA_CHUNK_SIZE = 64
    module.chunk_gated_delta_rule_fwd_h = original
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setattr(_gdn_backend, "_prefill_requested", "hcu-boltops")
    monkeypatch.setattr(
        adapter,
        "make_boltops_gdn_resolver",
        lambda _name: lambda: boltops_kernel,
    )
    assert adapter.apply_to_module(module) is True

    k = torch.empty(1, 1, 4, 128, dtype=torch.bfloat16)
    module.chunk_gated_delta_rule_fwd_h(
        k, k, torch.empty(1, 1, 12, 128, dtype=torch.bfloat16)
    )
    assert calls == ["official"]

    module.chunk_gated_delta_rule_fwd_h(
        k, k, torch.empty(1, 1, 8, 128, dtype=torch.bfloat16)
    )
    assert calls == ["official", "boltops"]


@pytest.mark.parametrize("lengths", [(64,), (33, 64)])
def test_three_to_one_official_chunk_h_and_boltops_chunk_o(
    monkeypatch: pytest.MonkeyPatch,
    lengths: tuple[int, ...],
):
    if not torch.cuda.is_available():
        pytest.skip("HCU is required")
    pytest.importorskip("boltops.fla.gdn")

    from vllm.third_party.flash_linear_attention.ops import chunk
    from vllm_hcu.patch.worker.op_opt import (
        patch_fla_chunk_delta_h,
        patch_fla_chunk_o,
    )
    from vllm_hcu.patch.worker.op_opt import _gdn_backend

    patch_fla_chunk_delta_h.apply_to_module(chunk)
    patch_fla_chunk_o.apply_to_module(chunk)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")

    total_tokens = sum(lengths)
    torch.manual_seed(17)
    q = torch.randn(
        1, total_tokens, 4, 128, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    k = torch.zeros_like(q)
    u = torch.zeros(
        1, total_tokens, 12, 128, device="cuda", dtype=torch.bfloat16
    )
    # w carries the 12 value heads (kernels index it with H, not Hg); a
    # 4-head w is read out of bounds.
    w = torch.zeros_like(u)
    initial_state = torch.randn(
        len(lengths), 12, 128, 128, device="cuda", dtype=torch.float32
    ) * 0.05
    if len(lengths) == 1:
        cu_seqlens = None
    else:
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.long)

    monkeypatch.setattr(_gdn_backend, "_prefill_requested", "hcu-boltops")
    h, v_new, final_state = chunk.chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    monkeypatch.setattr(_gdn_backend, "_prefill_requested", "triton")
    expected = chunk.chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        cu_seqlens=cu_seqlens,
    )

    output_buffer = torch.empty(
        u.numel() + 16, device="cuda", dtype=u.dtype
    )
    monkeypatch.setattr(_gdn_backend, "_prefill_requested", "hcu-boltops")
    actual = chunk.chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        cu_seqlens=cu_seqlens,
        core_attn_out=output_buffer,
    )
    torch.cuda.synchronize()

    assert actual.data_ptr() == output_buffer.data_ptr()
    assert final_state is not None
    assert torch.isfinite(final_state).all()
    assert expected.abs().max() > 0.02
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-4)


def test_aiter_hip_decode_matches_official_with_null_block_padding():
    """aiter HIP decode kernels + HCU index remap match vLLM FLA.

    vLLM pads CUDA-graph rows with NULL_BLOCK_ID=0, which aiter HIP would
    otherwise treat as a real slot; the remap must keep slot 0 untouched.
    """
    if not torch.cuda.is_available():
        pytest.skip("HCU is required")
    hip = pytest.importorskip("aiter.ops.fla")

    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode as official_recurrent,
    )
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update as official_sigmoid,
    )
    from vllm_hcu.patch.worker.op_opt.patch_gdn_linear_attention import (
        _aiter_state_indices,
    )

    torch.manual_seed(0)
    dev, bf16 = "cuda", torch.bfloat16
    L, H, HV, K, V = 6, 16, 48, 128, 128
    indices = torch.tensor([3, 7, 1, 9, 0, 0], device=dev, dtype=torch.int32)
    valid = indices > 0
    A_log = torch.randn(HV, device=dev) * 0.1
    dt_bias = torch.randn(HV, device=dev, dtype=bf16) * 0.1
    a = torch.randn(L, HV, device=dev, dtype=bf16)
    b = torch.randn(L, HV, device=dev, dtype=bf16)
    state0 = torch.randn(10, HV, V, K, device=dev) * 0.05

    q = torch.randn(1, L, H, K, device=dev, dtype=bf16) * 0.1
    k = torch.randn_like(q)
    v = torch.randn(1, L, HV, V, device=dev, dtype=bf16) * 0.1
    sigmoid_kwargs = dict(
        A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v,
        inplace_final_state=True,
        cu_seqlens=torch.arange(L + 1, device=dev, dtype=torch.int32),
        use_qk_l2norm_in_kernel=True,
    )
    ref_state, hip_state = state0.clone(), state0.clone()
    ref_out, _ = official_sigmoid(
        **sigmoid_kwargs, initial_state=ref_state, ssm_state_indices=indices
    )
    hip_out, _ = hip.vllm_fused_sigmoid_gating_delta_rule_update(
        **sigmoid_kwargs,
        initial_state=hip_state,
        ssm_state_indices=_aiter_state_indices(indices),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        hip_out[:, valid], ref_out[:, valid], rtol=5e-2, atol=5e-3
    )
    torch.testing.assert_close(hip_state, ref_state, rtol=5e-2, atol=5e-3)
    torch.testing.assert_close(hip_state[0], state0[0], rtol=0, atol=0)

    mixed = torch.randn(L, 2 * H * K + HV * V, device=dev, dtype=bf16) * 0.1
    ref_state, hip_state = state0.clone(), state0.clone()
    ref_out = torch.empty(L, 1, HV, V, device=dev, dtype=bf16)
    hip_out = torch.empty_like(ref_out)
    official_recurrent(
        mixed, a, b, A_log, dt_bias, K**-0.5, ref_state, ref_out, indices,
        use_qk_l2norm_in_kernel=True,
    )
    hip.aiter_fused_recurrent_gated_delta_rule_packed_decode(
        mixed, a, b, A_log, dt_bias, K**-0.5, hip_state, hip_out,
        _aiter_state_indices(indices), use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        hip_out[valid], ref_out[valid], rtol=5e-2, atol=5e-3
    )
    torch.testing.assert_close(hip_state, ref_state, rtol=5e-2, atol=5e-3)
    torch.testing.assert_close(hip_state[0], state0[0], rtol=0, atol=0)
