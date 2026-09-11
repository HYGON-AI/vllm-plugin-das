# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU numerical parity and routing for optional FLA providers."""

from __future__ import annotations

from types import ModuleType

import pytest
import torch


@pytest.mark.parametrize("lengths", [(64,), (32, 64)])
def test_boltops_chunk_pipeline_matches_official(
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
        patch_fla_recompute_w_u,
    )
    from vllm_hcu.platforms import envs as henvs

    patch_fla_chunk_delta_h.apply_to_module(chunk)
    patch_fla_chunk_o.apply_to_module(chunk)
    patch_fla_recompute_w_u.apply_to_module(chunk)

    total_tokens = sum(lengths)
    torch.manual_seed(7)
    q = torch.randn(
        1, total_tokens, 2, 64, device="cuda", dtype=torch.bfloat16
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
            len(lengths), 2, 64, 64, device="cuda", dtype=torch.float32
        ) * 0.01

    def run(enabled: bool):
        monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
        monkeypatch.setattr(
            henvs, "VLLM_HCU_USE_CUSTOM_AITER_FLA", enabled
        )
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

    expected_output, expected_state = run(False)
    actual_output, actual_state = run(True)
    torch.testing.assert_close(
        actual_output, expected_output, rtol=5e-2, atol=5e-3
    )
    torch.testing.assert_close(
        actual_state, expected_state, rtol=5e-2, atol=5e-3
    )


def test_chunk_o_prefers_aiter_hip_and_preserves_output_buffer(monkeypatch):
    from vllm_hcu.patch.worker.op_opt import patch_fla_chunk_o as adapter
    from vllm_hcu.platforms import envs as henvs

    calls = []

    def original(
        q,
        k,
        v,
        h,
        g=None,
        scale=None,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=64,
        core_attn_out=None,
    ):
        calls.append("official")
        return torch.zeros_like(v)

    def aiter_kernel(**kwargs):
        calls.append(("aiter", kwargs["transpose_state_layout"]))
        return torch.ones_like(kwargs["v"])

    module = ModuleType(adapter.TARGET_MODULE)
    module.FLA_CHUNK_SIZE = 64
    module.chunk_fwd_o = original
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_AITER_FLA", True)
    monkeypatch.setattr(
        adapter,
        "make_aiter_fla_resolver",
        lambda _name: lambda: aiter_kernel,
    )
    monkeypatch.setattr(
        adapter,
        "make_boltops_gdn_resolver",
        lambda _name: lambda: None,
    )
    assert adapter.apply_to_module(module) is True

    q = torch.empty(1, 1, 2, 128, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty(1, 1, 4, 128, dtype=torch.bfloat16)
    h = torch.empty(1, 1, 4, 128, 128, dtype=torch.bfloat16)
    output_buffer = torch.empty(v.numel() + 8, dtype=v.dtype)
    output = module.chunk_fwd_o(q, k, v, h, core_attn_out=output_buffer)

    assert calls == [("aiter", True)]
    assert output.data_ptr() == output_buffer.data_ptr()
    torch.testing.assert_close(output, torch.ones_like(v))


def test_chunk_h_uses_official_for_three_to_one_value_head_ratio(monkeypatch):
    from vllm_hcu.patch.worker.op_opt import patch_fla_chunk_delta_h as adapter
    from vllm_hcu.platforms import envs as henvs

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
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_OPS", True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_CUSTOM_AITER_FLA", True)
    monkeypatch.setattr(
        adapter,
        "make_boltops_gdn_resolver",
        lambda _name: lambda: boltops_kernel,
    )
    assert adapter.apply_to_module(module) is True

    k = torch.empty(1, 1, 4, 128)
    module.chunk_gated_delta_rule_fwd_h(k, k, torch.empty(1, 1, 12, 128))
    assert calls == ["official"]

    module.chunk_gated_delta_rule_fwd_h(k, k, torch.empty(1, 1, 8, 128))
    assert calls == ["official", "boltops"]
