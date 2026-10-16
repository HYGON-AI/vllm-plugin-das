# SPDX-License-Identifier: Apache-2.0
"""Tests for VLLM_HCU_MHC_FUSED_RMSNORM and
VLLM_HCU_GLM5NEXT_KDA_STRIDED_INPUTS runtime patches."""

import sys
from types import ModuleType

import pytest
import torch

from vllm_hcu.patch.worker.core_fix import patch_mhc_fused_rmsnorm
from vllm_hcu.patch.worker.op_opt import (
    patch_glm5next_kda_strided_recurrent as kda_patch,
)
from vllm_hcu.platforms import envs as henvs

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU for Triton kernels"
)


# ---------------------------------------------------------------- mHC norm


def test_elementwise_fusion_switches_enabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_HCU_MHC_FUSED_RMSNORM", raising=False)
    monkeypatch.delenv("VLLM_HCU_GLM5NEXT_KDA_STRIDED_INPUTS", raising=False)
    assert henvs.VLLM_HCU_MHC_FUSED_RMSNORM is True
    assert henvs.VLLM_HCU_GLM5NEXT_KDA_STRIDED_INPUTS is True


def _reference_rms_norm(x, weight, eps):
    # Mirrors vllm.ir.ops.rms_norm's native implementation.
    orig_dtype = x.dtype
    xf = x.to(torch.float32)
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    xf = xf.to(weight.dtype) * weight
    return xf.to(orig_dtype)


def _fake_mhc_module() -> ModuleType:
    module = ModuleType(patch_mhc_fused_rmsnorm.TARGET_MODULE)

    def _apply_mhc_norm(layer_input, norm_weight, norm_eps):
        if norm_weight is None:
            return layer_input
        return _reference_rms_norm(layer_input, norm_weight, norm_eps)

    module._apply_mhc_norm = _apply_mhc_norm
    return module


def test_mhc_fused_rmsnorm_disabled_leaves_module_untouched(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_MHC_FUSED_RMSNORM", "0")
    module = _fake_mhc_module()
    original = module._apply_mhc_norm
    assert patch_mhc_fused_rmsnorm.apply_to_module(module) is False
    assert module._apply_mhc_norm is original


def test_mhc_fused_rmsnorm_respects_custom_ops_master_switch(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv("VLLM_HCU_MHC_FUSED_RMSNORM", "1")
    module = _fake_mhc_module()
    original = module._apply_mhc_norm
    assert patch_mhc_fused_rmsnorm.apply_to_module(module) is False
    assert module._apply_mhc_norm is original


def test_mhc_fused_rmsnorm_matches_reference(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_MHC_FUSED_RMSNORM", "1")
    module = _fake_mhc_module()
    assert patch_mhc_fused_rmsnorm.apply_to_module(module) is True
    # idempotent
    assert patch_mhc_fused_rmsnorm.apply_to_module(module) is False

    torch.manual_seed(0)
    x = torch.randn(32, 2048, dtype=torch.bfloat16)
    w = torch.randn(2048, dtype=torch.bfloat16)
    eps = 1e-6

    out = module._apply_mhc_norm(x, w, eps)
    ref = _reference_rms_norm(x, w, eps)
    assert out.dtype == x.dtype
    # Single- vs double-rounded weight multiply: <= 1 bf16 ulp.
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=2**-7)

    # None weight passes through.
    assert module._apply_mhc_norm(x, None, eps) is x


def test_mhc_fused_rmsnorm_dtype_mismatch_falls_back(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_MHC_FUSED_RMSNORM", "1")
    module = _fake_mhc_module()
    patch_mhc_fused_rmsnorm.apply_to_module(module)

    x = torch.randn(4, 64, dtype=torch.bfloat16)
    w = torch.randn(64, dtype=torch.float32)
    out = module._apply_mhc_norm(x, w, 1e-6)
    ref = _reference_rms_norm(x, w, 1e-6)
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


# ------------------------------------------------------------- KDA strided


def test_kda_strided_disabled_leaves_module_untouched(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_GLM5NEXT_KDA_STRIDED_INPUTS", "0")
    kda = pytest.importorskip("vllm.models.glm5next.nvidia.kda")
    original = kda.fused_recurrent_kda
    try:
        assert kda_patch.apply_to_module(sys.modules[kda.__name__]) is False
        assert kda.fused_recurrent_kda is original
    finally:
        kda.fused_recurrent_kda = original


def test_kda_strided_respects_custom_ops_master_switch(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv("VLLM_HCU_GLM5NEXT_KDA_STRIDED_INPUTS", "1")
    kda = pytest.importorskip("vllm.models.glm5next.nvidia.kda")
    original = kda.fused_recurrent_kda
    try:
        assert kda_patch.apply_to_module(sys.modules[kda.__name__]) is False
        assert kda.fused_recurrent_kda is original
    finally:
        kda.fused_recurrent_kda = original


@requires_gpu
@pytest.mark.parametrize("spec", [False, True])
def test_kda_strided_matches_contiguous_reference(monkeypatch, spec):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_GLM5NEXT_KDA_STRIDED_INPUTS", "1")
    kda = pytest.importorskip("vllm.models.glm5next.nvidia.kda")
    original = getattr(
        kda, "_vllm_hcu_original_fused_recurrent_kda", kda.fused_recurrent_kda
    )
    try:
        kda_patch.apply_to_module(sys.modules[kda.__name__])

        torch.manual_seed(0)
        N, Tp, H, K = 4, 4, 8, 128
        T = N * Tp
        # Split-qkv views: non-contiguous, like the conv output in _forward.
        qkv = torch.randn(T, 3 * H * K, device="cuda", dtype=torch.bfloat16)
        q, k, v = (x.reshape(1, T, H, K) for x in qkv.split(H * K, dim=-1))
        assert not q.is_contiguous()
        g = torch.randn(1, T, H, K, device="cuda", dtype=torch.bfloat16)
        beta = torch.randn(1, T, H, device="cuda", dtype=torch.bfloat16)
        a_log = torch.randn(H, device="cuda")
        g_bias = torch.randn(H * K, device="cuda")
        cu = torch.arange(0, T + 1, Tp, device="cuda", dtype=torch.int32)
        idx = torch.arange(1, T + 1, device="cuda", dtype=torch.int32).view(N, Tp)
        nacc = (
            torch.full((N,), 2, device="cuda", dtype=torch.int32) if spec else None
        )
        state0 = torch.randn(T + 1, H, K, K, device="cuda") * 0.1

        def run(fn, qq, kk, vv):
            st = state0.clone()
            o, _ = fn(
                q=qq,
                k=kk,
                v=vv,
                g=g,
                beta=beta,
                initial_state=st,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu,
                ssm_state_indices=idx,
                num_accepted_tokens=nacc,
                sigmoid_beta=True,
                a_log=a_log,
                g_bias=g_bias,
                compute_gate=True,
                lower_bound=-5.0,
            )
            torch.cuda.synchronize()
            return o, st

        ref_o, ref_s = run(
            original, q.contiguous(), k.contiguous(), v.contiguous()
        )
        out_o, out_s = run(kda.fused_recurrent_kda, q, k, v)
        assert torch.equal(ref_o, out_o)
        assert torch.equal(ref_s, out_s)
    finally:
        kda.fused_recurrent_kda = original
        for attr in (kda_patch._MARKER, "_vllm_hcu_original_fused_recurrent_kda"):
            if hasattr(kda, attr):
                delattr(kda, attr)


@requires_gpu
def test_kda_strided_k_nonpow2_no_nan(monkeypatch):
    """K not a power of 2 triggers the BK > K tail path; upstream bug would
    read g beyond allocation and risk NaN pollution."""
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_GLM5NEXT_KDA_STRIDED_INPUTS", "1")
    kda = pytest.importorskip("vllm.models.glm5next.nvidia.kda")
    original = kda.fused_recurrent_kda
    try:
        kda_patch.apply_to_module(sys.modules[kda.__name__])
        torch.manual_seed(0)
        B, T, H, K, V = 2, 8, 4, 96, 96  # K=96, BK=128 rounds up
        qkv = torch.randn(B, T, 3 * H * K, device="cuda", dtype=torch.bfloat16)
        q, k, v = (x.reshape(B, T, H, K) for x in qkv.split(H * K, dim=-1))
        g = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
        beta = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
        state0 = torch.randn(B * T + 1, H, V, K, device="cuda") * 0.1
        o, fs = kda.fused_recurrent_kda(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=state0,
            inplace_final_state=False,
            use_qk_l2norm_in_kernel=True, sigmoid_beta=True,
        )
        torch.cuda.synchronize()
        # Verify both bugs are fixed: no NaN from OOB gate reads, and
        # fs has B*T slots (vs upstream bug that allocated T).
        assert fs.shape[0] == B * T, f"final_state shape[0]={fs.shape[0]} != B*T={B*T}"
        assert not torch.isnan(o).any(), "output contains NaN"
        assert not torch.isinf(o).any(), "output contains Inf"
    finally:
        kda.fused_recurrent_kda = original
        for attr in (kda_patch._MARKER, "_vllm_hcu_original_fused_recurrent_kda"):
            if hasattr(kda, attr):
                delattr(kda, attr)


@requires_gpu
def test_kda_strided_final_state_non_inplace_alloc(monkeypatch):
    """B > 1, inplace_final_state=False: upstream bug would allocate T slots
    but write B*T, causing OOB writes for i_n >= 1."""
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_GLM5NEXT_KDA_STRIDED_INPUTS", "1")
    kda = pytest.importorskip("vllm.models.glm5next.nvidia.kda")
    original = kda.fused_recurrent_kda
    try:
        kda_patch.apply_to_module(sys.modules[kda.__name__])
        torch.manual_seed(0)
        B, T, H, K, V = 3, 4, 4, 64, 64
        q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
        beta = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
        state0 = torch.randn(B * T + 1, H, V, K, device="cuda") * 0.1
        o, fs = kda.fused_recurrent_kda(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=state0,
            inplace_final_state=False,
            use_qk_l2norm_in_kernel=True, sigmoid_beta=True,
        )
        torch.cuda.synchronize()
        # The fix allocates B*T slots; verify shape and no CUDA errors.
        assert fs.shape[0] == B * T, f"final_state.shape[0]={fs.shape[0]} != B*T={B*T}"
        assert not torch.isnan(o).any()
        assert not torch.isnan(fs).any()
    finally:
        kda.fused_recurrent_kda = original
        for attr in (kda_patch._MARKER, "_vllm_hcu_original_fused_recurrent_kda"):
            if hasattr(kda, attr):
                delattr(kda, attr)


def test_inner_dense_rules():
    x = torch.randn(16, 3 * 8 * 32)
    q = x.split(8 * 32, dim=-1)[0].reshape(1, 16, 8, 32)
    # Strided tokens, dense inner (head, dim): eligible.
    assert kda_patch._inner_dense(q)
    # Transposed inner layout: not eligible.
    assert not kda_patch._inner_dense(
        x.view(1, 16, 32, 24).transpose(2, 3)
    )
    # 3D dense: eligible.
    assert kda_patch._inner_dense(torch.randn(1, 16, 8))
    # Non-unit last-dim stride: not eligible.
    assert not kda_patch._inner_dense(torch.randn(1, 16, 8, 4)[..., ::2])
