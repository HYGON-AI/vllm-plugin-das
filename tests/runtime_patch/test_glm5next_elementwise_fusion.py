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
    """Smoke test: patch applies, is idempotent, and callable on CPU."""
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


@requires_gpu
@pytest.mark.parametrize("shape,dtype", [
    ((32, 2048), torch.bfloat16),
    ((128, 4096), torch.bfloat16),
    ((1, 2048), torch.float16),
    ((256, 2048), torch.float16),
    ((64, 8192), torch.bfloat16),
])
def test_mhc_fused_rmsnorm_cuda_numerical(monkeypatch, shape, dtype):
    """Numerical correctness of HCU fused RMSNorm vs reference on GPU."""
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_MHC_FUSED_RMSNORM", "1")

    module = _fake_mhc_module()
    patch_mhc_fused_rmsnorm.apply_to_module(module)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    w = torch.randn(shape[-1], dtype=dtype, device="cuda")
    eps = 1e-6

    y_fused = module._apply_mhc_norm(x, w, eps)
    y_ref = _reference_rms_norm(x, w, eps)

    # bfloat16 has ~7 mantissa bits, expect rtol ~1e-2
    # float16 has 10 mantissa bits, expect rtol ~1e-3
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    atol = 1e-3 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(y_fused, y_ref, rtol=rtol, atol=atol)
    assert not torch.isnan(y_fused).any()
    assert not torch.isinf(y_fused).any()


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
    """Bit-exact match between strided-input and contiguous-ref paths."""
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
@pytest.mark.parametrize("B,T,H,K,dtype", [
    (2, 8, 4, 128, torch.bfloat16),
    (4, 16, 8, 64, torch.bfloat16),
    (1, 32, 4, 128, torch.float16),
    (8, 4, 8, 256, torch.bfloat16),
])
def test_kda_strided_numerical_correctness(monkeypatch, B, T, H, K, dtype):
    """Numerical correctness of strided KDA vs contiguous reference across
    shapes and dtypes."""
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_GLM5NEXT_KDA_STRIDED_INPUTS", "1")
    kda = pytest.importorskip("vllm.models.glm5next.nvidia.kda")
    original = getattr(
        kda, "_vllm_hcu_original_fused_recurrent_kda", kda.fused_recurrent_kda
    )
    try:
        kda_patch.apply_to_module(sys.modules[kda.__name__])

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        V = K  # Keep V=K for simplicity
        # Create non-contiguous split views as in real usage
        qkv = torch.randn(B, T, 3 * H * K, device="cuda", dtype=dtype)
        q, k, v = (x.reshape(B, T, H, K) for x in qkv.split(H * K, dim=-1))
        assert not q.is_contiguous()

        g = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
        beta = torch.randn(B, T, H, device="cuda", dtype=dtype)
        state0 = torch.randn(B * T + 1, H, V, K, device="cuda") * 0.1

        def run(fn, qq, kk, vv):
            st = state0.clone()
            o, fs = fn(
                q=qq, k=kk, v=vv, g=g, beta=beta,
                initial_state=st,
                inplace_final_state=False,
                use_qk_l2norm_in_kernel=True,
                sigmoid_beta=True,
            )
            torch.cuda.synchronize()
            return o, fs

        ref_o, ref_fs = run(original, q.contiguous(), k.contiguous(), v.contiguous())
        out_o, out_fs = run(kda.fused_recurrent_kda, q, k, v)

        # Numerical tolerance based on dtype
        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        atol = 1e-3 if dtype == torch.bfloat16 else 1e-4
        torch.testing.assert_close(out_o, ref_o, rtol=rtol, atol=atol)
        torch.testing.assert_close(out_fs, ref_fs, rtol=rtol, atol=atol)

        # Check for NaN (always unexpected)
        assert not torch.isnan(out_o).any()
        # Inf can occur legitimately in float16 with long sequences;
        # verify it matches the reference pattern if present.
        if torch.isinf(ref_o).any():
            assert torch.equal(torch.isinf(out_o), torch.isinf(ref_o)), \
                "Inf pattern differs from reference"
        else:
            assert not torch.isinf(out_o).any(), "output has Inf when ref does not"
        assert not torch.isnan(out_fs).any()
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



@requires_gpu
def test_kda_strided_noncontiguous_gate_beta(monkeypatch):
    """Verify strided kernel handles non-contiguous g/beta correctly."""
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_GLM5NEXT_KDA_STRIDED_INPUTS", "1")
    kda = pytest.importorskip("vllm.models.glm5next.nvidia.kda")
    original = getattr(
        kda, "_vllm_hcu_original_fused_recurrent_kda", kda.fused_recurrent_kda
    )
    try:
        kda_patch.apply_to_module(sys.modules[kda.__name__])

        torch.manual_seed(42)
        B, T, H, K = 2, 8, 4, 128

        # Create non-contiguous g and beta by slicing from a larger buffer
        g_buf = torch.randn(B, T, 2 * H * K, device="cuda", dtype=torch.bfloat16)
        g = g_buf[..., :H * K].reshape(B, T, H, K)  # Take first half
        assert not g.is_contiguous()

        beta_buf = torch.randn(B, T, 3 * H, device="cuda", dtype=torch.bfloat16)
        beta = beta_buf[..., H:2*H]  # Take middle slice
        assert not beta.is_contiguous()

        # qkv also non-contiguous as before
        qkv = torch.randn(B, T, 3 * H * K, device="cuda", dtype=torch.bfloat16)
        q, k, v = (x.reshape(B, T, H, K) for x in qkv.split(H * K, dim=-1))

        state0 = torch.randn(B * T + 1, H, K, K, device="cuda") * 0.1

        def run(fn, qq, kk, vv, gg, bb):
            st = state0.clone()
            o, fs = fn(
                q=qq, k=kk, v=vv, g=gg, beta=bb,
                initial_state=st,
                inplace_final_state=False,
                use_qk_l2norm_in_kernel=True,
                sigmoid_beta=True,
            )
            torch.cuda.synchronize()
            return o, fs

        # Reference: make everything contiguous
        ref_o, ref_fs = run(
            original,
            q.contiguous(), k.contiguous(), v.contiguous(),
            g.contiguous(), beta.contiguous()
        )

        # Strided: pass non-contiguous directly
        out_o, out_fs = run(kda.fused_recurrent_kda, q, k, v, g, beta)

        torch.testing.assert_close(out_o, ref_o, rtol=1e-2, atol=1e-3)
        torch.testing.assert_close(out_fs, ref_fs, rtol=1e-2, atol=1e-3)
    finally:
        kda.fused_recurrent_kda = original
        for attr in (kda_patch._MARKER, "_vllm_hcu_original_fused_recurrent_kda"):
            if hasattr(kda, attr):
                delattr(kda, attr)


@requires_gpu
def test_kda_strided_preallocated_output(monkeypatch):
    """Verify strided kernel writes to caller-provided output buffer."""
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_GLM5NEXT_KDA_STRIDED_INPUTS", "1")
    kda = pytest.importorskip("vllm.models.glm5next.nvidia.kda")
    original = getattr(
        kda, "_vllm_hcu_original_fused_recurrent_kda", kda.fused_recurrent_kda
    )
    try:
        kda_patch.apply_to_module(sys.modules[kda.__name__])

        torch.manual_seed(42)
        B, T, H, K = 2, 8, 4, 128
        qkv = torch.randn(B, T, 3 * H * K, device="cuda", dtype=torch.bfloat16)
        q, k, v = (x.reshape(B, T, H, K) for x in qkv.split(H * K, dim=-1))
        g = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
        beta = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
        state0 = torch.randn(B * T + 1, H, K, K, device="cuda") * 0.1

        # Pre-allocate output buffer (fill with sentinel to detect no-write)
        out_buf = torch.full((B, T, H, K), -999.0, device="cuda", dtype=torch.bfloat16)

        st = state0.clone()
        returned_o, fs = kda.fused_recurrent_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=st,
            out=out_buf,  # Pass pre-allocated buffer
            inplace_final_state=False,
            use_qk_l2norm_in_kernel=True,
            sigmoid_beta=True,
        )
        torch.cuda.synchronize()

        # Verify kernel wrote to the buffer (no sentinel left)
        assert not torch.any(out_buf == -999.0), "output buffer not fully written"

        # Verify returned pointer is the same buffer
        assert returned_o.data_ptr() == out_buf.data_ptr(), \
            "returned output is not the provided buffer"

        # Compare against reference (no pre-allocated output)
        st_ref = state0.clone()
        ref_o, ref_fs = original(
            q=q.contiguous(), k=k.contiguous(), v=v.contiguous(),
            g=g, beta=beta,
            initial_state=st_ref,
            inplace_final_state=False,
            use_qk_l2norm_in_kernel=True,
            sigmoid_beta=True,
        )

        torch.testing.assert_close(returned_o, ref_o, rtol=1e-2, atol=1e-3)
        torch.testing.assert_close(fs, ref_fs, rtol=1e-2, atol=1e-3)
    finally:
        kda.fused_recurrent_kda = original
        for attr in (kda_patch._MARKER, "_vllm_hcu_original_fused_recurrent_kda"):
            if hasattr(kda, attr):
                delattr(kda, attr)


@requires_gpu
@pytest.mark.parametrize("scenario", [
    "decode",           # B > 1, T=1 per seq
    "mixed_spec_accept", # Some seqs prefill, some decode, all accepted
    "mixed_spec_reject", # Some seqs have num_accepted < Tp (reject)
])
def test_kda_strided_decode_and_spec(monkeypatch, scenario):
    """Verify strided kernel in decode and speculative decoding scenarios."""
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_GLM5NEXT_KDA_STRIDED_INPUTS", "1")
    kda = pytest.importorskip("vllm.models.glm5next.nvidia.kda")
    original = getattr(
        kda, "_vllm_hcu_original_fused_recurrent_kda", kda.fused_recurrent_kda
    )
    try:
        kda_patch.apply_to_module(sys.modules[kda.__name__])

        torch.manual_seed(42)

        if scenario == "decode":
            # Pure decode: 4 seqs, each generating 1 token
            N, Tp = 4, 1
            cu = torch.arange(0, N * Tp + 1, Tp, device="cuda", dtype=torch.int32)
            idx = torch.arange(1, N * Tp + 1, device="cuda", dtype=torch.int32).view(N, Tp)
            nacc = None  # No speculation

        elif scenario == "mixed_spec_accept":
            # 3 seqs: [prefill=4, decode=1, decode=1]
            cu = torch.tensor([0, 4, 5, 6], device="cuda", dtype=torch.int32)
            idx = torch.tensor([
                [1, 2, 3, 4],  # seq 0: prefill 4 tokens
                [5, 0, 0, 0],  # seq 1: decode 1 token (pad with 0)
                [6, 0, 0, 0],  # seq 2: decode 1 token
            ], device="cuda", dtype=torch.int32)
            nacc = torch.tensor([4, 1, 1], device="cuda", dtype=torch.int32)
            N = 3
            Tp = 4  # Max tokens per seq in this batch

        elif scenario == "mixed_spec_reject":
            # 2 seqs with spec decode, one rejected 2 tokens
            cu = torch.tensor([0, 3, 6], device="cuda", dtype=torch.int32)
            idx = torch.tensor([
                [1, 2, 3, 0],  # seq 0: generated 4, accepted 3
                [4, 5, 6, 0],  # seq 1: generated 4, accepted 3
            ], device="cuda", dtype=torch.int32)
            nacc = torch.tensor([3, 3], device="cuda", dtype=torch.int32)  # Both reject 1
            N = 2
            Tp = 4

        T = cu[-1].item()
        H, K = 8, 128

        qkv = torch.randn(T, 3 * H * K, device="cuda", dtype=torch.bfloat16)
        q, k, v = (x.reshape(1, T, H, K) for x in qkv.split(H * K, dim=-1))
        g = torch.randn(1, T, H, K, device="cuda", dtype=torch.bfloat16)
        beta = torch.randn(1, T, H, device="cuda", dtype=torch.bfloat16)
        state0 = torch.randn(T + 1, H, K, K, device="cuda") * 0.1

        def run(fn, qq, kk, vv):
            st = state0.clone()
            o, fs = fn(
                q=qq, k=kk, v=vv, g=g, beta=beta,
                initial_state=st,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu,
                ssm_state_indices=idx,
                num_accepted_tokens=nacc,
                sigmoid_beta=True,
            )
            torch.cuda.synchronize()
            return o, fs

        ref_o, ref_fs = run(original, q.contiguous(), k.contiguous(), v.contiguous())
        out_o, out_fs = run(kda.fused_recurrent_kda, q, k, v)

        torch.testing.assert_close(out_o, ref_o, rtol=1e-2, atol=1e-3)
        torch.testing.assert_close(out_fs, ref_fs, rtol=1e-2, atol=1e-3)
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
