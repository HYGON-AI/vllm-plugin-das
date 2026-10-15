# SPDX-License-Identifier: Apache-2.0
"""Run the production router kernel with Triton's CPU interpreter."""

import os
import subprocess
import sys
import textwrap

import pytest


def test_router_topk_interpreter_matches_reference_and_preserves_padding():
    # A fresh process sets interpreter mode before any Triton decorators run.
    # Bypass autotuning only: the production kernel body executes unchanged.
    script = textwrap.dedent(
        """
        import torch
        from vllm_hcu.models.step5.kernels import router_bias_func
        from vllm_hcu.ops.step5.router_bias import router_bias_topk_kernel

        assert not torch.cuda.is_initialized()
        torch.manual_seed(29)
        cases = ((1, 13), (3, 13), (6, 13), (8, 16), (13, 13))
        checked = 0
        for topk, experts in cases:
            for renormalize in (False, True):
                for index_dtype in (torch.int32, torch.int64):
                    logits = torch.randn(4, experts, dtype=torch.float32)
                    bias = torch.randn(experts, dtype=torch.float32) * 0.1
                    logits[2, 1] = float("nan")
                    scale = 1.7
                    sentinel = -7
                    want_w, want_i = router_bias_func(
                        torch.empty(4, 1), logits, topk, renormalize,
                        router_bias=bias, routed_scaling_factor=scale,
                        nan_row_i_out=sentinel, indices_dtype=index_dtype,
                    )
                    # Extra lanes detect stores from the padded top-k vector.
                    weight_buffer = torch.full((4, topk + 4), 54321.0)
                    id_buffer = torch.full((4, topk + 4), 1234, dtype=index_dtype)
                    weights, ids = weight_buffer[:, :topk], id_buffer[:, :topk]
                    router_bias_topk_kernel.fn[(4,)](
                        logits, bias, weights, ids, logits.stride(0),
                        weights.stride(0), E=experts, TOPK=topk,
                        RENORM=renormalize, CHECK_NAN=True,
                        BLOCK_E=1 << (experts - 1).bit_length(),
                        ROUTED_SCALING_FACTOR=scale, NAN_ROW_I_OUT=sentinel,
                    )
                    torch.testing.assert_close(weights, want_w, atol=1e-6, rtol=1e-6)
                    assert torch.equal(ids, want_i), (topk, ids, want_i)
                    assert (weight_buffer[:, topk:] == 54321.0).all()
                    assert (id_buffer[:, topk:] == 1234).all()
                    if renormalize:
                        torch.testing.assert_close(
                            weights[[0, 1, 3]].sum(dim=1),
                            torch.full((3,), scale), atol=1e-6, rtol=1e-6,
                        )
                    assert not weights[2].any() and (ids[2] == sentinel).all()
                    checked += 1
        assert checked == 20
        assert not torch.cuda.is_initialized()
        print("20 router interpreter cases passed")
        """
    )
    env = os.environ.copy()
    env.update(
        TRITON_INTERPRET="1",
        VLLM_PLUGINS="__disabled__",
        CUDA_VISIBLE_DEVICES="",
        HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="",
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "20 router interpreter cases passed" in result.stdout


@pytest.mark.parametrize("topk", [0, 14])
def test_router_rejects_topk_outside_expert_range_before_launch(topk, monkeypatch):
    from vllm_hcu.ops.step5 import router_bias

    class DeviceInput:
        is_cuda = True
        ndim = 2
        shape = (2, 13)

    def forbidden_launch(_spec):
        pytest.fail("invalid top-k must not reach the kernel launcher")

    monkeypatch.setattr(
        router_bias, "_router_bias_topk_driver_launcher", forbidden_launch
    )
    with pytest.raises(AssertionError):
        router_bias.router_bias_triton_func(
            DeviceInput(), object(), topk, renormalize=True
        )
