# SPDX-License-Identifier: Apache-2.0
"""Small, non-gating GPU timing check for the KDA state stage only."""
import pytest
import torch

from vllm_hcu.model_executor.layers import kimi_k3_kda as kda

pytestmark = pytest.mark.hcu


@torch.inference_mode()
def test_4096_state_stage_timing():
    if not torch.cuda.is_available():
        pytest.skip("requires HCU")
    torch.manual_seed(42)
    shape = (1, 4096, 12, 128)
    k, w = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.01
            for _ in range(2))
    u = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    gk = torch.full(shape, -0.01, device="cuda", dtype=torch.float32)
    initial = torch.randn((1, 12, 128, 128), device="cuda") * 0.1
    cu = torch.tensor([0, 4096], device="cuda", dtype=torch.int32)
    ci = torch.tensor([(0, i) for i in range(64)], device="cuda", dtype=torch.int32)
    _, state = kda.boltops_kernels()
    common = dict(k=k, w=w, u=u, gk=gk, initial_state=initial,
                  cu_seqlens=cu, chunk_indices=ci)
    paths = {
        "boltops_fp32": lambda: kda._chunk_gated_delta_rule_fwd_h_fp32(state, **common),
        "aiter_bf16": lambda: kda._chunk_gated_delta_rule_fwd_h_aiter(**common),
    }
    for name, run in paths.items():
        run()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            result = run()
        end.record()
        end.synchronize()
        print(name, "average_ms", start.elapsed_time(end) / 5, "history_dtype", result[0].dtype)
        assert torch.isfinite(result[0]).all()
