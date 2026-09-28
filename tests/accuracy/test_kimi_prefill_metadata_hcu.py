# SPDX-License-Identifier: Apache-2.0
"""Real HCU conv: CPU metadata reuse preserves output and recurrent cache."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.hcu


@pytest.mark.parametrize("width", [1536, 3072])
def test_cpu_lengths_conv_math_and_state(monkeypatch, width):
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    from causal_conv1d import causal_conv1d_fn_hcu
    from vllm.v1.attention.backends import gdn_attn
    from vllm_hcu.patch.worker.op_opt import patch_kimi_prefill_metadata as patch
    from vllm_hcu.models.kimi_k3.amd.ops.prefill_metadata import (
        LENGTHS_KEY, prefill_sequence_lengths,
    )
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_KIMI_PREFILL_CPU_LENGTHS", "1")
    # Undo module mutation after this test; do not leak into other patch tests.
    monkeypatch.setattr(gdn_attn, "compute_causal_conv1d_metadata", gdn_attn.compute_causal_conv1d_metadata)
    monkeypatch.setattr(gdn_attn, patch._MARKER, getattr(gdn_attn, patch._MARKER, False), raising=False)
    patch.apply(gdn_attn)
    cpu = torch.tensor([0, 1, 18], dtype=torch.int32)
    starts = cpu.cuda()
    meta = SimpleNamespace(nums_dict=gdn_attn.compute_causal_conv1d_metadata(cpu, device=starts.device)[0])
    assert meta.nums_dict[LENGTHS_KEY] == [1, 17]
    torch.manual_seed(41)
    x = (torch.randn(18, width * 3, device="cuda", dtype=torch.bfloat16) * .2)[:, width:2*width]
    w = torch.randn(width, 4, device="cuda", dtype=torch.bfloat16) * .2
    initial = torch.randn(2, width, 3, device="cuda", dtype=torch.bfloat16) * .2
    state = initial.clone()
    indices = torch.arange(2, device="cuda", dtype=torch.int32)
    has = torch.tensor([True, False], device="cuda")
    def run(lengths):
        state.copy_(initial)
        return causal_conv1d_fn_hcu(x.T, w, initial_states=state,
            query_start_loc=starts, cache_indices=indices, has_initial_state=has,
            seq_lens_cpu=lengths, activation="silu").T
    baseline = run(starts.diff().tolist()).clone()
    baseline_state = state.clone()
    # Prove the enabled accessor does not touch the device boundary tensor.
    class NoDeviceRead:
        def diff(self):
            raise AssertionError("device read in optimized path")
    actual = run(prefill_sequence_lengths(meta, NoDeviceRead()))
    assert torch.equal(actual, baseline)
    assert torch.equal(state, baseline_state)
    for seq, (begin, end) in enumerate(((0, 1), (1, 18))):
        history = initial[seq].float() if seq == 0 else torch.zeros_like(initial[seq], dtype=torch.float32)
        full = torch.cat((history, x[begin:end].T.float()), dim=1)
        y = sum(full[:, j:j+end-begin] * w[:, j:j+1].float() for j in range(4))
        expected = F.silu(y).T.to(x.dtype)
        assert torch.isfinite(actual[begin:end]).all()
        torch.testing.assert_close(actual[begin:end], expected, atol=.002, rtol=.02)
        torch.testing.assert_close(state[seq], full[:, -3:].to(state.dtype), atol=0, rtol=0)
