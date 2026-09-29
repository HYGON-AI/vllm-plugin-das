# SPDX-License-Identifier: Apache-2.0
"""KDA speculative Conv1D history against FP32 math; no model/service."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.hcu


@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("channels", [96, 4608])
@pytest.mark.parametrize("drafts", [1, 4])
@pytest.mark.parametrize("graph", [False, True])
def test_speculative_conv_acceptance_and_guard_blocks(monkeypatch, dim_first, channels, drafts, graph):
    if not torch.cuda.is_available():
        pytest.skip("HCU required")
    from vllm.model_executor.layers.mamba import mamba_utils
    from vllm_hcu.models.kimi_k3.amd import kimi_gdn_linear_attn as model

    monkeypatch.setattr(mamba_utils, "is_conv_state_dim_first", lambda: dim_first)
    torch.manual_seed(93)
    device, dtype = "cuda", torch.bfloat16
    sequence = drafts + 1
    head_dim = 128 if channels == 4608 else 4
    conv_shape, _ = model.KimiGatedDeltaNetAttention.get_state_shape(SimpleNamespace(
        tp_size=1, num_heads=channels // (3 * head_dim), head_dim=head_dim,
        conv_size=4, num_spec=drafts))
    assert conv_shape == ((channels, 3 + drafts) if dim_first else (3 + drafts, channels))
    # Model cache orientation plus inter-layer padding in a shared block slab.
    slab = torch.full((8, 2, *conv_shape), 13., dtype=dtype, device=device)
    cache = slab[:, 0]
    state = cache if dim_first else cache.transpose(-1, -2)
    indices = torch.tensor([2, 5, 7], dtype=torch.int32, device=device)
    starts = torch.tensor([0, sequence, 2 * sequence, 2 * sequence], dtype=torch.int32, device=device)
    accepted = torch.ones(3, dtype=torch.int32, device=device)
    weights = torch.randn(channels, 4, device=device) * .1
    x = torch.zeros(2 * sequence, channels, device=device, dtype=dtype)
    out = torch.empty_like(x)

    def launch():
        return model._causal_conv1d_update_compat(
            x, state, weights, None, activation="silu", conv_state_indices=indices,
            num_accepted_tokens=accepted, query_start_loc=starts,
            max_query_len=sequence, validate_data=False, out=out)

    state[2].zero_()
    state[5].zero_()
    launch()  # compile before capture
    replay = None
    if graph:
        replay = torch.cuda.CUDAGraph()
        with torch.cuda.graph(replay):
            launch()
    max_error = 0.
    for count in range(1, sequence + 1):
        # The count includes one mandatory target token. count=1 rejects all
        # drafts; count=sequence accepts all. Both rounds use the same addresses.
        state[2].copy_(torch.randn_like(state[2]) * .1)
        state[5].copy_(torch.randn_like(state[5]) * .1)
        accepted.copy_(torch.tensor([count, sequence + 1 - count, 1], device=device))
        for _ in range(2):
            x.copy_(torch.randn_like(x) * .1)
            original_x, original_state, original_slab = x.clone(), state.clone(), slab.clone()
            expected_out = []
            expected_states = []
            for row, (block, keep) in enumerate(((2, count), (5, sequence + 1 - count))):
                history = original_state[block, :, keep - 1:keep + 2].float()
                tokens = original_x[row * sequence:(row + 1) * sequence].T.float()
                full = torch.cat((history, tokens), dim=1)
                y = sum(full[:, j:j + sequence] * weights[:, j:j + 1] for j in range(4))
                expected_out.append(F.silu(y).T.to(dtype))
                expected_states.append(full[:, 1:].to(dtype))
            replay.replay() if graph else launch()
            torch.cuda.synchronize()
            expected = torch.cat(expected_out)
            assert torch.isfinite(out).all()
            torch.testing.assert_close(out, expected, atol=.002, rtol=.02)
            max_error = max(max_error, (out.float() - expected.float()).abs().max().item())
            for block, expected_state in zip((2, 5), expected_states):
                torch.testing.assert_close(state[block], expected_state, atol=0, rtol=0)
            # Idle rank/request and every unrelated block/padding stay intact.
            torch.testing.assert_close(state[[0, 1, 3, 4, 6, 7]],
                                       original_state[[0, 1, 3, 4, 6, 7]], atol=0, rtol=0)
            torch.testing.assert_close(slab[:, 1], original_slab[:, 1], atol=0, rtol=0)
    print(f"SPEC_CONV channels={channels} drafts={drafts} DS={dim_first} graph={graph} "
          f"max_abs={max_error} state_exact=True guards_exact=True")
