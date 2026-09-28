"""Allocation/copy contracts for KDA speculative history, without a service."""
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba import mamba_utils
from vllm_hcu.models.kimi_k3.amd.linear import KimiLinearForCausalLM
from vllm_hcu.models.kimi_k3.amd.kimi_gdn_linear_attn import KimiGatedDeltaNetAttention
from vllm_hcu.models.kimi_k3.amd.ops.state_shape import kda_state_shape


@pytest.mark.parametrize("invalid", [-1, True, 1.5, None])
def test_invalid_draft_count_fails_before_allocation(invalid):
    with pytest.raises(ValueError, match="num_spec"):
        kda_state_shape(8, 96, 128, conv_kernel_size=4, num_spec=invalid)


@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("drafts", [0, 1, 4, 8])
def test_model_and_layer_allocate_complete_speculative_conv_history(monkeypatch, dim_first, tp, drafts):
    monkeypatch.setattr(mamba_utils, "is_conv_state_dim_first", lambda: dim_first)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(linear_attn_config={
            "num_heads": 96, "head_dim": 128, "short_conv_kernel_size": 4})),
        speculative_config=SimpleNamespace(num_speculative_tokens=drafts) if drafts else None)
    layer = SimpleNamespace(tp_size=tp, num_heads=96, head_dim=128, conv_size=4, num_spec=drafts)
    channels, history = 3 * 96 * 128 // tp, 3 + drafts
    expected_conv = (channels, history) if dim_first else (history, channels)
    expected = (expected_conv, (96 // tp, 128, 128))
    assert KimiLinearForCausalLM.get_mamba_state_shape_from_config(config) == expected
    assert KimiGatedDeltaNetAttention.get_state_shape(layer) == expected


@pytest.mark.parametrize("accepted_drafts", [0, 1, 2, 4])
def test_sd_copy_offsets_follow_accepted_history_with_gapped_block_stride(monkeypatch, accepted_drafts):
    monkeypatch.setattr(mamba_utils, "is_conv_state_dim_first", lambda: False)
    # A padded block stride occurs in a shared cache slab. Each individual
    # state remains contiguous, as required by the byte-copy consumer.
    conv = torch.arange(8 * 2 * 7 * 12).reshape(8, 2, 7, 12)[:, 0]
    recurrent = torch.arange(8 * 2 * 2 * 4 * 4).reshape(8, 2, 2, 4, 4)[:, 0]
    blocks = [6, 1, 5, 2, 4, 0]
    conv_copy, recurrent_copy = KimiLinearForCausalLM.get_mamba_state_copy_func()
    # vLLM counts the mandatory target/bonus token too: rejecting every draft
    # means num_accepted_tokens=1, not 0.
    accepted_tokens = accepted_drafts + 1
    c = conv_copy(conv, blocks, 0, accepted_tokens)
    r = recurrent_copy(recurrent, blocks, 0, accepted_tokens)
    expected_c = conv[6, accepted_drafts:]
    expected_r = recurrent[blocks[accepted_drafts]]
    assert c.start_addr == expected_c.data_ptr()
    assert c.num_elements == expected_c.numel()
    assert r.start_addr == expected_r.data_ptr()
    assert r.num_elements == expected_r.numel()
    assert c.num_elements >= 3 * 12  # the next convolution still has 3 old tokens


def test_ds_nonzero_offset_requires_fused_postprocess(monkeypatch):
    monkeypatch.setattr(mamba_utils, "is_conv_state_dim_first", lambda: True)
    conv_copy, _ = KimiLinearForCausalLM.get_mamba_state_copy_func()
    state = torch.zeros(3, 12, 7)
    assert conv_copy(state, [2, 0], 0, 1).start_addr == state[2].data_ptr()
    with pytest.raises(AssertionError, match="fused postprocess"):
        conv_copy(state, [2, 0], 0, 2)
