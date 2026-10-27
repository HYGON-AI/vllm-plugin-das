# SPDX-License-Identifier: Apache-2.0
"""Exercise real vLLM checkpoint rules, convolution windows and cache modes."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers import kimi_k3_kda as kda


@pytest.mark.parametrize("length,position", [(4096, 3072), (8192, 7680), (8175, 7680)])
def test_checkpoint_uses_absolute_position_and_separate_slot(length, position):
    rows, cols, hi, conv, pos = kda.checkpoint_plan(
        [0, length], [length], num_decodes=0, block_size=1536, hash_size=1536)
    assert rows == [0]
    assert cols == [(length + 1535) // 1536 - 2]
    assert hi == [position // 64]
    assert conv == pos == [position]


def test_prefix_resume_varlen_and_decode_first():
    # A length-1 decode, a prefix hit at 3072, then an unrelated long prefill.
    rows, cols, hi, conv, pos = kda.checkpoint_plan(
        [0, 1, 4097, 12272], [42, 7168, 8175], num_decodes=1,
        block_size=1536, hash_size=1536)
    assert rows == [1, 2]
    assert cols == [3, 4]
    assert hi == [48, 64 + 120]
    assert conv == [3073, 4097 + 7680]
    assert pos == [6144, 7680]


def test_chunk_budget_keeps_existing_boundary_rules():
    # A short budget chunk has no internal checkpoint and remains a normal
    # align-mode chunk. Resume from a partial hash unit is never advertised.
    for start, length in [(0, 512), (3072, 512), (3111, 4096)]:
        assert kda.checkpoint_plan(
            [0, length], [start + length], num_decodes=0,
            block_size=1536, hash_size=1536)[0] == []


def test_conv_checkpoints_survive_final_state_update():
    metadata = SimpleNamespace(nums_dict={kda.CHECKPOINT_KEY: kda.Checkpoints(
        torch.tensor([2, 4]), torch.tensor([1, 3]), (64, 192), (64, 192))})
    raw = torch.arange(3 * 256, dtype=torch.float32).reshape(3, 256)
    states = torch.full((6, 3, 3), -1.0)
    kda.save_conv_checkpoints(raw, states, metadata)
    raw_second = raw + 1000
    states_second = torch.full_like(states, -1.0)
    kda.save_conv_checkpoints(raw_second, states_second, metadata)
    states[2].fill_(1000)  # The conv kernel may overwrite a reserved slot.
    states[4].fill_(2000)
    states[5].fill_(3000)
    states_second[2].fill_(4000)
    kda.finalize_checkpoints(metadata)
    torch.testing.assert_close(states[2], raw[:, 61:64])
    torch.testing.assert_close(states[4], raw[:, 189:192])
    torch.testing.assert_close(states_second[2], raw_second[:, 61:64])
    assert states[0].eq(-1).all()


def test_invalid_strategy_fails_explicitly(monkeypatch):
    monkeypatch.setenv("VLLM_KIMI_KDA_CACHE_STRATEGY", "extra_buffer")
    monkeypatch.setenv("VLLM_KIMI_KDA_PREFILL_BACKEND", "legacy")
    with pytest.raises(ValueError, match="requires the boltops"):
        kda.cache_strategy()


def test_spec_only_advertises_supported_configuration(monkeypatch):
    @dataclass(frozen=True)
    class Spec:
        block_size: int = 64
        num_prefill_checkpoint_blocks: int = 0
        prefill_checkpoint_alignment: int | None = None

    class Layer:
        def get_kv_cache_spec(self, config):
            return Spec()

    kda.install_checkpoint_spec(Layer)
    monkeypatch.setenv("VLLM_KIMI_KDA_PREFILL_BACKEND", "boltops")
    monkeypatch.setenv("VLLM_KIMI_KDA_CACHE_STRATEGY", "extra_buffer")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(kda, "boltops_kernels", lambda: None)
    config = SimpleNamespace(cache_config=SimpleNamespace(
        enable_prefix_caching=True, mamba_cache_mode="align", block_size=1536,
        prefix_match_unit=None),
        parallel_config=SimpleNamespace(), speculative_config=None)
    assert Layer().get_kv_cache_spec(config) == Spec(1536, 1, 64)
    config.speculative_config = object()
    with pytest.raises(ValueError, match="speculative decoding disabled"):
        Layer().get_kv_cache_spec(config)
    config.speculative_config = None
    config.cache_config.enable_prefix_caching = False
    with pytest.raises(ValueError, match="prefix caching"):
        Layer().get_kv_cache_spec(config)
