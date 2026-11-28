# SPDX-License-Identifier: Apache-2.0
"""Actual stock MRotaryEmbedding reference; no native/runtime modules doubled."""

import pytest
import torch
from hcu_vllm_das.models.hy_v4_flash.dfly_rotary import (
    HYV4StandardRotaryEmbedding,
    validate_dfly_rotary_pair,
)
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding


@pytest.fixture(autouse=True)
def native_cpu_config():
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
            custom_ops=["none"],
        ),
    )
    with set_current_vllm_config(config):
        yield config


class CPUMRotaryEmbedding(MRotaryEmbedding):
    """Use stock's real native math on CPU even on an HCU-installed host."""

    def forward(self, positions, query, key):
        return self.forward_native(positions, query, key)


def rope(*, section=None, base=10000.0, interleaved=True):
    return CPUMRotaryEmbedding(
        head_size=6,
        rotary_dim=6,
        max_position_embeddings=32,
        base=base,
        is_neox_style=False,
        dtype=torch.float32,
        mrope_section=section or [1, 1, 1],
        mrope_interleaved=interleaved,
    )


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("interleaved", [False, True])
def test_native_caller_assignment_preserves_rotary_math(
    multimodal, strided, interleaved
):
    rotary = rope(interleaved=interleaved)
    wrapper = HYV4StandardRotaryEmbedding(rotary)
    query = torch.arange(80, dtype=torch.float32).reshape(4, 2, 10) / 100
    key = torch.arange(32, dtype=torch.float32).reshape(4, 1, 8) / 100
    q_view = query[..., 4:]
    k_view = key[..., 2:]
    if not strided:
        q_view = q_view.contiguous()
        k_view = k_view.contiguous()
    original_nope = query[..., :4].clone()
    positions = torch.arange(4, dtype=torch.int64)
    if multimodal:
        positions = torch.stack([positions, positions + 2, positions + 4])
    q_ref, k_ref = rotary.forward_native(
        positions, q_view.reshape(4, -1).clone(), k_view.reshape(4, -1).clone()
    )
    q_out, k_out = wrapper(positions, q_view, k_view)
    # These are the assignments made by unchanged stock HY4 attention.forward.
    q_view[...], k_view[...] = q_out, k_out
    torch.testing.assert_close(q_view, q_ref.reshape(q_view.shape), rtol=0, atol=0)
    torch.testing.assert_close(k_view, k_ref.reshape(k_view.shape), rtol=0, atol=0)
    torch.testing.assert_close(query[..., :4], original_nope, rtol=0, atol=0)


def test_adjacent_layers_reusing_rotary_do_not_reuse_old_positions():
    rotary = rope()
    first, second = (
        HYV4StandardRotaryEmbedding(rotary),
        HYV4StandardRotaryEmbedding(rotary),
    )
    q = torch.arange(48, dtype=torch.float32).reshape(4, 2, 6) / 100
    k = torch.arange(24, dtype=torch.float32).reshape(4, 1, 6) / 100
    a = torch.arange(4, dtype=torch.int64).expand(3, -1)
    b = a + 3
    a_q, a_k = first(a, q.clone(), k.clone())
    b_q, b_k = second(b, q.clone(), k.clone())
    ref_q, ref_k = rotary.forward_native(
        b, q.reshape(4, -1).clone(), k.reshape(4, -1).clone()
    )
    assert not torch.equal(a_q, b_q) and not torch.equal(a_k, b_k)
    torch.testing.assert_close(b_q, ref_q.reshape(q.shape), rtol=0, atol=0)
    torch.testing.assert_close(b_k, ref_k.reshape(k.shape), rtol=0, atol=0)


def test_target_and_draft_wrappers_are_unwrapped_symmetrically():
    source = HYV4StandardRotaryEmbedding(rope())
    dest = HYV4StandardRotaryEmbedding(rope())
    validate_dfly_rotary_pair(source, dest)
    with pytest.raises(ValueError, match="base"):
        validate_dfly_rotary_pair(
            source, HYV4StandardRotaryEmbedding(rope(base=20000.0))
        )
    with pytest.raises(ValueError, match="mrope_section"):
        validate_dfly_rotary_pair(
            source, HYV4StandardRotaryEmbedding(rope(section=[2, 1, 0]))
        )


@pytest.mark.parametrize("shape", [(2, 4), (1, 3, 4)])
def test_noncanonical_mrope_position_shapes_are_rejected(shape):
    wrapper = HYV4StandardRotaryEmbedding(rope())
    with pytest.raises(ValueError, match="shape"):
        wrapper(
            torch.zeros(shape, dtype=torch.int64),
            torch.zeros(4, 2, 6),
            torch.zeros(4, 1, 6),
        )
