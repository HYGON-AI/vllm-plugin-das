# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""gfx938 numerical regression for the optional PCP Q rotation."""

import pytest
import torch

from hcu_vllm_das import pcp_q_rope


def _check_component(candidate, native, a, b, c, d):
    """Check both FP32 accumulation choices before the common BF16 cast."""
    ab, cd = a * b, c * d
    separate = (ab + cd).to(torch.bfloat16)
    torch.testing.assert_close(candidate, separate, rtol=0, atol=0)

    # The native build can contract either product into its sum. Model the
    # individual FP32 FMA round with FP64 operands and one cast back to FP32.
    fused_ab = (a.double() * b.double() + cd.double()).float().to(torch.bfloat16)
    fused_cd = (ab.double() + c.double() * d.double()).float().to(torch.bfloat16)
    matches = (native == separate) | (native == fused_ab) | (native == fused_cd)
    assert matches.all().item(), "native output lies outside the legal FP32 rounding modes"


@pytest.mark.hcu
@pytest.mark.parametrize("tokens", [1, 128, 4096])
def test_device_matches_native_q_only(tokens):
    assert torch.version.hip is not None and torch.cuda.is_available(), "requires ROCm"
    device = torch.device("cuda", torch.cuda.current_device())
    assert pcp_q_rope._supported_device(device), "requires gfx938"

    from vllm.platforms import current_platform

    current_platform.import_kernels()
    with torch.inference_mode():
        generator = torch.Generator(device=device).manual_seed(33)
        q = torch.randn(tokens, 64, 512, dtype=torch.bfloat16, device=device, generator=generator)
        positions = torch.arange(tokens, dtype=torch.int64, device=device)
        angles = torch.randn(tokens, 32, device=device, generator=generator)
        cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
        kv = torch.empty(tokens, 512, dtype=q.dtype, device=device)
        cache = torch.empty(1, 64 * 584, dtype=torch.uint8, device=device)
        slots = positions[:0]
        native = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q, kv, cache, slots, positions, cos_sin, 64, 1e-6, 64,
            False, False, True, False,
        )
        even = q[..., 448::2].float()
        odd = q[..., 449::2].float()
        cosine, sine = (part[:, None, :] for part in cos_sin.chunk(2, dim=-1))
        prepare = pcp_q_rope.initialize(cos_sin)
        assert prepare is not None
        rotate = prepare(q, positions, cos_sin)
        assert rotate is not None
        rotate()
        torch.testing.assert_close(q[..., :448], native[..., :448], rtol=0, atol=0)
        _check_component(q[..., 448::2], native[..., 448::2], even, cosine, -odd, sine)
        _check_component(q[..., 449::2], native[..., 449::2], odd, cosine, even, sine)
