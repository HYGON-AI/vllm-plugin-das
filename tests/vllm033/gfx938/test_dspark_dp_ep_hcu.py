# SPDX-License-Identifier: Apache-2.0
"""gfx938 DSpark receive quantization and native activation parity."""
import pytest
import torch

from hcu_vllm_das import dspark_ll_quant

pytestmark = [
    pytest.mark.hcu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU"),
]


@pytest.mark.parametrize("width", [2, 4, 8])
def test_receiver_valid_rows_match_public_quantizer(width):
    from vllm.model_executor.layers.fused_moe.utils import per_token_quant_int8
    experts, tokens, hidden = 128 // width, 17 * width, 5120
    torch.manual_seed(width)
    x = torch.randn(experts, tokens, hidden, device="cuda", dtype=torch.bfloat16)
    counts = torch.arange(experts, device="cuda", dtype=torch.int32) % (tokens + 1)
    counts[-1] = tokens
    expected_q, expected_s = per_token_quant_int8(x.view(-1, hidden))
    actual_q, actual_s = dspark_ll_quant.quantize_receiver(x, counts)
    valid = torch.arange(tokens, device="cuda")[None, :] < counts[:, None]
    assert torch.equal(actual_q[valid], expected_q.view_as(x)[valid])
    assert torch.equal(actual_s[valid], expected_s.view(experts, tokens, 1)[valid])


@pytest.mark.parametrize("experts,rows,hint", [
    (32, 1024, 16), (96, 1024, 80), (32, 1024, 1024), (32, 7, 3)])
def test_activation_passes_native_byte_qualification_at_actual_geometry(experts, rows, hint):
    from hcu_vllm_das.moe.clamped_int8_admission import admitted
    from hcu_vllm_das.moe.ct_w4a8_runtime import get_ops
    placeholder = torch.empty(1, device="cuda")
    w1_scale = torch.empty(experts, 4608, 1, device="cuda")
    w2_scale = torch.empty(experts, 5120, 1, device="cuda")
    assert admitted(get_ops(), (placeholder, placeholder), w1_scale, w2_scale,
                    10, masked=True, rows=rows, expected_m=hint)
