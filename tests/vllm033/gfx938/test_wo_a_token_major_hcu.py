"""BF16 WO-A numerical contract on the supported gfx938 device."""

import pytest

from hcu_vllm_das import wo_a_token_major as wo


@pytest.mark.hcu
@pytest.mark.parametrize('rows', [2, 6, 16, 32])
def test_hcu_bitwise(rows):
    import torch
    from hcu_adapter.bridges.vllm import detect_hcu

    assert torch.cuda.is_available() and detect_hcu().detected, 'requires an HCU device'
    torch.manual_seed(625914 + rows)
    x = torch.randn(rows, 8, 4096, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(8, 1024, 4096, device='cuda', dtype=torch.bfloat16) / 64
    result = wo._token_major_bmm(torch, x, w)
    expected = torch.einsum('tgd,grd->tgr', x, w)
    assert result.is_contiguous()
    assert result.flatten(1).data_ptr() == result.data_ptr()
    assert torch.equal(result.view(torch.int16), expected.contiguous().view(torch.int16))
