# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_hcu.v1.attention.ops.deepseek_v4_topk import compute_global_topk_indices_and_lens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_map_topk_bounds_and_valid_prefix():
    device = "cuda"
    table = torch.tensor([[3, 99, -1, 99, 8, 99], [6, 99, 7, 99, 9, 99]],
                         device=device, dtype=torch.int32)[:, ::2]
    local = torch.full((5, 1031), -1, device=device, dtype=torch.int32)
    local[:, :4] = torch.tensor([2, 8, 0, 5], device=device)
    local[:, 1026] = 3
    requests = torch.tensor([1, 0, 2, -1, 0], device=device, dtype=torch.int32)
    valid = torch.tensor([True, True, True, True, False], device=device)

    slots, lengths = compute_global_topk_indices_and_lens(local, requests, table, 2, valid)

    expected = torch.full_like(local, -1)
    expected[0, :4] = torch.tensor([14, 12, 19, 15], device=device)
    expected[1, :2] = torch.tensor([6, 17], device=device)
    torch.testing.assert_close(slots, expected)
    assert lengths.tolist() == [4, 2, 0, 0, 0]
