# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Live-HCU graph safety for the missing vLLM MoE-align ABI fallback."""

from __future__ import annotations

import pytest
import torch

from vllm.utils.math_utils import round_up
from vllm_hcu.patch.worker.op_opt.moe.patch_moe_align_block_size import (
    _torch_moe_align_block_size,
)


pytestmark = pytest.mark.hcu


def test_torch_moe_align_fallback_replays_in_hcu_graph() -> None:
    topk_ids = torch.tensor([[0, 1], [1, 2]], device="cuda", dtype=torch.int32)

    # Warm allocator and kernels before capture.
    _torch_moe_align_block_size(
        torch, topk_ids, 2, 3, None, False, False, round_up
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sorted_ids, expert_ids, count = _torch_moe_align_block_size(
            torch, topk_ids, 2, 3, None, False, False, round_up
        )

    topk_ids.copy_(torch.tensor([[2, 2], [0, 1]], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    assert count.item() == 6
    torch.testing.assert_close(
        sorted_ids[: count.item()].cpu(),
        torch.tensor([2, 4, 3, 4, 0, 1], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        expert_ids[:3].cpu(),
        torch.tensor([0, 1, 2], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
