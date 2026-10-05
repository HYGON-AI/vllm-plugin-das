# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Padding rows must not enter DeepEP low-latency expert dispatch."""

from types import SimpleNamespace

import torch
from vllm import envs, forward_context
from vllm_hcu.model_executor.layers.fused_moe.router_runtime import (
    make_hcu_grouped_topk_router,
)


def test_hcu_grouped_router_masks_padding_after_routing(monkeypatch):
    class BaseRouter:
        pass

    router = make_hcu_grouped_topk_router(BaseRouter)()
    weights = torch.tensor([[0.25, 0.75], [0.4, 0.6], [0.3, 0.7]])
    ids = torch.tensor([[1, 2], [3, 4], [5, 6]])
    router._compute_routing_unmasked = lambda *args, **kwargs: (
        weights.clone(),
        ids.clone(),
    )
    monkeypatch.setattr(envs, "VLLM_MOE_SKIP_PADDING", True)
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(is_padding=torch.tensor([False, True, False])),
    )

    masked_weights, masked_ids = router._compute_routing(None, None, None)
    torch.testing.assert_close(masked_ids[0], ids[0])
    torch.testing.assert_close(masked_ids[1], torch.tensor([-1, -1]))
    torch.testing.assert_close(masked_ids[2], ids[2])
    torch.testing.assert_close(masked_weights[1], torch.zeros(2))

    monkeypatch.setattr(envs, "VLLM_MOE_SKIP_PADDING", False)
    normal_weights, normal_ids = router._compute_routing(None, None, None)
    torch.testing.assert_close(normal_weights, weights)
    torch.testing.assert_close(normal_ids, ids)
