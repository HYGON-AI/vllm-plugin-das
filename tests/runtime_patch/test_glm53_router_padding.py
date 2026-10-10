# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Padding rows must not enter DeepEP low-latency expert dispatch."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import torch
from vllm import envs, forward_context

from vllm_hcu.model_executor.layers.fused_moe import router_runtime
from vllm_hcu.model_executor.layers.fused_moe.router_runtime import (
    make_hcu_grouped_topk_router,
)


def test_hcu_grouped_router_masks_padding_after_lightop(monkeypatch):
    from vllm_hcu.platforms import envs as henvs

    class BaseRouter:
        def _compute_routing(self, *args, **kwargs):
            raise AssertionError("LightOp success should not use official fallback")

    weights = torch.tensor([[0.25, 0.75], [0.4, 0.6], [0.3, 0.7]])
    ids = torch.tensor([[1, 2], [3, 4], [5, 6]])
    lightop = ModuleType("lightop")
    lightop.__path__ = []
    moe = ModuleType("lightop.moe")
    moe.moe_fused_gate = lambda *args, **kwargs: (weights.clone(), ids.clone())
    lightop.moe = moe
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.moe", moe)
    monkeypatch.setattr(router_runtime, "lightop_moe_gate_kwargs", lambda *args: {})
    monkeypatch.setattr(henvs, "optional_custom_op_enabled", lambda *args: True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_FUSE_MOE_GATE", True)

    router = make_hcu_grouped_topk_router(BaseRouter)()
    router.num_expert_group = 2
    router.topk_group = 1
    router.top_k = 2
    router.e_score_correction_bias = torch.ones(4)
    router.routed_scaling_factor = 1.0
    router.scoring_func = "sigmoid"
    router.renormalize = True
    monkeypatch.setattr(envs, "VLLM_MOE_SKIP_PADDING", True)
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(is_padding=torch.tensor([False, True, False])),
    )

    masked_weights, masked_ids = router._compute_routing(
        None, torch.ones((3, 4)), None
    )
    torch.testing.assert_close(masked_ids[0], ids[0])
    torch.testing.assert_close(masked_ids[1], torch.tensor([-1, -1]))
    torch.testing.assert_close(masked_ids[2], ids[2])
    torch.testing.assert_close(masked_weights[1], torch.zeros(2))

    monkeypatch.setattr(envs, "VLLM_MOE_SKIP_PADDING", False)
    normal_weights, normal_ids = router._compute_routing(
        None, torch.ones((3, 4)), None
    )
    torch.testing.assert_close(normal_weights, weights)
    torch.testing.assert_close(normal_ids, ids)


def test_hcu_grouped_router_fallback_does_not_remask(monkeypatch):
    from vllm_hcu.platforms import envs as henvs

    weights = torch.tensor([[0.25, 0.75]])
    ids = torch.tensor([[1, 2]])

    class BaseRouter:
        def _compute_routing(self, *args, **kwargs):
            return weights, ids

    router = make_hcu_grouped_topk_router(BaseRouter)()
    router.e_score_correction_bias = None
    router.num_expert_group = 2
    monkeypatch.setattr(henvs, "optional_custom_op_enabled", lambda *args: False)
    monkeypatch.setattr(envs, "VLLM_MOE_SKIP_PADDING", True)

    def unexpected_context_check():
        raise AssertionError("fallback routing must not apply a second padding mask")

    monkeypatch.setattr(
        forward_context, "is_forward_context_available", unexpected_context_check
    )
    actual_weights, actual_ids = router._compute_routing(
        None, torch.ones((1, 4)), None
    )
    assert actual_weights is weights
    assert actual_ids is ids


def test_hcu_grouped_router_ignores_short_local_padding_mask(monkeypatch):
    """PCP-gathered routing rows must not use one rank's shorter mask."""
    from vllm.utils import torch_utils

    from vllm_hcu.platforms import envs as henvs

    register_custom_op = torch_utils.direct_register_custom_op

    def register_without_duplicate_moe_ops(op_name, *args, **kwargs):
        if op_name in {"moe_forward", "moe_forward_shared"}:
            return None
        return register_custom_op(op_name, *args, **kwargs)

    monkeypatch.setattr(
        torch_utils,
        "direct_register_custom_op",
        register_without_duplicate_moe_ops,
    )
    moe_runner = importlib.import_module(
        "vllm_hcu.model_executor.layers.fused_moe.moe_runner"
    )

    class BaseRouter:
        def _compute_routing(self, *args, **kwargs):
            raise AssertionError("LightOp success should not use official fallback")

    weights = torch.tensor(
        [[0.25, 0.75], [0.4, 0.6], [0.3, 0.7], [0.45, 0.55]]
    )
    ids = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    lightop = ModuleType("lightop")
    lightop.__path__ = []
    moe = ModuleType("lightop.moe")
    moe.moe_fused_gate = lambda *args, **kwargs: (weights.clone(), ids.clone())
    lightop.moe = moe
    monkeypatch.setitem(sys.modules, "lightop", lightop)
    monkeypatch.setitem(sys.modules, "lightop.moe", moe)
    monkeypatch.setattr(router_runtime, "lightop_moe_gate_kwargs", lambda *args: {})
    monkeypatch.setattr(henvs, "optional_custom_op_enabled", lambda *args: True)
    monkeypatch.setattr(henvs, "VLLM_HCU_USE_FUSE_MOE_GATE", True)

    router = make_hcu_grouped_topk_router(BaseRouter)()
    router.num_expert_group = 2
    router.topk_group = 1
    router.top_k = 2
    router.e_score_correction_bias = torch.ones(4)
    router.routed_scaling_factor = 1.0
    router.scoring_func = "sigmoid"
    router.renormalize = True

    class PCPGroup:
        def all_gather(self, tensor, dim=0):
            return torch.cat((tensor, tensor.clone()), dim=dim)

    runner = object.__new__(moe_runner.MoERunner)
    runner.moe_config = SimpleNamespace(
        pcp_size=2,
        dp_size=1,
        is_sequence_parallel=False,
        moe_parallel_config=SimpleNamespace(use_all2all_kernels=False),
    )
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(supports_internal_mk=False)
    )
    runner._shared_experts = None
    monkeypatch.setattr(moe_runner, "get_pcp_group", lambda: PCPGroup())
    _, gathered_logits = runner._maybe_dispatch(
        torch.ones((2, 4)), torch.ones((2, 4))
    )
    assert gathered_logits is not None
    assert gathered_logits.shape == (4, 4)

    monkeypatch.setattr(envs, "VLLM_MOE_SKIP_PADDING", True)
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(is_padding=torch.tensor([False, True])),
    )

    actual_weights, actual_ids = router._compute_routing(None, gathered_logits, None)

    torch.testing.assert_close(actual_weights, weights)
    torch.testing.assert_close(actual_ids, ids)
