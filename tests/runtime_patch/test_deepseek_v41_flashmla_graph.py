# SPDX-License-Identifier: Apache-2.0
"""Graph rollout gates and fresh-plan ownership; no GPU needed."""

import pytest
import torch

from tests.runtime_patch.test_deepseek_v41_flashmla_decode import (
    enabled,  # noqa: F401 -- pytest fixture
    forbidden,
    inputs,
)
from vllm_hcu.platforms import envs
from vllm_hcu.v1.attention.ops import deepseek_v41_flashmla_decode as runtime
from vllm_hcu.v1.attention.ops.deepseek_v41_flashmla_metadata import DecodeScheduler


@pytest.mark.parametrize("graph,decode,master,expected", [
    (None, None, None, False), ("1", None, None, True),
    ("0", "1", "1", False), ("1", "0", "1", False),
    ("1", "1", "0", False), ("true", "true", "true", True),
])
def test_graph_rollout_switch(monkeypatch, graph, decode, master, expected):
    for name, value in (
        ("VLLM_HCU_DSV41_FLASHMLA_DECODE_GRAPH", graph),
        ("VLLM_HCU_DSV41_FLASHMLA_DECODE", decode),
        ("VLLM_HCU_USE_CUSTOM_OPS", master),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert envs.dsv41_flashmla_decode_graph_enabled() is expected
    assert envs.hcu_vllm_environment_variables["VLLM_HCU_DSV41_FLASHMLA_DECODE_GRAPH"]() is expected


@pytest.mark.parametrize("opt_in,managed,allow_graph", [
    (False, True, True), (True, False, False), (True, True, False),
])
def test_capture_fallback_leaves_plan_and_output_untouched(enabled, monkeypatch, opt_in, managed, allow_graph):
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_DECODE_GRAPH", str(int(opt_in)))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(runtime, "_decode_bindings", forbidden)
    plan = object()
    scheduler = DecodeScheduler(lambda: (plan, None), allow_graph=allow_graph)
    args = inputs()
    args["scheduler"] = scheduler if managed else object()
    assert not runtime.try_flashmla_decode(**args)
    assert torch.all(args["output"] == 7)
    assert scheduler.take() is plan


def test_warmup_and_capture_always_initialize_different_plans(enabled, monkeypatch):
    monkeypatch.setenv("VLLM_HCU_DSV41_FLASHMLA_DECODE_GRAPH", "1")
    launched = []
    created = []

    def factory():
        plan = object()
        created.append(plan)
        return plan, None

    def kernel(**kwargs):
        launched.append(kwargs["tile_scheduler_metadata"])
        return torch.zeros_like(kwargs["q"]), None

    args = inputs()
    args["scheduler"] = DecodeScheduler(factory, allow_graph=True)
    monkeypatch.setattr(runtime, "_decode_bindings", lambda: (kernel, forbidden, forbidden))
    assert runtime.try_flashmla_decode(**args)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert runtime.try_flashmla_decode(**args)
    assert created == launched and len(created) == 2
    assert created[0] is not created[1]
