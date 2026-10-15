# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""EPLB topology diagnostics must preserve the detected topology."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    "num_gpus,num_nodes,expected_nodes",
    [(6, 4, 1), (8, 4, 4)],
)
def test_rearrange_logs_detected_nodes_before_fallback(
    monkeypatch, num_gpus, num_nodes, expected_nodes
):
    from vllm_hcu.distributed import eplb_runtime as runtime

    warnings = []
    monkeypatch.setattr(runtime, "get_node_count", lambda: num_nodes)
    monkeypatch.setattr(
        runtime,
        "get_ep_group",
        lambda: SimpleNamespace(
            device_group=SimpleNamespace(size=lambda: num_gpus, rank=lambda: 1)
        ),
    )
    monkeypatch.setattr(
        runtime, "logger", SimpleNamespace(warning_once=warnings.append)
    )
    monkeypatch.setattr(runtime, "EplbStats", SimpleNamespace)
    model_state = SimpleNamespace(
        model=SimpleNamespace(
            num_logical_experts=2,
            num_physical_experts=2,
            num_expert_groups=1,
            num_moe_layers=1,
        ),
        expert_load_window=torch.ones(1, 1, 2),
        physical_to_logical_map=torch.tensor([[0, 1]]),
    )
    state = SimpleNamespace(
        is_async=True,
        model_states={"model": model_state},
        expert_load_window_size=1,
        _allreduce_list=lambda windows: windows,
        rearrange_event=SimpleNamespace(record=lambda: None),
    )

    runtime.rearrange(state)

    assert model_state.eplb_stats.num_nodes == expected_nodes
    assert model_state.eplb_stats.num_gpus == num_gpus
    if expected_nodes == 1:
        assert len(warnings) == 1
        assert f"num_gpus={num_gpus}, num_nodes={num_nodes}" in warnings[0]
    else:
        assert warnings == []
