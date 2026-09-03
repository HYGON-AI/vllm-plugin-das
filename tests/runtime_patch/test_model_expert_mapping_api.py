# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import ast
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers import fused_moe


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_method(
    relative_path: str,
    class_name: str,
    method_name: str,
    namespace: dict[str, object],
):
    path = REPO_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    function_module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias("annotations")],
                level=0,
            ),
            method,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(function_module)
    exec(compile(function_module, str(path), "exec"), namespace)
    return namespace[method_name]


@pytest.mark.parametrize(
    ("relative_path", "class_name", "config", "expert_names"),
    [
        (
            "vllm_hcu/models/hy_v3.py",
            "HYV3Model",
            SimpleNamespace(num_experts=2),
            ("gate_proj", "down_proj", "up_proj"),
        ),
        (
            "vllm_hcu/models/deepseek_v4.py",
            "DeepseekV4Model",
            SimpleNamespace(n_routed_experts=2),
            ("w1", "w2", "w3"),
        ),
    ],
)
def test_model_expert_mapping_uses_vllm_0251_function_api(
    relative_path: str,
    class_name: str,
    config: SimpleNamespace,
    expert_names: tuple[str, str, str],
) -> None:
    get_expert_mapping = _load_method(
        relative_path,
        class_name,
        "get_expert_mapping",
        {
            "FusedMoE": fused_moe.FusedMoE,
            "fused_moe_make_expert_params_mapping": (
                fused_moe.fused_moe_make_expert_params_mapping
            ),
            "islice": islice,
            "make_deepseek_v4_expert_params_mapping": lambda _count: [],
        },
    )
    model = SimpleNamespace(config=config)
    if class_name == "DeepseekV4Model":
        model.start_layer = 0
        model.end_layer = 1
        model.layers = [SimpleNamespace(ffn=SimpleNamespace(use_mega_moe=False))]
    model.named_parameters = lambda: []

    mapping = get_expert_mapping(model)

    gate_name, down_name, up_name = expert_names
    assert mapping == [
        ("experts.routed_experts.w13_", f"experts.0.{gate_name}.", 0, "w1"),
        ("experts.routed_experts.w2_", f"experts.0.{down_name}.", 0, "w2"),
        ("experts.routed_experts.w13_", f"experts.0.{up_name}.", 0, "w3"),
        ("experts.routed_experts.w13_", f"experts.1.{gate_name}.", 1, "w1"),
        ("experts.routed_experts.w2_", f"experts.1.{down_name}.", 1, "w2"),
        ("experts.routed_experts.w13_", f"experts.1.{up_name}.", 1, "w3"),
    ]


def test_hy_v3_mtp_precomputes_expert_mapping_with_vllm_0251_api() -> None:
    load_weights = _load_method(
        "vllm_hcu/models/hy_v3_mtp.py",
        "HYV3MTP",
        "load_weights",
        {
            "FusedMoE": fused_moe.FusedMoE,
            "fused_moe_make_expert_params_mapping": (
                fused_moe.fused_moe_make_expert_params_mapping
            ),
            "_get_cla_factor": lambda _config: 1,
            "_is_moe": lambda _config: True,
            "torch": torch,
        },
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=8,
            num_experts=2,
            num_hidden_layers=1,
        ),
        quant_config=None,
        use_pp=False,
        named_parameters=lambda: [],
        _split_qkv_weight=lambda value: value,
    )

    assert load_weights(model, []) is None


def test_deepseek_v4_mtp_precomputes_expert_mapping_with_vllm_0251_api() -> None:
    load_weights = _load_method(
        "vllm_hcu/models/deepseek_v4_mtp.py",
        "DeepSeekV4MTP",
        "load_weights",
        {
            "FusedMoE": fused_moe.FusedMoE,
            "fused_moe_make_expert_params_mapping": (
                fused_moe.fused_moe_make_expert_params_mapping
            ),
            "get_tensor_model_parallel_world_size": lambda: 1,
            "get_tensor_model_parallel_rank": lambda: 0,
            "make_deepseek_v4_expert_params_mapping": lambda _count: [],
            "logger": SimpleNamespace(info_once=lambda *_args: None),
            "torch": torch,
        },
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_attention_heads=4,
            n_routed_experts=2,
            expert_dtype="fp8",
        ),
        quant_config=None,
        named_parameters=lambda: [],
        model=SimpleNamespace(
            layers={
                "1": SimpleNamespace(
                    mtp_block=SimpleNamespace(
                        ffn=SimpleNamespace(use_mega_moe=False)
                    )
                )
            },
            mtp_start_layer_idx=1,
            num_mtp_layers=0,
        ),
        finalize_mega_moe_weights=lambda: None,
    )

    assert load_weights(model, []) == set()
