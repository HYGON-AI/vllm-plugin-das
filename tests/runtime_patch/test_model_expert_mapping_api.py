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


@pytest.mark.parametrize("num_redundant_experts", [0, 1])
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
    num_redundant_experts: int,
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
        model.layers = [
            SimpleNamespace(
                ffn=SimpleNamespace(
                    use_mega_moe=False,
                    n_redundant_experts=num_redundant_experts,
                )
            )
        ]
    else:
        model.num_redundant_experts = num_redundant_experts
    model.named_parameters = lambda: []

    mapping = get_expert_mapping(model)

    gate_name, down_name, up_name = expert_names
    expected_mapping = [
        ("experts.routed_experts.w13_", f"experts.0.{gate_name}.", 0, "w1"),
        ("experts.routed_experts.w2_", f"experts.0.{down_name}.", 0, "w2"),
        ("experts.routed_experts.w13_", f"experts.0.{up_name}.", 0, "w3"),
        ("experts.routed_experts.w13_", f"experts.1.{gate_name}.", 1, "w1"),
        ("experts.routed_experts.w2_", f"experts.1.{down_name}.", 1, "w2"),
        ("experts.routed_experts.w13_", f"experts.1.{up_name}.", 1, "w3"),
    ]
    if num_redundant_experts == 1:
        expected_mapping.extend(
            [
                (
                    "experts.routed_experts.w13_",
                    f"experts.0.{gate_name}.",
                    2,
                    "w1",
                ),
                (
                    "experts.routed_experts.w2_",
                    f"experts.0.{down_name}.",
                    2,
                    "w2",
                ),
                (
                    "experts.routed_experts.w13_",
                    f"experts.0.{up_name}.",
                    2,
                    "w3",
                ),
            ]
        )
    assert mapping == expected_mapping


def test_hy_v3_mtp_precomputes_expert_mapping_with_vllm_0251_api() -> None:
    mapping_calls: list[dict[str, object]] = []

    def record_expert_mapping(model: object, **kwargs: object):
        mapping_calls.append(kwargs)
        return fused_moe.fused_moe_make_expert_params_mapping(model, **kwargs)

    load_weights = _load_method(
        "vllm_hcu/models/hy_v3_mtp.py",
        "HYV3MTP",
        "load_weights",
        {
            "FusedMoE": fused_moe.FusedMoE,
            "fused_moe_make_expert_params_mapping": record_expert_mapping,
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
        num_redundant_experts=1,
        named_parameters=lambda: [],
        _split_qkv_weight=lambda value: value,
    )

    assert load_weights(model, []) is None
    assert mapping_calls == [
        {
            "ckpt_gate_proj_name": "gate_proj",
            "ckpt_down_proj_name": "down_proj",
            "ckpt_up_proj_name": "up_proj",
            "num_experts": 2,
            "num_redundant_experts": 1,
        }
    ]


def test_hy_v3_mtp_loads_local_redundant_expert_weights() -> None:
    loaded_shards: dict[tuple[int, str], float] = {}

    class ExpertParam:
        def weight_loader(
            self,
            _param: object,
            loaded_weight: torch.Tensor,
            _weight_name: str,
            *,
            shard_id: str,
            expert_id: int,
            return_success: bool = False,
        ) -> bool | None:
            if expert_id not in (2, 3):
                return False if return_success else None
            loaded_shards[expert_id, shard_id] = loaded_weight.item()
            return True if return_success else None

    load_weights = _load_method(
        "vllm_hcu/models/hy_v3_mtp.py",
        "HYV3MTP",
        "load_weights",
        {
            "fused_moe_make_expert_params_mapping": (
                fused_moe.fused_moe_make_expert_params_mapping
            ),
            "_get_cla_factor": lambda _config: 1,
            "_is_moe": lambda _config: True,
            "get_spec_layer_idx_from_weight_name": lambda _config, _name: 1,
            "is_pp_missing_parameter": lambda _name, _model: False,
            "torch": torch,
        },
    )
    expert_param = ExpertParam()
    params = {
        "model.layers.1.mlp.experts.routed_experts.w13_weight": expert_param,
        "model.layers.1.mlp.experts.routed_experts.w2_weight": expert_param,
    }
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=8,
            num_experts=2,
            num_hidden_layers=1,
            tie_word_embeddings=False,
        ),
        quant_config=None,
        use_pp=False,
        num_redundant_experts=2,
        named_parameters=lambda: params.items(),
        _split_qkv_weight=lambda value: value,
        _rewrite_spec_layer_name=lambda _spec_layer, name: name,
    )
    weights = [
        (
            f"model.layers.1.mlp.experts.{logical_id}.{projection}.weight",
            torch.tensor(value),
        )
        for logical_id, values in enumerate(((10.0, 11.0, 12.0), (20.0, 21.0, 22.0)))
        for projection, value in zip(
            ("gate_proj", "down_proj", "up_proj"), values, strict=True
        )
    ]

    assert load_weights(model, weights) is None
    assert loaded_shards == {
        (2, "w1"): 10.0,
        (2, "w2"): 11.0,
        (2, "w3"): 12.0,
        (3, "w1"): 20.0,
        (3, "w2"): 21.0,
        (3, "w3"): 22.0,
    }


def test_deepseek_v4_mtp_precomputes_expert_mapping_with_vllm_0251_api() -> None:
    mapping_calls: list[dict[str, object]] = []

    def record_expert_mapping(model: object, **kwargs: object):
        mapping_calls.append(kwargs)
        return fused_moe.fused_moe_make_expert_params_mapping(model, **kwargs)

    load_weights = _load_method(
        "vllm_hcu/models/deepseek_v4_mtp.py",
        "DeepSeekV4MTP",
        "load_weights",
        {
            "FusedMoE": fused_moe.FusedMoE,
            "fused_moe_make_expert_params_mapping": record_expert_mapping,
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
        num_redundant_experts=1,
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
    assert mapping_calls == [
        {
            "ckpt_gate_proj_name": "w1",
            "ckpt_down_proj_name": "w2",
            "ckpt_up_proj_name": "w3",
            "num_experts": 2,
            "num_redundant_experts": 1,
        }
    ]
