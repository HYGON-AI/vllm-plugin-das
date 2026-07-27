# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CASES = (
    ("vllm_hcu/models/deepseek_v2.py", "DeepseekV2MLP", "deepseek"),
    ("vllm_hcu/models/deepseek_v2.py", "DeepseekV2SharedMLP", "plain"),
    ("vllm_hcu/models/hy_v3.py", "HYV3FeedForward", "sp"),
    ("vllm_hcu/models/glm4_moe.py", "Glm4MoeMLP", "sp"),
)


class _GateUp:
    def __init__(self, gate_up: torch.Tensor) -> None:
        self.gate_up = gate_up

    def __call__(self, value, x_and_scale_quanted=None):
        return self.gate_up, None

    def _forward_with_hcu_quanted(self, value, x_and_scale_quanted):
        return self(value, x_and_scale_quanted=x_and_scale_quanted)


class _FusedAct:
    def __init__(self, xq: torch.Tensor, xs: torch.Tensor) -> None:
        self.xq = xq
        self.xs = xs

    def __call__(self, value, quant_dtype=None):
        return self.xq, self.xs


class _FloatAct:
    def __init__(self, activated: torch.Tensor) -> None:
        self.activated = activated

    def __call__(self, value):
        return self.activated


class _Down:
    def __init__(self, output: torch.Tensor) -> None:
        self.output = output
        self.calls: list[tuple[torch.Tensor, object]] = []

    def __call__(self, value, x_and_scale_quanted=None):
        self.calls.append((value, x_and_scale_quanted))
        return self.output, None

    def _forward_with_hcu_quanted(self, value, x_and_scale_quanted):
        return self(value, x_and_scale_quanted=x_and_scale_quanted)


def _load_forward(relative_path: str, class_name: str, mode: str):
    path = REPO_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    forward = next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "forward"
    )
    namespace = {
        "torch": torch,
        "get_forward_context": lambda: SimpleNamespace(enable_lightly_cp=False),
        "tensor_model_parallel_all_gather": lambda value, dim: value,
        "tensor_model_parallel_all_reduce": lambda value: value,
        "tensor_model_parallel_reduce_scatter": lambda value, dim: value,
        "prepare_mlp_inputs_for_sp": lambda value, quanted, enabled: (
            value,
            quanted,
        ),
        "finalize_mlp_output_for_sp": lambda value, enabled: value,
    }
    function_module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0), forward],
        type_ignores=[],
    )
    ast.fix_missing_locations(function_module)
    exec(compile(function_module, str(path), "exec"), namespace)
    return namespace["forward"]


def _load_top_level_function(relative_path: str, function_name: str):
    path = REPO_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    namespace = {}
    ast.fix_missing_locations(function)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function_name]


def _make_layer(*, fused: bool, mode: str):
    gate_up = torch.ones(2, 8)
    xq = torch.ones(2, 4, dtype=torch.int8)
    xs = torch.ones(2, 1, dtype=torch.float32)
    activated = torch.full((2, 4), 2.0)
    output = torch.full((2, 3), 3.0)
    down = _Down(output)
    layer = SimpleNamespace(
        gate_up_proj=_GateUp(gate_up),
        down_proj=down,
        enable_fuse_silu_mul_quant=fused,
        quant_dtype=torch.int8,
        act_fn=_FusedAct(xq, xs) if fused else _FloatAct(activated),
    )
    if mode == "deepseek":
        layer.tp_size = 1
    elif mode == "sp":
        layer.use_sp_token_gather = False
    return layer, down, xq, xs, activated, output


@pytest.mark.parametrize("relative_path,class_name,mode", CASES)
def test_fused_mlp_down_projection_uses_quantized_activation_contract(
    relative_path, class_name, mode
):
    forward = _load_forward(relative_path, class_name, mode)
    layer, down, xq, xs, _, output = _make_layer(fused=True, mode=mode)

    result = forward(layer, torch.zeros(2, 3))

    assert result is output
    assert len(down.calls) == 1
    assert down.calls[0][0] is xq
    assert down.calls[0][1] == (xq, xs)


@pytest.mark.parametrize("relative_path,class_name,mode", CASES)
def test_feature_off_mlp_down_projection_uses_float_activation(
    relative_path, class_name, mode
):
    forward = _load_forward(relative_path, class_name, mode)
    layer, down, _, _, activated, output = _make_layer(fused=False, mode=mode)

    result = forward(layer, torch.zeros(2, 3))

    assert result is output
    assert down.calls == [(activated, None)]


@pytest.mark.parametrize(
    "relative_path",
    ("vllm_hcu/models/hy_v3.py", "vllm_hcu/models/glm4_moe.py"),
)
def test_moe_models_reject_triton_cache_layout_for_fused_qkv_store(relative_path):
    supports = _load_top_level_function(
        relative_path, "fused_qkv_cache_layout_supported"
    )

    assert not supports(SimpleNamespace(get_name=lambda: "TRITON_ATTN"))
    assert supports(SimpleNamespace(get_name=lambda: "FLASH_ATTN"))
