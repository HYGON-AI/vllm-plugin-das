# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO = Path(__file__).resolve().parents[2]
MODEL_SOURCE = (REPO / "vllm_hcu/models/deepseek_v2.py").read_text(
    encoding="utf-8"
)


def _load_model_helpers():
    tree = ast.parse(MODEL_SOURCE)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name
        in {"_try_load_quantized_indexer_wk", "_rewrite_stacked_param_name"}
    ]
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "GroupShape": lambda rows, columns: (rows, columns),
        "scaled_dequantize": lambda weight, scale, *, group_shape, out_dtype: (
            weight.to(out_dtype) * 0 + scale.flatten()[0].to(out_dtype)
        ),
    }
    exec(compile(module, "deepseek_v2_helpers", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_quantized_indexer_wk_loader_source_contract_supports_int8_and_fp8(
    dtype: torch.dtype,
):
    helpers = _load_model_helpers()
    loader = helpers["_try_load_quantized_indexer_wk"]
    calls: list[tuple[torch.Tensor, int]] = []
    param = SimpleNamespace(
        weight_loader=lambda _param, tensor, shard: calls.append((tensor, shard))
    )
    params = {"model.layers.0.self_attn.indexer.wk_weights_proj.weight": param}
    pending: dict[str, dict[str, torch.Tensor]] = {}
    loaded: set[str] = set()
    weight = torch.ones((4, 8), dtype=dtype)
    scale = torch.full((2, 2), 2.0)

    assert loader(
        "model.layers.0.self_attn.indexer.wk.weight",
        weight,
        pending,
        params,
        loaded,
    )
    assert calls == []
    assert loader(
        "model.layers.0.self_attn.indexer.wk.weight_scale",
        scale,
        pending,
        params,
        loaded,
    )
    assert calls[0][0].dtype == torch.bfloat16
    assert calls[0][1] == 0
    assert loaded == set(params)
    assert pending == {}


def test_quantized_indexer_wk_loader_source_contract_rejects_bad_scale_shape():
    helpers = _load_model_helpers()
    loader = helpers["_try_load_quantized_indexer_wk"]
    prefix = "model.layers.0.self_attn.indexer"
    pending = {
        prefix: {
            "weight": torch.ones((4, 8), dtype=torch.int8),
        }
    }
    with pytest.raises(ValueError, match="not divisible"):
        loader(
            f"{prefix}.wk.weight_scale",
            torch.ones((3, 2)),
            pending,
            {},
            set(),
        )


def test_stacked_name_rewrite_source_contract_is_component_bounded():
    rewrite = _load_model_helpers()["_rewrite_stacked_param_name"]
    name = "model.gate_proj_alias.gate_proj.weight"
    assert rewrite(name, "gate_proj", "gate_up_proj") == (
        "model.gate_proj_alias.gate_up_proj.weight"
    )
