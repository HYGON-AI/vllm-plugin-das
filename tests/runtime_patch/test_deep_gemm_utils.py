# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_permute_function():
    source_path = (
        Path(__file__).parents[2]
        / "vllm_hcu/model_executor/layers/fused_moe/deep_gemm_utils.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "deepgemm_moe_permute"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    namespace: dict[str, object] = {
        "torch": torch,
        "mk": SimpleNamespace(ExpertTokensMetadata=object),
        "_HCU_TOKEN_ALIGNMENT": 256,
        "round_up": lambda value, multiple: (
            (value + multiple - 1) // multiple * multiple
        ),
    }
    exec(compile(ast.fix_missing_locations(module), source_path, "exec"), namespace)
    return namespace


def test_rocm_permute_uses_hcu_alignment_without_upstream_query():
    namespace = _load_permute_function()
    namespace["current_platform"] = SimpleNamespace(is_rocm=lambda: True)
    namespace["get_mk_alignment_for_contiguous_layout"] = lambda: (
        _ for _ in ()
    ).throw(AssertionError("upstream query invoked"))
    namespace["count_expert_num_tokens"] = lambda *_args: torch.ones(
        2, dtype=torch.int32
    )
    namespace["compute_aligned_M_and_alignment"] = (
        lambda **kwargs: (512, kwargs["alignment"])
    )
    scatter: dict[str, object] = {}
    namespace["ep_scatter"] = lambda **kwargs: scatter.update(kwargs)

    result = namespace["deepgemm_moe_permute"](
        aq=torch.zeros((2, 4), dtype=torch.int8),
        aq_scale=torch.ones((2, 1), dtype=torch.float32),
        topk_ids=torch.tensor([[0], [1]], dtype=torch.int32),
        local_num_experts=2,
        expert_map=None,
        expert_tokens_meta=None,
    )

    assert result[-1] == 256
    assert scatter["align_m"] == 256
