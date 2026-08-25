# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

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


def _load_deep_gemm_apply():
    source_path = (
        Path(__file__).parents[2]
        / "vllm_hcu/model_executor/layers/fused_moe/experts/deep_gemm_moe.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    experts = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DeepGemmExperts"
    )
    function = next(
        node
        for node in experts.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply"
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
        "MoEActivation": SimpleNamespace(SILU="silu"),
        "nullcontext": nullcontext,
        "_HCU_HT_EP_TOKEN_ALIGNMENT": 256,
        "compute_aligned_M_and_alignment": lambda **_kwargs: (2, 256),
        "get_mk_alignment_for_contiguous_layout": lambda: (128, 128),
        "_resize_cache": lambda tensor, shape: torch.zeros(
            shape, dtype=tensor.dtype
        ),
        "deepgemm_moe_permute": lambda **kwargs: (
            kwargs["aq"],
            kwargs["aq_scale"],
            torch.zeros(2, dtype=torch.int32),
            torch.zeros_like(kwargs["topk_ids"], dtype=torch.int32),
            256,
        ),
        "deepgemm_unpermute_and_reduce": lambda **_kwargs: None,
        "m_grouped_fp8_gemm_nt_contiguous": lambda *_args, **_kwargs: None,
    }
    exec(compile(ast.fix_missing_locations(module), source_path, "exec"), namespace)
    return namespace


def _run_w8a8_apply(namespace: dict[str, object]) -> None:
    experts = SimpleNamespace(
        block_shape=(128, 128),
        quant_config=SimpleNamespace(use_int8_w8a8=True, use_fp8_w8a8=False),
        w1_scale=torch.ones((1, 1), dtype=torch.float32),
        w2_scale=torch.ones((1, 1), dtype=torch.float32),
        _hcu_logical_n=4,
        _hcu_logical_k=4,
        mxfp8=False,
        adjust_N_for_activation=lambda size, _activation: size // 2,
    )
    namespace["apply"](
        experts,
        output=torch.empty((2, 4)),
        hidden_states=torch.zeros((2, 4), dtype=torch.int8),
        w1=torch.zeros((1, 4, 4), dtype=torch.int8),
        w2=torch.zeros((1, 4, 4), dtype=torch.int8),
        topk_weights=torch.ones((2, 1)),
        topk_ids=torch.zeros((2, 1), dtype=torch.int32),
        activation="silu",
        global_num_experts=1,
        expert_map=None,
        a1q_scale=torch.ones((2, 1)),
        a2_scale=None,
        workspace13=torch.empty(32, dtype=torch.int8),
        workspace2=torch.empty(32),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )


def test_w8a8_apply_skips_alignment_scope_only_on_rocm(monkeypatch):
    lightop = ModuleType("lightop")
    lightop.m_grouped_w8a8_gemm_nt_contig_asm = lambda *_args: None
    lightop.fuse_silu_mul_quant = lambda tensor, **kwargs: (
        kwargs["output"],
        torch.ones((tensor.shape[0], 1)),
    )
    monkeypatch.setitem(sys.modules, "lightop", lightop)

    hcu = _load_deep_gemm_apply()
    hcu["current_platform"] = SimpleNamespace(is_rocm=lambda: True)
    hcu["mk_alignment_scope"] = lambda _alignment: (_ for _ in ()).throw(
        AssertionError("upstream alignment scope invoked")
    )
    _run_w8a8_apply(hcu)

    events: list[object] = []

    class RecordingScope:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_args):
            events.append("exit")

    non_hcu = _load_deep_gemm_apply()
    non_hcu["current_platform"] = SimpleNamespace(is_rocm=lambda: False)
    non_hcu["mk_alignment_scope"] = lambda alignment: (
        events.append(("scope", alignment)) or RecordingScope()
    )
    _run_w8a8_apply(non_hcu)

    assert events == [("scope", 256), "enter", "exit"]
