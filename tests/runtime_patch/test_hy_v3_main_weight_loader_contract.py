# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_hy_v3_uses_official_main_auto_weights_loader_api() -> None:
    source = (ROOT / "vllm_hcu/models/hy_v3.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    causal_cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HYV3ForCausalLM"
    )
    load_weights = next(
        node
        for node in causal_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_weights"
    )
    loader_call = next(
        node
        for node in ast.walk(load_weights)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AutoWeightsLoader"
    )
    assert len(loader_call.args) == 1
    assert isinstance(loader_call.args[0], ast.Name)
    assert loader_call.args[0].id == "self"
    assert loader_call.keywords == []


def test_hy_v3_loaders_do_not_use_removed_cache_scale_api() -> None:
    model_source = (ROOT / "vllm_hcu/models/hy_v3.py").read_text(encoding="utf-8")
    mtp_source = (ROOT / "vllm_hcu/models/hy_v3_mtp.py").read_text(
        encoding="utf-8"
    )
    assert ".get_cache_scale(" not in model_source
    assert ".get_cache_scale(" not in mtp_source
    assert "maybe_remap_kv_scale_name" in model_source
    assert ".get_cache_scale_mapper()" in mtp_source
    assert "maybe_remap_kv_scale_name" in mtp_source
