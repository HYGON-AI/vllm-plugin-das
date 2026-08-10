# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_hy_v3_decoder_layer_passes_enable_eplb_to_moe() -> None:
    source = (ROOT / "vllm_hcu/models/hy_v3.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    decoder_cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HYV3DecoderLayer"
    )
    init_fn = next(
        node
        for node in decoder_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    moe_call = next(
        node
        for node in ast.walk(init_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "HYV3MoEFused"
    )

    enable_eplb_kw = next(
        (keyword for keyword in moe_call.keywords if keyword.arg == "enable_eplb"),
        None,
    )
    assert enable_eplb_kw is not None
    assert ast.unparse(enable_eplb_kw.value) == "parallel_config.enable_eplb"


def test_hy_v3_uses_v0251_expert_mapping_function() -> None:
    for relative_path in (
        "vllm_hcu/models/hy_v3.py",
        "vllm_hcu/models/hy_v3_mtp.py",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source)

        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "vllm.model_executor.layers.fused_moe"
            for alias in node.names
        }
        assert "fused_moe_make_expert_params_mapping" in imported_names

        obsolete_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "FusedMoE"
            and node.func.attr == "make_expert_params_mapping"
        ]
        assert obsolete_calls == []


def test_hy_v3_uses_v0251_cache_scale_mapper_api() -> None:
    sources = {
        relative_path: (ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in (
            "vllm_hcu/models/hy_v3.py",
            "vllm_hcu/models/hy_v3_mtp.py",
        )
    }

    for source in sources.values():
        tree = ast.parse(source)
        obsolete_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_cache_scale"
        ]
        assert obsolete_calls == []

    assert "get_cache_scale_mapper()" in sources[
        "vllm_hcu/models/hy_v3_mtp.py"
    ]


def test_fp8_marlin_moe_method_allows_eplb() -> None:
    source = (
        ROOT
        / "vllm_hcu/model_executor/layers/quantization/compressed_tensors/"
        "compressed_tensors_moe_marlin.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    method_cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "CompressedTensorsW8A8FP8MarlinMoEMethod"
    )
    supports_prop = next(
        (
            node
            for node in method_cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "supports_eplb"
            and any(
                isinstance(decorator, ast.Name) and decorator.id == "property"
                for decorator in node.decorator_list
            )
        ),
        None,
    )
    assert supports_prop is not None
    assert any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
        for node in ast.walk(supports_prop)
    )

    apply_fn = next(
        node
        for node in method_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply"
    )
    assert "if enable_eplb:" not in ast.unparse(apply_fn)


def test_hy_v3_models_register_eplb_state() -> None:
    source = (ROOT / "vllm_hcu/models/hy_v3.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    hy_model_cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HYV3Model"
    )
    hy_model_setter = next(
        (
            node
            for node in hy_model_cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "set_eplb_state"
        ),
        None,
    )
    assert hy_model_setter is not None
    hy_model_setter_source = ast.unparse(hy_model_setter)
    assert "self.expert_weights.clear()" in hy_model_setter_source
    assert "layer.set_eplb_state" in hy_model_setter_source
    assert "expert_load_view=expert_load_view" in hy_model_setter_source

    causal_cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HYV3ForCausalLM"
    )
    causal_init = next(
        node
        for node in causal_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    causal_init_source = ast.unparse(causal_init)
    assert "self.expert_weights = self.model.expert_weights" in causal_init_source
    assert "self.moe_layers = self.model.moe_layers" in causal_init_source
    assert "self.num_moe_layers = self.model.num_moe_layers" in causal_init_source

    causal_setter = next(
        (
            node
            for node in causal_cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "set_eplb_state"
        ),
        None,
    )
    assert causal_setter is not None
    assert "self.model.set_eplb_state" in ast.unparse(causal_setter)


def test_hy_v3_mtp_registers_eplb_state() -> None:
    source = (ROOT / "vllm_hcu/models/hy_v3_mtp.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    predictor_cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HYV3MultiTokenPredictor"
    )
    predictor_init = next(
        node
        for node in predictor_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    predictor_init_source = ast.unparse(predictor_init)
    assert "self.expert_weights = []" in predictor_init_source
    assert "self.moe_layers = []" in predictor_init_source
    assert "self.num_moe_layers = len(self.moe_layers)" in predictor_init_source

    predictor_setter = next(
        (
            node
            for node in predictor_cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "set_eplb_state"
        ),
        None,
    )
    assert predictor_setter is not None
    predictor_setter_source = ast.unparse(predictor_setter)
    assert "self.expert_weights.clear()" in predictor_setter_source
    assert "layer.set_eplb_state" in predictor_setter_source

    mtp_cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HYV3MTP"
    )
    mtp_init = next(
        node
        for node in mtp_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    mtp_init_source = ast.unparse(mtp_init)
    assert "self.expert_weights = self.model.expert_weights" in mtp_init_source
    assert "self.moe_layers = self.model.moe_layers" in mtp_init_source

    mtp_setter = next(
        (
            node
            for node in mtp_cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "set_eplb_state"
        ),
        None,
    )
    assert mtp_setter is not None
    assert "self.model.set_eplb_state" in ast.unparse(mtp_setter)
