# SPDX-License-Identifier: Apache-2.0
"""Verify dense MTP tail configuration with CPU constructor stubs.

These tests do not cover expert checkpoint loading or full MTP execution.
"""

import ast
from copy import copy
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).parents[2]


def load_constructor(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    cls.decorator_list = []
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])),
            path,
            "exec",
        ),
        namespace,
    )
    return namespace[name]


class RecordedAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.args = kwargs


def reject_moe_mtp(*args, **kwargs):
    raise AssertionError("Dense MTP must not construct an MoE block")


class Projection(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.prefix = kwargs.get("prefix")


def constructors():
    ns = dict(
        torch=torch,
        nn=nn,
        copy=copy,
        extract_layer_index=lambda p: int(re.search(r"layers\.(\d+)", p).group(1)),
        Step5Attention=RecordedAttention,
        FusedMoEBlock=reject_moe_mtp,
        get_tp_group=lambda: None,
        get_dp_group=lambda: SimpleNamespace(world_size=2),
        get_tensor_model_parallel_world_size=lambda: 4,
        _fp32_gemma_rms_norm=lambda *a: nn.Identity(),
        MergedColumnParallelLinear=Projection,
        RowParallelLinear=Projection,
        SiluAndMul=nn.Identity,
        SwigluStepAndMul=lambda **k: nn.Identity(),
        FP8_DENSE=False,
        Step5MTPSharedHead=lambda **k: nn.Identity(),
    )
    text = "vllm_hcu/models/step5_text.py"
    load_constructor(text, "Step5MLP", ns)
    load_constructor(text, "Step5DecoderLayer", ns)
    mtp = "vllm_hcu/models/step5_mtp.py"
    tree = ast.parse((ROOT / mtp).read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_get_mtp_config"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, fn], type_ignores=[])),
            mtp,
            "exec",
        ),
        ns,
    )
    return load_constructor(mtp, "Step5MultiTokenPredictorLayer", ns), ns


def config(nested):
    # Synthetic tail settings differ from the main stack to catch index reuse.
    # The main stack includes MoE, but all three MTP layers are dense.
    main_types = ["sliding_attention"] * 92
    text = SimpleNamespace(
        hidden_size=8,
        rms_norm_eps=1e-6,
        intermediate_size=16,
        num_hidden_layers=92,
        num_nextn_predict_layers=3,
        att_impl_type="GQA",
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=2,
        max_position_embeddings=1048576,
        layer_types=main_types,
        layer_types_with_mtp=main_types
        + ["full_attention", "sliding_attention", "full_attention"],
        rope_theta=[10000] * 92 + [500000, 20000, 700000],
        partial_rotary_factors=[1.0] * 92 + [1 / 3, 1 / 2, 1 / 4],
        use_rope_layers=[True] * 92 + [False, True, True],
        swiglu_limits_shared=[7.0] * 92 + [5.0, 6.0, 8.0],
        swiglu_limits=[7.0] * 95,
        moe_layers_enum=",".join(str(idx) for idx in range(3, 91)),
        attention_other_setting=dict(
            attention_type="full_attention",
            num_attention_heads=8,
            num_attention_groups=4,
            head_dim=2,
        ),
    )
    hf = SimpleNamespace(text_config=text) if nested else text
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, dtype=torch.bfloat16),
        quant_config=None,
        cache_config=None,
    )
    return cfg, hf, text


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("idx", [92, 93, 94])
def test_dense_mtp_uses_absolute_tail_layer_configuration(nested, idx):
    cls, _ = constructors()
    cfg, original, text = config(nested)
    layer = cls(cfg, prefix=f"model.layers.{idx}")
    decoder = layer.mtp_block
    assert decoder.layer_idx == idx
    assert decoder.prefix == f"model.layers.{idx}.mtp_block"
    attn = decoder.self_attn.args
    assert attn["prefix"] == f"model.layers.{idx}.mtp_block.self_attn"
    assert attn["layer_types"][idx] == text.layer_types_with_mtp[idx]
    assert attn["rope_theta"][idx] == text.rope_theta[idx]
    assert attn["partial_rotary_factor"] == text.partial_rotary_factors[idx]
    assert attn["use_rope_layers"][idx] == text.use_rope_layers[idx]
    assert attn["num_heads"] == (4 if idx == 93 else 8)
    assert not decoder.use_moe
    assert decoder.mlp.limit == text.swiglu_limits_shared[idx]
    assert (
        decoder.mlp.gate_up_proj.prefix
        == f"model.layers.{idx}.mtp_block.mlp.gate_up_proj"
    )
    assert cfg.model_config.hf_config is original
    assert len(text.layer_types) == 92


def test_failed_mtp_construction_does_not_replace_shared_hf_config():
    cls, ns = constructors()
    cfg, original, _ = config(True)

    def fail(*a, **k):
        raise RuntimeError("decoder failure")

    ns["Step5DecoderLayer"] = fail
    with pytest.raises(RuntimeError, match="decoder failure"):
        cls(cfg, prefix="model.layers.92")
    assert cfg.model_config.hf_config is original
