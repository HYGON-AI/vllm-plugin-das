# SPDX-License-Identifier: Apache-2.0
"""DFly dtype audit using its production loader AST and real Torch storage.

Only the final AutoWeightsLoader assignment and CT config identity are doubles;
these tests do not claim installed Core loading or HCU checkpoint inference.
"""
import ast
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Iterator

import pytest
import torch
from torch import nn

SOURCE = Path(__file__).resolve().parents[3] / "vllm_hcu/models/hy_v4_flash/nvidia/dfly.py"


class CTConfig:
    pass


class RecordingLoader:
    def __init__(self, model):
        self.model = model

    def load_weights(self, weights, mapper):
        self.model.received = {name: weight.clone() for name, weight in weights}
        return set(self.model.received)


def draft(weight_dtype=torch.int8, ct=True, scale_suffix="weight_scale"):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "HYV4DFlyForCausalLM")
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef)
             and n.name in ("_required_weights", "load_weights")]
    ns = dict(torch=torch, Iterable=Iterable, Iterator=Iterator, chain=chain,
              CompressedTensorsConfig=CTConfig, AutoWeightsLoader=RecordingLoader,
              _slice_sink_for_tp=lambda weight, **kw: weight,
              get_tensor_model_parallel_world_size=lambda: 1,
              get_tensor_model_parallel_rank=lambda: 0)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    loader = type("ProductionDFlyLoader", (nn.Module,), {
        "_required_weights": ns["_required_weights"], "load_weights": ns["load_weights"],
        "checkpoint_prefix": "model.dfly_layers.", "_ignored_prefixes": ("confidence_head.",),
        "_weight_mapper": object(),
    })()
    loader.model = nn.Module()
    loader.model.quant_config = CTConfig() if ct else None
    loader.model.mtp_fusion = nn.Module()
    proj = loader.model.mtp_fusion.eh_proj = nn.Module()
    proj.weight = nn.Parameter(torch.zeros((2, 4), dtype=weight_dtype), requires_grad=False)
    proj.register_parameter(scale_suffix, nn.Parameter(torch.zeros((2, 1)), requires_grad=False))
    loader.model.norm = nn.Module()
    loader.model.norm.weight = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.learnable_sink_param = nn.Parameter(torch.zeros(2))
    loader.model.layers = nn.ModuleList([layer])
    loader.model.ffn_width = 8
    loader.model.resolve_post_proj_norm = lambda seen: set()
    loader.config = SimpleNamespace(hidden_size=4, num_attention_heads=2)
    return loader


def checkpoint(loader, scale_dtype, scale_suffix="weight_scale"):
    return [("model.dfly_layers.mtp_fusion.eh_proj.weight",
             torch.ones((2, 4), dtype=loader.model.mtp_fusion.eh_proj.weight.dtype)),
            ("model.dfly_layers.mtp_fusion.eh_proj." + scale_suffix,
             torch.tensor([[1.125], [.03125]], dtype=scale_dtype)),
            ("model.dfly_layers.norm.weight", torch.ones(4, dtype=torch.bfloat16)),
            ("model.dfly_layers.0.self_attn.learnable_sink_param", torch.ones(2, dtype=torch.bfloat16))]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ct_int8_channel_scale_loads_losslessly(dtype):
    loader = draft()
    weights = checkpoint(loader, dtype)
    loader.load_weights(weights)
    actual = loader.received["model.mtp_fusion.eh_proj.weight_scale"]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, weights[1][1].float(), rtol=0, atol=0)
    assert loader.received["model.mtp_fusion.eh_proj.weight"].dtype == torch.int8


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int8])
def test_other_int8_scale_dtypes_still_fail(dtype):
    with pytest.raises(ValueError, match="weight_scale requires"):
        draft().load_weights(checkpoint(draft(), dtype))


@pytest.mark.parametrize("weight_dtype,ct", [
    (torch.bfloat16, True), (torch.float8_e4m3fn, True), (torch.int8, False),
])
def test_bf16_promotion_is_limited_to_ct_int8_weights(weight_dtype, ct):
    loader = draft(weight_dtype, ct)
    with pytest.raises(ValueError, match="weight_scale requires"):
        loader.load_weights(checkpoint(loader, torch.bfloat16))


@pytest.mark.parametrize("leaf", ["norm.weight", "0.self_attn.learnable_sink_param"])
def test_norm_and_sink_checks_stay_strict(leaf):
    loader = draft()
    weights = checkpoint(loader, torch.float32)
    weights = [(name, value.float() if name == "model.dfly_layers." + leaf else value)
               for name, value in weights]
    with pytest.raises(ValueError, match="requires"):
        loader.load_weights(weights)


def test_existing_fp8_inverse_scale_promotion_is_preserved():
    loader = draft(torch.float8_e4m3fn, scale_suffix="weight_scale_inv")
    weights = checkpoint(loader, torch.bfloat16, "weight_scale_inv")
    loader.load_weights(weights)
    actual = loader.received["model.mtp_fusion.eh_proj.weight_scale_inv"]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, weights[1][1].float(), rtol=0, atol=0)
