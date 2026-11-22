# SPDX-License-Identifier: Apache-2.0
"""DFly CT FP8 dtype audit using its production loader AST and real Torch storage.

The production loader methods are extracted from their AST. Most cases use a
recording assignment and CT config identity double; one also exercises installed
Core AutoWeightsLoader assignment. None constructs or runs the full model.
"""

import ast
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Iterator

import pytest
import torch
from torch import nn

SOURCE = (
    Path(__file__).resolve().parents[3] / "vllm_hcu/models/hy_v4_flash/nvidia/dfly.py"
)


class CTConfig:
    pass


class RecordingLoader:
    def __init__(self, model):
        self.model = model

    def load_weights(self, weights, mapper):
        self.model.received = {name: weight.clone() for name, weight in weights}
        return set(self.model.received)


def draft(
    weight_dtype=torch.float8_e4m3fn,
    ct=True,
    scale_suffix="weight_scale",
    *,
    auto_loader=RecordingLoader,
    mapper=None
):
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "HYV4DFlyForCausalLM"
    )
    nodes = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_required_weights", "load_weights")
    ]
    ns = dict(
        torch=torch,
        Iterable=Iterable,
        Iterator=Iterator,
        chain=chain,
        CompressedTensorsConfig=CTConfig,
        AutoWeightsLoader=auto_loader,
        _slice_sink_for_tp=lambda weight, **kw: weight,
        get_tensor_model_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_rank=lambda: 0,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    loader = type(
        "ProductionDFlyLoader",
        (nn.Module,),
        {
            "_required_weights": ns["_required_weights"],
            "load_weights": ns["load_weights"],
            "checkpoint_prefix": "model.dfly_layers.",
            "_ignored_prefixes": ("confidence_head.",),
            "_weight_mapper": mapper,
        },
    )()
    loader.model = nn.Module()
    loader.model.quant_config = CTConfig() if ct else None
    loader.model.mtp_fusion = nn.Module()
    proj = loader.model.mtp_fusion.eh_proj = nn.Module()
    proj.weight = nn.Parameter(
        torch.zeros((2, 4), dtype=weight_dtype), requires_grad=False
    )
    proj.register_parameter(
        scale_suffix, nn.Parameter(torch.zeros((2, 1)), requires_grad=False)
    )
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
    return [
        (
            "model.dfly_layers.mtp_fusion.eh_proj.weight",
            torch.ones((2, 4), dtype=loader.model.mtp_fusion.eh_proj.weight.dtype),
        ),
        (
            "model.dfly_layers.mtp_fusion.eh_proj." + scale_suffix,
            torch.tensor([[1.125], [0.03125]], dtype=scale_dtype),
        ),
        ("model.dfly_layers.norm.weight", torch.ones(4, dtype=torch.bfloat16)),
        (
            "model.dfly_layers.0.self_attn.learnable_sink_param",
            torch.ones(2, dtype=torch.bfloat16),
        ),
    ]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ct_fp8_channel_scale_loads_losslessly(dtype):
    loader = draft()
    weights = checkpoint(loader, dtype)
    loader.load_weights(weights)
    actual = loader.received["model.mtp_fusion.eh_proj.weight_scale"]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, weights[1][1].float(), rtol=0, atol=0)
    assert (
        loader.received["model.mtp_fusion.eh_proj.weight"].dtype == torch.float8_e4m3fn
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int8])
def test_other_fp8_scale_dtypes_still_fail(dtype):
    with pytest.raises(ValueError, match="weight_scale requires"):
        draft().load_weights(checkpoint(draft(), dtype))


@pytest.mark.parametrize(
    "weight_dtype,ct",
    [
        (torch.bfloat16, True),
        (torch.float8_e4m3fn, False),
    ],
)
def test_bf16_promotion_is_limited_to_ct_fp8_weights(weight_dtype, ct):
    loader = draft(weight_dtype, ct)
    with pytest.raises(ValueError, match="weight_scale requires"):
        loader.load_weights(checkpoint(loader, torch.bfloat16))


@pytest.mark.parametrize("leaf", ["norm.weight", "0.self_attn.learnable_sink_param"])
def test_norm_and_sink_checks_stay_strict(leaf):
    loader = draft()
    weights = checkpoint(loader, torch.float32)
    weights = [
        (name, value.float() if name == "model.dfly_layers." + leaf else value)
        for name, value in weights
    ]
    with pytest.raises(ValueError, match="requires"):
        loader.load_weights(weights)


def test_existing_fp8_inverse_scale_promotion_is_preserved():
    loader = draft(torch.float8_e4m3fn, scale_suffix="weight_scale_inv")
    weights = checkpoint(loader, torch.bfloat16, "weight_scale_inv")
    loader.load_weights(weights)
    actual = loader.received["model.mtp_fusion.eh_proj.weight_scale_inv"]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, weights[1][1].float(), rtol=0, atol=0)


@pytest.mark.parametrize("scale_first", [False, True])
def test_scale_does_not_depend_on_checkpoint_iteration_order(scale_first):
    loader = draft()
    weights = checkpoint(loader, torch.bfloat16)
    if scale_first:
        weights[0], weights[1] = weights[1], weights[0]
    before = {name: value.clone() for name, value in weights}
    loader.load_weights(iter(weights))
    actual = loader.received["model.mtp_fusion.eh_proj.weight_scale"]
    torch.testing.assert_close(
        actual, torch.tensor([[1.125], [0.03125]]), rtol=0, atol=0
    )
    for name, value in weights:
        # Byte comparison works even where FP8 equality kernels are unavailable.
        assert torch.equal(value.view(torch.uint8), before[name].view(torch.uint8))


def test_scale_without_fp8_sibling_is_not_promoted():
    loader = draft()
    del loader.model.mtp_fusion.eh_proj.weight
    weights = checkpoint(draft(), torch.bfloat16)[1:]
    with pytest.raises(ValueError, match="weight_scale requires"):
        loader.load_weights(weights)


def test_real_core_auto_weights_loader_assigns_promoted_scale():
    # This is real Core assignment into Torch parameters, not a model forward.
    utils = pytest.importorskip("vllm.model_executor.models.utils")

    class CTLoaderConfig(CTConfig):
        _ignore_unexpected_suffixes = []

        def get_cache_scale_mapper(self):
            return utils.WeightsMapper()

        def get_checkpoint_weight_mapper(self):
            return utils.WeightsMapper()

    loader = draft(auto_loader=utils.AutoWeightsLoader, mapper=utils.WeightsMapper())
    loader.model.quant_config = CTLoaderConfig()
    weights = checkpoint(loader, torch.bfloat16)
    loaded = loader.load_weights(weights)
    name = "model.mtp_fusion.eh_proj.weight_scale"
    assert name in loaded
    actual = loader.model.mtp_fusion.eh_proj.weight_scale
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, weights[1][1].float(), rtol=0, atol=0)
    assert loader.model.mtp_fusion.eh_proj.weight.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize("use_core_loader", [False, True])
def test_hcu_bf16_scale_promotion_preserves_exact_values(use_core_loader):
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("requires a HIP device")
    loader = draft()
    if use_core_loader:
        utils = pytest.importorskip("vllm.model_executor.models.utils")

        # Core's mapper consults the config; identity mappings are sufficient
        # here because checkpoint names have already been normalized by DFly.
        class CTLoaderConfig(CTConfig):
            _ignore_unexpected_suffixes = []

            def get_cache_scale_mapper(self):
                return utils.WeightsMapper()

            def get_checkpoint_weight_mapper(self):
                return utils.WeightsMapper()

        loader = draft(
            auto_loader=utils.AutoWeightsLoader, mapper=utils.WeightsMapper()
        )
        loader.model.quant_config = CTLoaderConfig()
    loader.model.to("cuda:0")
    weights = [
        (name, value.to("cuda:0")) for name, value in checkpoint(loader, torch.bfloat16)
    ]
    original = weights[1][1].clone()
    loader.load_weights(weights)
    actual = (
        loader.model.mtp_fusion.eh_proj.weight_scale
        if use_core_loader
        else loader.received["model.mtp_fusion.eh_proj.weight_scale"]
    )
    assert actual.dtype == torch.float32
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual, original.float(), rtol=0, atol=0)
    torch.testing.assert_close(weights[1][1], original, rtol=0, atol=0)
