# SPDX-License-Identifier: Apache-2.0
"""CPU production-body contracts; native kernels and checkpoint loading are separate."""
import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType

import pytest
import torch

MODEL = Path(__file__).resolve().parents[3] / "vllm_hcu/models"


def function(relative, name):
    tree = ast.parse((MODEL / relative).read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    scope = dict(torch=torch, contextmanager=contextmanager, nullcontext=nullcontext,
                 default_weight_loader=lambda p, w: p.data.copy_(w))
    if relative == "hy_v4/vl_model.py":
        from vllm.model_executor.models.utils import AutoWeightsLoader
        scope["AutoWeightsLoader"] = AutoWeightsLoader
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])),
                 str(MODEL / relative), "exec"), scope)
    return scope[name]


@pytest.mark.parametrize("rows", [1, 33])
@pytest.mark.parametrize("strided", [False, True])
def test_negative_padding_ids_ignore_nan_weights_and_hidden_rows(rows, strided):
    fn = function("hy_v4_flash/nvidia/w4a8.py", "routed_reference")
    ids = torch.tensor([[0, -1, 2], [-1, -1, -1], [1, 2, -1]], dtype=torch.int32)
    ids = ids.repeat((rows + 2) // 3, 1)[:rows]
    weights = torch.where(ids >= 0, 0.25, float("nan")).float()
    if strided:
        ids = ids.repeat_interleave(2, 1)[:, ::2]
        weights = weights.repeat_interleave(2, 1)[:, ::2]
    x = (torch.arange(rows * 4).reshape(rows, 4) / 8).to(torch.bfloat16)
    x[~(ids >= 0).any(-1)] = float("nan")

    def experts(value, *args, **kwargs):
        counts = args[-1]
        valid = torch.arange(value.shape[1])[None, :] < counts[:, None]
        assert value.shape[1] <= 32
        result = value * torch.arange(1, 4)[:, None, None]
        return torch.where(valid[:, :, None], result, float("nan"))

    actual = fn(x, weights, ids, torch.empty(3, 2, 2), None, None, None,
                expert_impl=experts)
    expected = torch.zeros_like(x, dtype=torch.float32)
    for row in range(rows):
        for slot in range(ids.shape[1]):
            expert = int(ids[row, slot])
            if expert >= 0:
                expected[row] += (x[row] * (expert + 1)).float() * weights[row, slot]
    torch.testing.assert_close(actual, expected.to(x.dtype), rtol=0, atol=0)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("invalid", [-2, 3])
def test_other_out_of_range_expert_ids_still_reject(invalid):
    fn = function("hy_v4_flash/nvidia/w4a8.py", "routed_reference")
    with pytest.raises(RuntimeError):
        fn(torch.ones(1, 4), torch.ones(1, 1), torch.tensor([[invalid]]),
           torch.empty(3, 2, 2), None, None, None,
           expert_impl=lambda *a, **kw: pytest.fail("invalid route reached expert"))


def vision_shell():
    params = {}
    for layer in range(2):
        for suffix, shape in (("weight", (6, 2)), ("bias", (6,))):
            param = torch.nn.Parameter(torch.full(shape, float("nan")), requires_grad=False)
            def loader(p, value, shard=None):
                if shard is None:
                    p.data.copy_(value)
                else:
                    p.data.chunk(3)[("q", "k", "v").index(shard)].copy_(value)
            param.weight_loader = loader
            params[f"layers.{layer}.attn.qkv.{suffix}"] = param
    params["norm.weight"] = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    shell = torch.nn.Module()
    for name, param in params.items():
        parent = shell
        *parts, leaf = name.split(".")
        for part in parts:
            if not hasattr(parent, part):
                parent.add_module(part, torch.nn.Module())
            parent = getattr(parent, part)
        parent.register_parameter(leaf, param)
    for name in ("load_weights", "weight_loading_session"):
        setattr(shell, name, MethodType(function("hy_v4/vision.py", name), shell))
    return shell, params


def visual_stream(*, omitted=None, fused=False):
    for layer in range(2):
        for suffix, shape in (("weight", (2, 2)), ("bias", (2,))):
            if fused:
                yield f"layers.{layer}.attn.qkv.{suffix}", torch.full((6, *shape[1:]), 7.0)
            else:
                for index, projection in enumerate(("v_proj", "q_proj", "k_proj")):
                    name = f"layers.{layer}.attn.{projection}.{suffix}"
                    if name != omitted:
                        yield name, torch.full(shape, float(index + 1))
    yield "norm.weight", torch.ones(2)


@pytest.mark.parametrize("projection", ["q_proj", "k_proj", "v_proj"])
@pytest.mark.parametrize("suffix", ["weight", "bias"])
def test_missing_visual_qkv_projection_is_rejected(projection, suffix):
    shell, _ = vision_shell()
    load = function("hy_v4/vision.py", "load_weights")
    with pytest.raises(ValueError) as error:
        load(shell, visual_stream(omitted=f"layers.1.attn.{projection}.{suffix}"))
    assert f"layers.1.attn.qkv.{suffix}" in str(error.value)


@pytest.mark.parametrize("fused", [False, True])
def test_complete_visual_qkv_stream_accepts_sharded_or_fused_parameters(fused):
    shell, params = vision_shell()
    load = function("hy_v4/vision.py", "load_weights")
    assert load(shell, visual_stream(fused=fused)) == set(params)
    for name, param in params.items():
        assert torch.isfinite(param).all()
        if ".qkv." in name:
            expected = (7, 7, 7) if fused else (2, 3, 1)
            assert all(torch.all(part == value) for part, value in zip(param.chunk(3), expected))


@pytest.mark.parametrize("missing", [False, True])
def test_real_core_loader_audits_after_interleaved_visual_text_stream(missing):
    visual, params = vision_shell()
    root = torch.nn.Module()
    root.visual = visual
    root.other = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    root.hf_to_vllm_mapper = None
    load = function("hy_v4/vl_model.py", "load_weights")

    def stream():
        for name, value in visual_stream(
            omitted="layers.1.attn.q_proj.weight" if missing else None
        ):
            yield "visual." + name, value
            yield "other", torch.ones(1)

    if missing:
        with pytest.raises(ValueError):
            load(root, stream())
    else:
        assert load(root, stream()) == {"visual." + name for name in params} | {"other"}
        # An earlier successful stream must not conceal a missing reload shard.
        with pytest.raises(ValueError):
            load(root, (("visual." + name, value) for name, value in visual_stream(
                omitted="layers.0.attn.v_proj.bias")))
    assert visual._vision_loading_depth == 0 and not visual._vision_qkv_shards
