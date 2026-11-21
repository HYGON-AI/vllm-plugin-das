# SPDX-License-Identifier: Apache-2.0
"""CPU contracts only: not GPU/model, Graph, or image qualification."""

import ast
from copy import deepcopy
from pathlib import Path

import pytest
import torch

FLASH = Path(__file__).resolve().parents[3] / "vllm_hcu/models/hy_v4_flash"


def definitions(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def recipe():
    def group(bits):
        return dict(
            format="pack-quantized" if bits == 4 else "int-quantized",
            targets=["re:.*experts.*" if bits == 4 else "re:.*q_b_proj$"],
            weights=dict(
                num_bits=bits,
                type="int",
                strategy="channel",
                symmetric=True,
                dynamic=False,
            ),
            input_activations=dict(
                num_bits=8, type="int", strategy="token", symmetric=True, dynamic=True
            ),
            output_activations=None,
        )

    return dict(
        quant_method="compressed-tensors",
        format="pack-quantized",
        quantization_status="compressed",
        config_groups={"int4": group(4), "int8": group(8)},
    )


def validator():
    return definitions(FLASH / "runtime.py", {"_validate_channel_w4a8_config"})[
        "_validate_channel_w4a8_config"
    ]


def test_integer_recipe_is_not_mutated():
    config = recipe()
    original = deepcopy(config)
    validator()(config)
    assert config == original
    assert config["config_groups"]["int4"]["input_activations"] is not None


@pytest.mark.parametrize(
    "part,key,value",
    [
        ("weights", "num_bits", 3),
        ("weights", "num_bits", True),
        ("weights", "type", "float"),
        ("weights", "strategy", "group"),
        ("weights", "symmetric", False),
        ("weights", "dynamic", True),
        ("weights", "group_size", 128),
        ("weights", "scale_dtype", "float16"),
        ("input_activations", "observer_kwargs", {"custom": True}),
        ("weights", "actorder", "group"),
        ("input_activations", "num_bits", 4),
        ("input_activations", "type", "float"),
        ("input_activations", "dynamic", False),
        ("input_activations", "strategy", "tensor"),
        ("input_activations", "symmetric", False),
        ("input_activations", "block_structure", [1, 128]),
    ],
)
def test_refuses_other_quantizers(part, key, value):
    config = recipe()
    config["config_groups"]["int4"][part][key] = value
    with pytest.raises(ValueError, match="requires symmetric channel"):
        validator()(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("kv_cache_scheme", {}),
        ("transform_config", {}),
        ("quantization_status", "frozen"),
        ("format", "int-quantized"),
        ("quant_method", "slimquant_w4a8"),
        ("config_groups", {}),
    ],
)
def test_refuses_other_containers(key, value):
    config = recipe()
    config[key] = value
    with pytest.raises(ValueError):
        validator()(config)


@pytest.mark.parametrize(
    "kind", ["no_int8", "missing_activation", "output", "wrong_format", "empty_targets"]
)
def test_refuses_partial_or_weight_only_recipes(kind):
    config = recipe()
    group = config["config_groups"]["int4"]
    if kind == "no_int8":
        del config["config_groups"]["int8"]
    elif kind == "missing_activation":
        group["input_activations"] = None
    elif kind == "output":
        group["output_activations"] = {"num_bits": 8}
    elif kind == "wrong_format":
        group["format"] = "int-quantized"
    else:
        group["targets"] = []
    with pytest.raises(ValueError):
        validator()(config)


def test_packing_preserves_biased_nibble_order_and_storage():
    fn = definitions(FLASH / "nvidia/w4a8.py", {"packed_byte_view"})["packed_byte_view"]
    words = torch.tensor([[[0x76543210, -19088744]]], dtype=torch.int32)
    actual = fn(words)
    assert actual.tolist() == [[[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]]]
    assert actual.data_ptr() == words.data_ptr()
    with pytest.raises(ValueError):
        fn(words.to(torch.int64))


@pytest.mark.parametrize("rows", [0, 1, 2, 32, 33, 67])
@pytest.mark.parametrize("strided", [False, True])
def test_dispatch_combine_slot_order_chunking_and_padding(rows, strided):
    fn = definitions(FLASH / "nvidia/w4a8.py", {"routed_reference"})["routed_reference"]
    x = torch.arange(rows * 6, dtype=torch.float32).reshape(rows, 6).to(torch.bfloat16)
    if strided:
        x = x[:, ::2]
    ids = torch.tensor([[3, 0, 2], [1, 3, 0]], dtype=torch.int32).repeat(
        (rows + 1) // 2, 1
    )[:rows]
    weights = torch.tensor([[0.1, 0.7, 0.2]], dtype=torch.float32).repeat(rows, 1)
    w13 = torch.empty(5, 2, 2, dtype=torch.uint8)
    calls = []

    def implementation(value, *args, swiglu_limit):
        counts = args[-1]
        calls.append(value.shape[1])
        assert value.shape[1] <= 32 and swiglu_limit == 10.0
        output = value * torch.arange(1, 6)[:, None, None]
        valid = torch.arange(value.shape[1])[None, :] < counts[:, None]
        return torch.where(valid[:, :, None], output, float("nan"))

    actual = fn(
        x,
        weights,
        ids,
        w13,
        None,
        None,
        None,
        swiglu_limit=10.0,
        expert_impl=implementation,
    )
    selected = x[:, None].float() * (ids.float() + 1)[:, :, None]
    # The expert's output store is BF16 before FP32 weighted combine.
    expected = (selected.to(x.dtype).float() * weights[:, :, None]).sum(1).to(x.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert sum(calls) == rows


def test_expert_regex_admits_packed_and_shape_without_1d_chunk():
    import re

    tree = ast.parse((FLASH / "nvidia/model.py").read_text())
    assignment = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_EXPERT_WEIGHT" for t in n.targets)
    )
    pattern = eval(
        compile(ast.Expression(assignment.value), "<regex>", "eval"), {"re": re}
    )
    for suffix in (".weight_packed", ".weight_scale", ".weight_shape"):
        assert pattern.fullmatch("layers.1.mlp.experts.0.gate_proj" + suffix)


@pytest.mark.parametrize(
    "backend,eager,ep,hcu,expected",
    [
        ("disabled", True, False, True, "requires HCU"),
        ("adapter_reference", False, False, True, "eager TP-only"),
        ("adapter_reference", True, True, True, "eager TP-only"),
        ("adapter_reference", True, False, False, "requires HCU"),
        ("adapter_reference", True, False, True, None),
    ],
)
def test_runtime_admission_is_explicit_eager_tp_only(
    monkeypatch, backend, eager, ep, hcu, expected
):
    import sys
    from types import ModuleType, SimpleNamespace

    runtime = ModuleType("vllm_hcu.models.hy_v4_flash.runtime")
    runtime.__dict__.update(
        definitions(
            FLASH / "runtime.py",
            {
                "_validate_fp8_config",
                "_validate_channel_w4a8_config",
                "validate_flash_quant_config",
            },
        )
    )
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(is_cuda=lambda: True)
    monkeypatch.setitem(sys.modules, platforms.__name__, platforms)
    tree = ast.parse((FLASH / "nvidia/model.py").read_text())
    method = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "validate_runtime"
    )
    scope = {"torch": torch, "VllmConfig": object, "is_hcu_platform": lambda: hcu}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "<runtime-gate>", "exec"),
        scope,
    )
    text = SimpleNamespace(
        validate_flash_config=lambda: None,
        index_topk=2048,
        sliding_window=512,
        engram_layer_ids=[],
        enable_ihc=True,
        num_attention_heads=64,
        swa_num_attention_heads=64,
        flash_w4a8_backend=backend,
    )
    parallel = SimpleNamespace(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_eplb=False,
        enable_dbo=False,
        tensor_parallel_size=8,
        enable_expert_parallel=ep,
    )
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=text,
            dtype=torch.bfloat16,
            quantization="compressed-tensors",
            enforce_eager=eager,
            model_arch_config=SimpleNamespace(quantization_config=recipe()),
        ),
        parallel_config=parallel,
        use_v2_model_runner=True,
        kv_transfer_config=None,
        attention_config=SimpleNamespace(hisparse_config=None),
        lora_config=None,
        speculative_config=None,
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    if expected:
        with pytest.raises(ValueError, match=expected):
            scope["validate_runtime"](cfg)
    else:
        scope["validate_runtime"](cfg)


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn, torch.bfloat16])
def test_dfly_required_dtypes_preserve_quant_weights_not_other_params(dtype):
    from types import SimpleNamespace

    tree = ast.parse((FLASH / "nvidia/dfly.py").read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_required_weights"
    )
    scope = {"torch": torch}
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]), "<dfly-dtype-audit>", "exec"
        ),
        scope,
    )
    values = {
        "layers.0.mlp.gate_up_proj.weight": dtype,
        "layers.0.mlp.down_proj.weight": dtype,
        "layers.0.mlp.down_proj.weight_scale": torch.bfloat16,
        "layers.0.self_attn.q_b_proj.weight_scale_inv": torch.float32,
        "layers.0.input_layernorm.weight": torch.float32,
        "layers.0.self_attn.learnable_sink_param": torch.float32,
        # An unexpected INT8 non-weight must not weaken BF16 auditing.
        "layers.0.unexpected_metadata": torch.int8,
        "layers.0.self_attn.mla_attn.k_scale": torch.float32,
    }
    model = SimpleNamespace(
        named_parameters=lambda: [
            (n, SimpleNamespace(dtype=d)) for n, d in values.items()
        ]
    )
    required = scope["_required_weights"](SimpleNamespace(model=model))
    assert required["layers.0.mlp.gate_proj.weight"] == dtype
    assert required["layers.0.mlp.up_proj.weight"] == dtype
    assert required["layers.0.mlp.down_proj.weight"] == dtype
    assert required["layers.0.mlp.down_proj.weight_scale"] == torch.bfloat16
    assert required["layers.0.self_attn.q_b_proj.weight_scale_inv"] == torch.float32
    assert required["layers.0.input_layernorm.weight"] == torch.bfloat16
    assert required["layers.0.self_attn.learnable_sink_param"] == torch.bfloat16
    assert required["layers.0.unexpected_metadata"] == torch.bfloat16
    assert "layers.0.self_attn.mla_attn.k_scale" not in required
