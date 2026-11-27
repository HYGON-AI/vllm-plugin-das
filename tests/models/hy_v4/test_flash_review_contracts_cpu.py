# SPDX-License-Identifier: Apache-2.0
"""Regressions for MR493 review findings, without GPU or plugin activation.

Real CPU Torch executes production routing/reference/session function bodies.
The expert kernel, forward context and final Core tracker are explicit doubles;
these tests do not qualify native kernels, the full Core loader or an engine.
Vision tests import the complete production module and real Core/HF configs.
"""

import ast
import builtins
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[3]
FLASH = ROOT / "vllm_hcu/models/hy_v4_flash"
VISION = (
    ROOT / "adapter_extension/src/hcu_vllm_das/models/" "hy_v4_flash/vision_config.py"
)


class _NonnegativeIndexAudit(TorchDispatchMode):
    """Check actual CPU index operands, not just the masked final values."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.index.Tensor:
            for index in args[1]:
                if index is not None and index.dtype in (torch.int32, torch.int64):
                    assert (index >= 0).all(), "negative gather index reached dispatch"
        return func(*args, **(kwargs or {}))


def _functions(path, names, **namespace):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    selected = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in selected} == set(names)
    namespace = {"torch": torch, **namespace}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


def _vision_module():
    pytest.importorskip("transformers")
    pytest.importorskip("vllm.transformers_utils.configs.hy_v4")
    spec = importlib.util.spec_from_file_location("mr493_vision_config", VISION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_hf_public_config_name_and_full_vision_module_import():
    module = _vision_module()
    from transformers.configuration_utils import PreTrainedConfig, PretrainedConfig
    from vllm.transformers_utils.configs.hy_v4 import HYV4Config

    assert PreTrainedConfig is PretrainedConfig
    assert issubclass(module.HYV4VLVisionConfig, PreTrainedConfig)
    assert issubclass(module.HYV4VLConfig, PreTrainedConfig)
    assert module.HYV4VLConfig.sub_configs["text_config"] is HYV4Config
    assert module.HYV4VLConfig.sub_configs["vision_config"] is module.HYV4VLVisionConfig


def test_real_vision_text_subconfigs_json_round_trip():
    module = _vision_module()
    from transformers.configuration_utils import PreTrainedConfig

    config = module.HYV4VLConfig(
        text_config={"hidden_size": 64},
        vision_config={"hidden_size": 32, "num_attention_heads": 4},
    )
    assert config.text_config.hidden_size == 64
    assert config.vision_config.out_hidden_size == 64
    assert config.vision_config.text_hidden_size == 64
    payload = json.loads(config.to_json_string(use_diff=False))
    again = module.HYV4VLConfig.from_dict(payload)
    assert isinstance(again.text_config, PreTrainedConfig)
    assert isinstance(again.vision_config, module.HYV4VLVisionConfig)
    assert again.text_config.hidden_size == 64
    assert again.vision_config.hidden_size == 32
    assert again.vision_config.out_hidden_size == 64
    assert again.vision_config.text_hidden_size == 64


def test_real_vision_projector_mismatch_still_rejected():
    module = _vision_module()
    with pytest.raises(ValueError, match="projector output must match"):
        module.HYV4VLConfig(
            text_config={"hidden_size": 64}, vision_config={"out_hidden_size": 63}
        )


@pytest.mark.parametrize("rows", [4, 33, 67])
@pytest.mark.parametrize("all_padding", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_router_sentinel_reaches_reference_without_negative_gather(
    rows, all_padding, index_dtype
):
    padding = (
        torch.ones(rows, dtype=torch.bool)
        if all_padding
        else torch.arange(rows) % 3 == 1
    )
    context = SimpleNamespace(is_padding=padding)
    namespace = _functions(
        FLASH / "nvidia/router.py",
        {"stable_sigmoid_topk", "_compute_routing"},
        envs=SimpleNamespace(VLLM_MOE_SKIP_PADDING=True),
        is_forward_context_available=lambda: True,
        get_forward_context=lambda: context,
    )
    router = SimpleNamespace(
        global_num_experts=4,
        top_k=2,
        renormalize=True,
        routed_scaling_factor=2.0,
        skip_padding=True,
        e_score_correction_bias=torch.zeros(4),
    )
    x = torch.arange(rows * 6).reshape(rows, 6).to(torch.bfloat16)
    x[padding] = float("nan")
    logits = torch.arange(rows * 4).reshape(rows, 4).float() % 7
    weights, ids = namespace["_compute_routing"](router, x, logits, None)
    assert (ids[padding] == -1).all()
    assert (weights[padding] == 0).all()
    ids = ids.to(index_dtype)
    calls = []

    def expert_kernel(values, *args, swiglu_limit):
        counts = args[-1]
        assert values.shape[1] <= 32
        # Valid experts alone receive non-padding tokens; uncomputed storage
        # is deliberately NaN so an unsafe gather/combine cannot look green.
        expected_counts = torch.stack(
            [
                (ids[calls_offset[0] : calls_offset[0] + values.shape[1]] == e)
                .any(1)
                .sum()
                for e in range(4)
            ]
        ).to(torch.int32)
        torch.testing.assert_close(counts, expected_counts, rtol=0, atol=0)
        calls_offset[0] += values.shape[1]
        calls.append(counts.clone())
        result = values * torch.arange(1, 5)[:, None, None]
        active = torch.arange(values.shape[1])[None, :] < counts[:, None]
        return torch.where(active[:, :, None], result, float("nan"))

    calls_offset = [0]
    reference = _functions(FLASH / "nvidia/w4a8.py", {"routed_reference"})[
        "routed_reference"
    ]
    with _NonnegativeIndexAudit():
        output = reference(
            x,
            weights,
            ids,
            torch.empty(4, 1, 1),
            None,
            None,
            None,
            expert_impl=expert_kernel,
        )
    valid = ids >= 0
    safe_x = torch.where(padding[:, None], 0, x)
    slot = (safe_x[:, None].float() * (ids.clamp_min(0).float() + 1)[:, :, None]).to(
        x.dtype
    )
    expected = (
        (
            torch.where(valid[:, :, None], slot.float(), 0)
            * torch.where(valid, weights, 0)[:, :, None]
        )
        .sum(1)
        .to(x.dtype)
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert torch.isfinite(output).all()
    assert (output[padding] == 0).all()
    assert calls_offset[0] == rows
    if all_padding:
        assert all((counts == 0).all() for counts in calls)


@pytest.mark.parametrize("bad_id", [-2, 4])
def test_reference_keeps_invalid_expert_id_rejection(bad_id):
    reference = _functions(FLASH / "nvidia/w4a8.py", {"routed_reference"})[
        "routed_reference"
    ]
    calls = []
    with pytest.raises(RuntimeError, match="expert ID out of range"):
        reference(
            torch.ones(1, 3),
            torch.ones(1, 1),
            torch.tensor([[bad_id]], dtype=torch.int32),
            torch.empty(4, 1, 1),
            None,
            None,
            None,
            expert_impl=lambda *a, **k: calls.append(True),
        )
    assert calls == []


def test_reference_masks_nan_weights_on_inactive_sentinel_slots():
    reference = _functions(FLASH / "nvidia/w4a8.py", {"routed_reference"})[
        "routed_reference"
    ]
    ids = torch.tensor([[1, -1], [-1, -1]], dtype=torch.int32)
    weights = torch.tensor([[0.5, float("nan")], [float("inf"), float("nan")]])
    x = torch.tensor([[2.0, 4.0], [float("nan"), float("nan")]])

    def expert_kernel(values, *args, **kwargs):
        counts = args[-1]
        result = values * torch.arange(1, 5)[:, None, None]
        active = torch.arange(values.shape[1])[None, :] < counts[:, None]
        return torch.where(active[:, :, None], result, float("nan"))

    with _NonnegativeIndexAudit():
        output = reference(
            x,
            weights,
            ids,
            torch.empty(4, 1, 1),
            None,
            None,
            None,
            expert_impl=expert_kernel,
        )
    torch.testing.assert_close(
        output, torch.tensor([[2.0, 4.0], [0.0, 0.0]]), rtol=0, atol=0
    )


def _loading_shell():
    calls = []

    class CoreTrackerDouble:
        def __init__(self, load_config):
            pass

        def track_weights_loading(self, shell, loaded, *, quantized):
            assert quantized is True
            assert loaded == shell._loaded_weights
            calls.append("core")

    original_import = builtins.__import__

    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm.model_executor.model_loader.default_loader":
            return SimpleNamespace(DefaultModelLoader=CoreTrackerDouble)
        return original_import(name, globals, locals, fromlist, level)

    # Only this extracted function's namespace sees the tracker double; no
    # sys.modules/global builtins patch or installed Core mutation is used.
    namespace = _functions(
        FLASH / "nvidia/model.py",
        {"weight_loading_session"},
        contextmanager=contextmanager,
        __builtins__={**vars(builtins), "__import__": local_import},
    )
    shell = type(
        "LoadingAuditShell",
        (torch.nn.Module,),
        {
            "weight_loading_session": namespace["weight_loading_session"],
        },
    )()
    shell._weight_loading_depth = 0
    shell._seen_weights = set()
    shell._loaded_weights = set()
    shell._load_config = object()
    shell.model = torch.nn.Module()
    shell.model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    for layer in shell.model.layers:
        layer.mlp = torch.nn.Module()
        layer.mlp.expert_bias = torch.nn.Parameter(torch.empty(4))
        layer.mlp.weight = torch.nn.Parameter(torch.ones(4))
    shell.model.loaded_parts = {}
    shell.model.verify_loaded_parts = lambda: calls.append("shards")
    return shell, calls


@pytest.mark.parametrize("layer", [0, 1])
def test_unobserved_correction_bias_is_required_before_quantized_core_tracker(layer):
    shell, calls = _loading_shell()
    missing = f"model.layers.{layer}.mlp.expert_bias"
    with pytest.raises(ValueError, match=missing):
        with shell.weight_loading_session():
            shell._loaded_weights.update(
                name for name, _ in shell.named_parameters() if name != missing
            )
            # Deliberately never record the missing parameter in the shard
            # auditor: a stream-observation-only check would miss this case.
            shell.model.loaded_parts.update(
                {name: True for name in shell._loaded_weights}
            )
    assert calls == []
    assert shell._weight_loading_depth == 0
    assert not shell._seen_weights and not shell._loaded_weights


def test_complete_correction_bias_stream_audits_once_after_nested_session():
    shell, calls = _loading_shell()
    with shell.weight_loading_session():
        shell._loaded_weights.update(name for name, _ in shell.named_parameters())
        with shell.weight_loading_session():
            assert shell._weight_loading_depth == 2
        assert calls == []
    assert calls == ["core", "shards"]
    assert shell._weight_loading_depth == 0 and not shell._loaded_weights
