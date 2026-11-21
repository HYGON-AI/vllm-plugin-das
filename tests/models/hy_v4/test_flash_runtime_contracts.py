# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU-only seam tests; these do not claim vLLM imports or HCU execution."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
FLASH = ROOT / "vllm_hcu/models/hy_v4_flash"


def runtime(monkeypatch, hcu=True):
    cls = type("Platform", (), {"__module__": "vllm_hcu.platforms.hcu" if hcu else "vllm.platforms.cuda"})
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = cls()
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    spec = importlib.util.spec_from_file_location("_hy4_flash_runtime_contract", FLASH / "runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hcu_provider_does_not_require_deepgemm(monkeypatch):
    module = runtime(monkeypatch)
    pair = (object(), object())
    bridge = ModuleType("hcu_adapter.bridges.vllm_paged_mqa")
    bridge.get_paged_mqa_provider = lambda: pair
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    assert module.get_paged_mqa_provider() == pair


def test_hcu_missing_adapter_reports_dependency(monkeypatch):
    module = runtime(monkeypatch)
    monkeypatch.setitem(sys.modules, "hcu_adapter.bridges.vllm_paged_mqa", None)
    with pytest.raises(RuntimeError, match="requires adapter-das"):
        module.get_paged_mqa_provider()


@pytest.mark.parametrize("available", [True, False])
def test_cuda_provider_still_uses_vllm(monkeypatch, available):
    module = runtime(monkeypatch, hcu=False)
    deepgemm = ModuleType("vllm.utils.deep_gemm")
    deepgemm.has_deep_gemm = lambda: available
    deepgemm.get_paged_mqa_logits_metadata = object()
    deepgemm.fp8_fp4_paged_mqa_logits = object()
    monkeypatch.setitem(sys.modules, deepgemm.__name__, deepgemm)
    if available:
        assert module.get_paged_mqa_provider() == (deepgemm.get_paged_mqa_logits_metadata, deepgemm.fp8_fp4_paged_mqa_logits)
    else:
        with pytest.raises(RuntimeError, match="requires DeepGEMM"):
            module.get_paged_mqa_provider()


def test_v033_mx_inputs_call_has_three_arguments():
    tree = ast.parse((FLASH / "nvidia/compressed_indexer.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "mqa_logits_mx_inputs"]
    assert len(calls) == 1
    call = compile(ast.Expression(calls[0]), "<v033-mx-call>", "eval")
    def v033_inputs(q, q_scale, weights):
        return q_scale, weights
    assert eval(call, {"mqa_logits_mx_inputs": v033_inputs, "quant_q": object(), "weights": 3}) == (None, 3)
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Attribute) and target.attr == "use_mx_scales" for target in node.targets))
    def forbidden():
        raise AssertionError("HCU must not infer MX scales from CUDA capability")
    assert eval(compile(ast.Expression(assignment.value), "<hcu-non-mx>", "eval"),
                {"is_hcu_platform": lambda: True, "mqa_logits_use_mx_scales": forbidden}) is False


@pytest.mark.parametrize("skip_topk", [False, True])
def test_flashmla_publishes_logical_indices_before_attention(skip_topk):
    tree = ast.parse((FLASH / "nvidia/attention.py").read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_indexer_and_attn")
    assert [item.id for item in method.decorator_list] == ["eager_break_during_capture"]
    method.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<flash-attn-seam>", "exec"), namespace)
    calls = []
    result = object()
    class Output:
        shape = (1, 2)
        def copy_(self, value):
            assert value is result
            calls.append("copy")
    def indexer(*args):
        calls.append("indexer")
    def attention(*args, **kwargs):
        assert kwargs == {"output_shape": (1, 2)}
        calls.append("attention")
        return result
    attention.impl = SimpleNamespace(
        record_logical_topk_ready=lambda: calls.append("logical_ready")
    )
    model = SimpleNamespace(indexer=indexer, is_sparse=True, skip_topk=skip_topk,
                            indexer_rope_emb=None, mla_attn=attention,
                            topk_snapshot=None)
    namespace["_indexer_and_attn"](model, None, None, None, None, None, None, Output())
    assert calls == (["indexer"] if not skip_topk else []) + [
        "logical_ready", "attention", "copy"
    ]


@pytest.mark.parametrize("module_name, expected", [
    ("hcu_adapter.bridges.vllm", True),
    ("vllm_hcu.platforms.hcu", True),
    ("vllm.platforms.rocm", False),
    ("vllm.platforms.cuda", False),
])
def test_flash_recognizes_adapter_platform_owner(monkeypatch, module_name, expected):
    module = runtime(monkeypatch)
    module.current_platform = type("Platform", (), {"__module__": module_name})()
    assert module.is_hcu_platform() is expected


def _channel_fp8_config():
    return {
        "quant_method": "compressed-tensors", "format": "float-quantized",
        "config_groups": {"group": {
            "weights": {"num_bits": 8, "type": "float", "strategy": "channel",
                        "dynamic": False, "symmetric": True},
            "input_activations": {"num_bits": 8, "type": "float", "strategy": "token",
                                  "dynamic": True, "symmetric": True},
        }},
    }


def test_channel_fp8_preserves_checkpoint_contract(monkeypatch):
    module = runtime(monkeypatch)
    module._validate_fp8_config(_channel_fp8_config())


@pytest.mark.parametrize("bad", ["weight_type", "activation_static", "activation_group", "kv", "transform"])
def test_channel_fp8_refuses_other_recipes(monkeypatch, bad):
    module = runtime(monkeypatch)
    config = _channel_fp8_config()
    group = config["config_groups"]["group"]
    if bad == "weight_type":
        group["weights"]["type"] = "int"
    elif bad == "activation_static":
        group["input_activations"]["dynamic"] = False
    elif bad == "activation_group":
        group["input_activations"]["group_size"] = 128
    elif bad == "kv":
        config["kv_cache_scheme"] = {}
    else:
        config["transform_config"] = {}
    with pytest.raises(ValueError, match="requires W8A8 FP8"):
        module._validate_fp8_config(config)

def test_flash_hc_construction_matches_shared_constructor():
    """Bind Flash's two HC calls to the real shared constructor signature."""
    import inspect

    shared = ast.parse((FLASH.parent / "hy_v4/hc.py").read_text())
    cls = next(n for n in shared.body if isinstance(n, ast.ClassDef)
               and n.name == "HYV4HCLayer")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name == "__init__")
    for arg in init.args.args:
        arg.annotation = None
    init.returns = None
    init.body = [ast.Pass()]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[init], type_ignores=[])),
                 "<shared-hc-signature>", "exec"), namespace)
    signature = inspect.signature(namespace["__init__"])
    tree = ast.parse((FLASH / "nvidia/model.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "HYV4HCLayer"]
    assert len(calls) == 2
    for call in calls:
        args = [object() for _ in call.args]
        kwargs = {keyword.arg: object() for keyword in call.keywords}
        bound = signature.bind(object(), *args, **kwargs)
        assert "config" in bound.arguments and "layer_idx" in bound.arguments

def test_flash_engram_defaults_to_device_tables(monkeypatch):
    module = runtime(monkeypatch)
    assert module.get_engram_cpu_offload(None) is False
    # v0.33 EngramConfig has no disk_offload attribute.
    storage = SimpleNamespace(cpu_offload=True, embedding_across_dp=False,
                              dp_shared_memory=False)
    assert module.get_engram_cpu_offload(storage) is True
    storage.cpu_offload = False
    assert module.get_engram_cpu_offload(storage) is False


@pytest.mark.parametrize("unsupported", ["disk_offload", "embedding_across_dp",
                                        "dp_shared_memory"])
def test_flash_engram_refuses_unsupported_storage(monkeypatch, unsupported):
    module = runtime(monkeypatch)
    storage = SimpleNamespace(cpu_offload=True)
    setattr(storage, unsupported, True)
    with pytest.raises(ValueError, match="TP-local tables"):
        module.get_engram_cpu_offload(storage)


@pytest.mark.parametrize("hcu", [True, False])
def test_flash_metadata_uses_matching_scheduler_owner(monkeypatch, hcu):
    tree = ast.parse((FLASH / "nvidia/swa.py").read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == "FlashSparseMetadataBuilder")
    method = next(n for n in original.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_build_fp8_mixed_decode_prefill")
    core_scheduler, hcu_scheduler = object(), object()

    class CoreBuilder:
        def _build_fp8_mixed_decode_prefill(self):
            return core_scheduler

    provider = ModuleType("vllm_hcu.v1.attention.ops.flashmla")
    provider.get_mla_metadata = lambda: (hcu_scheduler, None)
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    cls = ast.ClassDef(name="FlashBuilder", bases=[ast.Name(id="CoreBuilder", ctx=ast.Load())],
                       keywords=[], body=[method], decorator_list=[])
    namespace = {"CoreBuilder": CoreBuilder, "is_hcu_platform": lambda: hcu,
                 "FlashMLASparseMetadata": SimpleNamespace(FP8KernelMetadata=SimpleNamespace)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 "<flash-scheduler-owner>", "exec"), namespace)
    builder = namespace["FlashBuilder"]()
    builder.max_model_len_tensor = [4096, 4096]
    builder.dummy_block_table = [[0], [1]]
    result = builder._build_fp8_mixed_decode_prefill()
    if hcu:
        assert result.scheduler_metadata is hcu_scheduler
        assert result.cache_lens == [4096]
        assert result.dummy_block_table == [[0]]
    else:
        assert result is core_scheduler
