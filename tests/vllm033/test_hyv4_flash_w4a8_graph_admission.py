"""CPU configuration contracts; GPU capture and model quality are separate."""
import importlib.util
import ast
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest


@pytest.fixture
def api(monkeypatch):
    root = Path(__file__).resolve().parents[2] / (
        "adapter_extension/src/hcu_vllm_das/models/hy_v4_flash"
    )
    package = ModuleType("_flash_w4a8_contract")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    name = package.__name__ + ".quantization"
    spec = importlib.util.spec_from_file_location(name, root / "quantization.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    # Only the external breakable-mode query is isolated; the real shared
    # admission predicate and quantization module are loaded without Core/GPU.
    breakable = ModuleType("vllm.compilation.breakable_cudagraph")
    breakable.is_breakable_cudagraph_enabled = lambda: False
    monkeypatch.setitem(sys.modules, breakable.__name__, breakable)
    return module, breakable


def current():
    return NS(
        use_v2_model_runner=True,
        model_config=NS(enforce_eager=False, enable_prompt_embeds=False,
                        hf_text_config=NS(model_type="hy_v4_flash"),
                        multimodal_config=None),
        compilation_config=NS(mode=NS(name="VLLM_COMPILE"),
                              cudagraph_mode=NS(name="FULL_AND_PIECEWISE")),
        parallel_config=NS(tensor_parallel_size=4, data_parallel_size=1,
            pipeline_parallel_size=1, prefill_context_parallel_size=1,
            decode_context_parallel_size=1, enable_dbo=False,
            enable_eplb=False, disable_custom_all_reduce=True,
            enable_expert_parallel=False, enable_elastic_ep=False,
            use_sequence_parallel_moe=False,
            eplb_config=NS(num_redundant_experts=0)),
        speculative_config=NS(method="dfly", num_speculative_tokens=5),
        cache_config=NS(cache_dtype="fp8_ds_mla", enable_prefix_caching=False),
        lora_config=None, kv_transfer_config=None,
    )


def validate(api, cfg):
    function = getattr(api[0], "validate_w4a8_execution", None)
    assert callable(function), "W4A8 Graph admission is missing"
    return function(cfg)


@pytest.mark.parametrize("disable_custom_all_reduce", [True, False])
def test_qualified_fullpiecewise_preserves_config(api, disable_custom_all_reduce):
    cfg = current()
    cfg.parallel_config.disable_custom_all_reduce = disable_custom_all_reduce
    before = deepcopy(cfg)
    assert validate(api, cfg) is True
    assert cfg == before


def test_eager_retains_reference_execution(api):
    cfg = current()
    cfg.model_config.enforce_eager = True
    assert validate(api, cfg) is False


@pytest.mark.parametrize("section,field,value", [
    ("compilation_config", "mode", NS(name="NONE")),
    ("compilation_config", "cudagraph_mode", NS(name="PIECEWISE")),
    ("compilation_config", "cudagraph_mode", NS(name="FULL")),
    ("parallel_config", "tensor_parallel_size", 2),
    ("parallel_config", "tensor_parallel_size", 8),
    ("parallel_config", "data_parallel_size", 2),
    ("parallel_config", "pipeline_parallel_size", 2),
    ("parallel_config", "prefill_context_parallel_size", 2),
    ("parallel_config", "decode_context_parallel_size", 2),
    ("parallel_config", "enable_expert_parallel", True),
    ("parallel_config", "enable_elastic_ep", True),
    ("parallel_config", "enable_eplb", True),
    ("parallel_config", "enable_dbo", True),
    ("parallel_config", "use_sequence_parallel_moe", True),
    ("parallel_config", "disable_custom_all_reduce", None),
    ("parallel_config", "disable_custom_all_reduce", 0),
    ("parallel_config", "disable_custom_all_reduce", 1),
    ("parallel_config", "disable_custom_all_reduce", "false"),
    ("speculative_config", "method", "mtp"),
    ("speculative_config", "num_speculative_tokens", 3),
    ("cache_config", "cache_dtype", "auto"),
    ("cache_config", "enable_prefix_caching", True),
    ("model_config", "enable_prompt_embeds", True),
])
def test_unqualified_graph_fails_closed(api, section, field, value):
    cfg = current()
    setattr(getattr(cfg, section), field, value)
    with pytest.raises(NotImplementedError):
        validate(api, cfg)


@pytest.mark.parametrize("field,value", [
    ("use_v2_model_runner", False), ("lora_config", object()),
    ("kv_transfer_config", object()),
])
def test_unqualified_runtime_fails_closed(api, field, value):
    cfg = current()
    setattr(cfg, field, value)
    with pytest.raises(NotImplementedError):
        validate(api, cfg)


def test_breakable_graph_is_rejected(api):
    api[1].is_breakable_cudagraph_enabled = lambda: True
    with pytest.raises(NotImplementedError):
        validate(api, current())


def test_eager_does_not_bypass_parallel_contract(api):
    cfg = current()
    cfg.model_config.enforce_eager = True
    cfg.parallel_config.enable_expert_parallel = True
    with pytest.raises(NotImplementedError):
        validate(api, cfg)


@pytest.mark.parametrize("admitted", [True, False])
def test_apply_capture_uses_same_admission_without_changing_math(admitted):
    # Isolate the actual apply body for CPU control flow only. Native arithmetic
    # and changed-input device capture are separate GPU qualification gates.
    path = Path(__file__).resolve().parents[2] / (
        "adapter_extension/src/hcu_vllm_das/models/hy_v4_flash/w4a8.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "HYV4FlashChannelW4A8Method")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "apply")
    scalar_type = object()
    tensor = NS(dtype=scalar_type, device=NS(type="cuda"))
    seen = []
    output = object()
    def routed(*args, **kwargs):
        assert args == (tensor, 1, 2, 3, 4, 5, 6)
        assert kwargs == {"swiglu_limit": 10}
        seen.append("math")
        return output
    ns = {"torch": NS(bfloat16=scalar_type,
                       cuda=NS(is_current_stream_capturing=lambda: True)),
          "routed_reference": routed, "packed_byte_view": lambda x: x}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
    layer = NS(w13_weight_packed=3, w2_weight_packed=4,
               w13_weight_scale=5, w2_weight_scale=6, swiglu_limit=10)
    owner = NS(_weights_processed=True, _graph_admitted=admitted)
    if admitted:
        assert ns["apply"](owner, layer, tensor, 1, 2, None, None) is output
        assert seen == ["math"]
    else:
        with pytest.raises(RuntimeError, match="eager configuration"):
            ns["apply"](owner, layer, tensor, 1, 2, None, None)
        assert seen == []
