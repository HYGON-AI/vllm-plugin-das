"""Pure eligibility checks; real runner/capture qualification is separate."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def api():
    path = Path(__file__).resolve().parents[2] / (
        "adapter_extension/src/hcu_vllm_das/models/hy_v4_flash/fullgraph_runner.py"
    )
    assert path.is_file(), "qualified Flash runner integration absent"
    spec = importlib.util.spec_from_file_location("flash_runner_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configuration():
    return NS(
        use_v2_model_runner=True,
        model_config=NS(enforce_eager=False, enable_prompt_embeds=False,
                        hf_text_config=NS(model_type="hy_v4_flash"),
                        multimodal_config=None),
        parallel_config=NS(tensor_parallel_size=4, data_parallel_size=1,
                           pipeline_parallel_size=1,
                           prefill_context_parallel_size=1,
                           decode_context_parallel_size=1,
                           enable_dbo=False, enable_eplb=False,
                           disable_custom_all_reduce=True),
        compilation_config=NS(mode=NS(name="VLLM_COMPILE"),
                              cudagraph_mode=NS(name="FULL_AND_PIECEWISE")),
        speculative_config=NS(method="dfly", num_speculative_tokens=5),
        cache_config=NS(cache_dtype="fp8_ds_mla", enable_prefix_caching=False),
        lora_config=None, kv_transfer_config=None,
    )


def test_current_configuration_and_text_only_multimodal_shell(api):
    cfg = configuration()
    assert api.is_qualified_config(cfg, breakable=False)
    cfg.model_config.multimodal_config = NS(get_limit_per_prompt=lambda _: 0)
    assert api.is_qualified_config(cfg, breakable=False)
    cfg.model_config.multimodal_config = NS(get_limit_per_prompt=lambda _: 1)
    assert not api.is_qualified_config(cfg, breakable=False)


@pytest.mark.parametrize("tp_size", [4, 8])
@pytest.mark.parametrize("disable_custom_all_reduce", [True, False])
def test_supported_tp_and_all_reduce_policy(api, tp_size, disable_custom_all_reduce):
    cfg = configuration()
    cfg.parallel_config.tensor_parallel_size = tp_size
    cfg.parallel_config.disable_custom_all_reduce = disable_custom_all_reduce

    assert api.is_qualified_config(cfg, breakable=False)


@pytest.mark.parametrize("owner,field,value", [
    ("model_config", "enforce_eager", True),
    ("model_config", "enable_prompt_embeds", True),
    ("parallel_config", "tensor_parallel_size", 2),
    ("parallel_config", "data_parallel_size", 2),
    ("parallel_config", "pipeline_parallel_size", 2),
    ("parallel_config", "prefill_context_parallel_size", 2),
    ("parallel_config", "decode_context_parallel_size", 2),
    ("parallel_config", "enable_dbo", True),
    ("parallel_config", "enable_eplb", True),
    ("parallel_config", "disable_custom_all_reduce", None),
    ("parallel_config", "disable_custom_all_reduce", 0),
    ("speculative_config", "method", "mtp"),
    ("speculative_config", "num_speculative_tokens", 6),
    ("cache_config", "cache_dtype", "auto"),
    ("cache_config", "enable_prefix_caching", True),
])
def test_unqualified_options_retain_original_runner(api, owner, field, value):
    cfg = configuration()
    setattr(getattr(cfg, owner), field, value)
    assert not api.is_qualified_config(cfg, breakable=False)


def test_eager_piecewise_breakable_and_unknown_never_select_bank(api):
    cfg = configuration()
    assert not api.is_qualified_config(cfg, breakable=True)
    for field, value in [("lora_config", object()), ("kv_transfer_config", object()),
                         ("use_v2_model_runner", False),
                         ("speculative_config", None)]:
        changed = copy.deepcopy(cfg)
        setattr(changed, field, value)
        assert not api.is_qualified_config(changed, breakable=False)
    for mode in ["NONE", "FULL", "PIECEWISE"]:
        changed = copy.deepcopy(cfg)
        changed.compilation_config.cudagraph_mode.name = mode
        assert not api.is_qualified_config(changed, breakable=False)
    assert not api.is_qualified_config(NS(), breakable=False)


def test_config_predicate_does_not_mutate_callers(api):
    cfg = configuration()
    before = repr(cfg)
    assert api.is_qualified_config(cfg, breakable=False)
    assert repr(cfg) == before
