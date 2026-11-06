"""Producer compilation stays scoped and cannot re-enable runtime tuning."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest


@pytest.fixture
def configure(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "vllm_hcu/runtime_compat/step5.py"
    spec = importlib.util.spec_from_file_location("producer_compile_config_tested", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vllm = ModuleType("vllm")
    vllm.envs = NS(VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=False,
                   VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=False)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    return module.configure_step5_pd_producer_compilation, vllm.envs


def config(model="step5", role="kv_producer", eager=False, graph="NONE",
           connector="MooncakeConnector"):
    return NS(model_config=NS(hf_config=NS(model_type=model), enforce_eager=eager),
              kv_transfer_config=NS(kv_connector=connector, kv_role=role),
              compilation_config=NS(cudagraph_mode=NS(name=graph),
                                    inductor_compile_config={"size_asserts": False}))


@pytest.mark.parametrize("case", [
    config(model="deepseek_v3"), config(model="step3p5"),
    config(role="kv_consumer", graph="FULL_AND_PIECEWISE"),
    config(role="kv_both"), config(connector="NixlConnector"),
    config(eager=True),
])
def test_other_models_roles_and_eager_keep_their_compiler_options(configure, case):
    fn, envs = configure
    envs.VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE = True
    envs.VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING = True
    before = dict(case.compilation_config.inductor_compile_config)
    assert fn(case) is False
    assert case.compilation_config.inductor_compile_config == before


def test_compiled_producer_is_fused_without_runtime_tuning_and_is_idempotent(configure):
    fn, _ = configure
    case = config()
    assert fn(case) is True
    options = case.compilation_config.inductor_compile_config
    assert options["size_asserts"] is False
    assert options["triton.autotune_at_compile_time"] is True
    for key in ("coordinate_descent_tuning", "triton.autotune_pointwise",
                "max_autotune", "max_autotune_pointwise", "max_autotune_gemm",
                "combo_kernels", "benchmark_combo_kernel"):
        assert options[key] is False
    assert fn(case) is False
    assert case.compilation_config.cudagraph_mode.name == "NONE"


@pytest.mark.parametrize("key", ["VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE",
                                 "VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING"])
def test_framework_static_range_tuning_must_be_disabled_before_compilation(configure, key):
    fn, envs = configure
    setattr(envs, key, True)
    case = config()
    case.compilation_config.inductor_compile_config.update(
        max_autotune=False, coordinate_descent_tuning=False)
    with pytest.raises(ValueError, match="VLLM_ENABLE_INDUCTOR"):
        fn(case)


@pytest.mark.parametrize("graph", [None, "FULL", "PIECEWISE", "FULL_AND_PIECEWISE"])
def test_producer_snapshots_never_opt_into_cuda_graph_capture(configure, graph):
    fn, _ = configure
    with pytest.raises(ValueError, match="cudagraph_mode=NONE"):
        fn(config(graph=graph))
