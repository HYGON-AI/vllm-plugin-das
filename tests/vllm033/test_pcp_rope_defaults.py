"""CPU confirmation of default and explicit PCP RoPE startup dispatch."""

import pytest

from hcu_adapter.kernels import pcp_fork, pcp_rope_policy
from hcu_vllm_das.adapter_api import adapter_module

hcu_config = adapter_module("hcu_config")
from hcu_vllm_das import inverse_rope_pcp, pcp_q_rope, v41_pcp, v41_pcp_activation
from hcu_vllm_das.inverse_rope_policy import selected_mode as inverse_mode


@pytest.fixture
def native_route(monkeypatch, tmp_path):
    monkeypatch.delenv("HCU_CONFIG", raising=False)
    monkeypatch.delenv(pcp_rope_policy.MODE_VARIABLE, raising=False)
    monkeypatch.setenv("HCU_DISABLE_OPS", "")
    monkeypatch.setenv("HCU_VLLM_PCP_IMPL", "native")
    monkeypatch.setenv("VLLM_HCU_DSV41_PCP_ATTENTION", "1")
    monkeypatch.setattr(hcu_config, "_STATE", {"loaded": False})
    monkeypatch.setattr(hcu_config, "_candidate_paths", lambda: [tmp_path / "hcu_adapter.json"])
    monkeypatch.setattr(pcp_rope_policy, "_process_arch", lambda _: "gfx938")
    assert hcu_config.path() is None
    return hcu_config.load()["ops"]


def _modes():
    return inverse_mode(), "hcu" if pcp_fork.q_rope_selected() else "stock"


def test_default_binds_both_providers_without_rope_config(monkeypatch, native_route):
    from hcu_adapter.extensions import ExtensionContext
    from vllm import envs
    from vllm.model_executor.models import ModelRegistry

    activation = v41_pcp_activation
    for name in ("_reference", "_engram_validator", "_forward_context_resolver",
                 "_forward_mqa_provider", "_kv_transfer_validator", "_classes",
                 "_q_rope_provider", "_q_rope_binding", "_q_rope_failure"):
        monkeypatch.setattr(pcp_fork, name, None)
    monkeypatch.setattr(v41_pcp, "_bound", False)
    monkeypatch.setattr(activation, "_state", None)
    monkeypatch.setattr(activation, "_failure", None)
    monkeypatch.setattr(activation, "_publishing", False)
    monkeypatch.setattr(activation, "_is_hcu", lambda: True)
    monkeypatch.setattr(activation, "_install_profile_worker", lambda: None)
    monkeypatch.setenv("VLLM_HCU_DSV41_ENGRAM_HOST", "0")
    monkeypatch.setenv("HCU_VLLM_FP4_INDEXER_IMPL", "stock")
    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
    monkeypatch.setattr(envs, "environment_variables", dict(envs.environment_variables))
    envs.environment_variables.pop(activation._FACTOR, None)

    activation.activate_extension(ExtensionContext("0.5.22", "0.33.0"))
    assert _modes() == ("hcu", "hcu")
    assert pcp_fork.bound_forward_mqa_provider() is inverse_rope_pcp.prepare_forward_mqa
    assert pcp_fork._q_rope_provider is pcp_q_rope.initialize
    assert activation.policy_factor()[-2:] == ("hcu", "q_rope_hcu")


def test_explicit_override_and_per_operator_kill_switch(monkeypatch, native_route):
    native_route["vllm_inverse_rope"] = "stock"
    assert _modes() == ("stock", "hcu")
    monkeypatch.setenv(pcp_rope_policy.MODE_VARIABLE, "hcu")
    assert _modes() == ("hcu", "hcu")
    monkeypatch.setenv("HCU_DISABLE_OPS", pcp_fork.Q_ROPE_OP)
    assert _modes() == ("hcu", "stock")
    monkeypatch.setenv(pcp_rope_policy.MODE_VARIABLE, "stock")
    native_route["vllm_inverse_rope"] = "hcu"
    assert _modes() == ("stock", "stock")


@pytest.mark.parametrize("mode", ["auto", "hcu"])
@pytest.mark.parametrize("unavailable", ["arch", "pcp_impl", "pcp_model"])
def test_unsupported_routes_stay_on_stock(monkeypatch, native_route, mode, unavailable):
    monkeypatch.setenv(pcp_rope_policy.MODE_VARIABLE, mode)
    if unavailable == "arch":
        monkeypatch.setattr(pcp_rope_policy, "_process_arch", lambda _: "gfx936")
    elif unavailable == "pcp_impl":
        monkeypatch.setenv("HCU_VLLM_PCP_IMPL", "stock")
    else:
        monkeypatch.setenv("VLLM_HCU_DSV41_PCP_ATTENTION", "0")
    assert _modes() == ("stock", "stock")
