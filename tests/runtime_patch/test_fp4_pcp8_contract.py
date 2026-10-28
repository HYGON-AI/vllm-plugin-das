# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the narrowly admitted V4.1 FP4 PCP8 topology."""

from types import SimpleNamespace

import pytest

from vllm_hcu.patch.worker.core_fix import patch_deepseek_v41_indexer_k_layout as patch


_PCP_REASON = "FP4 PCP requires experimental eager MRV2 V4.1"
_PARALLEL_REASON = "FP4 requires TP1/PP1/DCP1"
_SPEC_REASON = "only non-adaptive DSpark(5) speculative decode is admitted"


def _config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=["DeepseekV41ForCausalLM"],
            use_mla=True,
            enforce_eager=True,
            hf_config=SimpleNamespace(model_type="deepseek_v41"),
            hf_text_config=SimpleNamespace(
                compress_ratios=[0, 1, 2],
                kv_source_layer_ids=[0, 2],
                index_source_layer_ids=[0, 1, 2],
            ),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            prefill_context_parallel_size=8,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=True,
            all2all_backend="deepep_high_throughput",
        ),
        speculative_config=SimpleNamespace(
            method="dspark",
            num_speculative_tokens=5,
            enable_adaptive_verification=False,
        ),
        use_v2_model_runner=True,
        kv_transfer_config=None,
    )


@pytest.fixture(autouse=True)
def _switches(monkeypatch):
    for name in (
        "VLLM_HCU_USE_CUSTOM_OPS",
        "VLLM_HCU_USE_FP4_INDEXER",
        "VLLM_HCU_DSV4_PCP_EXPERIMENTAL",
    ):
        monkeypatch.setenv(name, "1")


def test_admits_pcp8_ep_dspark_contract():
    assert patch._fp4_domain_blockers(SimpleNamespace(), _config()) == ()


@pytest.mark.parametrize(
    "section,name,value,expected_reason",
    [
        pytest.param("parallel_config", "decode_context_parallel_size", 8,
                     _PARALLEL_REASON, id="pcp8-dcp8"),
        pytest.param("parallel_config", "tensor_parallel_size", 2,
                     _PARALLEL_REASON, id="pcp8-tp2"),
        pytest.param("parallel_config", "pipeline_parallel_size", 2,
                     _PARALLEL_REASON, id="pcp8-pp2"),
        pytest.param("parallel_config", "prefill_context_parallel_size", 2,
                     _PCP_REASON, id="pcp2"),
        pytest.param("parallel_config", "data_parallel_size", 8,
                     _PCP_REASON, id="pcp8-dp8"),
        pytest.param("parallel_config", "enable_expert_parallel", False,
                     _PCP_REASON, id="pcp8-without-ep"),
        pytest.param("parallel_config", "all2all_backend", "deepep_low_latency",
                     _PCP_REASON, id="pcp8-wrong-all2all"),
        pytest.param("model_config", "enforce_eager", False,
                     _PCP_REASON, id="pcp8-without-eager"),
        pytest.param("model_config", "architectures", ["DeepseekV4ForCausalLM"],
                     _PCP_REASON, id="pcp8-wrong-model"),
        pytest.param("speculative_config", "method", None,
                     _PCP_REASON, id="pcp8-without-dspark"),
        pytest.param("speculative_config", "num_speculative_tokens", 4,
                     _SPEC_REASON, id="pcp8-dspark4"),
        pytest.param("speculative_config", "enable_adaptive_verification", True,
                     _SPEC_REASON, id="pcp8-adaptive-verification"),
    ],
)
def test_rejects_unproven_topology_or_compute_contract(
    section, name, value, expected_reason
):
    config = _config()
    assert patch._fp4_domain_blockers(SimpleNamespace(), config) == ()
    setattr(getattr(config, section), name, value)
    reasons = patch._fp4_domain_blockers(SimpleNamespace(), config)
    assert any(reason.startswith(expected_reason) for reason in reasons), reasons


def test_rejects_pcp_when_experimental_off(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_DSV4_PCP_EXPERIMENTAL", "0")
    reasons = patch._fp4_domain_blockers(SimpleNamespace(), _config())
    assert any(reason.startswith(_PCP_REASON) for reason in reasons), reasons


def test_rejects_pcp_without_mrv2():
    config = _config()
    config.use_v2_model_runner = False
    reasons = patch._fp4_domain_blockers(SimpleNamespace(), config)
    assert any(reason.startswith(_PCP_REASON) for reason in reasons), reasons


@pytest.mark.parametrize("dp", [1, 8], ids=["pcp1-dp1", "pcp1-dp8"])
def test_existing_pcp1_dp1_dp8_remain_admitted(dp):
    config = _config()
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.data_parallel_size = dp
    assert patch._fp4_domain_blockers(SimpleNamespace(), config) == ()


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_admits_fixed_mooncake_deployment(role):
    config = _config()
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="MooncakeConnector", kv_role=role
    )
    if role == "kv_consumer":
        config.parallel_config.prefill_context_parallel_size = 1
        config.parallel_config.data_parallel_size = 8
        config.parallel_config.all2all_backend = "deepep_low_latency"
    assert patch._fp4_domain_blockers(SimpleNamespace(), config) == ()


@pytest.mark.parametrize("connector,role", [
    ("MooncakeConnector", "kv_consumer"),
    ("MooncakeConnector", "kv_both"),
    ("OtherConnector", "kv_producer"),
])
def test_pd_gate_remains_limited_to_fixed_deployment(connector, role):
    config = _config()
    config.kv_transfer_config = SimpleNamespace(
        kv_connector=connector, kv_role=role
    )
    assert patch._fp4_domain_blockers(SimpleNamespace(), config)
