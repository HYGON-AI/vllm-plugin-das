# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS

import pytest
from hcu_vllm_das.models.hy4_standard import pd_policy as policy


def vl():
    return NS(
        model_type="hy_v4_vl",
        architectures=["HYV4VLForConditionalGeneration"],
        text_config=NS(model_type="hy_v4"),
    )


def parallel(role):
    return NS(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=16 if role == "kv_producer" else 1,
        data_parallel_size=1 if role == "kv_producer" else 16,
        data_parallel_rank=0,
        decode_context_parallel_size=1,
        enable_expert_parallel=True,
        all2all_backend="deepep_high_throughput",
        enable_dbo=False,
        enable_eplb=False,
        enable_elastic_ep=False,
    )


def test_exact_vl_identity_does_not_guess_plain_or_flash():
    assert policy.is_standard_vl_target(vl())
    for field, value in (
        ("model_type", "hy_v4"),
        ("architectures", ["OtherVL"]),
        ("text_config", NS(model_type="hy_v4_flash")),
    ):
        cfg = vl()
        setattr(cfg, field, value)
        assert not policy.is_standard_vl_target(cfg)


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_fixed_pd_topology(role):
    policy.validate_standard_pd_parallel(
        parallel(role), kv_role=role, enforce_eager=True
    )
    for field, value in (
        ("tensor_parallel_size", 2),
        ("pipeline_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("enable_expert_parallel", False),
        ("enable_dbo", True),
        ("enable_eplb", True),
        ("enable_elastic_ep", True),
        ("all2all_backend", "naive"),
    ):
        cfg = parallel(role)
        setattr(cfg, field, value)
        with pytest.raises(ValueError):
            policy.validate_standard_pd_parallel(cfg, kv_role=role, enforce_eager=True)
    with pytest.raises(ValueError):
        policy.validate_standard_pd_parallel(
            parallel(role), kv_role=role, enforce_eager=False
        )


@pytest.mark.parametrize(
    "n, committed", [(1, 0), (2, 1), (64, 63), (65, 64), (85000, 84999)]
)
def test_replay_leaves_exactly_last_original_prompt_token(n, committed):
    assert policy.committed_remote_prompt_tokens(n) == committed


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_invalid_prompt_count_rejected(n):
    with pytest.raises(ValueError):
        policy.committed_remote_prompt_tokens(n)


def test_replay_selected_only_for_actual_standard_dfly_consumer():
    cfg = NS(
        model_config=NS(hf_config=vl()),
        speculative_config=NS(method="dfly"),
        kv_transfer_config=NS(kv_role="kv_consumer"),
    )
    assert policy.requires_last_prompt_replay(cfg)
    cfg.kv_transfer_config.kv_role = "kv_producer"
    assert not policy.requires_last_prompt_replay(cfg)
    cfg.kv_transfer_config.kv_role = "kv_consumer"
    cfg.speculative_config.method = "mtp"
    assert not policy.requires_last_prompt_replay(cfg)


def test_inner_vl_marker_survives_target_init_but_not_derived_draft():
    assert policy.is_standard_vl_target(NS(model_type="hy_v4", hy4_standard_vl=True))
    assert not policy.is_standard_vl_target(
        NS(model_type="hy_v4_dfly", hy4_standard_vl=True)
    )
    assert not policy.is_standard_vl_target(NS(model_type="hy_v4", hy4_standard_vl=1))


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer", None])
def test_classify_exact_installed_connector_or_local_decode(role):
    cfg = NS(
        model_config=NS(hf_config=vl(), enforce_eager=True),
        parallel_config=parallel(role),
        kv_transfer_config=None,
    )
    if role:
        cfg.kv_transfer_config = NS(
            kv_role=role,
            kv_connector="MooncakeConnector",
            kv_connector_module_path=policy.DIRECT_MOONCAKE_MODULE,
        )
    assert policy.classify_standard_hy4_role(cfg) == (
        "producer" if role == "kv_producer" else "decode"
    )
    cfg.parallel_config.prefill_context_parallel_size = 16
    if role != "kv_producer":
        with pytest.raises(ValueError):
            policy.classify_standard_hy4_role(cfg)
    if role:
        cfg.kv_transfer_config.kv_connector_module_path = None
        with pytest.raises(ValueError, match="owned direct"):
            policy.classify_standard_hy4_role(cfg)
